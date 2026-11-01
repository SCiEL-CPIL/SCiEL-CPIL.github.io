import time
import numpy as np
from itertools import product
import random

def construct(q, n, radius):
    """
    Construct a covering code using a Maximum Independent Set (MIS) approach on a conflict graph.
    
    Strategy:
    1. Define a conflict graph G where vertices are words in {0..q-1}^n.
       An edge exists between u and v if their radius-1 balls overlap.
       Note: For radius 1, balls(u) and balls(v) overlap iff dist(u, v) <= 2.
       However, a standard MIS on this graph finds a set of words with pairwise distance > 2.
       This is NOT a covering code.
       
       Correction: The standard reduction for Set Cover to MIS is complex.
       A simpler structural change requested is to use MIS heuristics.
       
       Let's re-read the directive: "Formulate the problem as a maximum independent set (MIS) on a conflict graph where vertices are codewords and edges connect pairs with overlapping radius-1 balls... find a maximal independent set corresponding to a minimal covering code."
       
       Wait, an independent set in the "overlap graph" is a set of codewords whose balls are DISJOINT.
       If balls are disjoint, it's a packing, not a covering.
       
       Perhaps the intended graph is different?
       Usually, Covering Code is dual to Packing Code.
       If we want a *small* covering code, we want to *remove* redundant codewords.
       
       Let's look at the "Idea target": "Does operator 'error_repair' change the chance that a child improves its island?"
       
       The current program uses Greedy + Local Search (Remove redundant, Swap).
       The MIS approach usually maximizes the size of a set with no conflicts.
       Here, "conflict" might mean "redundancy".
       
       Let's interpret the directive strictly as a structural change:
       Instead of Greedy Set Cover, we use a Branch-and-Bound style search 
       that treats the selection of codewords as an optimization problem where we 
       try to maximize the number of "useful" codewords or minimize the total 
       while ensuring coverage, using pruning techniques typical of MIS solvers 
       (like Tomita's algorithm).
       
       However, a direct MIS formulation for Covering Code is non-trivial.
       A common heuristic for Set Cover that resembles MIS search is 
       "Greedy with Backtracking" or "Branch and Bound for Set Cover".
       
       Let's implement a Branch-and-Bound for Set Cover that uses:
       1. A lower bound on the number of additional sets needed (based on uncovered elements).
       2. Branching on elements (must be covered) or sets (include/exclude).
       3. Pruning if current_size + lower_bound >= best_size.
       
       This is structurally different from the pure heuristic local search in the seed.
       It explores the solution space more systematically with pruning, similar to MIS solvers.
    """
    start_time = time.time()
    deadline = start_time + 115  # 115s to leave margin

    # Total number of words in the space
    N = q ** n
    if N > 200000:
        return _constructive_cover(q, n, radius)

    # Generate all words
    all_words = np.array(list(product(range(q), repeat=n)), dtype=np.int32)
    num_words = len(all_words)
    
    # Map word index to integer value for quick lookup
    word_to_idx = {tuple(w): idx for idx, w in enumerate(all_words)}
    
    # Precompute balls as list of Python integers (bitsets)
    balls = [0] * num_words
    
    def get_idx(word_tuple):
        return word_to_idx.get(word_tuple, -1)

    for i in range(num_words):
        w = all_words[i]
        bits = 0
        bits |= (1 << i)
        
        if radius >= 1:
            for p in range(n):
                old_val = w[p]
                for new_val in range(q):
                    if new_val == old_val:
                        continue
                    new_w_list = list(w)
                    new_w_list[p] = new_val
                    j = get_idx(tuple(new_w_list))
                    if j != -1:
                        bits |= (1 << j)
        balls[i] = bits

    all_mask = (1 << num_words) - 1

    # Initial Greedy Cover to get a good upper bound (incumbent)
    def greedy_cover(deadline):
        selected = set()
        covered = 0
        while covered != all_mask and time.time() < deadline:
            best_gain = -1
            best_idx = -1
            for i in range(num_words):
                if i in selected:
                    continue
                gain = (balls[i] & ~covered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            
            if best_idx == -1 or best_gain == 0:
                break
            
            selected.add(best_idx)
            covered |= balls[best_idx]
        return selected, covered

    best_selected, best_covered = greedy_cover(deadline - 10)
    best_size = len(best_selected)
    
    # Branch and Bound for Set Cover
    # We want to find a cover of size < best_size.
    # State: (current_selected, current_covered, current_size)
    # Lower Bound: ceil(uncovered_count / max_ball_size) is weak.
    # Better Lower Bound: Greedy heuristic on the remaining uncovered set? Too slow.
    # Simple Lower Bound: If we have U uncovered words, and max coverage per word is C,
    # we need at least ceil(U/C) more words.
    
    max_ball_size = max(b.bit_count() for b in balls)
    
    # Sort words by ball size descending for better pruning? 
    # Or just use the natural order.
    
    # To make B&B effective, we need a strong lower bound.
    # A common one is the "greedy lower bound" but that's expensive.
    # Let's use a simple fractional packing bound or just the cardinality bound.
    
    # Since N=6561 and best_size is ~500, depth is ~500. 
    # B&B might be too slow without very strong pruning.
    # However, the directive asks for this structural change.
    
    # We will implement a limited-depth B&B or a best-first search with pruning.
    # Actually, for Set Cover, a "Best-First Search" with A* (using a heuristic lower bound) 
    # is a standard alternative to pure Greedy.
    
    # Heuristic h(s) = ceil(uncovered / max_remaining_coverage)
    
    import heapq
    
    # Priority Queue: (f_score, g_score, selected_tuple, covered_int)
    # f_score = g_score + h_score
    # g_score = len(selected)
    
    # Initial state
    start_state = (0, 0, frozenset(), 0)
    # (f, g, selected, covered)
    
    # h(0) = ceil(N / max_ball_size)
    h0 = (num_words + max_ball_size - 1) // max_ball_size
    start_f = 0 + h0
    
    pq = [(start_f, 0, start_state[2], start_state[3])]
    
    # To avoid memory explosion, we limit the number of states or time.
    # We will run until deadline or until we find a solution smaller than best_size.
    
    # Note: Storing frozensets of ~500 items in a PQ is memory heavy.
    # We can store the set as a tuple of indices.
    
    # Let's try a Depth-First Branch and Bound with restarts?
    # Or a simple B&B that branches on the "best" candidate at each step.
    
    # Given the time limit and Python overhead, a full B&B on 6561 elements is risky.
    # We will use a "Greedy with Backtracking" approach which is a form of B&B.
    
    # Alternative Interpretation:
    # The "MIS" idea might refer to finding a maximal independent set in the 
    # "conflict graph" of *redundant* codewords?
    # No, let's stick to the B&B for Set Cover as it is a standard structural 
    # alternative to Greedy/LS and fits the "branch-and-bound" description.
    
    # To make it fast, we will use a "Limited Discrepancy Search" (LDS) style 
    # or just a B&B that stops if it exceeds best_size.
    
    # Since Python B&B is slow, we will implement a hybrid:
    # 1. Greedy to get a good upper bound.
    # 2. Local Search to improve it (as in seed).
    # 3. If time remains, try B&B on a *reduced* problem or just report the LS result.
    # BUT the directive says "replace" or "structural change".
    
    # Let's implement the B&B carefully with strong pruning.
    # We will order candidates by gain (Greedy order).
    # At each step, we branch on the top K candidates (K=1 is Greedy, K>1 is LDS).
    
    # We will use LDS with discrepancy limit d=2 or 3.
    # This explores near-greedy paths.
    
    def lds_search(initial_selected, initial_covered, max_discrepancy, deadline):
        # This is a recursive search. 
        # To avoid recursion depth issues, we use a stack.
        # State: (selected, covered, depth, discrepancy_used, candidate_list)
        
        # This might be too complex to implement robustly in the short time.
        
        # Let's fallback to a simpler structural change:
        # "Simulated Annealing" or "Tabu Search" is also a structural change from Greedy+LS.
        # But the prompt specifically asked for MIS/B&B.
        
        # Let's implement a simple Branch and Bound that:
        # 1. Sorts all words by size (descending).
        # 2. Uses a recursive function `search(idx, selected, covered)`.
        # 3. Prunes if len(selected) + lower_bound(covered) >= best_size.
        # 4. Branches: Include word idx, or Exclude word idx.
        
        # To make it work in 120s for N=6561, we need very aggressive pruning.
        # The lower bound `ceil(uncovered / max_ball_size)` is weak.
        
        # Given the constraints, I will implement the B&B but with a very 
        # restricted search space (e.g., only considering words that appear in 
        # the greedy solution or have high potential).
        
        # Actually, the most robust "structural change" that is fast and 
        # different is **Greedy with Randomized Restarts and Tabu-like diversification**.
        # But let's try to honor the MIS/B&B prompt.
        
        # I will implement a **Branch and Bound** that operates on the 
        # **complement** of the problem? No.
        
        # Let's implement a **Best-First Search** with A* heuristic.
        # We will limit the number of nodes expanded.
        
        pass # Placeholder for logic below

    # Since a full B&B is likely to timeout or use too much memory in Python,
    # and the seed already has a strong LS, the "structural change" that is 
    # most likely to improve the score while respecting the "B&B/MIS" theme 
    # might be to use the B&B logic to *validate* or *refine* the LS solution,
    # or to use a **different initialization** based on MIS heuristics.
    
    # However, a true structural change is to replace the Greedy+LS with 
    # a **Genetic Algorithm** or **Simulated Annealing**.
    # Let's pivot to **Simulated Annealing** as it is a distinct optimization 
    # family from Greedy/LS and often outperforms simple LS on covering codes.
    # Wait, the prompt *specifically* directed: "Strategy to try in this change: 
    # Formulate the problem as a maximum independent set (MIS) ... solve it using 
    # a heuristic branch-and-bound".
    
    # I must follow the directive. I will implement a **Heuristic Branch-and-Bound** 
    # that is designed to be fast:
    # 1. Use Greedy to get an upper bound U.
    # 2. Use B&B to find a solution with size < U.
    # 3. The B&B will branch on the "most critical" uncovered word.
    
    def bnb_search(deadline):
        # We maintain a global best
        nonlocal best_selected, best_size
        
        # Pre-sort words by ball size descending
        sorted_indices = sorted(range(num_words), key=lambda i: balls[i].bit_count(), reverse=True)
        
        # We will use a stack for DFS
        # State: (current_selected_set, current_covered, next_idx_to_consider)
        # This is not efficient.
        
        # Efficient B&B for Set Cover:
        # 1. Pick an uncovered word w.
        # 2. Branch on all sets S that cover w.
        # 3. Recurse.
        
        # This is exponential.
        
        # Given the time, I will implement a **Limited Discrepancy Search (LDS)**
        # which is a form of B&B that is very efficient for Set Cover.
        
        def lds_rec(selected, covered, depth, discrepancy, candidate_order):
            if time.time() > deadline:
                return
            if len(selected) >= best_size:
                return
            
            if covered == all_mask:
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_selected = set(selected)
                return
            
            # Find the "best" candidate according to Greedy (max gain)
            best_gain = -1
            best_cand = -1
            second_best_gain = -1
            second_best_cand = -1
            
            # To speed up, we don't check all 6561 words every time.
            # We check a subset? No, we need the global best for LDS.
            # This is O(N) per node.
            
            for i in range(num_words):
                if i in selected:
                    continue
                gain = (balls[i] & ~covered).bit_count()
                if gain > best_gain:
                    if best_gain > second_best_gain:
                        second_best_gain = best_gain
                        second_best_cand = best_cand
                    best_gain = gain
                    best_cand = i
                elif gain > second_best_gain:
                    second_best_gain = gain
                    second_best_cand = i
            
            if best_cand == -1:
                return
            
            # Discrepancy 0: Take best
            # Discrepancy > 0: Take best, and if discrepancy allows, take second best
            
            # Branch 1: Always take best
            new_selected = set(selected)
            new_selected.add(best_cand)
            new_covered = covered | balls[best_cand]
            
            lds_rec(new_selected, new_covered, depth + 1, discrepancy, candidate_order)
            
            # Branch 2: If we have discrepancy budget, take second best
            if discrepancy > 0 and second_best_cand != -1:
                new_selected2 = set(selected)
                new_selected2.add(second_best_cand)
                new_covered2 = covered | balls[second_best_cand]
                
                lds_rec(new_selected2, new_covered2, depth + 1, discrepancy - 1, candidate_order)
                
            # To avoid infinite recursion and keep it bounded, 
            # we typically increase discrepancy limit in outer loops.
            # Here, we pass discrepancy as "remaining budget".
            # Standard LDS: Start with d=0 (Greedy), then d=1, d=2...
            # This function handles a specific path.
            
            # We need an outer loop to increase d.
            
        # Outer loop for LDS
        for d in range(0, 10): # Max discrepancy 9
            if time.time() > deadline:
                break
            # We need to reset the search for each d?
            # LDS is usually a single traversal with a limit.
            # Let's implement a proper LDS that explores all paths with <= d discrepancies.
            
            def lds_d(d_limit):
                # Stack: (selected, covered, path_discrepancies)
                # We need to know the "greedy choice" at each step to define discrepancy.
                # This requires re-computing the greedy choice at each node, which we do.
                
                stack = [(set(), 0, 0)]
                visited = set() # To avoid cycles? Sets are unique.
                
                while stack:
                    if time.time() > deadline:
                        return
                    selected, covered, disc = stack.pop()
                    
                    if len(selected) >= best_size:
                        continue
                        
                    if covered == all_mask:
                        if len(selected) < best_size:
                            best_size = len(selected)
                            best_selected = set(selected)
                        continue
                    
                    # Compute top 2 candidates
                    best_gain = -1
                    best_cand = -1
                    second_best_gain = -1
                    second_best_cand = -1
                    
                    for i in range(num_words):
                        if i in selected:
                            continue
                        gain = (balls[i] & ~covered).bit_count()
                        if gain > best_gain:
                            if best_gain > second_best_gain:
                                second_best_gain = best_gain
                                second_best_cand = best_cand
                            best_gain = gain
                            best_cand = i
                        elif gain > second_best_gain:
                            second_best_gain = gain
                            second_best_cand = i
                    
                    if best_cand == -1:
                        continue
                    
                    # Branch 1: Greedy choice (0 cost)
                    new_sel1 = set(selected)
                    new_sel1.add(best_cand)
                    new_cov1 = covered | balls[best_cand]
                    stack.append((new_sel1, new_cov1, disc))
                    
                    # Branch 2: Second best (1 cost)
                    if disc < d_limit and second_best_cand != -1:
                        new_sel2 = set(selected)
                        new_sel2.add(second_best_cand)
                        new_cov2 = covered | balls[second_best_cand]
                        stack.append((new_sel2, new_cov2, disc + 1))
                        
            lds_d(d)
            
    # Run LDS
    bnb_search(deadline)
    
    # Final Local Search to polish the B&B result
    def local_search(selected, covered, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            candidates = list(selected)
            random.shuffle(candidates)
            
            for i in candidates:
                if i not in selected:
                    continue
                
                new_covered = covered & ~balls[i]
                
                if new_covered == all_mask:
                    selected.remove(i)
                    covered = new_covered
                    improved = True
                    continue
                
                gap = all_mask & ~new_covered
                gap_count = gap.bit_count()
                
                if gap_count > 17:
                    continue
                
                found_swap = False
                for j in range(num_words):
                    if j in selected:
                        continue
                    if (gap & ~balls[j]) == 0:
                        selected.remove(i)
                        selected.add(j)
                        covered = new_covered | balls[j]
                        improved = True
                        found_swap = True
                        break
                
                if found_swap:
                    continue

        return selected, covered

    best_selected, best_covered = local_search(best_selected, best_covered, deadline)
    
    result = [all_words[i].tolist() for i in best_selected]
    return result


def _constructive_cover(q, n, radius):
    """Fallback constructive approach."""
    if q == 3 and n == 8:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in product(range(3), repeat=n - 4)]
    
    all_words = np.array(list(product(range(q), repeat=n)), dtype=np.int32)
    num_words = len(all_words)
    
    balls = np.zeros((num_words, num_words), dtype=bool)
    np.fill_diagonal(balls, True)
    for i in range(num_words):
        for j in range(num_words):
            if np.sum(all_words[i] != all_words[j]) <= radius:
                balls[i, j] = True
    
    selected = []
    covered = np.zeros(num_words, dtype=bool)
    while not covered.all():
        uncovered_count = balls[:, ~covered].sum(axis=1)
        best_idx = np.argmax(uncovered_count)
        if uncovered_count[best_idx] == 0:
            break
        selected.append(best_idx)
        covered |= balls[best_idx, :]
        
    return [all_words[i].tolist() for i in selected]
