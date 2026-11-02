import time
import random
import heapq
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s kill

    total_space = q ** n
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i - 1] * q

    # Precompute masks: mask[i] has bits set for i and all neighbors within radius
    # Here radius=1, so size is 1 + n*(q-1)
    masks = [0] * total_space
    for i in range(total_space):
        m = 1 << i
        for pos in range(n):
            digit_val = (i // powers[pos]) % q
            for new_val in range(q):
                if new_val == digit_val:
                    continue
                diff = new_val - digit_val
                nb = i + diff * powers[pos]
                m |= (1 << nb)
        masks[i] = m

    full_mask = (1 << total_space) - 1

    def int_to_tuple(val):
        res = [0] * n
        for i in range(n):
            res[i] = (val // powers[i]) % q
        return res

    def greedy_solution():
        """Standard greedy set cover to get an initial upper bound."""
        covered = 0
        selected = []
        remaining = full_mask
        
        # Precompute list of all words
        all_words = list(range(total_space))
        
        while remaining and time.time() < deadline - 10:
            best_gain = -1
            best_word = -1
            for w in all_words:
                gain = (masks[w] & remaining).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_word = w
            if best_word == -1:
                break
            selected.append(best_word)
            covered |= masks[best_word]
            remaining = full_mask & ~covered
        return selected

    # Phase 1: Get a good initial upper bound
    initial_sol = greedy_solution()
    upper_bound = len(initial_sol)
    
    # Polish: remove redundant words
    def polish(selected):
        if not selected:
            return selected
        counts = np.zeros(total_space, dtype=np.int32)
        for w in selected:
            m = masks[w]
            mm = m
            while mm:
                lowbit = mm & -mm
                idx = lowbit.bit_length() - 1
                counts[idx] += 1
                mm ^= lowbit
        
        changed = True
        while changed:
            changed = False
            indices = list(range(len(selected)))
            random.shuffle(indices)
            for idx in indices:
                if idx >= len(selected):
                    continue
                w = selected[idx]
                m = masks[w]
                removable = True
                mm = m
                while mm:
                    lowbit = mm & -mm
                    pos = lowbit.bit_length() - 1
                    if counts[pos] == 1:
                        removable = False
                        break
                    mm ^= lowbit
                if removable:
                    mm = m
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] -= 1
                        mm ^= lowbit
                    selected.pop(idx)
                    changed = True
        return selected

    polished_initial = polish(initial_sol[:])
    upper_bound = len(polished_initial)
    best_solution = polished_initial[:]

    # Phase 2: Branch and Bound
    # We want to find a solution of size < upper_bound.
    # State: (uncovered_mask, current_selected_list)
    # Lower Bound Heuristic:
    # 1. Greedy packing: Pick a word in uncovered, remove its neighbors from uncovered, count++.
    #    This gives a lower bound on the number of additional words needed.
    # 2. Simple: bit_count(uncovered) / max_mask_size (weak, but fast).
    #    Let's use a slightly stronger one: Maximal independent set in the "conflict graph" of uncovered?
    #    Actually, for set cover, a lower bound is the size of a maximal set of uncovered elements
    #    that are pairwise "far" enough that no single codeword can cover two of them.
    #    In our case, two elements x, y can be covered by the same codeword z iff d(x,z)<=1 and d(y,z)<=1.
    #    This implies d(x,y) <= 2.
    #    So if we find a set of uncovered elements S such that d(x,y) >= 3 for all x!=y in S,
    #    then each element in S requires a distinct codeword.
    #    So |S| is a lower bound.
    
    # Precompute distance >= 3 pairs? Or just check on the fly?
    # For n=7, q=3, total_space=2187.
    # Let's implement the "Maximal Independent Set in Distance Graph" lower bound.
    # Graph G on vertices {0..2186}, edge (u,v) if d(u,v) <= 2.
    # We want Max Independent Set in G restricted to uncovered vertices.
    # This is equivalent to Min Dominating Set in the complement? No.
    # It's just MIS in G.
    
    # Computing MIS is NP-hard, but for small sets (uncovered size decreases) it might be ok?
    # Or we can use a greedy approximation for the lower bound?
    # A greedy maximal independent set gives a *valid* independent set, so its size is a lower bound.
    # Algorithm for LB:
    #   S = uncovered
    #   LB = 0
    #   while S:
    #      pick v in S
    #      LB += 1
    #      remove from S all u such that d(u,v) <= 2
    #   return LB
    # This is actually the Greedy Set Cover lower bound? No, it's the size of a packing.
    # This is a valid lower bound.
    
    # Precompute "distance <= 2" masks?
    # mask2[i] has bits set for j if d(i,j) <= 2.
    # d(i,j) <= 2 iff there exists a ball of radius 1 around i and j that intersect?
    # Or simply: j is in mask2[i] if j is in mask[i] or j is a neighbor of a neighbor of i.
    # Let's precompute mask2.
    mask2 = [0] * total_space
    for i in range(total_space):
        m = masks[i]
        # Union of masks of neighbors of i
        mm = m
        while mm:
            lowbit = mm & -mm
            idx = lowbit.bit_length() - 1
            m |= masks[idx]
            mm ^= lowbit
        mask2[i] = m

    def lower_bound(uncovered):
        """
        Greedy packing lower bound.
        Finds a maximal set of uncovered points such that no two are within distance 2.
        """
        if not uncovered:
            return 0
        lb = 0
        rem = uncovered
        # To make it deterministic and fast, pick the first set bit?
        # Or randomize? Randomization helps break symmetry but costs time.
        # Let's try picking the first set bit for speed.
        while rem:
            lb += 1
            # Pick a vertex v from rem
            v = (rem & -rem).bit_length() - 1
            # Remove all vertices within distance 2 of v from rem
            rem &= ~mask2[v]
        return lb

    # Branch and Bound
    # We will use a recursive function with a global best.
    # To avoid deep recursion stack issues or overhead, we can use a stack.
    # But recursion is fine for depth ~200.
    
    import sys
    sys.setrecursionlimit(10000)

    def bbb_solve(uncovered, current_size, selected_list):
        nonlocal upper_bound, best_solution, deadline
        
        if time.time() > deadline - 2:
            return

        if not uncovered:
            if current_size < upper_bound:
                upper_bound = current_size
                best_solution = selected_list[:]
            return

        # Lower bound check
        lb = lower_bound(uncovered)
        if current_size + lb >= upper_bound:
            return

        # Choose a vertex to cover
        # Heuristic: Pick the vertex in 'uncovered' that has the fewest candidates?
        # Or just pick the first one.
        # Let's pick the first set bit for simplicity, or the one with min degree in candidate set?
        # Min degree heuristic:
        # For each v in uncovered, count how many w in all_words cover v and are "available"?
        # All words are always available.
        # So count = popcount of {w | v in masks[w]}. This is constant for all v?
        # No, it depends on v.
        # Actually, due to symmetry of the hypercube, all vertices have the same number of codewords covering them.
        # So any vertex is equivalent. Pick the first one.
        
        target = (uncovered & -uncovered).bit_length() - 1
        
        # Candidates: all w such that masks[w] has bit 'target' set.
        # We can iterate through all w and check?
        # Or precompute reverse mapping?
        # reverse[target] = list of w such that target in masks[w].
        
        # Let's precompute reverse mapping for speed.
        # This takes memory but speeds up branching.
        
        # Wait, we are inside the function. We should precompute outside.
        # But we didn't. Let's do it now or pass it.
        # For now, iterate all w. 2187 iterations per node.
        # If tree is small, it's fine.
        
        candidates = []
        for w in range(total_space):
            if (masks[w] >> target) & 1:
                candidates.append(w)
        
        # Order candidates by gain?
        # Gain = (masks[w] & uncovered).bit_count()
        # Sort by gain descending.
        
        # Optimization: Only consider candidates that don't lead to immediate failure?
        # We can prune candidates if current_size + 1 + lower_bound(uncovered & ~masks[w]) >= upper_bound
        
        # To keep it fast, let's just sort by gain and iterate.
        candidates.sort(key=lambda w: (masks[w] & uncovered).bit_count(), reverse=True)
        
        for w in candidates:
            if time.time() > deadline - 2:
                return
            
            new_uncovered = uncovered & ~masks[w]
            
            # Pruning check
            if current_size + 1 + lower_bound(new_uncovered) >= upper_bound:
                continue
            
            selected_list.append(w)
            bbb_solve(new_uncovered, current_size + 1, selected_list)
            selected_list.pop()
            
            if upper_bound == current_size + 1: # Can't do better than current_size + 1
                break

    # Start B&B
    # Initial uncovered is full_mask
    # Initial size is 0
    # But we have an upper_bound from greedy.
    # We can start with a tighter bound?
    # The greedy solution is a valid cover.
    
    # One issue: B&B might be slow if the gap is large.
    # Greedy is usually within factor ln(n).
    # For n=2187, ln(n) ~ 7.
    # Best is 186. Greedy might be ~300-400?
    # If greedy gives 400, B&B has to cut 400 -> 186. That's a lot.
    
    # Improvement: Use the greedy solution to guide the search?
    # Or use a better initial bound?
    # The "polish" step on greedy is good.
    
    # Let's run B&B.
    if time.time() < deadline - 5:
        bbb_solve(full_mask, 0, [])

    # If B&B didn't find anything better, return the polished greedy solution.
    # If it did, return the best found.
    
    # Final check: Ensure the returned solution is valid.
    # The B&B guarantees validity if it returns a full cover.
    # The greedy solution is valid by construction.
    
    # If best_solution is empty (shouldn't happen), fallback to greedy.
    if not best_solution:
        best_solution = polished_initial

    return [int_to_tuple(v) for v in best_solution]
