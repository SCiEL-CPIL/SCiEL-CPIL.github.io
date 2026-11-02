import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    total_space = q ** n

    # Precompute positional powers
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    def int_to_tuple(val):
        res = [0] * n
        for i in range(n):
            res[i] = (val // powers[i]) % q
        return res

    # Build forbidden graph adjacency as Python ints (bitset)
    # Two vertices are adjacent if their Hamming distance > radius
    # For radius=1: adjacent if dist >= 2
    # We'll build the "compatible" graph (distance <= radius) and use its complement for MIS
    # But for B&B on MIS in the forbidden graph, we need neighbors = words at dist > radius
    
    # For efficiency with 2187 nodes, we can store adjacency as Python ints
    # adj[i] = bitset of j such that dist(i,j) > radius
    
    # First, build distance-based adjacency for the "covered" relation
    # covered_by[i] = bitset of j such that dist(i,j) <= radius
    covered_by = [0] * total_space
    
    # Precompute digit arrays for speed
    digits = np.zeros((total_space, n), dtype=np.int8)
    for i in range(total_space):
        for pos in range(n):
            digits[i, pos] = (i // powers[pos]) % q
    
    # For radius=1, dist <= 1 means same word or differ in exactly 1 position
    # Build covered_by efficiently
    for i in range(total_space):
        m = 1 << i
        for pos in range(n):
            dv = digits[i, pos]
            for nv in range(q):
                if nv == dv:
                    continue
                j = i + (nv - dv) * powers[pos]
                m |= (1 << j)
        covered_by[i] = m
    
    # Forbidden graph: adj_forbidden[i] = all nodes NOT in covered_by[i]
    full_mask = (1 << total_space) - 1
    adj_forbidden = [0] * total_space
    for i in range(total_space):
        adj_forbidden[i] = full_mask & ~covered_by[i]
    
    # Greedy seed for initial solution
    def greedy_cover():
        """Greedy set cover: repeatedly pick word covering most uncovered words."""
        remaining = full_mask
        selected = []
        while remaining and time.time() < deadline - 20:
            best_gain = -1
            best_word = -1
            # Sample-based greedy to save time
            # Try all words but with early termination heuristics
            for w in range(total_space):
                gain = (covered_by[w] & remaining).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_word = w
            if best_word == -1 or best_gain <= 0:
                break
            selected.append(best_word)
            remaining &= ~covered_by[best_word]
        return selected
    
    # Local search: swap-based improvement
    def local_search(selected, max_iters=500):
        """Try to remove or swap codewords to reduce count."""
        selected = list(selected)
        improved = True
        iterations = 0
        while improved and iterations < max_iters and time.time() < deadline - 10:
            improved = False
            iterations += 1
            # Try removal
            for i in range(len(selected)):
                w = selected[i]
                # Check if w is redundant
                others_cover = 0
                for j in range(len(selected)):
                    if j != i:
                        others_cover |= covered_by[selected[j]]
                if (covered_by[w] & ~others_cover) == 0:
                    selected.pop(i)
                    improved = True
                    break
            if improved:
                continue
            # Try swap: replace selected[i] with some w not in selected
            for i in range(len(selected)):
                w_old = selected[i]
                # Find best replacement
                others_cover = 0
                for j in range(len(selected)):
                    if j != i:
                        others_cover |= covered_by[selected[j]]
                # What does w_old uniquely cover?
                unique = covered_by[w_old] & ~others_cover
                if unique == 0:
                    # Can remove
                    selected.pop(i)
                    improved = True
                    break
                # Try to find w that covers unique and allows removal of another
                best_new = -1
                best_score = -1
                # Sample candidates
                candidates = list(range(total_space))
                random.shuffle(candidates)
                candidates = candidates[:min(200, total_space)]
                for w_new in candidates:
                    if w_new == w_old or w_new in selected:
                        continue
                    # Check if w_new covers unique
                    if (covered_by[w_new] & unique) != unique:
                        continue
                    # Check if adding w_new allows removing another codeword
                    new_selected = selected[:i] + [w_new] + selected[i+1:]
                    # Try to remove any redundant
                    for j in range(len(new_selected)):
                        if j == i:
                            continue
                        wj = new_selected[j]
                        others = 0
                        for k in range(len(new_selected)):
                            if k != j:
                                others |= covered_by[new_selected[k]]
                        if (covered_by[wj] & ~others) == 0:
                            best_new = w_new
                            best_score = 1  # can remove one
                            break
                    if best_new != -1:
                        break
                if best_new != -1:
                    selected[i] = best_new
                    # Now try to remove the redundant one
                    for j in range(len(selected)):
                        wj = selected[j]
                        others = 0
                        for k in range(len(selected)):
                            if k != j:
                                others |= covered_by[selected[k]]
                        if (covered_by[wj] & ~others) == 0:
                            selected.pop(j)
                            break
                    improved = True
                    break
        return selected
    
    # Branch and bound for MIS in forbidden graph
    # We want maximum independent set in forbidden graph = minimum vertex cover in covered graph
    # = minimum dominating set for the covering code problem
    
    best_solution = [None]
    best_size = [total_space + 1]
    
    def branch_and_bound(candidates, current_set, depth=0):
        """
        candidates: bitset of remaining candidate vertices
        current_set: list of vertices in current independent set
        """
        if time.time() > deadline - 5:
            return
        
        # Lower bound: |current_set| + greedy coloring bound on candidates
        if candidates == 0:
            if len(current_set) > best_size[0]:
                best_size[0] = len(current_set)
                best_solution[0] = list(current_set)
            return
        
        # Upper bound: |current_set| + popcount(candidates)
        upper_bound = len(current_set) + candidates.bit_count()
        if upper_bound <= best_size[0]:
            return
        
        # Greedy coloring to get a lower bound on the size of MIS in candidates
        # This gives an upper bound on how many more we can add
        # Actually, we need a lower bound on the remaining MIS size to prune
        # Use greedy coloring of the induced subgraph on candidates
        # The chromatic number is a lower bound on the clique number, but we need...
        # For MIS, we use: if we can color candidates with k colors, then MIS <= k
        # So upper bound = |current| + k
        
        def greedy_color_bound(cand):
            """Returns an upper bound on MIS size in cand using greedy coloring."""
            if cand == 0:
                return 0
            # Simple greedy: repeatedly pick a vertex, remove it and its neighbors
            # This gives an upper bound on MIS
            count = 0
            remaining = cand
            while remaining:
                # Pick a vertex with minimum degree in remaining (heuristic)
                # For speed, just pick the lowest set bit
                v_bit = remaining & -remaining
                v = v_bit.bit_length() - 1
                count += 1
                remaining &= ~adj_forbidden[v]
                remaining &= ~v_bit
            return count
        
        # Use a simpler bound: just popcount for now, refine if time permits
        # For 2187 nodes, this B&B may be too slow without strong bounds
        # Let's use a different approach: recursive with good branching
        
        # Pick a vertex to branch on
        # Heuristic: pick vertex with highest degree in candidates
        v_bit = candidates & -candidates
        v = v_bit.bit_length() - 1
        
        # Branch 1: include v
        # New candidates = candidates without v and without its neighbors
        new_cands_include = candidates & ~adj_forbidden[v] & ~v_bit
        current_set.append(v)
        branch_and_bound(new_cands_include, current_set, depth + 1)
        current_set.pop()
        
        # Branch 2: exclude v
        new_cands_exclude = candidates & ~v_bit
        branch_and_bound(new_cands_exclude, current_set, depth + 1)
    
    # The B&B above is too naive for 2187 nodes. Let's use a different strategy:
    # Start from greedy solution, then do local search with swaps
    
    # First, get a good initial solution via greedy
    initial = greedy_cover()
    
    # Apply local search
    result = local_search(initial, max_iters=200)
    
    # Try multiple restarts with different random seeds
    best_result = result
    best_len = len(result)
    
    while time.time() < deadline - 10:
        # Restart with perturbed greedy
        initial2 = greedy_cover()
        # Perturb: randomly remove some and re-greedy
        if len(initial2) > 10:
            perturbed = list(initial2)
            num_remove = max(1, len(perturbed) // 10)
            for _ in range(num_remove):
                if perturbed:
                    idx = random.randint(0, len(perturbed) - 1)
                    perturbed.pop(idx)
            # Re-greedy to fill gaps
            remaining = full_mask
            for w in perturbed:
                remaining &= ~covered_by[w]
            while remaining and time.time() < deadline - 10:
                best_gain = -1
                best_word = -1
                for w in range(total_space):
                    gain = (covered_by[w] & remaining).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_word = w
                if best_word == -1 or best_gain <= 0:
                    break
                perturbed.append(best_word)
                remaining &= ~covered_by[best_word]
        
        improved = local_search(perturbed, max_iters=100)
        if len(improved) < best_len:
            best_len = len(improved)
            best_result = improved
    
    return [int_to_tuple(v) for v in best_result]
