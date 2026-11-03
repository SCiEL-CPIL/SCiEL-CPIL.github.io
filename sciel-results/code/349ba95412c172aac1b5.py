import time
import random
import numpy as np
from itertools import product

def construct(q, n, radius):
    total_words = q ** n
    
    def idx_to_tuple(idx):
        t = []
        for _ in range(n):
            t.append(idx % q)
            idx //= q
        return tuple(reversed(t))
    
    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    # Precompute neighbors as numpy arrays for vectorized operations
    cover_count = 1 + n * (q - 1)
    neighbors_arr = np.zeros((total_words, cover_count), dtype=np.int32)
    for i in range(total_words):
        w = idx_to_tuple(i)
        idxs = [i]
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                new_w = list(w)
                new_w[pos] = new_val
                j = tuple_to_idx(tuple(new_w))
                idxs.append(j)
        for j, idx in enumerate(idxs):
            neighbors_arr[i, j] = idx

    # Precompute reverse mapping: for each word, which codewords cover it
    reverse_cover_arr = np.zeros((total_words, cover_count), dtype=np.int32)
    reverse_cover_counts = np.zeros(total_words, dtype=np.int32)
    for i in range(total_words):
        for j in range(cover_count):
            w = neighbors_arr[i, j]
            reverse_cover_arr[w, reverse_cover_counts[w]] = i
            reverse_cover_counts[w] += 1

    def greedy_construct_vectorized(random_seed):
        if random_seed is not None:
            random.seed(random_seed)
        
        uncovered = np.ones(total_words, dtype=np.bool_)
        code = []
        
        while uncovered.any():
            # Vectorized gain computation
            uncovered_expanded = uncovered[neighbors_arr]
            gains = uncovered_expanded.sum(axis=1)
            
            valid = gains > 0
            if not valid.any():
                break
            
            max_gain = gains[valid].max()
            best_candidates = np.where((gains == max_gain) & valid)[0]
            
            chosen = random.choice(best_candidates.tolist())
            code.append(chosen)
            uncovered[neighbors_arr[chosen, :]] = False
        
        return code

    def reduce_code_fast(code):
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        
        changed = True
        while changed:
            changed = False
            new_code = []
            for c in code:
                counts[neighbors_arr[c, :]] -= 1
                if np.any(counts == 0):
                    counts[neighbors_arr[c, :]] += 1
                    new_code.append(c)
                else:
                    changed = True
            code = new_code
        return code

    def branch_and_bound(uncovered_mask, current_code, best_size_ref, deadline):
        # uncovered_mask: boolean array of size total_words, True if uncovered
        # current_code: list of chosen codewords
        # best_size_ref: [best_size_found]
        
        if time.time() > deadline:
            return
        
        if not uncovered_mask.any():
            if len(current_code) < best_size_ref[0]:
                best_size_ref[0] = len(current_code)
            return

        # Lower Bound: Greedy packing
        # Find a set of vertices in U such that their neighborhoods are disjoint.
        # This is an independent set in the "conflict graph" where edges connect vertices with overlapping neighborhoods?
        # Simpler LB: ceil(|U| / max_cover)
        num_uncovered = np.sum(uncovered_mask)
        # Max coverage for a single node is cover_count, but in U it's less.
        # A safe LB is 1 if U is not empty.
        # Better LB: 
        # If we pick a vertex v in U, it must be covered by one of its neighbors.
        # Let's use a simple packing bound.
        # We can greedily pick vertices from U such that their N(v) are disjoint.
        # If we find k such vertices, we need at least k more codewords.
        
        # For speed, let's just use num_uncovered / cover_count as a weak bound, 
        # or try to find a small packing.
        
        # Let's try a simple packing:
        # Pick v in U. Add to packing. Remove N(v) from consideration for packing.
        # This is O(N^2) worst case, but N=2187. 
        # Let's limit packing search to first 100 vertices to save time.
        
        temp_uncovered = uncovered_mask.copy()
        packing_size = 0
        # To make it faster, we can just estimate.
        # Actually, a very strong and fast bound for set cover is:
        # Sort elements by frequency.
        # But let's stick to the simple one first.
        
        if len(current_code) + 1 >= best_size_ref[0]:
             return

        # Select an uncovered vertex to branch on.
        # Heuristic: Pick the one with fewest candidates (degree in the bipartite graph).
        # Candidates for v are neighbors of v.
        # The number of candidates is fixed (cover_count) for all v in regular graph.
        # So pick any uncovered vertex.
        uncovered_indices = np.where(uncovered_mask)[0]
        # Randomize to avoid bias? Or pick the one with max gain?
        # Let's pick the one with max gain in current U.
        gains = np.sum(uncovered_mask[neighbors_arr], axis=1)
        # Only consider neighbors of uncovered nodes? No, any node can cover.
        # But nodes with 0 gain in U are useless.
        valid_nodes = np.where(gains > 0)[0]
        
        if len(valid_nodes) == 0:
             return # Should not happen if U is not empty and graph is connected
        
        # Pick the node with max gain? Or min gain?
        # In Set Cover B&B, we usually branch on the *element* with fewest sets covering it.
        # Here all elements are covered by the same number of sets (regular graph).
        # So we branch on a specific element v, and try all sets c that cover v.
        
        # Let's pick a random uncovered element v.
        v = random.choice(uncovered_indices.tolist())
        
        # Candidates are neighbors of v.
        candidates = neighbors_arr[v, :].tolist()
        
        # Sort candidates by how many NEW vertices they cover (gain in U)
        # gain(c) = |N(c) intersect U|
        # We already have gains array.
        # Sort candidates by gains[c] descending.
        candidates.sort(key=lambda c: gains[c], reverse=True)
        
        for c in candidates:
            if time.time() > deadline:
                return
            
            if len(current_code) + 1 >= best_size_ref[0]:
                continue
                
            # Check if c is already in current_code? 
            # In Set Cover, sets are distinct. Codewords are distinct.
            # If c is already in current_code, it's already applied.
            # But we are branching on "which set covers v".
            # If c is already in current_code, then v would be covered.
            # Since v is uncovered, c cannot be in current_code.
            
            new_code = current_code + [c]
            
            # Update uncovered mask
            # We need to remove N(c) from uncovered_mask
            # Since we are recursing, we should pass a copy or modify and revert.
            # Copying 2187 bools is fast.
            new_uncovered = uncovered_mask.copy()
            new_uncovered[neighbors_arr[c, :]] = False
            
            branch_and_bound(new_uncovered, new_code, best_size_ref, deadline)

    start_time = time.time()
    deadline = start_time + 115
    
    best_code = None
    best_size = float('inf')
    
    # 1. Get a good initial upper bound using greedy + reduction
    init_code = greedy_construct_vectorized(0)
    init_code = reduce_code_fast(init_code)
    best_size = len(init_code)
    best_code = init_code
    best_size_ref = [best_size]
    
    # 2. Run Branch and Bound to improve
    # Start with empty code and full uncovered set
    full_uncovered = np.ones(total_words, dtype=np.bool_)
    
    # To make B&B more effective, we can start with the greedy solution as the "current" state?
    # No, B&B explores the space. 
    # However, standard B&B from scratch might be slow if the bound is weak.
    # Let's run B&B from scratch but with the strong upper bound from greedy.
    
    # Optimization: 
    # The B&B above is a depth-first search. 
    # We can add a timeout check.
    
    # Let's run it.
    branch_and_bound(full_uncovered, [], best_size_ref, deadline)
    
    # If B&B found a smaller size, we need the actual code.
    # The function above only updates the size. We need to modify it to store the best code.
    
    # Let's rewrite the B&B to store the best code.
    
    def branch_and_bound_v2(uncovered_mask, current_code, state, deadline):
        if time.time() > deadline:
            return
        
        if not uncovered_mask.any():
            if len(current_code) < state['best_size']:
                state['best_size'] = len(current_code)
                state['best_code'] = list(current_code)
            return

        if len(current_code) >= state['best_size']:
            return

        # Lower Bound
        # Simple: ceil(|U| / max_gain_in_U)
        # Let's compute max gain in U quickly? 
        # gains = sum(uncovered_mask[neighbors_arr], axis=1)
        # max_gain = gains.max()
        # lb = ceil(|U| / max_gain)
        # This is O(N * cover_count) which is ~2187 * 14 ~ 30k ops. Acceptable.
        
        gains = np.sum(uncovered_mask[neighbors_arr], axis=1)
        max_gain = gains.max()
        if max_gain == 0:
             return
        num_uncovered = np.sum(uncovered_mask)
        lb = (num_uncovered + max_gain - 1) // max_gain
        
        if len(current_code) + lb >= state['best_size']:
            return

        # Branching
        # Pick uncovered vertex v
        uncovered_indices = np.where(uncovered_mask)[0]
        v = random.choice(uncovered_indices.tolist())
        
        # Candidates: neighbors of v
        candidates = neighbors_arr[v, :].tolist()
        
        # Sort by gain in U
        candidates.sort(key=lambda c: gains[c], reverse=True)
        
        for c in candidates:
            if time.time() > deadline:
                return
            
            if len(current_code) + 1 >= state['best_size']:
                continue

            new_code = current_code + [c]
            
            new_uncovered = uncovered_mask.copy()
            new_uncovered[neighbors_arr[c, :]] = False
            
            branch_and_bound_v2(new_uncovered, new_code, state, deadline)

    state = {'best_size': best_size, 'best_code': best_code}
    branch_and_bound_v2(full_uncovered, [], state, deadline)
    
    # 3. Final local search on the best code found
    if time.time() < deadline - 5:
        final_code = reduce_code_fast(state['best_code'])
        # Simple local search improvement
        counts = np.zeros(total_words, dtype=np.int16)
        for c in final_code:
            counts[neighbors_arr[c, :]] += 1
        
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            indices = list(range(len(final_code)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() >= deadline - 1:
                    break
                
                c_remove = final_code[i]
                counts[neighbors_arr[c_remove, :]] -= 1
                
                if np.any(counts == 0):
                    counts[neighbors_arr[c_remove, :]] += 1
                    # Try to swap
                    critical_words = neighbors_arr[c_remove, :][counts[neighbors_arr[c_remove, :]] == 1]
                    if len(critical_words) == 0:
                        continue
                    
                    candidate_set = None
                    for cw in critical_words:
                        rc = reverse_cover_arr[cw, :reverse_cover_counts[cw]]
                        if candidate_set is None:
                            candidate_set = set(rc.tolist())
                        else:
                            candidate_set &= set(rc.tolist())
                    
                    if candidate_set is None or len(candidate_set) == 0:
                        continue
                    
                    candidate_set.discard(c_remove)
                    if len(candidate_set) == 0:
                        continue
                    
                    best_w = random.choice(list(candidate_set))
                    final_code[i] = best_w
                    counts[neighbors_arr[c_remove, :]] -= 1
                    counts[neighbors_arr[best_w, :]] += 1
                    improved = True
                    break
                else:
                    final_code.pop(i)
                    improved = True
                    break
        
        if len(final_code) < len(state['best_code']):
            state['best_code'] = final_code

    result = []
    for idx in state['best_code']:
        w = idx_to_tuple(idx)
        result.append(list(w))
    
    return result
