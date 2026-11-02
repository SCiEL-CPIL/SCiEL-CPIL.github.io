"""Improved Greedy with Local Search for ternary covering codes K_3(7,1)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _precompute_neighbors(q, n, radius):
    """Precompute neighbors for all words."""
    N = q ** n
    if radius == 1:
        max_nbrs = 1 + n * (q - 1)
    else:
        max_nbrs = q ** n 

    nbrs_list = [None] * N
    
    if q == 3 and n == 7:
        # Unrolled fast path for specific case
        for idx in range(N):
            word = [0] * 7
            tmp = idx
            word[0] = tmp % 3; tmp //= 3
            word[1] = tmp % 3; tmp //= 3
            word[2] = tmp % 3; tmp //= 3
            word[3] = tmp % 3; tmp //= 3
            word[4] = tmp % 3; tmp //= 3
            word[5] = tmp % 3; tmp //= 3
            word[6] = tmp % 3; tmp //= 3
            
            nbrs = [idx]
            for pos in range(7):
                old = word[pos]
                for v in range(3):
                    if v != old:
                        new_word = word.copy()
                        new_word[pos] = v
                        new_idx = new_word[0] + new_word[1]*3 + new_word[2]*9 + new_word[3]*27 + new_word[4]*81 + new_word[5]*243 + new_word[6]*729
                        nbrs.append(new_idx)
            nbrs_list[idx] = np.array(nbrs, dtype=np.int32)
    else:
        for idx in range(N):
            word = [0] * n
            tmp = idx
            for i in range(n):
                word[i] = tmp % q
                tmp //= q
            nbrs = [idx]
            for pos in range(n):
                old = word[pos]
                for v in range(q):
                    if v != old:
                        new_word = word.copy()
                        new_word[pos] = v
                        new_idx = 0
                        mul = 1
                        for i in range(n):
                            new_idx += new_word[i] * mul
                            mul *= q
                        nbrs.append(new_idx)
            nbrs_list[idx] = np.array(nbrs, dtype=np.int32)
            
    return nbrs_list

def _greedy_init(nbrs_list, N, rng, time_limit, start_time):
    """Run randomized greedy to get an initial solution."""
    covered_count = np.zeros(N, dtype=np.int32)
    selected = []
    selected_set = set()
    
    max_nbrs = len(nbrs_list[0])
    nbrs_mat = np.zeros((N, max_nbrs), dtype=np.int32)
    for w in range(N):
        nbrs_mat[w, :len(nbrs_list[w])] = nbrs_list[w]
        
    while True:
        if time.time() - start_time > time_limit - 60.0: # Leave 60s for local search
            break
            
        is_uncovered = (covered_count == 0).astype(np.int32)
        gain = is_uncovered[nbrs_mat].sum(axis=1)
        
        max_gain = np.max(gain)
        if max_gain == 0:
            break
        
        candidates = np.where(gain == max_gain)[0]
        idx = rng.choice(candidates)
        
        selected.append(idx)
        selected_set.add(idx)
        
        nbrs_idx = nbrs_list[idx]
        covered_count[nbrs_idx] += 1
        
    return selected, covered_count

def _local_search(nbrs_list, N, selected, covered_count, time_limit, start_time):
    """Perform 2-1 exchange local search."""
    selected_set = set(selected)
    improved = True
    
    # Precompute reverse neighbors: for each point v, which codewords cover it?
    # This helps in finding candidates quickly if we have a small set of uncovered points.
    # But for 2-1 exchange, we remove 2, get uncovered U. We need c such that U subset N(c).
    # If |U| is small, we can just check all c not in selected_set?
    # N=2187 is small enough that checking all c for a given U is fast if |U| is small.
    # However, doing this for all pairs (i,j) is O(K^2 * N * |U|). K~190, N~2187.
    # 190^2 * 2187 is ~78 million operations. Might be tight in Python.
    # We can optimize by only trying swaps if the current solution is "tight".
    
    while improved and time.time() - start_time < time_limit - 2.0:
        improved = False
        K = len(selected)
        
        # 1. Redundancy Removal (1-0)
        # Find any w where removing it doesn't uncover anything.
        for i in range(K):
            w = selected[i]
            nbrs_w = nbrs_list[w]
            if not np.any(covered_count[nbrs_w] == 1):
                covered_count[nbrs_w] -= 1
                selected.pop(i)
                selected_set.remove(w)
                improved = True
                break
        
        if improved:
            continue
            
        # 2. 2-1 Exchange
        # Iterate over pairs. To speed up, we can limit the number of pairs checked per pass
        # or randomize the order.
        # Let's iterate deterministically but break early if time is low.
        
        # Optimization: If K is large, checking all pairs is slow.
        # We can prioritize pairs that cover "critical" points?
        # For now, simple iteration.
        
        for i in range(K):
            if time.time() - start_time > time_limit - 2.0:
                return selected, covered_count
                
            w1 = selected[i]
            nbrs_w1 = nbrs_list[w1]
            
            for j in range(i + 1, K):
                w2 = selected[j]
                nbrs_w2 = nbrs_list[w2]
                
                # Tentatively remove w1 and w2
                # We need to check if the union of their neighbors is covered by others.
                # Instead of decrementing and checking all N, we can just check the neighbors of w1 and w2.
                # A point v in N(w1) is safe if covered_count[v] > 1 (since w1 is removed, it needs another cover).
                # But if v is also in N(w2), and covered_count[v] was 2, after removing both it becomes 0.
                
                # Let's do the actual decrement to be safe, then rollback if no swap found.
                # To avoid modifying covered_count permanently until we find a swap, we can use a temporary check.
                # But modifying and rolling back is easier to code correctly.
                
                # Decrement
                covered_count[nbrs_w1] -= 1
                covered_count[nbrs_w2] -= 1
                
                # Find uncovered points
                uncovered = np.where(covered_count == 0)[0]
                
                if len(uncovered) == 0:
                    # We can just remove both? No, we must add back one?
                    # If removing both leaves it valid, we reduced size by 2.
                    # This is a great improvement.
                    selected.pop(j) # Remove larger index first
                    selected.pop(i)
                    selected_set.remove(w1)
                    selected_set.remove(w2)
                    improved = True
                    break
                elif len(uncovered) <= 15: # Max neighbors is 15, so a single codeword can cover at most 15 points
                    # Find a codeword c not in selected_set that covers all uncovered points
                    # c must be such that N(c) contains all points in 'uncovered'
                    found_swap = False
                    # We can iterate through all possible c?
                    # Or better: For each point v in uncovered, c must be in "covers[v]".
                    # Intersection of covers[v] for v in uncovered.
                    # Since we didn't precompute covers, let's just iterate c in range(N)
                    # and check if it covers all uncovered.
                    # To speed up, we can precompute a bitmask or just check.
                    # Given N=2187, and this loop runs for each pair, it might be slow.
                    # Let's try a heuristic: Only check c if it is "close" to w1 or w2?
                    # No, just iterate all c not in selected_set.
                    
                    # Optimization: If len(uncovered) == 1, we just need any c covering that point.
                    # If len(uncovered) > 1, it's harder.
                    
                    # Let's just iterate all c.
                    for c in range(N):
                        if c in selected_set:
                            continue
                        # Check if c covers all uncovered
                        # N(c) is nbrs_list[c]
                        # Check if set(uncovered) is subset of set(nbrs_list[c])
                        # Efficient check:
                        # Count how many uncovered are in nbrs_list[c]
                        # If count == len(uncovered), it's a match.
                        # Use numpy sum
                        if np.sum(covered_count[nbrs_list[c]] == 0) == len(uncovered):
                            # Wait, covered_count is modified (w1, w2 removed).
                            # So covered_count[v] == 0 means v is uncovered by remaining set.
                            # If nbrs_list[c] contains all these v, then adding c covers them.
                            # We also need to ensure c doesn't break anything else?
                            # Adding c only increases counts, so it's safe.
                            
                            # Perform swap
                            selected.pop(j)
                            selected.pop(i)
                            selected.append(c)
                            selected_set.remove(w1)
                            selected_set.remove(w2)
                            selected_set.add(c)
                            
                            # Update covered_count
                            covered_count[nbrs_list[c]] += 1
                            
                            improved = True
                            found_swap = True
                            break
                    
                    if found_swap:
                        break
                else:
                    # Rollback
                    covered_count[nbrs_w1] += 1
                    covered_count[nbrs_w2] += 1
                    continue

                # If we broke due to improvement, we need to rollback the decrement if we didn't perform the swap?
                # In the len(uncovered)==0 case, we removed them, so covered_count is correct (decremented).
                # In the swap case, we added c, so covered_count is correct (decremented w1,w2, incremented c).
                # In the no-swap case (loop finished without break), we must rollback.
                
                if not improved:
                    covered_count[nbrs_w1] += 1
                    covered_count[nbrs_w2] += 1
            
            if improved:
                break
                
    return selected, covered_count

def construct(q, n, radius):
    start_time = time.time()
    time_limit = 118.0
    
    nbrs_list = _precompute_neighbors(q, n, radius)
    N = q ** n
    
    best_code = None
    best_size = float('inf')
    
    # Run a few greedy initializations
    # We don't need 1000 runs, 5-10 should be enough to find a good start.
    for seed in range(10):
        if time.time() - start_time > time_limit - 60.0:
            break
            
        rng = np.random.RandomState(seed)
        selected, covered_count = _greedy_init(nbrs_list, N, rng, time_limit, start_time)
        
        # Local Search
        selected, covered_count = _local_search(nbrs_list, N, selected, covered_count, time_limit, start_time)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_code = selected.copy()
            
    # If time remains, we could try more local search on the best code?
    # The local search inside the loop already runs until time is low.
    
    # Convert indices to words
    res = []
    for idx in best_code:
        word = [0] * n
        tmp = idx
        for i in range(n):
            word[i] = tmp % q
            tmp //= q
        res.append(word)
        
    return res
# EVOLVE-BLOCK-END
