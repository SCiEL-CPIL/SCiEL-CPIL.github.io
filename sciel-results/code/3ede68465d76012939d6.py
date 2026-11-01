"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    if n == 7 and q == 3 and radius == 1:
        N = 3**7
        # Precompute all possible neighbors (radius 1 ball) for each word index
        # We use base-q representation.
        # To speed up, we precompute the list of neighbors for each index? 
        # That would be N * (1 + n*(q-1)) integers. 
        # For N=2187, n=7, q=3: 2187 * (1 + 7*2) = 2187 * 15 = 32805 integers.
        # This is small.
        
        # Generate all words as indices 0..N-1
        # Convert index to digits
        # We need to map index <-> word
        
        # Let's represent words as indices 0..N-1.
        # Neighbor calculation:
        # For a given index, get digits. For each position i, for each val != digit[i],
        # compute new index.
        
        # Precomputation
        neighbors = np.empty((N, 1 + n*(q-1)), dtype=np.int32)
        for idx in range(N):
            # Get digits
            digits = [0]*n
            temp = idx
            for i in range(n-1, -1, -1):
                digits[i] = temp % q
                temp //= q
            
            nbrs = [idx]
            for i in range(n):
                for val in range(q):
                    if val != digits[i]:
                        # Compute new index
                        # new_idx = idx - digits[i]*q^i + val*q^i
                        pow_q = q**i
                        new_idx = idx - digits[i]*pow_q + val*pow_q
                        nbrs.append(new_idx)
            
            neighbors[idx] = nbrs
        
        # Initial solution: Greedy Set Cover
        # uncovered: boolean array of size N
        uncovered = np.ones(N, dtype=bool)
        # We want to pick a set of codewords.
        # Greedy: pick the word that covers the most uncovered words.
        
        # To make this fast, we can maintain a count of how many uncovered neighbors each word has.
        # cover_count[c] = number of uncovered words in ball(c).
        
        # Initial cover_count
        # For each word c, cover_count[c] = sum(uncovered[neighbors[c]])
        # Since initially all uncovered, cover_count[c] = 1 + n*(q-1) for all c.
        # But as we pick words, we need to update.
        
        # Let's use a different representation for SA:
        # We fix the number of codewords, M.
        # Start with a greedy solution of size M0.
        # Then try to reduce M or improve coverage with same M?
        # No, the objective is to minimize M such that coverage is 100%.
        
        # Strategy:
        # 1. Run greedy to get a valid covering set S_greedy.
        # 2. M = len(S_greedy).
        # 3. Use SA to try to find a covering set of size M-1, M-2, etc.?
        #    Or, fix M and try to minimize uncovered?
        #    Better: Start with a slightly larger set, and try to remove words?
        #    Or: Start with a random set of size M, and use SA to minimize uncovered.
        #    If uncovered becomes 0, we have a valid code of size M.
        #    Then try M-1.
        
        # Let's estimate M from greedy.
        
        # Greedy Implementation
        # We need efficient updates.
        # When we add word c to the code:
        # For each neighbor x of c:
        #   If x was uncovered:
        #     Mark x as covered.
        #     For each codeword d in the current code (and c itself):
        #       If x is a neighbor of d, decrement cover_count[d].
        
        # This is O(N * M * BallSize) which might be slow.
        # Alternative:
        # Maintain an array `uncovered_by` or just `uncovered` bool.
        # To find the best next word, we need max cover_count.
        # Updating cover_count for all d when x becomes covered is expensive.
        
        # Let's try a simpler greedy first to get M.
        # Then use SA on a fixed M.
        
        # Greedy:
        # uncovered = np.ones(N, dtype=bool)
        # counts = np.full(N, 1 + n*(q-1), dtype=np.int32)
        # code = []
        
        # For each step:
        #   c = argmax(counts)
        #   code.append(c)
        #   for x in neighbors[c]:
        #       if uncovered[x]:
        #           uncovered[x] = False
        #           # Update counts for all d that cover x
        #           # This requires knowing which d cover x.
        #           # This is the dual problem.
        
        # Given time constraints, maybe a simpler heuristic for initialization is enough,
        # and the SA does the heavy lifting.
        
        # Let's initialize with a random set of size M_est.
        # M_est = N / (1 + n*(q-1)) is the lower bound.
        # Greedy usually gets close to optimal.
        
        # Let's run a fast greedy using bitsets if possible?
        # N=2187 fits in 2187 bits. Python ints can handle this.
        # But numpy is faster for vector ops.
        
        # Let's try a simple greedy with numpy.
        # It might be slow if not optimized, but N is small.
        
        # Optimization:
        # Precompute `covers` matrix? N x N boolean? 2187^2 = 4.7M booleans. 4.7 MB.
        # This is feasible!
        # covers[i, j] = 1 if j is in ball(i).
        
        print("Precomputing cover matrix...", flush=True)
        covers = np.zeros((N, N), dtype=np.bool_)
        for i in range(N):
            covers[i, neighbors[i]] = True
            
        # Now Greedy is fast:
        # uncovered_mask = np.ones(N, dtype=bool)
        # remaining_cover = covers.sum(axis=1) # Initially
        # Actually, remaining_cover[i] = sum(covers[i, :][uncovered_mask])
        
        uncovered_mask = np.ones(N, dtype=bool)
        code = []
        
        # To speed up greedy, we can just iterate.
        # At each step, find i that maximizes sum(covers[i, :][uncovered_mask])
        
        # This is O(N^2 * steps).
        # Steps ~ M ~ 200. N=2187. 200 * 4.7M = 1B ops. Too slow in Python/NumPy loops?
        # NumPy sum is fast.
        
        # Let's try.
        
        for _ in range(1000): # Max iterations
            if not uncovered_mask.any():
                break
            # Calculate coverage for each candidate
            # cov = covers[:, :].T @ uncovered_mask.astype(np.int8)
            # This is matrix multiplication. N x N @ N x 1.
            # Fast in BLAS.
            cov = covers.T.dot(uncovered_mask.astype(np.int8))
            best_idx = np.argmax(cov)
            
            if cov[best_idx] == 0:
                break # No improvement possible (should not happen if uncovered exists)
                
            code.append(best_idx)
            # Update uncovered_mask
            uncovered_mask[covers[best_idx, :]] = False
            
            if time.time() > deadline - 30:
                break # Leave time for SA
        
        M_greedy = len(code)
        print(f"Greedy found {M_greedy} words", flush=True)
        
        # Now SA to improve
        # We want to find a code of size M < M_greedy?
        # Or just minimize size.
        
        # Let's fix M = M_greedy - 1 and see if we can cover everything.
        # If yes, try M-2, etc.
        
        best_code = code
        best_size = M_greedy
        
        # SA Parameters
        # State: a set of M indices.
        # We represent state as a boolean array `in_code` of size N.
        # Or a list of indices.
        # To make SA fast, we need to evaluate energy (uncovered count) quickly.
        
        # Energy = sum(uncovered_mask)
        # uncovered_mask = ~ (OR of covers[i] for i in code)
        
        # Let's use a count array `cover_counts` of size N.
        # cover_counts[x] = number of codewords that cover x.
        # uncovered = sum(cover_counts == 0)
        
        # When we add/remove a codeword c:
        # For each x in neighbors[c]:
        #   cover_counts[x] += 1 or -= 1
        #   if cover_counts[x] becomes 0, uncovered_count += 1
        #   if cover_counts[x] becomes 1 (from 0), uncovered_count -= 1
        
        # This update is O(BallSize) = O(15). Very fast!
        
        def sa_search(M, time_limit=10.0):
            start = time.time()
            
            # Initialize with random M words
            # Or initialize with greedy result truncated?
            # Let's try random first, maybe greedy truncated is better.
            # Greedy code is size M_greedy. If M < M_greedy, we can't just truncate.
            # Let's start with a random set.
            
            # Actually, starting from a valid covering set of size M_greedy and removing words is a good strategy?
            # No, we want to find a set of size M.
            
            # Let's try multiple restarts.
            
            best_energy = N
            best_state = None
            
            for restart in range(20):
                if time.time() > time_limit + start:
                    break
                
                # Init: Random M words
                code_set = set(np.random.choice(N, size=M, replace=False))
                
                cover_counts = np.zeros(N, dtype=np.int8)
                for c in code_set:
                    cover_counts[neighbors[c]] += 1
                
                uncovered_count = np.sum(cover_counts == 0)
                
                if uncovered_count == 0:
                    return list(code_set)
                
                current_energy = uncovered_count
                T = 50.0
                T_min = 0.1
                cooling = 0.999
                
                while time.time() < time_limit + start:
                    if uncovered_count == 0:
                        return list(code_set)
                        
                    # Pick a random move:
                    # 1. Remove a random word from code_set
                    # 2. Add a random word not in code_set
                    # 3. Swap
                    # Let's do Add/Remove to keep size M fixed?
                    # If we keep size M fixed, we must Swap.
                    
                    # Swap: Remove c_old, Add c_new
                    c_old = np.random.choice(list(code_set))
                    c_new = np.random.choice(N)
                    while c_new in code_set:
                        c_new = np.random.choice(N)
                    
                    # Calculate delta energy
                    # This requires updating counts.
                    # Let's simulate.
                    
                    # We can precompute neighbors.
                    # Delta for c_old:
                    # For x in neighbors[c_old]:
                    #   if cover_counts[x] == 1: becomes 0 -> +1 energy
                    #   if cover_counts[x] > 1: becomes >0 -> 0 energy change
                    # Delta for c_new:
                    # For x in neighbors[c_new]:
                    #   if cover_counts[x] == 0: becomes 1 -> -1 energy
                    #   if cover_counts[x] > 0: becomes >0 -> 0 energy change
                    # Note: if x is in both neighbors, effects might cancel or compound?
                    # No, we update counts.
                    # Let's just do the update and rollback if rejected.
                    
                    # Optimization:
                    # We can compute delta quickly.
                    # But rollback is easy.
                    
                    # Apply
                    for x in neighbors[c_old]:
                        cover_counts[x] -= 1
                    for x in neighbors[c_new]:
                        cover_counts[x] += 1
                    
                    # Recalculate uncovered count?
                    # We can track it incrementally.
                    # But it's easier to just count zeros in cover_counts?
                    # N=2187. np.sum(cover_counts == 0) is fast.
                    
                    new_energy = np.sum(cover_counts == 0)
                    delta = new_energy - current_energy
                    
                    if delta < 0 or np.random.rand() < np.exp(-delta / T):
                        # Accept
                        code_set.remove(c_old)
                        code_set.add(c_new)
                        current_energy = new_energy
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_state = set(code_set)
                    else:
                        # Reject: Rollback
                        for x in neighbors[c_old]:
                            cover_counts[x] += 1
                        for x in neighbors[c_new]:
                            cover_counts[x] -= 1
                    
                    T *= cooling
                    if T < T_min:
                        T = T_min
                        
            return list(best_state) if best_state else None

        # Try to reduce size
        current_M = M_greedy
        while current_M > 150: # Lower bound is 156
            if time.time() > deadline - 10:
                break
            
            print(f"Trying SA with M={current_M}", flush=True)
            # Allocate time based on remaining budget
            time_left = deadline - time.time() - 10
            t_alloc = min(10.0, time_left / 5.0)
            
            sol = sa_search(current_M, time_limit=t_alloc)
            
            if sol is not None:
                # Verify
                cover_counts = np.zeros(N, dtype=np.int8)
                for c in sol:
                    cover_counts[neighbors[c]] += 1
                if np.sum(cover_counts == 0) == 0:
                    print(f"Found valid code of size {current_M}", flush=True)
                    best_code = sol
                    best_size = current_M
                    current_M -= 1
                else:
                    break # If SA failed to find valid code, larger M might be needed? 
                          # But we are decreasing M. If M fails, M-1 is harder.
                          # So we should stop decreasing.
            else:
                break
                
        # Convert indices back to words
        final_words = []
        for idx in best_code:
            digits = [0]*n
            temp = idx
            for i in range(n-1, -1, -1):
                digits[i] = temp % q
                temp //= q
            final_words.append(digits)
            
        return final_words

    # Fallback for other parameters
    # Simple greedy
    N = q**n
    # ... (implement generic fallback if needed, but task is specific)
    return []
# EVOLVE-BLOCK-END
