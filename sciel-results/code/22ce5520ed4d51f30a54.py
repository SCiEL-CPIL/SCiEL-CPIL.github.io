import numpy as np
import time

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    # Precompute all arithmetic progressions as index lists (0-based)
    # An AP: start a (0-based), step s >= 1, length k >= 1
    # indices: a, a+s, a+2s, ..., a+(k-1)s where a+(k-1)s < n
    progs = []
    for a in range(n):
        for s in range(1, n):
            idx = list(range(a, n, s))
            if len(idx) >= 1:
                progs.append(idx)
    
    Np = len(progs)
    
    # Build incidence matrix M: M[p, i] = 1 if position i is in AP p
    M = np.zeros((Np, n), dtype=np.int8)
    for pid, idx in enumerate(progs):
        for i in idx:
            M[pid, i] = 1
    
    # Convert to float for faster matrix-vector ops if needed, but int8 is fine for logic
    # Actually, for the vectorized flip evaluation, we need to compute:
    # new_sums[i] = S + 2 * s[i] * M[:, i]
    # This is S[:, None] + 2 * s[None, :] * M
    # M is (Np, n), s is (n,), so s[None, :] * M is (Np, n)
    
    rng = np.random.RandomState(42)
    
    best_signs = None
    best_score = n  # Upper bound
    
    def compute_max_abs_sum(signs):
        """Compute max |sum| over all APs for a given sign vector."""
        sums = M @ signs  # (Np,)
        return int(np.max(np.abs(sums)))
    
    def evaluate_all_flips(signs, current_sums):
        """
        For each position i, compute the new max_abs_sum if we flip signs[i].
        Returns array of shape (n,) with the new max abs sum for each flip.
        """
        # current_sums: (Np,)
        # signs: (n,)
        # M: (Np, n)
        
        # If we flip position i, the change in sum for AP p is:
        # delta_p = -2 * signs[i] if M[p, i] == 1 else 0
        # Wait, flipping means new_sign = -signs[i].
        # Old contribution: signs[i]. New contribution: -signs[i].
        # Change: -2 * signs[i].
        # So new_sums_p = current_sums_p - 2 * signs[i] * M[p, i]
        
        # Vectorized:
        # changes = 2 * signs[None, :] * M  # (Np, n), element (p, i) is 2 * signs[i] * M[p,i]
        # new_sums = current_sums[:, None] - changes  # (Np, n)
        # new_maxes = np.max(np.abs(new_sums), axis=0)  # (n,)
        
        # To save memory/time, we can do it in chunks if Np*n is too large, but 10k * 256 is ~2.5M floats, which is fine.
        changes = 2.0 * signs[None, :] * M.astype(np.float32)
        new_sums = current_sums[:, None] - changes
        new_maxes = np.max(np.abs(new_sums), axis=0)
        return new_maxes
    
    # Simulated Annealing
    # We will run multiple restarts.
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Initialize
        if restart == 0:
            signs = rng.choice([-1, 1], size=n).astype(np.float32)
        else:
            signs = rng.choice([-1, 1], size=n).astype(np.float32)
        
        current_sums = M @ signs
        current_score = int(np.max(np.abs(current_sums)))
        
        if current_score < best_score:
            best_score = current_score
            best_signs = signs.copy()
        
        # Annealing parameters
        T_start = 5.0
        T_end = 0.1
        # We want to do as many iterations as possible.
        # Each iteration is O(Np * n) vectorized.
        
        # Estimate time per iteration
        # Let's just run a fixed number of iterations or until time runs out.
        
        iters = 0
        T = T_start
        
        while time.time() < deadline - 2:
            # Evaluate all flips
            new_maxes = evaluate_all_flips(signs, current_sums)
            
            # Find the best move (minimum new_maxes)
            best_idx = np.argmin(new_maxes)
            best_new_score = new_maxes[best_idx]
            
            # Simulated Annealing acceptance
            # If best_new_score < current_score, always accept.
            # Else, accept with probability exp(-(best_new_score - current_score) / T)
            
            if best_new_score < current_score:
                accept = True
            else:
                delta = best_new_score - current_score
                if delta == 0:
                    accept = True
                else:
                    prob = np.exp(-delta / T)
                    accept = rng.random() < prob
            
            if accept:
                # Apply flip at best_idx
                signs[best_idx] = -signs[best_idx]
                # Update current_sums
                # current_sums[p] -= 2 * old_signs[best_idx] * M[p, best_idx]
                # But we already computed new_maxes based on this flip.
                # Let's just recompute current_sums to be safe, or update incrementally.
                # Incremental:
                old_sign = -signs[best_idx]  # Because we just flipped
                current_sums -= 2.0 * old_sign * M[:, best_idx]
                current_score = int(best_new_score)
                
                if current_score < best_score:
                    best_score = current_score
                    best_signs = signs.copy()
            
            # Cool down
            # Geometric cooling
            T = max(T * 0.999, T_end)
            
            iters += 1
            
            # If we are stuck at a very low score, we can break early for this restart
            if current_score <= 3:  # Theoretical lower bound is 1 (singletons), but 3 is very good
                # Try a few more to see if we can get to 2 or 1
                if current_score <= 2:
                    break

    # Final polish: try all single flips greedily
    if time.time() < deadline - 1 and best_signs is not None:
        signs = best_signs.astype(np.float32)
        current_sums = M @ signs
        current_score = int(np.max(np.abs(current_sums)))
        
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            new_maxes = evaluate_all_flips(signs, current_sums)
            best_idx = np.argmin(new_maxes)
            best_new_score = new_maxes[best_idx]
            
            if best_new_score < current_score:
                signs[best_idx] = -signs[best_idx]
                old_sign = -signs[best_idx]
                current_sums -= 2.0 * old_sign * M[:, best_idx]
                current_score = int(best_new_score)
                improved = True
                
                if current_score < best_score:
                    best_score = current_score
                    best_signs = signs.copy()

    return [int(x) for x in best_signs]
