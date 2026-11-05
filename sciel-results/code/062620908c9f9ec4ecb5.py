"""Program for low discrepancy on all arithmetic progressions.
Uses efficient discrete local search minimizing sum of squared discrepancies.
"""
import numpy as np
import time

def construct():
    n = 256
    
    # 1. Precompute Arithmetic Progressions
    aps = []
    pos_to_aps = [[] for _ in range(n)]
    
    for d in range(1, n + 1):
        for s in range(0, n):
            current = s
            ap_indices = []
            while current < n:
                ap_indices.append(current)
                pos_to_aps[current].append(len(aps))
                current += d
            if ap_indices:
                aps.append(ap_indices)
                
    num_aps = len(aps)
    
    # Build M = A.T @ A
    # M[i, j] = number of APs containing both i and j.
    M = np.zeros((n, n), dtype=np.int32)
    for ap in aps:
        for i in ap:
            for j in ap:
                M[i, j] += 1
                
    M_diag = np.diag(M).copy()
    
    # 2. Helper functions
    
    def compute_ap_sums(signs):
        """Compute sums for all APs given signs. Returns array of length num_aps."""
        sums = np.zeros(num_aps, dtype=np.int32)
        for ap_idx, ap in enumerate(aps):
            sums[ap_idx] = np.sum(signs[ap])
        return sums
        
    def get_max_abs(sums):
        return np.max(np.abs(sums))
    
    # 3. Local Search with Vectorized Gradient Evaluation
    
    best_signs = None
    best_max_disc = float('inf')
    
    start_time = time.time()
    max_time = 55.0 # Leave 5s buffer
    
    num_restarts = 200 # Can afford more restarts now
    
    for restart in range(num_restarts):
        if time.time() - start_time > max_time:
            break
            
        # Initialize randomly
        signs = np.random.choice([-1, 1], size=n, dtype=np.int32)
        
        # Initial computations
        sums = compute_ap_sums(signs)
        max_disc = get_max_abs(sums)
        
        if max_disc < best_max_disc:
            best_max_disc = max_disc
            best_signs = signs.copy()
            
        # G = M @ signs
        G = M @ signs
        
        improved = True
        steps_since_kick = 0
        while improved and (time.time() - start_time < max_time):
            improved = False
            
            # Calculate delta for flipping each position i
            # Delta_i = -4 * signs[i] * G[i] + 4 * M_diag[i]
            deltas = -4 * signs * G + 4 * M_diag
            
            # Find the index with the most negative delta (largest improvement)
            min_delta_idx = np.argmin(deltas)
            min_delta = deltas[min_delta_idx]
            
            if min_delta < 0:
                i = min_delta_idx
                s_i = signs[i]
                
                # Flip sign
                signs[i] = -s_i
                
                # Update G (vectorized)
                G -= 2 * s_i * M[:, i]
                
                # Update sums
                for ap_idx in pos_to_aps[i]:
                    sums[ap_idx] -= 2 * s_i
                    
                improved = True
                steps_since_kick = 0
            else:
                # Stuck in local minimum
                # Random kick
                kick_pos = np.random.randint(n)
                s_kick = signs[kick_pos]
                signs[kick_pos] = -s_kick
                G -= 2 * s_kick * M[:, kick_pos]
                for ap_idx in pos_to_aps[kick_pos]:
                    sums[ap_idx] -= 2 * s_kick
                    
                # Recalculate max_disc after kick
                max_disc = get_max_abs(sums)
                if max_disc < best_max_disc:
                    best_max_disc = max_disc
                    best_signs = signs.copy()
                    
                # Allow a few kicks before restarting?
                # Or just break and restart?
                # Breaking and restarting is simpler and often effective.
                improved = False 
                break
                
            # Periodically check max_disc to update best
            # Doing it every step is O(num_aps) ~ 30k.
            # With N=256, convergence is fast. Let's do it every 10 steps.
            if steps_since_kick % 10 == 0 and improved:
                max_disc = get_max_abs(sums)
                if max_disc < best_max_disc:
                    best_max_disc = max_disc
                    best_signs = signs.copy()
                    
    if best_signs is None:
        return [1] * n
        
    return [int(s) for s in best_signs]
