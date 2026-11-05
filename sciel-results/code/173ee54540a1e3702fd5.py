import numpy as np
import time
import random
from scipy.optimize import differential_evolution

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 128
    target_min = 128
    target_max = 256
    
    def get_counts(A_mask):
        # A_mask is bool array of length 129
        v = A_mask.astype(np.int32)
        # Convolution of v with v
        # Result length 257
        conv = np.convolve(v, v, mode='full')
        # conv[k] is count of ordered pairs (i,j) with i+j=k
        # We need unordered pairs with i<=j
        # N_k = (conv[k] - self_pair_count) / 2 + self_pair_count
        # self_pair_count is 1 if k is even and k/2 is in A, else 0
        
        N = np.zeros(target_max + 1, dtype=np.int32)
        # We only care about target range
        for s in range(target_min, target_max + 1):
            ordered = conv[s]
            # Check if s is even and s/2 is in A
            if s % 2 == 0:
                half = s // 2
                if 0 <= half <= n and A_mask[half]:
                    self_count = 1
                else:
                    self_count = 0
            else:
                self_count = 0
            
            # Unordered count
            # Ordered pairs = 2 * (pairs with i != j) + (pairs with i == j)
            # conv[s] = 2 * (pairs i!=j) + self_count
            # We want (pairs i!=j) + self_count
            # pairs i!=j = (conv[s] - self_count) / 2
            # Unordered = (conv[s] - self_count) / 2 + self_count
            #           = (conv[s] + self_count) / 2
            
            N[s] = (ordered + self_count) // 2
        return N

    def objective(x):
        # x is float array of length 129
        A_mask = (x > 0.5).astype(bool)
        
        # Check if non-empty
        if not np.any(A_mask):
            return 1e9
            
        N = get_counts(A_mask)
        target_N = N[target_min:target_max+1]
        
        missing = np.sum(target_N == 0)
        if missing > 0:
            return 1000 * missing + np.max(target_N)
        
        peak = np.max(target_N)
        # Add small penalty for variance to encourage flatness
        sum_sq = np.sum(target_N**2)
        
        return peak + 0.01 * sum_sq

    # Run Differential Evolution
    bounds = [(0, 1) for _ in range(n+1)]
    
    # Parameters tuned for time
    # popsize=20, maxiter=100 might be too slow?
    # Let's try smaller popsize and maxiter first, then local search.
    
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            strategy='best1bin', 
            maxiter=50, 
            popsize=15, 
            tol=0.01, 
            seed=42,
            polish=False,
            init='latinhypercube'
        )
    except Exception:
        result = None

    best_x = result.x if result else np.full(n+1, 0.5)
    best_A = (best_x > 0.5).astype(bool)
    
    # If DE failed or produced bad result, fallback to greedy
    N = get_counts(best_A)
    target_N = N[target_min:target_max+1]
    if np.sum(target_N == 0) > 0:
        # Greedy repair
        # ... implement simple greedy ...
        pass

    # Local Search
    def local_search(A_mask, time_limit):
        A_mask = A_mask.copy()
        N = get_counts(A_mask)
        target_N = N[target_min:target_max+1]
        
        current_peak = np.max(target_N) if np.all(target_N > 0) else 999
        current_sum_sq = np.sum(target_N**2)
        current_missing = np.sum(target_N == 0)
        
        def get_score(N):
            tN = N[target_min:target_max+1]
            m = np.sum(tN == 0)
            if m > 0: return (m, 999, 999)
            return (0, np.max(tN), np.sum(tN**2))
        
        current_score = get_score(N)
        best_score = current_score
        best_A = A_mask.copy()
        
        A_indices = list(np.where(A_mask)[0])
        not_A_indices = [i for i in range(n+1) if not A_mask[i]]
        
        while time.time() < time_limit:
            improved = False
            # Sample swaps
            # To speed up, don't check all 129*129
            # Sample 100 swaps
            for _ in range(100):
                if not A_indices or not not_A_indices: break
                a = random.choice(A_indices)
                b = random.choice(not_A_indices)
                
                # Try swap
                A_mask[a] = False
                A_mask[b] = True
                new_N = get_counts(A_mask)
                new_score = get_score(new_N)
                
                if new_score < current_score:
                    # Accept
                    A_indices.remove(a)
                    A_indices.append(b)
                    not_A_indices.remove(b)
                    not_A_indices.append(a)
                    
                    N = new_N
                    current_score = new_score
                    improved = True
                    
                    if current_score < best_score:
                        best_score = current_score
                        best_A = A_mask.copy()
                else:
                    # Revert
                    A_mask[a] = True
                    A_mask[b] = False
            
            if not improved:
                # Maybe do a random restart or break?
                # Let's break if no improvement for a while
                break
                
        return best_A

    # Run local search with remaining time
    time_left = deadline - time.time() - 1
    if time_left > 1:
        best_A = local_search(best_A, deadline - 1)
        
    # Final check and return
    A_list = [i for i in range(n+1) if best_A[i]]
    if not A_list:
        # Fallback
        A_list = list(range(64, 129))
        
    return A_list
