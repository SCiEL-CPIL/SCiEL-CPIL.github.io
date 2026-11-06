"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 256
    
    # Precompute all arithmetic progressions with length >= 2
    aps = []
    for d in range(1, N // 2 + 1):
        for start in range(N - d):
            if start + d < N:
                indices = list(range(start, N, d))
                if len(indices) >= 2:
                    aps.append(indices)
    
    num_aps = len(aps)
    
    # Build sparse matrix M where M[i, j] = 1 if position j is in AP i
    # Use a dense matrix since N=256 and num_aps ~ 30k, but we can use int8
    # Actually, let's use a list of arrays and compute sums efficiently
    # Better: build a matrix and use matvec
    
    # Build the AP matrix as a numpy array of shape (num_aps, N) with int8
    # This allows fast matrix-vector product
    M = np.zeros((num_aps, N), dtype=np.int8)
    for i, ap in enumerate(aps):
        for idx in ap:
            M[i, idx] = 1
    
    # Convert to float32 for fast matvec
    Mf = M.astype(np.float32)
    
    def compute_max_abs_sum(signs):
        """Compute maximum absolute sum over all APs using matrix-vector product."""
        # signs is int8 array of shape (N,)
        # Mf @ signs gives shape (num_aps,)
        sums = Mf.dot(signs.astype(np.float32))
        return int(np.max(np.abs(sums)))
    
    # Simulated Annealing with efficient cost computation
    rng = random.Random(42)
    
    # Initial solution: random signs
    signs = np.array([rng.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
    current_cost = compute_max_abs_sum(signs)
    best_signs = signs.copy()
    best_cost = current_cost
    
    # Also try a few deterministic initializations
    for seed in [0, 1, 2, 3, 4, 5, 6, 7]:
        rng2 = random.Random(seed)
        s = np.array([rng2.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
        c = compute_max_abs_sum(s)
        if c < best_cost:
            best_cost = c
            best_signs = s.copy()
        if c < current_cost:
            current_cost = c
            signs = s.copy()
    
    # Also try alternating pattern
    alt = np.array([1 if i % 2 == 0 else -1 for i in range(N)], dtype=np.int8)
    c_alt = compute_max_abs_sum(alt)
    if c_alt < best_cost:
        best_cost = c_alt
        best_signs = alt.copy()
    if c_alt < current_cost:
        current_cost = c_alt
        signs = alt.copy()
    
    # Simulated Annealing
    T = 5.0  # Initial temperature
    cooling_rate = 0.9995
    
    while time.time() < deadline:
        if T < 0.01:
            break
            
        # Pick a random index to flip
        idx = rng.randrange(N)
        old_sign = signs[idx]
        signs[idx] = -old_sign
        
        new_cost = compute_max_abs_sum(signs)
        
        if new_cost <= current_cost:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_signs = signs.copy()
        else:
            # Accept with probability exp(-(new_cost - current_cost) / T)
            delta = new_cost - current_cost
            if delta > 0:
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    current_cost = new_cost
                else:
                    signs[idx] = old_sign
        
        T *= cooling_rate
    
    # Final local search: try flipping each position and keep if improves
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(N):
            old_sign = signs[idx]
            signs[idx] = -old_sign
            new_cost = compute_max_abs_sum(signs)
            if new_cost < best_cost:
                best_cost = new_cost
                best_signs = signs.copy()
                current_cost = new_cost
                improved = True
            else:
                signs[idx] = old_sign
    
    return [int(x) for x in best_signs]
# EVOLVE-BLOCK-END
