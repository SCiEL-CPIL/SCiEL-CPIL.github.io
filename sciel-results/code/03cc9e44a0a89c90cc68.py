import numpy as np
import time

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 90s limit

    # 1. Generate "critical" Arithmetic Progressions
    # We ignore length 1 (max sum 1) and length 2 (max sum 2).
    # If the global max discrepancy is > 2, these don't constrain the optimum.
    # If it converges to 2, we can re-check, but typically low-discrepancy sequences
    # for n=256 have max > 2.
    progs = []
    for s in range(1, n + 1):
        for a in range(n):
            # Check length
            # Length L satisfies: a + (L-1)*s < n  =>  L-1 < (n-a)/s
            L = (n - 1 - a) // s + 1
            if L >= 3:
                progs.append((a, s))
    
    num_progs = len(progs)
    
    # 2. Build Incidence Matrix M (n x num_progs)
    # M[i, p] = 1 if index i is in AP p, else 0
    # Using sparse logic to build dense for fast matmul
    M = np.zeros((n, num_progs), dtype=np.int8)
    for p_idx, (a, s) in enumerate(progs):
        # Indices: a, a+s, ...
        for idx in range(a, n, s):
            M[idx, p_idx] = 1
            
    # 3. Initialize signs
    # Thue-Morse is a strong heuristic start for low discrepancy
    def thue_morse(n):
        return 1 if bin(n).count('1') % 2 == 0 else -1
        
    signs = np.array([thue_morse(i) for i in range(n)], dtype=np.int8)
    
    # 4. Compute initial sums and max
    sums = signs @ M
    current_max = np.max(np.abs(sums))
    best_max = current_max
    best_signs = signs.copy()
    
    # 5. Simulated Annealing with Vectorized Evaluation
    # We will flip one index at a time, but use the matrix to update sums quickly?
    # Actually, signs @ M is O(n * num_progs). 
    # num_progs for L>=3:
    # s=1: 254 starts, L>=3 -> ~254*254/2 ~ 32k? No.
    # Let's estimate num_progs.
    # For s=1, a in 0..253, L>=3 -> a <= 253. ~254 APs.
    # For s=2, a in 0..255, L>=3 -> a+4<n -> a<252. ~252 APs.
    # Total APs is roughly sum_{s=1}^{n} (n - 2s) ~ n^2/2 - 2n^2/2 ... 
    # Actually sum_{s=1}^{n/3} (n-2s) ~ n^2/6 + n^2/2 ... 
    # Let's just count.
    # If num_progs is ~10k-20k, M is 256 x 20k = 5M elements. 
    # signs @ M is 256 * 20k = 5M ops. Very fast in numpy.
    
    T = 5.0
    cooling_rate = 0.999
    rng = np.random.default_rng(42)
    
    steps = 0
    while time.time() < deadline:
        # Pick a random index to flip
        i = rng.integers(n)
        
        # Flip sign
        signs[i] = -signs[i]
        
        # Compute new sums
        # Optimization: We only need to update the sums of APs containing i.
        # But signs @ M is so fast, maybe just recompute?
        # 5M int8 ops is ~5ms? No, int8 matmul is very fast. 
        # Let's try full recompute first. If too slow, we switch to delta updates.
        new_sums = signs @ M
        new_max = np.max(np.abs(new_sums))
        
        delta_e = new_max - current_max
        
        # Acceptance criterion
        if delta_e < 0 or (T > 1e-4 and rng.random() < np.exp(-delta_e / T)):
            current_max = new_max
            sums = new_sums # Keep reference for next step? No, we recompute anyway.
            if current_max < best_max:
                best_max = current_max
                best_signs = signs.copy()
        else:
            # Rollback
            signs[i] = -signs[i]
            
        T *= cooling_rate
        steps += 1
        
        # If we hit 0, we can stop early or continue to polish?
        # 0 is impossible for n=256 likely. 
        # If best_max is small, we can tighten T faster.
        
    return [int(s) for s in best_signs]
