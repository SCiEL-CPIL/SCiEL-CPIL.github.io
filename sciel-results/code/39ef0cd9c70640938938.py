import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit, allow 5s buffer
    
    N = 24
    M = 2**15 # FFT size
    sqrt_N = np.sqrt(N)
    
    # Precompute DFT matrix D where F = D @ c
    # F[k] = sum_{n=0}^{N-1} c_n exp(-2j pi k n / M)
    k = np.arange(M)
    n = np.arange(N)
    D = np.exp(-2j * np.pi * (k[:, None] * n[None, :] / M))
    
    def get_score(F):
        return np.max(np.abs(F)) / sqrt_N

    def flip_bit_F(F, c, idx):
        # F_new = F - 2 * c[idx] * D[:, idx]
        return F - 2.0 * c[idx] * D[:, idx]

    def flip_two_bits_F(F, c, idx1, idx2):
        return F - 2.0 * (c[idx1] * D[:, idx1] + c[idx2] * D[:, idx2])
    
    def flip_three_bits_F(F, c, idx1, idx2, idx3):
        return F - 2.0 * (c[idx1] * D[:, idx1] + c[idx2] * D[:, idx2] + c[idx3] * D[:, idx3])

    # Golay sequence of length 24 (one of the known optimal or near-optimal sequences)
    # This is a known good starting point for this problem
    golay = np.array([1, 1, 1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1], dtype=float)
    
    best_score = float('inf')
    best_c = None
    best_F = None
    
    def local_search(c_init, F_init, max_iter=1000):
        c = c_init.copy()
        F = F_init.copy()
        current_score = get_score(F)
        
        # 1-flip local search
        improved = True
        iter_count = 0
        while improved and time.time() < deadline and iter_count < max_iter:
            improved = False
            best_delta = 0
            best_idx = -1
            best_F_candidate = None
            
            for idx in range(N):
                F_candidate = flip_bit_F(F, c, idx)
                new_score = get_score(F_candidate)
                delta = new_score - current_score
                if delta < best_delta:
                    best_delta = delta
                    best_idx = idx
                    best_F_candidate = F_candidate
            
            if best_idx != -1:
                c[best_idx] *= -1
                F = best_F_candidate
                current_score += best_delta
                improved = True
                iter_count += 1
        
        # 2-flip local search
        improved = True
        iter_count = 0
        while improved and time.time() < deadline and iter_count < max_iter:
            improved = False
            best_delta = 0
            best_idx1 = -1
            best_idx2 = -1
            best_F_candidate = None
            
            for n1 in range(N):
                for n2 in range(n1+1, N):
                    F_candidate = flip_two_bits_F(F, c, n1, n2)
                    new_score = get_score(F_candidate)
                    delta = new_score - current_score
                    if delta < best_delta:
                        best_delta = delta
                        best_idx1 = n1
                        best_idx2 = n2
                        best_F_candidate = F_candidate
            
            if best_idx1 != -1:
                c[best_idx1] *= -1
                c[best_idx2] *= -1
                F = best_F_candidate
                current_score += best_delta
                improved = True
                iter_count += 1
                
        # 3-flip local search (slower, use sparingly or if time permits)
        if time.time() < deadline - 5.0:
            improved = True
            iter_count = 0
            while improved and time.time() < deadline - 2.0 and iter_count < 50:
                improved = False
                best_delta = 0
                best_idx1 = -1
                best_idx2 = -1
                best_idx3 = -1
                best_F_candidate = None
                
                # Sample a subset of 3-flips to save time, or do all if N is small enough
                # C(24,3) = 2024, which is manageable
                for n1 in range(N):
                    for n2 in range(n1+1, N):
                        for n3 in range(n2+1, N):
                            F_candidate = flip_three_bits_F(F, c, n1, n2, n3)
                            new_score = get_score(F_candidate)
                            delta = new_score - current_score
                            if delta < best_delta:
                                best_delta = delta
                                best_idx1 = n1
                                best_idx2 = n2
                                best_idx3 = n3
                                best_F_candidate = F_candidate
                
                if best_idx1 != -1:
                    c[best_idx1] *= -1
                    c[best_idx2] *= -1
                    c[best_idx3] *= -1
                    F = best_F_candidate
                    current_score += best_delta
                    improved = True
                    iter_count += 1

        return c, F, current_score

    # Initialize with the Golay sequence
    c = golay.copy()
    F = D @ c
    current_score = get_score(F)
    
    if current_score < best_score:
        best_score = current_score
        best_c = c.copy()
        best_F = F.copy()
        
    # Run local search on the initial Golay sequence
    c_ls, F_ls, score_ls = local_search(c, F)
    if score_ls < best_score:
        best_score = score_ls
        best_c = c_ls.copy()
        best_F = F_ls.copy()

    # Random restarts
    rng = np.random.RandomState(42)
    num_restarts = 0
    while time.time() < deadline - 10.0:
        c_rand = rng.choice([-1.0, 1.0], N)
        F_rand = D @ c_rand
        score_rand = get_score(F_rand)
        
        # Run local search on this random start
        c_ls, F_ls, score_ls = local_search(c_rand, F_rand)
        
        if score_ls < best_score:
            best_score = score_ls
            best_c = c_ls.copy()
            best_F = F_ls.copy()
            
        num_restarts += 1
        # Check time frequently to avoid overshooting
        if time.time() > deadline - 2.0:
            break

    if best_c is None:
        return [1]*24
        
    return [int(x) for x in best_c]
