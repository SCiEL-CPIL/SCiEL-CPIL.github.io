import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
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
    
    # Golay sequence of length 24 (one of the known optimal or near-optimal sequences)
    # This is a known good starting point for this problem
    golay = np.array([1, 1, 1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1], dtype=float)
    
    best_score = float('inf')
    best_c = None
    best_F = None
    
    # Initialize with the Golay sequence
    c = golay.copy()
    F = D @ c
    current_score = get_score(F)
    
    if current_score < best_score:
        best_score = current_score
        best_c = c.copy()
        best_F = F.copy()
    
    # 1-flip local search
    improved = True
    while improved and time.time() < deadline:
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
            if current_score < best_score:
                best_score = current_score
                best_c = c.copy()
                best_F = F.copy()
            improved = True
    
    # 2-flip local search
    improved = True
    while improved and time.time() < deadline:
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
            if current_score < best_score:
                best_score = current_score
                best_c = c.copy()
                best_F = F.copy()
            improved = True
    
    # If we have time, try a few random restarts with local search
    rng = np.random.RandomState(42)
    while time.time() < deadline:
        c_rand = np.random.choice([-1.0, 1.0], N)
        F_rand = D @ c_rand
        score_rand = get_score(F_rand)
        
        if score_rand < best_score:
            # Do local search on this random start
            c = c_rand.copy()
            F = F_rand.copy()
            current_score = score_rand
            
            # 1-flip local search
            improved = True
            while improved and time.time() < deadline:
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
                    if current_score < best_score:
                        best_score = current_score
                        best_c = c.copy()
                        best_F = F.copy()
                    improved = True
            
            # 2-flip local search
            improved = True
            while improved and time.time() < deadline:
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
                    if current_score < best_score:
                        best_score = current_score
                        best_c = c.copy()
                        best_F = F.copy()
                    improved = True
    
    if best_c is None:
        return [1]*24
        
    return [int(x) for x in best_c]
