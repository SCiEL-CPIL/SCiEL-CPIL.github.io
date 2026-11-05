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
        return F - 2.0 * c[idx] * D[:, idx]

    def flip_two_bits_F(F, c, idx1, idx2):
        return F - 2.0 * (c[idx1] * D[:, idx1] + c[idx2] * D[:, idx2])

    def local_search_1flip(F, c):
        current_score = get_score(F)
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
                improved = True
        return F, c, current_score

    def local_search_2flip(F, c):
        current_score = get_score(F)
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
                improved = True
        return F, c, current_score

    def spectral_init(seed_offset):
        """Generate coefficients from inverse FFT of a target spectrum."""
        rng_local = np.random.RandomState(42 + seed_offset)
        
        # Try different target spectrum shapes
        shape_type = rng_local.randint(3)
        
        if shape_type == 0:
            # Flat top
            target = np.ones(M)
        elif shape_type == 1:
            # Cosine taper
            theta = np.linspace(0, np.pi, M)
            target = np.cos(theta) + 1.0
        else:
            # Random phases
            target = np.exp(2j * np.pi * rng_local.rand(M))
        
        # Inverse FFT to get time domain
        c_complex = np.fft.ifft(target * (M / N))[:N]
        
        # Threshold to +/- 1
        c = np.sign(np.real(c_complex))
        c[c == 0] = 1.0
        
        return c

    rng = np.random.RandomState(42)
    
    best_score = float('inf')
    best_c = None
    best_F = None
    
    # Initialize with spectral synthesis and random sequences
    for i in range(10):
        if time.time() > deadline:
            break
        
        if i % 2 == 0:
            c = spectral_init(i)
        else:
            c = np.random.choice([-1.0, 1.0], N)
        
        F = D @ c
        current_score = get_score(F)
        
        # Simulated Annealing
        T = 0.05
        alpha = 0.999
        T_min = 1e-6
        
        while time.time() < deadline and T > T_min:
            # Perturb
            if rng.rand() < 0.7:
                # 1 flip
                idx = rng.randint(N)
                F_new = flip_bit_F(F, c, idx)
                c_new = c.copy()
                c_new[idx] *= -1
            else:
                # 2 flip
                idx1, idx2 = rng.choice(N, 2, replace=False)
                F_new = flip_two_bits_F(F, c, idx1, idx2)
                c_new = c.copy()
                c_new[idx1] *= -1
                c_new[idx2] *= -1
            
            new_score = get_score(F_new)
            delta = new_score - current_score
            
            if delta < 0 or rng.rand() < np.exp(-delta / T):
                F = F_new
                c = c_new
                current_score = new_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_c = c.copy()
                    best_F = F.copy()
            
            T *= alpha

    # Final Local Search on best_c
    if best_c is not None:
        c = best_c.copy()
        F = best_F.copy()
        current_score = best_score
        
        # 1-flip local search
        F, c, current_score = local_search_1flip(F, c)
        if current_score < best_score:
            best_score = current_score
            best_c = c.copy()
            best_F = F.copy()
        
        # 2-flip local search
        F, c, current_score = local_search_2flip(F, c)
        if current_score < best_score:
            best_score = current_score
            best_c = c.copy()
            best_F = F.copy()

    if best_c is None:
        return [1]*24
        
    return [int(x) for x in best_c]
