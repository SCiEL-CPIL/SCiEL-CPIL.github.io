import numpy as np
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    N = 24
    M = 1 << 15  # 32768 FFT size for high-resolution screening
    sqrt_N = math.sqrt(N)
    
    # Precompute DFT columns for fast updates
    # D[k, n] = exp(-2j * pi * k * n / M)
    # F = D @ c
    k_idx = np.arange(M)
    n_idx = np.arange(N)
    # Memory efficient: we only need to update F, so we precompute the columns
    # D_col[n] = exp(-2j * pi * k_idx * n / M)
    D_cols = np.empty((N, M), dtype=complex)
    for n in range(N):
        D_cols[n] = np.exp(-2j * np.pi * k_idx * n / M)
    
    def get_score(F):
        return np.max(np.abs(F)) / sqrt_N
    
    def flip_bit_F(F, c, idx):
        # F_new = F - 2 * c[idx] * D_cols[idx]
        return F - 2.0 * c[idx] * D_cols[idx]
    
    def flip_two_bits_F(F, c, idx1, idx2):
        return F - 2.0 * (c[idx1] * D_cols[idx1] + c[idx2] * D_cols[idx2])

    def flip_four_bits_F(F, c, idxs):
        # idxs is a list of 4 indices
        delta = 0.0
        for idx in idxs:
            delta += c[idx] * D_cols[idx]
        return F - 2.0 * delta

    def _eval_at(c, t):
        """Evaluate |P(exp(2j*pi*t))| using Horner's method."""
        z = np.exp(2j * np.pi * t)
        acc = 0j
        for coeff in reversed(c):
            acc = acc * z + coeff
        return np.abs(acc)

    def _peak(c):
        """
        Accurate peak: dense FFT sampling + golden-section refinement of top peaks.
        Used for final evaluation and initial candidate ranking.
        """
        n = len(c)
        N_fft = 8192
        cf = np.zeros(N_fft, dtype=complex)
        cf[:n] = c
        F = np.fft.ifft(cf) * N_fft
        mags = np.abs(F)
        
        top_k = 8
        idx = np.argpartition(mags, -top_k)[-top_k:]
        best_peak = float(mags[idx].max())
        
        gr = (math.sqrt(5) - 1) / 2
        for k in idx:
            lo = max(0.0, (k - 5) / N_fft)
            hi = min(1.0, (k + 6) / N_fft)
            a, b = lo, hi
            c1 = b - gr * (b - a)
            c2 = a + gr * (b - a)
            f1 = _eval_at(c, c1)
            f2 = _eval_at(c, c2)
            
            for _ in range(35):
                if f1 < f2:
                    a = c1
                    c1 = c2
                    f1 = f2
                    c2 = a + gr * (b - a)
                    f2 = _eval_at(c, c2)
                else:
                    b = c2
                    c2 = c1
                    f2 = f1
                    c1 = b - gr * (b - a)
                    f1 = _eval_at(c, c1)
            
            v = max(f1, f2)
            if v > best_peak:
                best_peak = v
                
        return best_peak

    def _local_search(c, F, current_score, deadline, rng):
        """
        Local search with 1-bit and 2-bit flips.
        Uses precomputed F for fast updates.
        """
        c = c.copy()
        F = F.copy()
        n = len(c)
        
        # 1-bit local search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_delta = 0
            best_idx = -1
            best_F_candidate = None
            
            for idx in range(n):
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
        
        # 2-bit local search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_delta = 0
            best_idx1 = -1
            best_idx2 = -1
            best_F_candidate = None
            
            for n1 in range(n):
                for n2 in range(n1+1, n):
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
                
        return c, F, current_score

    rng = np.random.default_rng(12345)
    
    best_score = float('inf')
    best_c = None
    best_F = None
    
    # Structured / known-good patterns
    candidates = []
    candidates.append(np.array([1]*N, dtype=float))
    candidates.append(np.array([1,-1]*12, dtype=float))
    candidates.append(np.array([1,1,-1,-1]*6, dtype=float))
    candidates.append(np.array([1,-1,-1]*8, dtype=float))
    candidates.append(np.array([1,1,1,-1,-1,-1]*4, dtype=float))
    candidates.append(np.array([1,-1,1,1,-1,1,1,-1]*3, dtype=float))
    candidates.append(np.array([1,1,1,1,-1,-1,-1,-1]*3, dtype=float))
    candidates.append(np.array([1,-1,-1,-1,1,1,1,1]*3, dtype=float))
    
    # GRS construction
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-x for x in q]
    candidates.append(np.array(p[:N], dtype=float))
    
    # Diverse random restarts
    for seed in range(30):
        r = np.random.default_rng(seed)
        candidates.append(r.choice([-1.0, 1.0], size=N))

    # Phase 1: Screen and refine top candidates
    screened = []
    for c in candidates:
        if time.time() > deadline - 30:
            break
        F = D_cols.T @ c
        score = get_score(F)
        screened.append((score, c, F))
        
    screened.sort(key=lambda x: x[0])
    top = screened[:10]
    
    for score, c, F in top:
        if time.time() > deadline - 15:
            break
        c_opt, F_opt, score_opt = _local_search(c, F, score, deadline - 10, rng)
        if score_opt < best_score:
            best_score = score_opt
            best_c = c_opt
            best_F = F_opt
            
    # Phase 2: Simulated Annealing with diverse starts
    if best_c is None:
        best_c = candidates[0]
        best_F = D_cols.T @ best_c
        best_score = get_score(best_F)
        
    for restart in range(5):
        if time.time() > deadline - 5:
            break
            
        # Start from best found or random
        if restart == 0:
            c = best_c.copy()
            F = best_F.copy()
            current_score = best_score
        else:
            c = rng.choice([-1.0, 1.0], size=N)
            F = D_cols.T @ c
            current_score = get_score(F)
            
        # Simulated Annealing
        T = 0.05
        alpha = 0.999
        T_min = 1e-6
        
        while time.time() < deadline - 5 and T > T_min:
            # Perturb
            if rng.rand() < 0.6:
                # 1 flip
                idx = rng.randint(N)
                F_new = flip_bit_F(F, c, idx)
                c_new = c.copy()
                c_new[idx] *= -1
            elif rng.rand() < 0.9:
                # 2 flip
                idx1, idx2 = rng.choice(N, 2, replace=False)
                F_new = flip_two_bits_F(F, c, idx1, idx2)
                c_new = c.copy()
                c_new[idx1] *= -1
                c_new[idx2] *= -1
            else:
                # 4 flip
                idxs = rng.choice(N, 4, replace=False)
                F_new = flip_four_bits_F(F, c, idxs)
                c_new = c.copy()
                for idx in idxs:
                    c_new[idx] *= -1
            
            new_score = get_score(F_new)
            delta = new_score - current_score
            
            if delta < 0 or rng.rand() < math.exp(-delta / T):
                F = F_new
                c = c_new
                current_score = new_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_c = c.copy()
                    best_F = F.copy()
                    
                # Occasionally run local search to polish
                if rng.rand() < 0.05:
                    c, F, current_score = _local_search(c, F, current_score, deadline - 10, rng)
                    if current_score < best_score:
                        best_score = current_score
                        best_c = c.copy()
                        best_F = F.copy()
            
            T *= alpha

    if best_c is None:
        return [1]*24
        
    return [int(x) for x in best_c]
