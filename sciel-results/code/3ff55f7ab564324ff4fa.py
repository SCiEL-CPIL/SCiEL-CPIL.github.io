"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_autocorr(s):
    n_fft = 1 << (2 * N - 1).bit_length()
    f_s = np.fft.rfft(s, n=n_fft)
    s_rev = s[::-1]
    f_s_rev = np.fft.rfft(s_rev, n=n_fft)
    conv = np.fft.irfft(f_s * f_s_rev, n=n_fft)
    R_full = conv[2*N-2::-1]
    return R_full

def _score_from_R(R_full):
    sidelobes = np.concatenate([R_full[:N-1], R_full[N:2*N-1]])
    return np.sum(sidelobes**2)

def _generate_m_sequence(m):
    L = (1 << m) - 1
    # Primitive polynomial for m=9: x^9 + x^5 + x^3 + x^2 + 1
    taps = [9, 5, 3, 2, 0]
    reg = 1
    seq = np.zeros(L, dtype=np.float64)
    for i in range(L):
        bit = reg & 1
        seq[i] = 1.0 if bit == 0 else -1.0
        fb = 0
        for t in taps:
            fb ^= (reg >> t) & 1
        reg = (reg >> 1) | (fb << (m - 1))
    return seq

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Precompute constants for vectorized search
    m_vals = np.arange(-(N-1), N)
    j_idx = np.arange(N)[:, np.newaxis]
    m_idx = m_vals[np.newaxis, :]
    idx_pos = j_idx + m_idx
    idx_neg = j_idx - m_idx
    valid_pos = (idx_pos >= 0) & (idx_pos < N)
    valid_neg = (idx_neg >= 0) & (idx_neg < N)
    idx_pos_safe = np.clip(idx_pos, 0, N-1)
    idx_neg_safe = np.clip(idx_neg, 0, N-1)
    
    mask_sidelobe = np.ones(2*N-1, dtype=bool)
    mask_sidelobe[N-1] = False
    R_full_masked_indices = np.where(mask_sidelobe)[0]
    
    best_s = None
    best_score = np.inf
    
    # Multiple restarts
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialization
        if r == 0:
            # M-sequence
            m_seq = _generate_m_sequence(9)
            s = m_seq[:N].copy()
        else:
            # Random or perturbed best
            if best_s is not None and r < 10:
                s = best_s.copy()
                # Perturb 10% of bits
                n_perturb = N // 10
                perturb_indices = np.random.choice(N, n_perturb, replace=False)
                s[perturb_indices] *= -1.0
            else:
                # Random
                s = np.random.choice([-1, 1], N).astype(np.float64)
        
        R_full = _compute_autocorr(s)
        current_score = _score_from_R(R_full)
        
        # Local Search Loop
        improved = True
        max_local_iters = 50
        local_iter = 0
        
        while improved and local_iter < max_local_iters and time.time() < deadline:
            improved = False
            local_iter += 1
            
            # Calculate Delta_R for all j
            term_pos = np.where(valid_pos, s[idx_pos_safe], 0.0)
            term_neg = np.where(valid_neg, s[idx_neg_safe], 0.0)
            Delta_R_all = -2.0 * s[:, np.newaxis] * (term_pos + term_neg)
            
            Delta_R_masked = Delta_R_all[:, R_full_masked_indices]
            R_full_masked = R_full[R_full_masked_indices]
            
            linear_term = 2.0 * np.sum(R_full_masked[np.newaxis, :] * Delta_R_masked, axis=1)
            quad_term = np.sum(Delta_R_masked**2, axis=1)
            
            new_scores = current_score + linear_term + quad_term
            
            best_idx = np.argmin(new_scores)
            best_new_score = new_scores[best_idx]
            
            if best_new_score < current_score - 1e-10:
                j = best_idx
                s[j] *= -1.0
                R_full += Delta_R_all[j]
                current_score = best_new_score
                improved = True
                
                if current_score < best_score:
                    best_score = current_score
                    best_s = s.copy()
            else:
                break
                
    if best_s is None:
        best_s = np.ones(N)
        
    return best_s
# EVOLVE-BLOCK-END
