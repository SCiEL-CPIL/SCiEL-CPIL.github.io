"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _eval_norm_fast(coeffs):
    """
    Fast approximation of sup norm for search.
    Uses smaller FFT and fewer refinement steps.
    """
    n = len(coeffs)
    M = 4096
    c_padded = np.zeros(M, dtype=complex)
    c_padded[:n] = coeffs
    P = M * np.fft.ifft(c_padded)
    mags = np.abs(P)
    
    K = 10
    idx = np.argsort(mags)[-K:][::-1]
    
    max_val = 0.0
    theta = 2 * np.pi * np.arange(M) / M
    
    # Precompute coeffs for Horner
    coeffs_rev = coeffs[::-1]
    
    for i in idx:
        w = 2 * np.pi / M
        a = theta[i] - w
        b = theta[i] + w
        
        gr = (5 ** 0.5 - 1) / 2
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        
        # Inline evaluation for speed
        def f(theta_val):
            z_val = np.exp(1j * theta_val)
            p_val = 0.0
            for coeff in coeffs_rev:
                p_val = p_val * z_val + coeff
            return np.abs(p_val)
        
        fc = f(c)
        fd = f(d)
        
        for _ in range(20):
            if fc < fd:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = f(d)
            else:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = f(c)
        
        peak_val = max(fc, fd)
        if peak_val > max_val:
            max_val = peak_val
            
    return max_val

def _eval_norm_precise(coeffs):
    """
    High precision sup norm evaluation.
    """
    n = len(coeffs)
    M = 16384
    c_padded = np.zeros(M, dtype=complex)
    c_padded[:n] = coeffs
    P = M * np.fft.ifft(c_padded)
    mags = np.abs(P)
    
    K = 20
    idx = np.argsort(mags)[-K:][::-1]
    
    max_val = 0.0
    theta = 2 * np.pi * np.arange(M) / M
    coeffs_rev = coeffs[::-1]
    
    for i in idx:
        w = 2 * np.pi / M
        a = theta[i] - w
        b = theta[i] + w
        
        gr = (5 ** 0.5 - 1) / 2
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        
        def f(theta_val):
            z_val = np.exp(1j * theta_val)
            p_val = 0.0
            for coeff in coeffs_rev:
                p_val = p_val * z_val + coeff
            return np.abs(p_val)
        
        fc = f(c)
        fd = f(d)
        
        for _ in range(40):
            if fc < fd:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = f(d)
            else:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = f(c)
        
        peak_val = max(fc, fd)
        if peak_val > max_val:
            max_val = peak_val
            
    return max_val

def _score_fast(coeffs):
    return _eval_norm_fast(coeffs) / np.sqrt(len(coeffs))

def _score_precise(coeffs):
    return _eval_norm_precise(coeffs) / np.sqrt(len(coeffs))

def construct():
    N = 24
    start_time = time.time()
    deadline = start_time + 140
    
    # 1. Generate Initial Candidates
    
    candidates = []
    
    # Golay-Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    candidates.append(np.array(p[:N], dtype=float))
    
    # Random
    for _ in range(5):
        candidates.append(np.random.choice([-1, 1], N))
        
    # Spectral Synthesis
    def generate_spectral_candidate():
        M = 4096
        phases = np.random.uniform(0, 2 * np.pi, M // 2 + 1)
        spectrum = np.ones(M // 2 + 1, dtype=complex) * np.exp(1j * phases)
        full_spectrum = np.zeros(M, dtype=complex)
        full_spectrum[:M // 2 + 1] = spectrum
        if M % 2 == 0:
            full_spectrum[M // 2] = np.real(full_spectrum[M // 2])
            for k in range(M // 2 + 1, M):
                full_spectrum[k] = np.conj(full_spectrum[M - k])
        else:
            for k in range(M // 2 + 1, M):
                full_spectrum[k] = np.conj(full_spectrum[M - k])
        x = np.fft.ifft(full_spectrum) * M
        x = np.real(x)
        coeffs = np.sign(x[:N])
        coeffs[coeffs == 0] = 1
        return coeffs

    for _ in range(10):
        candidates.append(generate_spectral_candidate())

    # 2. Local Search Initialization (Polish each candidate)
    
    def local_search_pair(coeffs, current_score, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try all pair flips
            for i in range(N):
                for j in range(i + 1, N):
                    coeffs[i] *= -1
                    coeffs[j] *= -1
                    new_score = _score_fast(coeffs)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                    else:
                        coeffs[i] *= -1
                        coeffs[j] *= -1
            # Try single flips
            for i in range(N):
                coeffs[i] *= -1
                new_score = _score_fast(coeffs)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    coeffs[i] *= -1
        return coeffs, current_score

    best_coeffs = None
    best_score = float('inf')
    
    for cand in candidates:
        # Quick polish
        c, s = local_search_pair(cand.copy(), _score_fast(cand), start_time + 5)
        if s < best_score:
            best_score = s
            best_coeffs = c.copy()

    # 3. Simulated Annealing with Block Flips
    
    # Precompute moves
    # Single flips
    single_moves = [i for i in range(N)]
    # Block flips (length 2 to 6)
    block_moves = []
    for length in range(2, 7):
        for start in range(N - length + 1):
            block_moves.append((start, length))
            
    T = 0.1
    cooling_rate = 0.999
    min_T = 1e-5
    
    current_coeffs = best_coeffs.copy()
    current_score = best_score
    best_coeffs_sa = current_coeffs.copy()
    best_score_sa = current_score
    
    while T > min_T and time.time() < deadline - 10:
        # Choose move
        if np.random.rand() < 0.5:
            # Single flip
            i = single_moves[np.random.randint(len(single_moves))]
            current_coeffs[i] *= -1
        else:
            # Block flip
            start, length = block_moves[np.random.randint(len(block_moves))]
            for k in range(length):
                current_coeffs[start + k] *= -1
                
        new_score = _score_fast(current_coeffs)
        delta = new_score - current_score
        
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_score = new_score
            if new_score < best_score_sa:
                best_score_sa = new_score
                best_coeffs_sa = current_coeffs.copy()
        else:
            # Revert
            if np.random.rand() < 0.5:
                i = single_moves[np.random.randint(len(single_moves))] # This is wrong, we need to revert the specific move
                # We need to store the move to revert it
                pass # The logic above is flawed for reversion. Let's fix.
                
    # Fix SA Loop
    current_coeffs = best_coeffs.copy()
    current_score = best_score
    best_coeffs_sa = current_coeffs.copy()
    best_score_sa = current_score
    
    while T > min_T and time.time() < deadline - 10:
        # Choose move and store it
        if np.random.rand() < 0.5:
            i = np.random.randint(N)
            move_type = 'single'
            move_data = i
            current_coeffs[i] *= -1
        else:
            start, length = block_moves[np.random.randint(len(block_moves))]
            move_type = 'block'
            move_data = (start, length)
            for k in range(length):
                current_coeffs[start + k] *= -1
                
        new_score = _score_fast(current_coeffs)
        delta = new_score - current_score
        
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_score = new_score
            if new_score < best_score_sa:
                best_score_sa = new_score
                best_coeffs_sa = current_coeffs.copy()
        else:
            # Revert
            if move_type == 'single':
                current_coeffs[move_data] *= -1
            else:
                start, length = move_data
                for k in range(length):
                    current_coeffs[start + k] *= -1
        
        T *= cooling_rate
        
        # Periodic local search on the best found so far
        if time.time() % 10 < 1.0:
             # We can't easily pause SA to do local search on best_coeffs_sa without disrupting SA state.
             # Instead, we can just let SA run.
             pass

    # 4. Final Polish
    best_coeffs = best_coeffs_sa.copy()
    best_score = best_score_sa
    best_coeffs, best_score = local_search_pair(best_coeffs, best_score, deadline - 2)
    
    # Final precise check
    final_score = _score_precise(best_coeffs)
    
    return [int(c) for c in best_coeffs]

# EVOLVE-BLOCK-END
