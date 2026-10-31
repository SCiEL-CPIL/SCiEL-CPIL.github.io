import numpy as np
import time

N = 400

def _compute_bound(f):
    """
    Computes the peak sidelobe magnitude of the linear autocorrelation.
    """
    n_fft = 2 * N
    F = np.fft.rfft(f, n=n_fft)
    PSD = F * np.conj(F)
    R = np.fft.irfft(PSD, n=n_fft)
    # R[0] is the peak (energy)
    # R[1:N] are the sidelobes
    return np.max(np.abs(R[1:N]))

def _gen_m_sequence(k):
    """
    Generate an m-sequence (maximal length sequence) of length 2^k - 1,
    then pad/truncate to N samples.
    """
    # Use a primitive polynomial for GF(2)
    # For k=9: x^9 + x^4 + 1 (primitive)
    # For k=10: x^10 + x^7 + 1 (primitive)
    
    # Let's use k=9, giving length 511, then we take first 400 or center
    # Actually, let's try different k values
    
    # Simple LFSR implementation for m-sequence
    def lfsr_mseq(k, poly):
        """Generate m-sequence of length 2^k - 1"""
        seq = [1] * k
        length = (1 << k) - 1
        result = []
        for i in range(length):
            result.append(seq[-1])
            # Feedback: XOR of bits specified by polynomial
            feedback = 0
            for bit in poly:
                feedback ^= seq[-1 - bit]
            seq.pop(0)
            seq.append(feedback)
        return np.array(result) * 2 - 1  # Map {0,1} to {-1,1}
    
    # Try k=9 with primitive polynomial x^9 + x^4 + 1
    # Polynomial bits: positions where x^i appears (excluding x^9)
    # x^9 + x^4 + 1 means taps at positions 4 and 0
    poly9 = [4, 0]
    
    # Try k=10 with x^10 + x^7 + 1
    poly10 = [7, 0]
    
    # Generate and select best
    candidates = []
    
    for k, poly in [(9, poly9), (10, poly10)]:
        try:
            mseq = lfsr_mseq(k, poly)
            # Take center N samples or first N
            if len(mseq) >= N:
                # Center crop
                start = (len(mseq) - N) // 2
                candidates.append(mseq[start:start+N])
                # Also try first N
                candidates.append(mseq[:N])
                # Also try last N
                candidates.append(mseq[-N:])
            else:
                # Pad with zeros or repeat
                padded = np.zeros(N)
                padded[:len(mseq)] = mseq
                candidates.append(padded)
        except Exception:
            continue
    
    return candidates

def _local_search(f, max_iter=500, time_limit=60):
    """
    Local search: try flipping individual samples to reduce peak sidelobe.
    """
    start_time = time.time()
    current_f = f.copy()
    current_bound = _compute_bound(current_f)
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        # Try flipping each sample
        for i in range(N):
            old_val = current_f[i]
            current_f[i] = -old_val
            new_bound = _compute_bound(current_f)
            if new_bound < current_bound:
                current_bound = new_bound
                improved = True
            else:
                current_f[i] = old_val
        
        if not improved:
            break
    
    return current_f, current_bound

def _spectral_design(beta=4.0):
    """
    Design signal in frequency domain with smooth window, convert to time domain.
    """
    # Create frequency domain signal with smooth window
    n_fft = 2 * N
    freqs = np.fft.rfftfreq(n_fft)
    
    # Raised cosine window in frequency domain
    # Band-limit to avoid Gibbs
    W = 0.4  # Bandwidth parameter
    H = np.zeros(len(freqs))
    for i, f in enumerate(freqs):
        if abs(f) <= W/2:
            H[i] = 1.0
        elif abs(f) <= W:
            # Raised cosine transition
            x = (abs(f) - W/2) / (W/2)
            H[i] = 0.5 * (1 + np.cos(np.pi * x))
    
    # Add phase for better time-domain properties
    phase = np.random.randn(len(freqs)) * 0.1
    F = H * np.exp(1j * phase)
    
    # Convert to time domain
    f = np.fft.irfft(F, n=n_fft)[:N]
    
    # Normalize
    norm = np.linalg.norm(f)
    if norm > 0:
        f = f / norm
    
    return f

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_f = None
    best_score = float('inf')
    
    # Strategy 1: m-sequences with local search
    mseq_candidates = _gen_m_sequence(9)
    for i, mseq in enumerate(mseq_candidates):
        if time.time() > deadline - 30:
            break
        # Normalize
        norm = np.linalg.norm(mseq)
        if norm > 0:
            mseq_norm = mseq / norm
        else:
            continue
        
        score = _compute_bound(mseq_norm)
        if score < best_score:
            best_score = score
            best_f = mseq_norm.copy()
        
        # Local search on best m-sequence
        if i == 0 or score < best_score * 1.1:
            if time.time() > deadline - 20:
                break
            f_ls, score_ls = _local_search(mseq_norm, max_iter=200, time_limit=15)
            if score_ls < best_score:
                best_score = score_ls
                best_f = f_ls.copy()
    
    # Strategy 2: Spectral design
    if time.time() < deadline - 25:
        for beta in [2.0, 3.0, 4.0, 5.0]:
            if time.time() > deadline - 20:
                break
            f_spec = _spectral_design(beta)
            score = _compute_bound(f_spec)
            if score < best_score:
                best_score = score
                best_f = f_spec.copy()
            
            # Local search on spectral design
            if time.time() < deadline - 15:
                f_ls, score_ls = _local_search(f_spec, max_iter=150, time_limit=10)
                if score_ls < best_score:
                    best_score = score_ls
                    best_f = f_ls.copy()
    
    # Strategy 3: Random signed sequences with local search
    if time.time() < deadline - 20:
        rng = np.random.default_rng(42)
        for trial in range(5):
            if time.time() > deadline - 15:
                break
            f_rand = rng.choice([-1, 1], size=N).astype(float)
            norm = np.linalg.norm(f_rand)
            f_rand = f_rand / norm
            score = _compute_bound(f_rand)
            if score < best_score:
                best_score = score
                best_f = f_rand.copy()
            
            # Local search
            if time.time() < deadline - 10:
                f_ls, score_ls = _local_search(f_rand, max_iter=100, time_limit=8)
                if score_ls < best_score:
                    best_score = score_ls
                    best_f = f_ls.copy()
    
    # Final local search on best found
    if best_f is not None and time.time() < deadline - 5:
        f_final, score_final = _local_search(best_f, max_iter=200, time_limit=10)
        if score_final < best_score:
            best_score = score_final
            best_f = f_final.copy()
    
    # Fallback
    if best_f is None:
        best_f = np.ones(N) / np.sqrt(N)
    
    return best_f
