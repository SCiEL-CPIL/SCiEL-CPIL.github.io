import numpy as np
import time

N = 400
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
DEADLINE = time.time() + TIME_LIMIT - SAFETY_MARGIN

# Precompute FFT size
FFT_SIZE = 1
while FFT_SIZE < 2 * N - 1:
    FFT_SIZE *= 2

def _autoconv_fft(f):
    """Compute linear autoconvolution using FFT."""
    F = np.fft.rfft(f, n=FFT_SIZE)
    C = np.fft.irfft(F * F, n=FFT_SIZE)
    return C[:2 * N - 1]

def _score(f):
    S = np.sum(f)
    if abs(S) < 1e-12:
        return 1e10
    C = _autoconv_fft(f)
    return np.max(np.abs(C)) / (S * S)

def _local_search_binary_fast(f, max_iter=2000, time_budget=10.0):
    """
    Local search on binary {-1, 1} sequences using incremental convolution updates.
    """
    f = np.sign(f).astype(np.float64)
    S = np.sum(f)
    if abs(S) < 1e-6:
        return f, 1e10
        
    C = _autoconv_fft(f)
    score = np.max(np.abs(C)) / (S * S)
    
    start_time = time.time()
    no_improve_count = 0
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        if time.time() > DEADLINE:
            break
            
        best_improvement = 0
        best_flip_idx = -1
        best_new_C = None
        best_new_S = S
        
        for i in range(N):
            v = f[i]
            delta = -2.0 * v
            
            C_new = C.copy()
            
            k_start = i
            k_end = i + N
            end_idx = min(k_end, len(C_new))
            start_idx = k_start
            
            if start_idx < len(C_new):
                f_segment = f[:end_idx - start_idx]
                C_new[start_idx:end_idx] += 2.0 * delta * f_segment
                
            if 2 * i < len(C_new):
                C_new[2 * i] += delta * delta
                
            new_S = S - 2.0 * v
            if abs(new_S) < 1e-6:
                continue
                
            new_score = np.max(np.abs(C_new)) / (new_S * new_S)
            
            if new_score < score - best_improvement:
                best_improvement = score - new_score
                best_flip_idx = i
                best_new_C = C_new
                best_new_S = new_S
        
        if best_flip_idx >= 0:
            v = f[best_flip_idx]
            f[best_flip_idx] = -v
            S = best_new_S
            C = best_new_C
            score = np.max(np.abs(C)) / (S * S)
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 10:
                break
                
    return f, score

def _soft_max_abs(C, beta=50.0):
    """
    Smooth approximation of max(|C|) using log-sum-exp.
    As beta -> inf, this approaches the true max.
    """
    x = np.abs(C)
    x_max = np.max(x)
    if x_max < 1e-12:
        return 1e-12
    x_scaled = x / x_max
    log_sum_exp = np.log(np.sum(np.exp(beta * (x_scaled - 1.0))))
    return x_max * (1.0 + np.log(np.exp(beta) + np.sum(np.exp(beta * (x_scaled - 1.0)) - np.exp(beta))) / beta)

def _score_soft(f, beta=50.0):
    """
    Smooth surrogate for the score using soft-max.
    """
    S = np.sum(f)
    if abs(S) < 1e-6:
        return 1e10
    C = _autoconv_fft(f)
    return _soft_max_abs(C, beta) / (S * S)

def _quadratic_phase_seq(T, shift=0):
    """
    Generate a signed sequence from quadratic phase chirp: sign(cos(pi * (n+shift)^2 / T)).
    T is a continuous parameter; optimizing it exploits stationary phase cancellation
    to bound autoconvolution sidelobes by O(sqrt(N)).
    """
    n = np.arange(N) + shift
    phase = np.pi * n * n / T
    c = np.cos(phase)
    return np.where(c >= 0, 1.0, -1.0)

def _linear_phase_seq(M, shift=0):
    """
    Generate a signed sequence based on linear phase: sign(exp(i*pi*n/M)).
    """
    n = np.arange(N) + shift
    phase = np.pi * n / M
    return np.sign(np.cos(phase)).astype(np.float64)

def _walsh_hadamard_seq(k):
    """
    Generate a Walsh-Hadamard sequence of order k.
    These have perfect autocorrelation (only peak at zero).
    """
    # Walsh-Hadamard matrix rows have good autocorrelation
    # For length 2^m, use the m-th row
    m = int(np.log2(k)) if k > 0 else 0
    if m < 1:
        return np.ones(N)
    
    # Generate the k-th row of the Walsh-Hadamard matrix
    # Using the recursive definition
    def _wh_row(row_idx, length):
        if length == 1:
            return np.array([1.0])
        half = length // 2
        left = _wh_row(row_idx % half, half)
        right = _wh_row(row_idx // 2, half) * (-1.0 if (row_idx % half) >= half else 1.0)
        # Actually, let's use the standard Walsh matrix construction
        # W(0) = [1], W(n+1) = [[W(n), W(n)], [W(n), -W(n)]]
        # Row k of W(m) for length 2^m
        # This is equivalent to: sign of product of (-1)^(bit_i * bit_j)
        pass
    
    # Simpler: use the definition directly
    # f(n) = (-1)^(popcount(n & k))
    # This is a Walsh function
    n = np.arange(N)
    # Compute popcount of (n & k)
    def _popcount(arr):
        arr = arr.astype(np.int32)
        count = np.zeros_like(arr, dtype=np.int32)
        while np.any(arr > 0):
            count += arr & 1
            arr >>= 1
        return count
    
    pc = _popcount(n & k)
    return np.where(pc % 2 == 0, 1.0, -1.0)

def _m_sequence_seq(p, c):
    """
    Generate a maximal-length sequence (m-sequence) from a linear feedback shift register.
    These have excellent autocorrelation properties (three-valued autocorrelation).
    """
    # For N=400, we need a suitable LFSR
    # Use a primitive polynomial of degree m where 2^m - 1 >= N
    # m=9 gives 511, which is close to 400
    # Primitive polynomial x^9 + x^4 + 1 (in GF(2))
    # This is one of the known primitive polynomials
    
    # Let's use a simpler approach: generate from a known m-sequence
    # and truncate to N
    
    # For degree 9, primitive polynomial x^9 + x^4 + 1
    # Tap positions: 9 and 4 (0-indexed from LSB)
    # LFSR: new_bit = bit[9-1] XOR bit[4-1] = bit[8] XOR bit[3]
    
    m = 9
    # Initial state: non-zero
    state = [1] * m  # all 1s
    
    seq = []
    for _ in range(2**m - 1):
        seq.append(1.0 if state[-1] == 1 else -1.0)
        # Compute new bit
        new_bit = state[m-1] ^ state[m-1-4]  # bit[8] XOR bit[3] for m=9
        # Shift right
        state = [new_bit] + state[:-1]
    
    # Truncate to N
    if len(seq) >= N:
        return np.array(seq[:N])
    else:
        # Pad with repeats
        repeats = int(np.ceil(N / len(seq)))
        return np.array(seq * repeats)[:N]

def _optimize_quadratic_T(f_template, T_lo=0.5, T_hi=50.0, n_iter=30, time_budget=5.0):
    """
    Golden section search over continuous T for quadratic phase chirp.
    T is the time-scaling factor; irrational relative to N exploits
    stationary phase cancellation for O(sqrt(N)) sidelobe bounds.
    """
    start = time.time()
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    a, b = T_lo, T_hi
    c = b - (b - a) / phi
    d = a + (b - a) / phi
    
    def eval_T(T):
        f = _quadratic_phase_seq(T)
        S = np.sum(f)
        if abs(S) < 1e-6:
            return 1e10
        C = _autoconv_fft(f)
        return np.max(np.abs(C)) / (S * S)
    
    fc = eval_T(c)
    fd = eval_T(d)
    
    for _ in range(n_iter):
        if time.time() - start > time_budget:
            break
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - (b - a) / phi
            fc = eval_T(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + (b - a) / phi
            fd = eval_T(d)
    
    T_best = (a + b) / 2.0
    f_best = _quadratic_phase_seq(T_best)
    S = np.sum(f_best)
    if S < 0:
        f_best = -f_best
    return f_best

def construct():
    start_time = time.time()
    
    best_f = np.ones(N)
    best_score = _score(best_f)
    
    rng = np.random.default_rng(42)
    
    # 1. Generate number-theoretic constructions
    candidates = []
    
    # All ones (baseline)
    candidates.append(np.ones(N))
    
    # Quadratic phase sequences with continuous T optimization
    # First, do a coarse grid to find good T regions
    T_coarse = np.linspace(0.5, 50.0, 50)
    for T in T_coarse:
        for shift in [0, 1]:
            candidates.append(_quadratic_phase_seq(T, shift))
    
    # Linear phase sequences
    for M in [2, 3, 4, 5, 7, 8, 10, 13, 16, 20, 25, 32, 40, 50, 64, 80, 100, 128, 160, 200, 256, 320, 400]:
        candidates.append(_linear_phase_seq(M, 0))
    
    # Walsh-Hadamard sequences
    for k in range(1, 32):
        candidates.append(_walsh_hadamard_seq(k))
    
    # M-sequences
    candidates.append(_m_sequence_seq(9, 0))
    
    # 2. Evaluate all candidates and keep the best few
    scored_candidates = []
    for c in candidates:
        if time.time() > DEADLINE:
            break
        s = np.sum(c)
        if abs(s) < N * 0.1:
            if s < 0:
                c = -c
            s = np.sum(c)
            if abs(s) < N * 0.1:
                c = np.sign(c + 0.5)
                s = np.sum(c)
                if abs(s) < N * 0.1:
                    continue
        score = _score(c)
        scored_candidates.append((score, c))
    
    # Sort by score and keep top 15
    scored_candidates.sort(key=lambda x: x[0])
    top_candidates = [c for _, c in scored_candidates[:15]]
    
    # 3. Local search from each top candidate
    for c in top_candidates:
        if time.time() > DEADLINE:
            break
        
        time_remaining = DEADLINE - time.time()
        budget = min(12.0, max(2.0, time_remaining / 4.0))
        
        f_local, score_local = _local_search_binary_fast(c, max_iter=3000, time_budget=budget)
        
        if score_local < best_score:
            best_score = score_local
            best_f = f_local.copy()
    
    # 4. Continuous T optimization for quadratic phase
    time_remaining = DEADLINE - time.time()
    if time_remaining > 15:
        # Use the best candidate as a template for T optimization
        # Try optimizing T around the best coarse-grid T values
        T_starts = [1.0, 2.0, 5.0, 10.0, 20.0, 35.0]
        for T0 in T_starts:
            if time.time() > DEADLINE:
                break
            t_budget = min(4.0, max(1.0, (DEADLINE - time.time()) / 3.0))
            f_opt = _optimize_quadratic_T(None, T_lo=max(0.3, T0 * 0.5), T_hi=min(60.0, T0 * 2.0), n_iter=25, time_budget=t_budget)
            s = np.sum(f_opt)
            if abs(s) < N * 0.1:
                continue
            score_opt = _score(f_opt)
            if score_opt < best_score:
                best_score = score_opt
                best_f = f_opt.copy()
            # Also run local search from the optimized T
            if time.time() < DEADLINE - 5:
                f_ls, score_ls = _local_search_binary_fast(f_opt, max_iter=2000, time_budget=min(8.0, DEADLINE - time.time() - 3))
                if score_ls < best_score:
                    best_score = score_ls
                    best_f = f_ls.copy()
    
    # 5. Additional random restarts with local search
    remaining_time = DEADLINE - time.time()
    if remaining_time > 15:
        n_restarts = min(5, int(remaining_time / 12))
        for _ in range(n_restarts):
            if time.time() > DEADLINE:
                break
            
            f_rand = rng.choice([-1.0, 1.0], size=N)
            S_rand = np.sum(f_rand)
            if abs(S_rand) < N * 0.2:
                continue
                
            time_remaining = DEADLINE - time.time()
            budget = min(8.0, max(2.0, time_remaining / 2.0))
            
            f_rand_local, score_rand_local = _local_search_binary_fast(f_rand, max_iter=1500, time_budget=budget)
            
            if score_rand_local < best_score:
                best_score = score_rand_local
                best_f = f_rand_local.copy()
    
    return best_f
