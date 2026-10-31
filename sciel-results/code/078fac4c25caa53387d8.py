import numpy as np
import time
from scipy.optimize import minimize

N = 400
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
DEADLINE = time.time() + TIME_LIMIT - SAFETY_MARGIN

def _legendre_symbol(n, p):
    """Compute the Legendre symbol (n/p) for prime p."""
    if n % p == 0:
        return 0
    return 1 if pow(n, (p - 1) // 2, p) == 1 else -1

def _generate_legendre_sequence(p):
    """
    Generate a Legendre sequence of length p-1.
    Legendre sequences have excellent auto-correlation properties:
    max |autocorr| <= (p+1)/2 for the periodic case.
    """
    seq = np.zeros(p - 1)
    for k in range(1, p):
        seq[k - 1] = _legendre_symbol(k, p)
    return seq

def _find_good_legendre_prime(target_len):
    """
    Find a prime p close to target_len that gives a good Legendre sequence.
    We want p-1 close to N, or we can pad/truncate.
    """
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    # Search for primes near target_len + 1
    p = target_len + 1
    while p < target_len + 200:
        if is_prime(p):
            return p
        p += 1
    return target_len + 1

# Precompute FFT size for linear convolution of two length N sequences
# Result length is 2N - 1
FFT_SIZE = 1
while FFT_SIZE < 2 * N - 1:
    FFT_SIZE *= 2

def _autoconv_fft(f):
    """Compute linear autoconvolution using FFT."""
    F = np.fft.rfft(f, n=FFT_SIZE)
    C = np.fft.irfft(F * F, n=FFT_SIZE)
    return C[:2 * N - 1]

def _objective_vectorized(f):
    """
    Compute max(|autoconv(f)|) / (sum(f))^2.
    If sum(f) == 0, return a large penalty.
    """
    S = np.sum(f)
    if abs(S) < 1e-12:
        return 1e10
    C = _autoconv_fft(f)
    return np.max(np.abs(C)) / (S * S)

def _objective_smooth(f, t=0.01):
    """
    Smooth surrogate using log-sum-exp for the max.
    """
    S = np.sum(f)
    if abs(S) < 1e-12:
        return 1e10
    C = _autoconv_fft(f)
    absC = np.abs(C)
    max_val = np.max(absC)
    # Numerical stability: subtract max before exp
    lse = max_val + np.log(np.sum(np.exp(t * (absC - max_val))))
    return (lse / t) / (S * S)

def _gradient_smooth(f, t=0.01):
    """
    Gradient of the smooth objective using FFT-based convolution.
    Uses FFT to compute grad[i] = 2 * sum_j A[i+j] * f[j] efficiently.
    """
    S = np.sum(f)
    if abs(S) < 1e-12:
        return np.zeros_like(f)
    
    C = _autoconv_fft(f)
    absC = np.abs(C)
    signC = np.sign(C)
    max_val = np.max(absC)
    
    exp_term = np.exp(t * (absC - max_val))
    sum_exp = np.sum(exp_term)
    weights = exp_term / sum_exp
    
    A = weights * signC
    
    # Compute grad[i] = 2 * sum_j A[i+j] * f[j] using FFT
    # This is a cross-correlation-like operation
    # grad[i] = 2 * sum_{j=0}^{N-1} A[i+j] * f[j] for i = 0..N-1
    # Let's pad A and f and use FFT
    # A has length 2N-1, f has length N
    # We need for each i: sum_{j=0}^{N-1} A[i+j] * f[j]
    # This is like a sliding dot product
    
    # Use FFT: create array B where B[k] = A[k] for k=0..2N-2
    # Then grad[i]/2 = sum_{j=0}^{N-1} B[i+j] * f[j]
    # This is the cross-correlation of B and reversed f, or we can use convolution
    
    # Actually, let's think of it as: for fixed i, sum_j A[i+j]*f[j]
    # Let's reverse f: fr[j] = f[N-1-j]
    # Then sum_j A[i+j]*f[j] = sum_j A[i+j]*fr[N-1-j]
    # Let m = N-1-j, j = N-1-m: sum_m A[i+N-1-m]*fr[m]
    # This is cross-correlation of A and fr at lag (i+N-1)
    
    # Simpler: just use the fact that this is a convolution
    # Define: for output index k, out[k] = sum_j A[k-j] * f[j] (convolution)
    # We want: sum_j A[i+j] * f[j]
    # Let's define A_rev[k] = A[2N-2-k]
    # Then sum_j A[i+j]*f[j] = sum_j A_rev[2N-2-i-j]*f[j]
    # This is convolution of A_rev and f at index (2N-2-i)
    
    # Even simpler: just compute it via FFT directly
    # Create signal of length 2N-1: A
    # Create signal of length N: f
    # We want h[i] = sum_{j=0}^{min(N-1, 2N-2-i)} A[i+j] * f[j]
    
    # Pad both to FFT_SIZE
    A_pad = np.zeros(FFT_SIZE)
    A_pad[:len(A)] = A
    f_pad = np.zeros(FFT_SIZE)
    f_pad[:N] = f
    
    # Cross-correlation via FFT
    # corr[k] = sum_j A[k+j] * f[j]
    # This is NOT standard convolution. Let's use:
    # corr = ifft(fft(A) * conj(fft(f))) gives sum_j A[j] * conj(f[j-k])
    # That's not quite right either.
    
    # Let's just do it the straightforward way but vectorized with numpy
    # For each i, we need dot(A[i:i+N], f) where A is truncated
    # We can use a stride trick or just accept the loop but make it faster
    
    # Actually for N=400, the loop is 400 iterations of dot products of length up to 400.
    # That's 400*400 = 160K multiply-adds, which is fast.
    # But let's try to vectorize using numpy's stride tricks
    
    from numpy.lib.stride_tricks import as_strided
    # A has length 2N-1 = 799
    # For i=0: A[0:400] dot f[0:400]
    # For i=1: A[1:401] dot f[0:400]
    # ...
    # For i=399: A[399:799] dot f[0:400]
    
    # Create a view with shape (N, N) where row i is A[i:i+N]
    # A length is 2N-1, so A[i:i+N] is valid for i=0..N-1
    A_view = as_strided(A, shape=(N, N), strides=(A.strides[0], A.strides[0]))
    
    # grad[i] = 2 * dot(A_view[i], f)
    grad = 2.0 * A_view.dot(f)
    
    LSE_val = max_val + np.log(sum_exp)
    G_val = LSE_val / t
    
    # d/d f [ G / S^2 ] = (1/S^2) dG - G * d(1/S^2)
    # d(1/S^2) = -2/S^3
    # dG/df[i] = grad[i]
    # d(1/S^2)/df[i] = -2/S^3
    return (1.0 / (S * S)) * grad - (G_val / (S * S * S)) * 2.0 * S

def _local_search_binary_fast(f_init, time_budget=10.0):
    """
    Local search on binary {-1, 1} sequences using incremental convolution updates.
    Optimized: precompute flip deltas and use vectorized max computation.
    """
    f = np.sign(f_init).astype(np.float64)
    f[f == 0] = 1.0
    
    S = np.sum(f)
    if abs(S) < 1e-6:
        return f, 1e10
        
    C = _autoconv_fft(f)
    score = np.max(np.abs(C)) / (S * S)
    
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 12
    
    while time.time() - start_time < time_budget:
        if time.time() > DEADLINE:
            break
            
        best_improvement = 0
        best_move = None
        
        # Precompute the delta for each single flip
        # Flipping f[i] from v to -v: C changes by -4*v*f[j] for k=i+j, plus +4 at k=2i
        # Vectorize: for each i, compute the new C and new score
        
        # Try all single flips in a batched manner
        # For each i, the update is: C[i:i+N] -= 4*v*f[:N], C[2i] += 4
        # We can compute all new scores more efficiently
        
        for i in range(N):
            if time.time() > DEADLINE:
                break
                
            v = f[i]
            # Compute the change to C
            # C_new = C - 4*v*f_conv where f_conv is the convolution of e_i with f
            # e_i * f has f[j] at position i+j
            # So C[i:i+N] -= 4*v*f[:N], and C[2i] += 4 (from e_i*e_i)
            
            new_S = S - 2.0 * v
            if abs(new_S) < 1e-6:
                continue
            
            # Apply update
            C_new = C.copy()
            end_idx = min(i + N, len(C_new))
            C_new[i:end_idx] -= 4.0 * v * f[:end_idx - i]
            if 2 * i < len(C_new):
                C_new[2 * i] += 4.0
                
            new_score = np.max(np.abs(C_new)) / (new_S * new_S)
            
            if new_score < score - best_improvement:
                best_improvement = score - new_score
                best_move = ([i], C_new, new_S)
        
        # If no single flip helps, try double flips (sampled)
        if best_move is None and no_improve_count >= 2:
            n_pairs = min(30, N)
            for _ in range(n_pairs):
                if time.time() > DEADLINE:
                    break
                i1 = np.random.randint(0, N)
                i2 = np.random.randint(0, N)
                if i1 == i2:
                    continue
                v1 = f[i1]
                v2 = f[i2]
                
                C_new = C.copy()
                
                # Flip at i1
                end1 = min(i1 + N, len(C_new))
                C_new[i1:end1] -= 4.0 * v1 * f[:end1 - i1]
                if 2 * i1 < len(C_new):
                    C_new[2 * i1] += 4.0
                    
                # Flip at i2 (use updated C_new for f[i2] if i2 affected)
                # But f hasn't changed yet, so use original f
                end2 = min(i2 + N, len(C_new))
                C_new[i2:end2] -= 4.0 * v2 * f[:end2 - i2]
                if 2 * i2 < len(C_new):
                    C_new[2 * i2] += 4.0
                    
                new_S = S - 2.0 * v1 - 2.0 * v2
                if abs(new_S) < 1e-6:
                    continue
                
                new_score = np.max(np.abs(C_new)) / (new_S * new_S)
                
                if new_score < score - best_improvement:
                    best_improvement = score - new_score
                    best_move = ([i1, i2], C_new, new_S)
        
        if best_move is not None:
            indices, C_new, new_S = best_move
            for idx in indices:
                f[idx] = -f[idx]
            S = new_S
            C = C_new
            score = np.max(np.abs(C)) / (S * S)
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
                
    return f, score

def _sparse_annealing(f_init, time_budget=15.0):
    """
    Simulated annealing on sparse signed sequences with adaptive temperature.
    """
    support = list(np.nonzero(f_init)[0])
    signs = [f_init[i] for i in support]
    
    def make_f(sup, sgn):
        f = np.zeros(N)
        for pos, s in zip(sup, sgn):
            f[pos] = s
        return f
    
    current_f = make_f(support, signs)
    current_score = _objective_vectorized(current_f)
    
    best_f = current_f.copy()
    best_score = current_score
    
    start_time = time.time()
    T = 1.0
    cooling = 0.99
    min_T = 0.005
    
    while time.time() - start_time < time_budget and T > min_T:
        if time.time() > DEADLINE:
            break
        
        move_type = np.random.random()
        old_state = (support[:], signs[:])
        
        if move_type < 0.25 and len(support) < 40:
            pos = np.random.randint(N)
            if pos not in support:
                sign = 1.0 if np.random.random() < 0.5 else -1.0
                support.append(pos)
                signs.append(sign)
        elif move_type < 0.50 and len(support) > 3:
            idx = np.random.randint(0, len(support))
            support.pop(idx)
            signs.pop(idx)
        elif move_type < 0.80 and len(support) > 0:
            idx = np.random.randint(0, len(support))
            signs[idx] = -signs[idx]
        else:
            if len(support) == 0:
                T *= cooling
                continue
            idx = np.random.randint(0, len(support))
            old_pos = support[idx]
            new_pos = np.random.randint(N)
            if new_pos not in support:
                support[idx] = new_pos
            else:
                T *= cooling
                continue
        
        new_f = make_f(support, signs)
        new_score = _objective_vectorized(new_f)
        
        delta = new_score - current_score
        if delta < 0 or (T > min_T and np.random.random() < np.exp(-delta / T)):
            current_f = new_f
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_f = new_f.copy()
        else:
            # Revert
            support, signs = old_state
        
        T *= cooling
    
    return best_f, best_score

def _polish_with_tighter_t(f, time_budget=8.0):
    """
    Final polish pass with tighter smoothness parameter for higher precision.
    """
    start_time = time.time()
    current_f = f.copy()
    current_score = _objective_vectorized(current_f)
    
    ts = [0.01, 0.005, 0.002, 0.001, 0.0005]
    
    for t in ts:
        if time.time() > DEADLINE or time.time() - start_time > time_budget:
            break
        
        try:
            res = minimize(
                _objective_smooth,
                current_f,
                args=(t,),
                method='L-BFGS-B',
                jac=_gradient_smooth,
                options={'maxiter': 80, 'maxls': 80, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            true_score = _objective_vectorized(res.x)
            if true_score < current_score:
                current_f = res.x
                current_score = true_score
        except Exception:
            continue
    
    return current_f, current_score

def _local_search_real_fast(f_init, time_budget=10.0, step=0.5):
    """
    Local search on real-valued sequences with incremental FFT updates.
    Optimized with adaptive step size and more trial positions.
    """
    f = f_init.copy().astype(np.float64)
    S = np.sum(f)
    C = _autoconv_fft(f)
    score = np.max(np.abs(C)) / (S * S)
    
    start_time = time.time()
    no_improve_count = 0
    current_step = step
    
    while time.time() - start_time < time_budget:
        if time.time() > DEADLINE:
            break
        
        best_improvement = 0
        best_change = None
        
        # Try more positions and include zero-delta neighbors
        n_trials = min(120, N)
        trial_positions = np.random.choice(N, size=n_trials, replace=False)
        
        for i in trial_positions:
            if time.time() > DEADLINE:
                break
                
            for delta in [-current_step, -current_step*0.5, current_step*0.5, current_step]:
                new_val = f[i] + delta
                if abs(new_val) < 1e-12:
                    continue
                    
                C_new = C.copy()
                end_idx = min(i + N, len(C_new))
                C_new[i:end_idx] += 2.0 * delta * f[:end_idx - i]
                if 2 * i < len(C_new):
                    C_new[2 * i] += delta * delta
                    
                new_S = S + delta
                if abs(new_S) < 1e-6:
                    continue
                
                new_score = np.max(np.abs(C_new)) / (new_S * new_S)
                
                if new_score < score - best_improvement:
                    best_improvement = score - new_score
                    best_change = (i, delta, C_new, new_S)
        
        if best_change is not None:
            i, delta, C_new, new_S = best_change
            f[i] += delta
            S = new_S
            C = C_new
            score = np.max(np.abs(C)) / (S * S)
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 15:
                current_step *= 0.8
                if current_step < 0.01:
                    break
                no_improve_count = 0
    
    return f, score

def construct():
    start_time = time.time()
    
    best_f = np.ones(N)
    best_score = _objective_vectorized(best_f)
    
    rng = np.random.default_rng(42)
    
    # Generate Legendre sequence initializations
    # Find good primes near N
    legendre_inits = []
    for prime_offset in [0, 2, 4, 6, 8, 10]:
        p = N + 1 + prime_offset
        # Find next prime >= p
        def _is_prime(n):
            if n < 2:
                return False
            if n == 2:
                return True
            if n % 2 == 0:
                return False
            for i in range(3, int(n**0.5) + 1, 2):
                if n % i == 0:
                    return False
            return True
        while not _is_prime(p):
            p += 1
        if p - 1 <= N + 20:  # Only use if close to N
            leg_seq = _generate_legendre_sequence(p)
            # Pad or truncate to length N
            if len(leg_seq) >= N:
                f_leg = leg_seq[:N].astype(np.float64)
            else:
                f_leg = np.zeros(N)
                f_leg[:len(leg_seq)] = leg_seq
                # Fill remaining with +1
                f_leg[len(leg_seq):] = 1.0
            legendre_inits.append(f_leg)
    
    # Phase 1: Continuous optimization with smooth objective
    for restart in range(8):
        if time.time() > DEADLINE:
            break
        
        if restart == 0:
            f0 = np.ones(N)
        elif restart == 1:
            f0 = np.sin(np.linspace(0, 2 * np.pi, N))
        elif restart == 2:
            f0 = np.cos(np.linspace(0, 4 * np.pi, N))
        elif restart == 3:
            f0 = np.sin(np.linspace(0, 3 * np.pi, N))
        elif restart == 4:
            f0 = np.cos(np.linspace(0, 6 * np.pi, N))
        elif restart == 5:
            f0 = np.sin(np.linspace(0, 5 * np.pi, N))
        elif restart == 6:
            f0 = np.cos(np.linspace(0, 8 * np.pi, N))
        else:
            f0 = rng.uniform(-1, 1, N)
        
        S0 = np.sum(f0)
        if abs(S0) > 1e-6:
            f0 = f0 / S0
        else:
            f0 = np.ones(N)
        
        ts = [5.0, 20.0, 50.0, 100.0, 200.0]
        current_f = f0
        
        for t in ts:
            if time.time() > DEADLINE:
                break
            
            try:
                res = minimize(
                    _objective_smooth,
                    current_f,
                    args=(t,),
                    method='L-BFGS-B',
                    jac=_gradient_smooth,
                    options={'maxiter': 60, 'maxls': 60, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                true_score = _objective_vectorized(res.x)
                if true_score < best_score:
                    best_score = true_score
                    best_f = res.x.copy()
                current_f = res.x
            except Exception:
                continue
    
    # Phase 2: Binary local search with FFT (Fast Incremental)
    # Use Legendre sequences as primary initializations (they have provably good autocorrelation)
    binary_inits = []
    
    # Add Legendre-based inits first
    for leg_f in legendre_inits:
        f_bin = np.sign(leg_f)
        f_bin[f_bin == 0] = 1.0
        binary_inits.append(f_bin)
    
    # Add traditional inits
    binary_inits.append(np.ones(N))
    
    sine_init = np.sign(np.sin(np.linspace(0, 2 * np.pi, N)))
    sine_init[sine_init == 0] = 1.0
    binary_inits.append(sine_init)
    
    cos_init = np.sign(np.cos(np.linspace(0, 4 * np.pi, N)))
    cos_init[cos_init == 0] = 1.0
    binary_inits.append(cos_init)
    
    sine3_init = np.sign(np.sin(np.linspace(0, 3 * np.pi, N)))
    sine3_init[sine3_init == 0] = 1.0
    binary_inits.append(sine3_init)
    
    cos6_init = np.sign(np.cos(np.linspace(0, 6 * np.pi, N)))
    cos6_init[cos6_init == 0] = 1.0
    binary_inits.append(cos6_init)
    
    # Add some random inits
    for _ in range(3):
        binary_inits.append(rng.choice([-1.0, 1.0], size=N))
    
    for f0 in binary_inits:
        if time.time() > DEADLINE:
            break
        
        S0 = np.sum(f0)
        
        if abs(S0) < 5:
            # Try to fix by flipping signs
            for _ in range(N // 2):
                idx = rng.integers(0, N)
                if f0[idx] * S0 < 0:
                    f0[idx] = -f0[idx]
                    S0 = np.sum(f0)
            if abs(S0) < 5:
                continue
        
        remaining_time = DEADLINE - time.time()
        time_budget = min(10.0, remaining_time * 0.4)
        
        if time_budget < 0.5:
            break
        
        f_local, score_local = _local_search_binary_fast(f0, time_budget=time_budget)
        
        if score_local < best_score:
            best_score = score_local
            best_f = f_local.copy()
    
    # Phase 3: Sparse annealing
    for restart in range(4):
        if time.time() > DEADLINE:
            break
        
        K = rng.integers(8, 30)
        support = list(rng.choice(N, size=K, replace=False))
        signs = [1.0 if rng.random() < 0.5 else -1.0 for _ in range(K)]
        
        S = sum(signs)
        if abs(S) < 3:
            signs = [1.0] * K
        
        f_sparse = np.zeros(N)
        for pos, s in zip(support, signs):
            f_sparse[pos] = s
        
        remaining_time = DEADLINE - time.time()
        time_budget = min(12.0, remaining_time * 0.25)
        
        if time_budget < 1.0:
            break
        
        f_sa, score_sa = _sparse_annealing(f_sparse, time_budget=time_budget)
        
        if score_sa < best_score:
            best_score = score_sa
            best_f = f_sa.copy()
    
    # Phase 4: Real-valued polish on best found
    if time.time() < DEADLINE:
        remaining_time = DEADLINE - time.time()
        if remaining_time > 5:
            time_budget = min(12.0, remaining_time * 0.4)
            f_start = best_f.copy()
            f_local, score_local = _local_search_real_fast(f_start, time_budget=time_budget, step=0.5)
            if score_local < best_score:
                best_score = score_local
                best_f = f_local.copy()
    
    # Phase 5: High-precision polish with tighter t
    if time.time() < DEADLINE:
        remaining_time = DEADLINE - time.time()
        if remaining_time > 3:
            time_budget = min(10.0, remaining_time * 0.5)
            f_polished, score_polished = _polish_with_tighter_t(best_f, time_budget=time_budget)
            if score_polished < best_score:
                best_score = score_polished
                best_f = f_polished.copy()
    
    return best_f
