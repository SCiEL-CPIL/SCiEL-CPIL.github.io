"""Rewritten program for the third autocorrelation inequality task."""
import numpy as np
import time

N = 400
# The interval is [-1/4, 1/4], so each step has width h = (1/2)/400 = 1/800.
# For a step function f(x) = sum_i s_i * h * 1_{[x_i, x_{i+1})}(x),
# the autoconvolution g(t) = (f*f)(t) is also a step function with steps of width h.
# Specifically, g_k = h^2 * sum_{i+j=k} s_i s_j for k=0..2N-2.
# The upper bound on C3 is max_k |g_k|.
# We want to minimize max_k |sum_{i+j=k} s_i s_j| over s_i in {-1, 1}.
# Note: sum_{i+j=k} s_i s_j is the autocorrelation at lag k (with s indexed 0..N-1).
# Actually, if we define the sequence a_0..a_{N-1} with a_i = s_i, then
# the autocorrelation at lag k (for k=0..N-1) is sum_{i=0}^{N-1-k} a_i a_{i+k}.
# But here the convolution sums over i+j=k, which is different.
# Let's re-examine: f(x) = sum_{i=0}^{N-1} s_i h * 1_{[x_i, x_{i+1})}(x) where x_i = -1/4 + i*h.
# (f*f)(t) = int f(u) f(t-u) du.
# For t in [t_k, t_{k+1}) where t_k = -1/2 + k*h, the overlap length is h * (number of pairs (i,j) with x_i + x_j in [t_k, t_{k+1})).
# x_i + x_j = -1/2 + (i+j)*h. So x_i+x_j is in [t_k, t_{k+1}) iff i+j = k.
# Thus (f*f)(t_k) = h^2 * sum_{i+j=k, 0<=i,j<N} s_i s_j.
# We want to minimize max_k |sum_{i+j=k} s_i s_j|.

def _compute_c3_bound(s):
    """Compute the upper bound on C3 for a given sign sequence s."""
    # s is a 1D array of shape (N,) with values +/-1.
    # Compute all convolution sums sum_{i+j=k} s_i s_j for k=0..2N-2.
    # This is equivalent to the convolution of s with s reversed? No.
    # Let's use FFT for efficiency.
    # The sum sum_{i+j=k} a_i b_j is the k-th coefficient of the product of polynomials A(z)B(z) where A(z)=sum a_i z^i, B(z)=sum b_j z^j.
    # Here a_i = s_i, b_j = s_j. So we want the coefficients of S(z)^2 where S(z) = sum_{i=0}^{N-1} s_i z^i.
    # We can compute this via FFT.
    n_fft = 2 * N
    S = np.zeros(n_fft)
    S[:N] = s
    S2 = np.fft.fft(S)
    coeffs = np.fft.ifft(S2 * S2).real
    # coeffs[k] = sum_{i+j=k} s_i s_j for k=0..2N-2 (approximately, due to floating point)
    # We take the max absolute value.
    bound = np.max(np.abs(coeffs[:2*N-1]))
    return bound

def _thue_morse(n):
    """Generate the Thue-Morse sequence of length n as +/-1."""
    # Thue-Morse: t(n) = (-1)^{popcount(n)}.
    # We want s_i = t(i) for i=0..n-1.
    # t(0)=1, t(1)=-1, t(2)=-1, t(3)=1, ...
    # This sequence has good autocorrelation properties.
    # Let's generate it.
    seq = np.ones(n, dtype=np.int8)
    for i in range(1, n):
        # t(i) = t(i >> 1) * (-1)^{(i & 1)}
        # Actually, t(2k) = t(k), t(2k+1) = -t(k).
        # Let's build it iteratively.
        pass
    # Better: use the property that t(n) = (-1)^{number of 1s in binary representation of n}.
    # We can compute popcount for each i.
    # For n=400, this is fast.
    seq = np.array([(-1) ** bin(i).count('1') for i in range(n)], dtype=np.int8)
    return seq

def _local_search(s, max_iters=5000, time_limit=10.0):
    """Local search by flipping individual signs to reduce the C3 bound."""
    s = s.copy()
    current_bound = _compute_c3_bound(s)
    start_time = time.time()
    
    # Precompute the FFT of s for efficiency.
    n_fft = 2 * N
    S = np.zeros(n_fft)
    S[:N] = s
    S_fft = np.fft.fft(S)
    
    # For a flip at position p: s_p -> -s_p.
    # S(z) -> S(z) - 2 s_p z^p.
    # S(z)^2 -> S(z)^2 - 4 s_p z^p S(z) + 4 s_p^2 z^{2p} = S(z)^2 - 4 s_p z^p S(z) + 4 z^{2p}.
    # The change in the k-th coefficient is -4 s_p * (coeff of z^{k-p} in S(z)) + 4 * delta_{k,2p}.
    # coeff of z^{k-p} in S(z) is s_{k-p} if 0 <= k-p < N, else 0.
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        best_delta = 0
        best_pos = -1
        # Try a random subset of positions for speed.
        # For N=400, trying all 400 positions each iteration is feasible.
        for p in range(N):
            # Compute the change in the bound if we flip position p.
            # new_coeffs = old_coeffs + delta_coeffs
            # delta_coeffs[k] = -4 * s[p] * (s[k-p] if 0<=k-p<N else 0) + 4 * (1 if k==2p else 0)
            # We need to find the max absolute value of (old_coeffs + delta_coeffs).
            # This is expensive to do for all k. Instead, we can approximate by looking at the current peak.
            # Alternatively, we can compute the full new bound for a few candidate positions.
            pass
        
        # A simpler approach: just try all flips and compute the new bound exactly.
        # For N=400, 400 flips * 4000 FFT operations might be too slow.
        # Let's use the incremental update.
        
        # Actually, let's just do a simple random restart local search.
        # Pick a random position, flip it, compute new bound. If better, keep it.
        p = np.random.randint(N)
        s_new = s.copy()
        s_new[p] = -s_new[p]
        new_bound = _compute_c3_bound(s_new)
        if new_bound < current_bound:
            s = s_new
            current_bound = new_bound
        else:
            # With some probability, accept worse solutions to escape local minima.
            if np.random.rand() < 0.1:
                s = s_new
                current_bound = new_bound
    
    return s, current_bound

def _pairwise_swap_search(s, max_iters=1000, time_limit=10.0):
    """Local search by swapping pairs of signs."""
    s = s.copy()
    current_bound = _compute_c3_bound(s)
    start_time = time.time()
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        p1 = np.random.randint(N)
        p2 = np.random.randint(N)
        if p1 == p2:
            continue
        s_new = s.copy()
        s_new[p1], s_new[p2] = s_new[p2], s_new[p1]
        new_bound = _compute_c3_bound(s_new)
        if new_bound < current_bound:
            s = s_new
            current_bound = new_bound
    
    return s, current_bound

def construct():
    """Construct a signed step function with 400 intervals to minimize the C3 bound."""
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, leave margin for safety.
    
    # Start with Thue-Morse sequence, which has good autocorrelation properties.
    best_s = _thue_morse(N)
    best_bound = _compute_c3_bound(best_s)
    
    # Perform local search from the Thue-Morse starting point.
    if time.time() < deadline - 30.0:
        s_ls, bound_ls = _local_search(best_s, max_iters=20000, time_limit=20.0)
        if bound_ls < best_bound:
            best_s = s_ls
            best_bound = bound_ls
    
    # Perform pairwise swap search.
    if time.time() < deadline - 30.0:
        s_sw, bound_sw = _pairwise_swap_search(best_s, max_iters=10000, time_limit=20.0)
        if bound_sw < best_bound:
            best_s = s_sw
            best_bound = bound_sw
    
    # Try a few random restarts.
    for restart in range(10):
        if time.time() > deadline - 30.0:
            break
        s_rand = np.random.choice([-1, 1], size=N)
        s_ls, bound_ls = _local_search(s_rand, max_iters=5000, time_limit=5.0)
        if bound_ls < best_bound:
            best_s = s_ls
            best_bound = bound_ls
    
    # Final high-precision polish: try all single flips.
    if time.time() < deadline - 10.0:
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            current_bound = _compute_c3_bound(best_s)
            for p in range(N):
                if time.time() > deadline - 5.0:
                    break
                s_new = best_s.copy()
                s_new[p] = -s_new[p]
                new_bound = _compute_c3_bound(s_new)
                if new_bound < current_bound:
                    best_s = s_new
                    current_bound = new_bound
                    improved = True
    
    return best_s
