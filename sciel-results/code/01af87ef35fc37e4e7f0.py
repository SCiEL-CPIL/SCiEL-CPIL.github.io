"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt


def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac))


def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(h) / (s * s)


def _local_sign_flip(h, max_passes=5, deadline=None):
    """
    Local optimization: try flipping each element's sign, keep if improves.
    """
    h = h.copy()
    best_val = _score(h)
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            h[i] = -h[i]
            new_val = _score(h)
            if new_val < best_val - 1e-12:
                best_val = new_val
                improved = True
            else:
                h[i] = -h[i]
        if not improved:
            break
    return h, best_val


def _pair_flip_search(h, max_passes=3, deadline=None):
    """
    Local optimization: try flipping pairs of elements to escape local minima.
    """
    h = h.copy()
    best_val = _score(h)
    rng = np.random.default_rng(12345)
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        # Try random pairs
        for _ in range(50):
            if deadline is not None and time.time() > deadline:
                break
            i, j = rng.integers(0, N, size=2)
            if i == j:
                continue
            orig_i, orig_j = h[i], h[j]
            h[i] = -orig_i
            h[j] = -orig_j
            new_val = _score(h)
            if new_val < best_val - 1e-12:
                best_val = new_val
                improved = True
            else:
                h[i] = orig_i
                h[j] = orig_j
        if not improved:
            break
    return h, best_val


def _gen_legendre(n, p):
    """Generate Legendre sequence of length n using prime p."""
    h = np.zeros(n)
    for i in range(n):
        if i % p == 0:
            h[i] = 0.0
        else:
            h[i] = 1.0 if pow(i, (p - 1) // 2, p) == 1 else -1.0
    return h


def _gen_m_sequence(n, poly):
    """
    Generate M-sequence (m-sequence) of length n using a primitive polynomial.
    poly is a list of exponents for the primitive polynomial.
    """
    # For GF(2^k), primitive polynomial x^k + ... + 1
    # This is a simplified generator.
    # For robustness, we'll use a standard known one or random if fails.
    # Actually, for N=400, we can just use a random binary sequence or a known good one.
    # Let's stick to Legendre and simple windows as they are easier to implement correctly for arbitrary N.
    pass


def _gen_sidon_set(n, max_iter=1000):
    """
    Generate a Sidon set (B_2 sequence) of size n using a greedy algorithm.
    A Sidon set has distinct pairwise sums, which helps minimize autoconvolution peaks.
    """
    # Greedy construction of a Sidon set
    # We want a set S such that all pairwise sums s_i + s_j (i <= j) are distinct.
    # This is equivalent to distinct differences for the autoconvolution.
    
    s = [0]
    used_sums = {0}
    
    for _ in range(n - 1):
        # Try to add the next smallest integer
        candidate = 1
        while True:
            # Check if adding candidate creates a duplicate sum
            new_sums = set()
            for x in s:
                new_sums.add(x + candidate)
            # Also need to check candidate + candidate
            new_sums.add(candidate + candidate)
            
            if new_sums & used_sums:
                candidate += 1
            else:
                s.append(candidate)
                used_sums |= new_sums
                break
    
    # Convert to a sequence of length N with +1 for elements in S and -1 otherwise
    s_set = set(s)
    h = np.array([1.0 if i in s_set else -1.0 for i in range(n)])
    return h


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill

    # Initialize best with the known baseline: all ones
    best_h = np.ones(N)
    best_val = _score(best_h)

    # Van der Corput sequence (base-2) for low-discrepancy initialization
    def van_der_corput(n, base=2):
        """Generate first n values of Van der Corput sequence in [0,1)."""
        seq = np.zeros(n)
        for i in range(n):
            f = 1.0 / base
            num = i
            val = 0.0
            while num > 0:
                val += f * (num % base)
                num //= base
                f /= base
            seq[i] = val
        return seq

    # Kronecker sequence
    def kronecker(n, alpha):
        """Generate Kronecker sequence {n*alpha mod 1} for n=0..N-1."""
        return np.mod(np.arange(n) * alpha, 1.0)

    rng = np.random.default_rng(42)

    # Try diverse initialization strategies with local search
    n_restarts = 60
    for r in range(n_restarts):
        if time.time() > deadline:
            break

        # Different initialization strategies
        if r % 5 == 0:
            # Random ±1
            x = rng.choice([-1.0, 1.0], size=N)
        elif r % 5 == 1:
            # Mostly +1 with some random -1
            x = np.ones(N)
            n_neg = rng.integers(N // 4, N // 2)
            neg_idx = rng.choice(N, n_neg, replace=False)
            x[neg_idx] = -1.0
        elif r % 5 == 2:
            # Chirp-like pattern
            t = np.arange(N)
            x = np.sign(np.cos(2 * np.pi * t * (t + 1) / (2 * N)))
        elif r % 5 == 3:
            # Van der Corput based sign pattern
            vdc = van_der_corput(N)
            x = np.sign(vdc - 0.5)
            x[x == 0] = 1.0
        else:
            # Kronecker sequence based
            alpha = (1 + np.sqrt(5)) / 2  # golden ratio
            kr = kronecker(N, alpha % 1)
            x = np.sign(kr - 0.5)
            x[x == 0] = 1.0

        # Ensure sum is not too small; flip if needed
        s = np.sum(x)
        if abs(s) < N * 0.1:
            if s < 0:
                x = -x
            # Add a slight bias toward +1
            x = np.sign(x + 0.3)
            x[x == 0] = 1.0

        val = _score(x)
        if val < best_val:
            best_val = val
            best_h = x.copy()

        # Local search: try flipping individual elements
        improved = True
        max_iters = 100
        for it in range(max_iters):
            if time.time() > deadline:
                break
            if not improved:
                break
            improved = False

            # Try random subset of positions
            trial_indices = rng.choice(N, size=min(40, N), replace=False)
            for idx in trial_indices:
                if time.time() > deadline:
                    break
                orig = x[idx]
                x[idx] = -orig
                new_val = _score(x)
                if new_val < best_val - 1e-12:
                    best_val = new_val
                    best_h = x.copy()
                    improved = True
                else:
                    x[idx] = orig

        # Pair flip search to escape local minima
        if time.time() < deadline - 5:
            x2, val2 = _pair_flip_search(best_h, max_passes=2, deadline=deadline)
            if val2 < best_val:
                best_val = val2
                best_h = x2

    # Final polish: use scipy.optimize on the best solution found
    if time.time() < deadline - 10:
        # Normalize to sum 1 for optimization stability
        s0 = np.sum(best_h)
        if abs(s0) > 1e-6:
            h_norm = best_h / s0

            def obj_constrained(x):
                s = np.sum(x)
                penalty = 100.0 * (s - 1.0) ** 2
                return _autoconv_max_abs(x) + penalty

            try:
                res = opt.minimize(obj_constrained, h_norm, method='BFGS',
                                   options={'maxiter': 50, 'disp': False, 'gtol': 1e-6})
                h_opt = res.x
                # Convert back to signed values
                h_signed = np.sign(h_opt)
                h_signed[h_signed == 0] = 1.0
                val_opt = _score(h_signed)
                if val_opt < best_val:
                    best_val = val_opt
                    best_h = h_signed
            except Exception:
                pass

    # Final local polish on the best solution
    if time.time() < deadline - 2:
        best_h, best_val = _local_sign_flip(best_h, max_passes=3, deadline=deadline)

    return best_h
# EVOLVE-BLOCK-END
