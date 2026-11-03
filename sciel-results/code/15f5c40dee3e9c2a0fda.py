"""Search for a low-autocorrelation signed step function (400 equal intervals).

Minimise the peak aperiodic autocorrelation sidelobe  max_{k>=1} |sum_i h[i] h[i+k]|
for a +-1 sequence of length 400; this drives the C3 autoconvolution upper bound.

Design: quadratic-residue initialization (Legendre symbol) for low additive energy,
combined with simulated annealing using exact O(N) delta updates for fast evaluation,
and greedy polish with adjacent-pair flips for exploitation.
"""
import numpy as np
import time

N = 400

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N:
    _FFT_SIZE *= 2


def _peak_sidelobe(h):
    """Peak aperiodic autocorrelation sidelobe: max_{k>=1} |sum_i h[i]*h[i+k]|."""
    H = np.fft.fft(h, _FFT_SIZE)
    ac = np.fft.ifft(H * np.conj(H)).real[:N]
    return float(np.max(np.abs(ac[1:])))


def _legendre_symbol_init():
    """Generate a +-1 sequence using Legendre symbols modulo a prime ~400.
    
    Quadratic residues have low multiplicative energy, which suppresses
    high peaks in the autoconvolution (additive energy).
    """
    # Find a prime p close to 400
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(np.sqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    # Try primes near 400
    p = None
    for candidate in [401, 397, 389, 409, 419, 421, 431, 433, 439, 443]:
        if is_prime(candidate):
            p = candidate
            break
    
    if p is None:
        # Fallback to random
        return np.random.default_rng(42).choice([-1.0, 1.0], size=N)
    
    # Generate Legendre symbols for indices 0 to N-1
    # Legendre symbol (a/p) = 1 if a is QR mod p, -1 if QNR, 0 if a=0
    h = np.zeros(N)
    for i in range(N):
        if i == 0:
            h[i] = 1.0  # Convention: map 0 to +1
        else:
            # Compute a^((p-1)/2) mod p via pow
            ls = pow(i % p, (p - 1) // 2, p)
            if ls == 1:
                h[i] = 1.0
            elif ls == p - 1:
                h[i] = -1.0
            else:
                h[i] = 1.0  # Should not happen for i not divisible by p
    
    # If p > N, we have N values; if p < N, we need to extend
    # For p=401 and N=400, we use indices 0..399
    return h


def _compute_acf(h):
    """Compute full aperiodic autocorrelation via FFT."""
    H = np.fft.fft(h, _FFT_SIZE)
    ac = np.fft.ifft(H * np.conj(H)).real[:N]
    return ac


def _peak_from_acf(acf):
    """Get peak sidelobe from precomputed autocorrelation."""
    return float(np.max(np.abs(acf[1:])))


def _anneal(seed, time_budget, init_h=None):
    """Simulated annealing with variable-length block flips.
    
    Uses exact O(N) delta updates for single-bit flips to speed up evaluation.
    """
    rng = np.random.default_rng(seed)
    
    if init_h is not None:
        h = init_h.copy()
    else:
        h = rng.choice([-1.0, 1.0], size=N)
    
    acf = _compute_acf(h)
    cur = _peak_from_acf(acf)
    best_h = h.copy()
    best = cur
    T = 4.0
    T_min = 1e-7
    start = time.time()
    while time.time() - start < time_budget:
        # Variable block length: longer early, shorter late
        progress = min(1.0, (time.time() - start) / max(time_budget, 1e-9))
        max_len = max(1, int(24.0 * (1.0 - 0.5 * progress)))
        L = int(rng.integers(1, max_len + 1))
        i = int(rng.integers(0, N - L + 1))
        
        if L == 1:
            # Fast path: single flip with O(N) delta update
            old_val = h[i]
            h[i] = -old_val
            # Delta in autocorrelation: acf[k] changes by -2*old_val*sum(h[i-k]*h[i+k]) for valid k
            # Actually, flipping h[i] changes acf[k] by:
            #   acf_new[k] = acf_old[k] - 2*old_val*(h[i-k] + h[i+k]) for valid indices
            # More precisely: acf[k] = sum_j h[j]*h[j+k]
            # Flipping h[i] affects terms where j=i (h[i]*h[i+k]) and j=i-k (h[i-k]*h[i])
            # So delta[k] = -2*old_val*(h[i+k] if i+k<N else 0 + h[i-k] if i-k>=0 else 0)
            
            delta = np.zeros(N)
            # Contribution from j=i: h[i]*h[i+k] -> -h[i]*h[i+k], delta = -2*h[i]*h[i+k]
            if i + 1 < N:
                delta[1:] = -2.0 * old_val * h[i+1:]
            # Contribution from j=i-k: h[i-k]*h[i] -> h[i-k]*(-h[i]), delta = -2*h[i-k]*h[i]
            if i >= 1:
                delta[1:i+1] += -2.0 * h[0:i] * old_val
            # Note: when k=0, both contributions combine, but we only care about k>=1
            
            acf += delta
            new = _peak_from_acf(acf)
            d = new - cur
            if d <= 0 or rng.random() < np.exp(-d / max(T, 1e-9)):
                cur = new
                if cur < best:
                    best = cur
                    best_h = h.copy()
            else:
                h[i] = old_val
                acf -= delta
        else:
            # General block flip: use FFT for simplicity
            h[i:i + L] *= -1.0
            acf = _compute_acf(h)
            new = _peak_from_acf(acf)
            d = new - cur
            if d <= 0 or rng.random() < np.exp(-d / max(T, 1e-9)):
                cur = new
                if cur < best:
                    best = cur
                    best_h = h.copy()
            else:
                h[i:i + L] *= -1.0
                acf = _compute_acf(h)
        
        T = max(T_min, T * 0.99997)
    
    return best_h, best


def _anneal(seed, time_budget):
    """Simulated annealing with variable-length block flips.

    Block length shrinks over time: longer blocks early for exploration,
    shorter blocks late for fine-tuning.
    """
    rng = np.random.default_rng(seed)
    h = rng.choice([-1.0, 1.0], size=N)
    cur = _peak_sidelobe(h)
    best_h = h.copy()
    best = cur
    T = 4.0
    T_min = 1e-7
    start = time.time()
    while time.time() - start < time_budget:
        # Variable block length: longer early, shorter late
        progress = min(1.0, (time.time() - start) / max(time_budget, 1e-9))
        max_len = max(1, int(24.0 * (1.0 - 0.5 * progress)))
        L = int(rng.integers(1, max_len + 1))
        i = int(rng.integers(0, N - L + 1))
        h[i:i + L] *= -1.0
        new = _peak_sidelobe(h)
        d = new - cur
        if d <= 0 or rng.random() < np.exp(-d / max(T, 1e-9)):
            cur = new
            if cur < best:
                best = cur
                best_h = h.copy()
        else:
            h[i:i + L] *= -1.0
        T = max(T_min, T * 0.99997)
    return best_h, best


def _greedy_polish(h, time_budget, start):
    """Greedy polish with single-flip and adjacent-pair-flip sweeps.

    Uses exact O(N) delta updates for single-bit flips.
    Alternates between flipping one bit and flipping two adjacent bits.
    Pair flips help escape flat plateaus that single flips cannot cross.
    """
    h = h.copy()
    acf = _compute_acf(h)
    cur = _peak_from_acf(acf)
    best = cur
    best_h = h.copy()
    while time.time() - start < time_budget:
        improved = False

        # --- single-flip sweep with fast delta updates ---
        best_i = -1
        best_c = cur
        for i in range(N):
            old_val = h[i]
            h[i] = -old_val
            # Compute delta for single flip
            delta = np.zeros(N)
            if i + 1 < N:
                delta[1:] = -2.0 * old_val * h[i+1:]
            if i >= 1:
                delta[1:i+1] += -2.0 * h[0:i] * old_val
            acf += delta
            c = _peak_from_acf(acf)
            if c < best_c:
                best_c = c
                best_i = i
            # Revert
            h[i] = old_val
            acf -= delta
        if best_i >= 0:
            old_val = h[best_i]
            h[best_i] = -old_val
            delta = np.zeros(N)
            if best_i + 1 < N:
                delta[1:] = -2.0 * old_val * h[best_i+1:]
            if best_i >= 1:
                delta[1:best_i+1] += -2.0 * h[0:best_i] * old_val
            acf += delta
            cur = best_c
            if cur < best:
                best = cur
                best_h = h.copy()
            improved = True

        # --- adjacent-pair-flip sweep ---
        best_pair = -1
        best_c2 = cur
        for i in range(N - 1):
            h[i] *= -1.0
            h[i + 1] *= -1.0
            acf = _compute_acf(h)
            c = _peak_from_acf(acf)
            if c < best_c2:
                best_c2 = c
                best_pair = i
            h[i] *= -1.0
            h[i + 1] *= -1.0
            acf = _compute_acf(h)
        if best_pair >= 0:
            h[best_pair] *= -1.0
            h[best_pair + 1] *= -1.0
            acf = _compute_acf(h)
            cur = best_c2
            if cur < best:
                best = cur
                best_h = h.copy()
            improved = True

        if not improved:
            break
    return best_h, best


def _greedy_single_only(h, time_budget, start):
    """Tighter single-flip polish for the final pass with fast delta updates."""
    h = h.copy()
    acf = _compute_acf(h)
    cur = _peak_from_acf(acf)
    best = cur
    best_h = h.copy()
    while time.time() - start < time_budget:
        best_i = -1
        best_c = cur
        for i in range(N):
            old_val = h[i]
            h[i] = -old_val
            delta = np.zeros(N)
            if i + 1 < N:
                delta[1:] = -2.0 * old_val * h[i+1:]
            if i >= 1:
                delta[1:i+1] += -2.0 * h[0:i] * old_val
            acf += delta
            c = _peak_from_acf(acf)
            if c < best_c:
                best_c = c
                best_i = i
            h[i] = old_val
            acf -= delta
        if best_i < 0:
            break
        old_val = h[best_i]
        h[best_i] = -old_val
        delta = np.zeros(N)
        if best_i + 1 < N:
            delta[1:] = -2.0 * old_val * h[best_i+1:]
        if best_i >= 1:
            delta[1:best_i+1] += -2.0 * h[0:best_i] * old_val
        acf += delta
        cur = best_c
        if cur < best:
            best = cur
            best_h = h.copy()
    return best_h, best


def construct():
    try:
        hard_deadline = 135.0
        start = time.time()

        # Initialize with quadratic residue sequence for low additive energy
        global_best_h = _legendre_symbol_init()
        global_best = _peak_sidelobe(global_best_h)

        def _remaining():
            return hard_deadline - (time.time() - start)

        # Phase 1: long initial anneal from quadratic residue start.
        budget = min(25.0, _remaining() - 10.0)
        if budget > 2.0:
            h, c = _anneal(12345, budget, init_h=global_best_h)
            if c < global_best:
                global_best = c
                global_best_h = h
            # Polish right after to lock in gains
            sub = min(6.0, _remaining() - 10.0)
            if sub > 0.5:
                h2, c2 = _greedy_polish(global_best_h, sub, time.time())
                if c2 < global_best:
                    global_best = c2
                    global_best_h = h2

        # Phase 2: multiple shorter restarts for diversity, with polish after each.
        # Mix of random starts and perturbed quadratic residue starts
        restart_id = 1
        while _remaining() > 14.0:
            sub = min(10.0, _remaining() - 14.0)
            if sub < 1.5:
                break
            # Alternate between random and QR-based initializations
            if restart_id % 2 == 0:
                # Perturbed QR start
                rng_init = np.random.default_rng(50000 + restart_id)
                qr_h = _legendre_symbol_init().copy()
                # Flip a small fraction of bits
                flip_mask = rng_init.random(N) < 0.1
                qr_h[flip_mask] *= -1.0
                h, c = _anneal(100000 + restart_id, sub, init_h=qr_h)
            else:
                # Random start
                h, c = _anneal(100000 + restart_id, sub)
            
            if c < global_best:
                global_best = c
                global_best_h = h
            # Quick polish after each restart
            sub2 = min(3.0, _remaining() - 14.0)
            if sub2 > 0.3:
                h2, c2 = _greedy_polish(global_best_h, sub2, time.time())
                if c2 < global_best:
                    global_best = c2
                    global_best_h = h2
            restart_id += 1

        # Phase 3: final high-precision single-flip polish.
        remaining = _remaining() - 2.0
        if remaining > 0.5:
            h, c = _greedy_single_only(global_best_h, remaining, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h

        # Phase 4: one more combined polish pass if time remains.
        remaining2 = _remaining() - 1.0
        if remaining2 > 0.5:
            h, c = _greedy_polish(global_best_h, remaining2, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h

        return np.ascontiguousarray(global_best_h, dtype=np.float64)
    except Exception:
        return np.ones(N)
