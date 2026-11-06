import math
import time
import numpy as np

_N = 24
_SQRTN = math.sqrt(_N)
_T0 = time.time()
_LIMIT = 150.0

def _time_left():
    return _LIMIT - (time.time() - _T0)

def _rs_poly(n):
    """Generate Rudin-Shapiro coefficients of length n."""
    if n <= 0:
        return []
    if n == 1:
        return [1]
    if n == 2:
        return [1, 1]
    
    # Recursive generation
    # R_{k+1} = R_k S_k
    # S_{k+1} = R_k (-S_k)
    # Start with R_1=[1], S_1=[1]
    R = [1]
    S = [1]
    while len(R) < n:
        new_R = R + S
        new_S = R + [-s for s in S]
        R = new_R
        S = new_S
    return R[:n]

def _calc_peak_fast(coeffs, M=2048):
    """
    Fast peak estimation using dense sampling only.
    Returns max |P(z)| on the grid.
    """
    c = np.array(coeffs, dtype=complex)
    k = np.arange(M)
    theta = 2.0 * np.pi * k / M
    powers = np.arange(_N)
    # Vectorized evaluation
    Z = np.exp(1j * np.outer(theta, powers))
    vals = Z @ c
    return np.max(np.abs(vals))

def _calc_peak(coeffs, M=2048, refine_top=5, refine_iters=20):
    """
    Calculate the peak magnitude of P(z) on the unit circle.
    Uses FFT for dense sampling and golden-section refinement for top peaks.
    """
    c = np.array(coeffs, dtype=complex)
    k = np.arange(M)
    theta = 2.0 * np.pi * k / M
    powers = np.arange(_N)
    Z = np.exp(1j * np.outer(theta, powers))
    vals = Z @ c
    
    mag2 = np.abs(vals) ** 2
    peak_val = np.max(mag2)
    
    # Find indices of top peaks
    if refine_top > 0 and M > 10:
        top_indices = np.argsort(mag2)[-refine_top:][::-1]
        best_peak = peak_val
        for idx in top_indices:
            # Refine around theta[idx]
            # Use a window of +/- 2 grid steps to be safe
            window = 2.0 * 2.0 * math.pi / M
            lo = theta[idx] - window
            hi = theta[idx] + window
            
            def f(a):
                z_a = np.exp(1j * a)
                z_powers = np.power(z_a, powers)
                s = np.dot(c, z_powers)
                return s.real * s.real + s.imag * s.imag

            # Golden section search
            gr = (math.sqrt(5) - 1) / 2
            x1 = hi - gr * (hi - lo)
            x2 = lo + gr * (hi - lo)
            f1 = f(x1)
            f2 = f(x2)
            
            for _ in range(refine_iters):
                if f1 < f2:
                    hi = x2
                    x2 = x1
                    f2 = f1
                    x1 = hi - gr * (hi - lo)
                    f1 = f(x1)
                else:
                    lo = x1
                    x1 = x2
                    f1 = f2
                    x2 = lo + gr * (hi - lo)
                    f2 = f(x2)
            
            local_peak = max(f1, f2)
            if local_peak > best_peak:
                best_peak = local_peak
        
        peak_val = best_peak
    
    return math.sqrt(peak_val)

def _local_search(seed, M=2048, max_time=10.0):
    """
    Perform local search to minimize the peak.
    Tries single flips, then double flips if stuck.
    """
    cur = list(seed)
    cur_peak = _calc_peak(cur, M)
    best = list(cur)
    best_peak = cur_peak
    
    start_time = time.time()
    
    while time.time() - start_time < max_time:
        improved = False
        
        # Single flip pass
        for j in range(_N):
            if time.time() - start_time > max_time:
                break
            cur[j] = -cur[j]
            p = _calc_peak(cur, M, refine_top=3)
            if p < cur_peak - 1e-12:
                cur_peak = p
                improved = True
                if cur_peak < best_peak - 1e-12:
                    best = list(cur)
                    best_peak = cur_peak
            else:
                cur[j] = -cur[j]
        
        if not improved:
            # Try double flips
            double_improved = False
            for j in range(_N):
                if time.time() - start_time > max_time:
                    break
                for k in range(j + 1, _N):
                    if time.time() - start_time > max_time:
                        break
                    cur[j] = -cur[j]
                    cur[k] = -cur[k]
                    p = _calc_peak(cur, M, refine_top=3)
                    if p < cur_peak - 1e-12:
                        cur_peak = p
                        double_improved = True
                        if cur_peak < best_peak - 1e-12:
                            best = list(cur)
                            best_peak = cur_peak
                    else:
                        cur[j] = -cur[j]
                        cur[k] = -cur[k]
            
            if not double_improved:
                break
            else:
                improved = True
        
    return best, best_peak

def _simulated_annealing(seed, max_time=60.0, M=1024):
    """
    Simulated Annealing to find a better starting point.
    Uses fast peak estimation.
    """
    cur = list(seed)
    cur_energy = _calc_peak_fast(cur, M)
    best = list(cur)
    best_energy = cur_energy
    
    start_time = time.time()
    T = 0.5 # Initial temperature (arbitrary scale, energy is ~3-5)
    cooling = 0.995
    
    while time.time() - start_time < max_time:
        if _time_left() < 20.0:
            break
            
        # Random move: flip a random coefficient
        j = np.random.randint(_N)
        cur[j] = -cur[j]
        new_energy = _calc_peak_fast(cur, M)
        
        delta = new_energy - cur_energy
        if delta < 0 or np.random.random() < math.exp(-delta / T):
            cur_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best = list(cur)
        else:
            cur[j] = -cur[j]
            
        T *= cooling
        if T < 1e-4:
            T = 1e-4
            
    return best, best_energy

def construct():
    global _T0
    _T0 = time.time()
    
    # Primary Seed: Rudin-Shapiro
    seed_rs = _rs_poly(_N)
    
    best_coeffs = seed_rs
    best_peak = _calc_peak(best_coeffs, M=2048)
    
    # Run Simulated Annealing to escape local minima
    # Use a generous time budget
    sa_time = 80.0
    if _time_left() > sa_time + 20.0:
        sa_coeffs, sa_energy = _simulated_annealing(seed_rs, max_time=sa_time, M=1024)
        # Verify with high precision
        sa_peak = _calc_peak(sa_coeffs, M=2048)
        if sa_peak < best_peak:
            best_coeffs = sa_coeffs
            best_peak = sa_peak
    
    # Run local search on the best candidate found
    if _time_left() > 10.0:
        res_coeffs, res_peak = _local_search(best_coeffs, M=2048, max_time=min(20.0, _time_left() - 5.0))
        if res_peak < best_peak:
            best_coeffs = res_coeffs
            best_peak = res_peak

    # If time remains, try a few random restarts with SA
    if _time_left() > 30.0:
        rng = np.random.default_rng(42)
        for _ in range(2):
            if _time_left() < 25.0:
                break
            seed_rand = rng.choice([-1, 1], size=_N).tolist()
            sa_coeffs, sa_energy = _simulated_annealing(seed_rand, max_time=min(20.0, _time_left() - 10.0), M=1024)
            sa_peak = _calc_peak(sa_coeffs, M=2048)
            if sa_peak < best_peak:
                best_coeffs = sa_coeffs
                best_peak = sa_peak
                # Polish
                if _time_left() > 10.0:
                    res_coeffs, res_peak = _local_search(best_coeffs, M=2048, max_time=min(10.0, _time_left() - 5.0))
                    if res_peak < best_peak:
                        best_coeffs = res_coeffs
                        best_peak = res_peak

    # Final validation and formatting
    best_coeffs = [1 if c >= 0 else -1 for c in best_coeffs]
    
    return best_coeffs

if __name__ == "__main__":
    # Self-test
    coeffs = construct()
    print(coeffs)
    print(len(coeffs))
    print(_calc_peak(coeffs, M=4096))
