<<<<<<< SEARCH
_GLOBAL_ZS_1024 = None

def _get_zs_1024():
    global _GLOBAL_ZS_1024
    if _GLOBAL_ZS_1024 is None:
        angles = 2 * np.pi * np.arange(1024) / 1024
        _GLOBAL_ZS_1024 = np.exp(1j * angles)
    return _GLOBAL_ZS_1024

def _score_fast(coeffs, num_samples=1024):
    """Faster score for local search: dense sampling without refinement."""
    n = len(coeffs)
    zs = _get_zs_1024()
    vals = np.zeros(num_samples, dtype=complex)
    for c in reversed(coeffs):
        vals = vals * zs + c
    return float(np.max(np.abs(vals))) / np.sqrt(n)

def _local_search(coeffs, time_limit=5.0, fast=True):
    """Coordinate descent local search. Flips single bits if it improves the score."""
    start = time.time()
    current = list(coeffs)
    n = len(current)
    score_func = _score_fast if fast else _score
    current_score = score_func(current)
    improved = True
    
    while improved and time.time() - start < time_limit:
        improved = False
        for i in range(n):
            if time.time() - start >= time_limit:
                break
            current[i] *= -1
            new_score = score_func(current)
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                current[i] *= -1
    return current, current_score
=======
_GLOBAL_ZS_1024 = None
_GLOBAL_POWERS = None

def _get_zs_1024():
    global _GLOBAL_ZS_1024
    if _GLOBAL_ZS_1024 is None:
        angles = 2 * np.pi * np.arange(1024) / 1024
        _GLOBAL_ZS_1024 = np.exp(1j * angles)
    return _GLOBAL_ZS_1024

def _get_powers(num_coeffs=24, num_samples=1024):
    global _GLOBAL_POWERS
    if _GLOBAL_POWERS is None:
        zs = _get_zs_1024()
        powers = np.zeros((num_coeffs, num_samples), dtype=complex)
        powers[0] = 1.0
        for k in range(1, num_coeffs):
            powers[k] = powers[k-1] * zs
        _GLOBAL_POWERS = powers
    return _GLOBAL_POWERS

def _score_fast(coeffs, num_samples=1024):
    """Faster score for local search: dense sampling without refinement."""
    n = len(coeffs)
    powers = _get_powers(n, num_samples)
    vals = np.sum(powers * np.array(coeffs)[:, None], axis=0)
    return float(np.max(np.abs(vals))) / np.sqrt(n)

def _local_search(coeffs, time_limit=5.0, fast=True):
    """Coordinate descent local search with incremental updates."""
    start = time.time()
    current = np.array(coeffs, dtype=float)
    n = len(current)
    
    if not fast:
        # Fallback to slow search if fast is disabled (unlikely to be used)
        score_func = _score
        current_score = score_func(current)
        improved = True
        while improved and time.time() - start < time_limit:
            improved = False
            for i in range(n):
                if time.time() - start >= time_limit:
                    break
                current[i] *= -1
                new_score = score_func(current)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    current[i] *= -1
        return current.tolist(), current_score

    # Fast path with incremental updates
    num_samples = 1024
    powers = _get_powers(n, num_samples)
    
    # Initial evaluation
    vals = np.sum(powers * current[:, None], axis=0)
    current_score = float(np.max(np.abs(vals))) / np.sqrt(n)
    
    improved = True
    while improved and time.time() - start < time_limit:
        improved = False
        # Randomize order to avoid getting stuck in specific patterns? 
        # Or just iterate. Let's iterate.
        for i in range(n):
            if time.time() - start >= time_limit:
                break
            
            # Flip coefficient i
            old_val = current[i]
            current[i] = -old_val
            
            # Update vals: P_new(z) = P_old(z) - 2 * c_i * z^i
            delta = -2.0 * old_val * powers[i]
            vals += delta
            
            new_score = float(np.max(np.abs(vals))) / np.sqrt(n)
            
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                # Revert
                current[i] = old_val
                vals -= delta
                
    return current.tolist(), current_score
>>>>>>> REPLACE

This change significantly speeds up `_local_search`. We should also ensure that `construct` uses this faster local search more effectively. The current `construct` uses `_simulated_annealing` which calls `_score_fast` repeatedly. We should update `_simulated_annealing` to also benefit from the fast evaluation, or replace SA with more Iterated Local Search (ILS) which is often more effective for this type of discrete optimization.

Let's update `_simulated_annealing` to use the same incremental logic or just rely on the much faster `_local_search`. Actually, ILS (Local Search + Perturbation) is very strong. Let's replace the SA phases with more ILS.

Also, the initial candidates should be good. GRS is a good start.

Let's modify `construct` to spend more time on ILS using the fast `_local_search`.

