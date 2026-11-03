<<<<<<< SEARCH
"""Rewritten program: MODEL-AND-OPTIMIZE for signed step function minimizing max|autoconv| / (sum)^2."""
import time
import numpy as np

N = 400
DEADLINE = 150.0
START_TIME = time.time()


def _score(f):
    """Compute the ratio max|autoconv| / (sum(f))^2."""
    s = np.sum(f)
    if abs(s) < 1e-12:
        return float('inf')
    # Compute full autoconvolution
    auto = np.correlate(f, f, mode='full')
    max_abs = np.max(np.abs(auto))
    return max_abs / (s * s)


def _time_left():
    return DEADLINE - (time.time() - START_TIME)


def construct():
    # Constant function: ratio = 1/N, which is provably optimal for |f_j|<=1
    # By Cauchy-Schwarz: autoconv[0] = sum(f^2) >= (sum f)^2 / N
    # So max|autoconv| >= (sum f)^2 / N, giving ratio >= 1/N
    # Equality holds iff all f_j equal, so all-ones achieves the lower bound.
    return np.ones(N)
=======
"""Spectral concentration approach: minimize max|autoconv|/(sum)^2 via continuous sign relaxation."""
import time
import numpy as np
from scipy.optimize import minimize

N = 400
DEADLINE = 145.0  # Safety margin before 150s kill
START_TIME = time.time()


def _autoconv_from_fft(f):
    """Compute autoconvolution using FFT for efficiency."""
    n = len(f)
    size = 2 * n
    F = np.fft.fft(f, size=size)
    power = F * np.conj(F)
    auto = np.fft.ifft(power).real[:n]
    return auto


def _score_continuous(x):
    """Score for continuous relaxation: maximize flatness of magnitude spectrum."""
    # x is in [-1, 1] range, representing soft signs
    f = x
    s = np.sum(f)
    if abs(s) < 1e-10:
        return 1e6
    
    # Compute autoconvolution
    auto = _autoconv_from_fft(f)
    max_abs = np.max(np.abs(auto))
    return max_abs / (s * s)


def _spectral_flatness_objective(x):
    """
    Objective: minimize variance of |F(f)|^2 across frequencies.
    This encourages a flat magnitude spectrum, reducing spectral concentration.
    """
    f = x
    F = np.fft.fft(f)
    power = np.abs(F) ** 2
    
    # Normalize by total power
    total = np.sum(power)
    if total < 1e-10:
        return 1e6
    
    normalized = power / total
    # Variance measures deviation from flatness
    var = np.var(normalized)
    
    # Add penalty for small sum (to avoid degenerate solutions)
    s = np.sum(f)
    sum_penalty = 1e4 * (1.0 / (abs(s) / N + 1e-10)) ** 2
    
    return var + sum_penalty


def _project_to_signs(x):
    """Project continuous values to hard signs with threshold."""
    return np.sign(x)


def _try_continuous_optimization():
    """Run continuous relaxation optimization."""
    if time.time() - START_TIME > DEADLINE:
        return None
    
    best_score = float('inf')
    best_x = None
    
    # Multiple random initializations
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() - START_TIME > DEADLINE:
            break
        
        # Initialize with random signs smoothed
        x0 = np.random.uniform(-1, 1, size=N)
        
        # Try L-BFGS-B with bounds
        try:
            res = minimize(
                _spectral_flatness_objective,
                x0,
                method='L-BFGS-B',
                bounds=[(-1.0, 1.0)] * N,
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if res.success or res.fun < best_score:
                x_opt = res.x
                # Project to hard signs
                f_hard = _project_to_signs(x_opt)
                score = _score_continuous(f_hard)
                
                if score < best_score:
                    best_score = score
                    best_x = f_hard.copy()
        except Exception:
            continue
    
    return best_x, best_score


def _try_structured_patterns():
    """Try various structured patterns as starting points."""
    if time.time() - START_TIME > DEADLINE:
        return None, float('inf')
    
    best_score = float('inf')
    best_f = None
    
    # Pattern 1: Alternating blocks
    for block_size in [1, 2, 4, 8, 16]:
        if time.time() - START_TIME > DEADLINE:
            break
        pattern = np.tile([1, -1], N // (2 * block_size) + 1)[:N]
        pattern = np.repeat(pattern, block_size)[:N]
        score = _score_continuous(pattern)
        if score < best_score:
            best_score = score
            best_f = pattern.copy()
    
    # Pattern 2: M-sequence-like (maximal length sequence)
    # Generate a simple pseudo-random ±1 sequence
    rng = np.random.RandomState(42)
    for _ in range(3):
        if time.time() - START_TIME > DEADLINE:
            break
        f = rng.choice([-1, 1], size=N)
        score = _score_continuous(f)
        if score < best_score:
            best_score = score
            best_f = f.copy()
    
    # Pattern 3: Sine wave modulation
    for freq in [1, 2, 3, 5, 7]:
        if time.time() - START_TIME > DEADLINE:
            break
        t = np.arange(N)
        f = np.sign(np.sin(2 * np.pi * freq * t / N))
        score = _score_continuous(f)
        if score < best_score:
            best_score = score
            best_f = f.copy()
    
    return best_f, best_score


def _local_refinement(f, max_flips=500):
    """Local refinement: try single and double flips to reduce score."""
    if time.time() - START_TIME > DEADLINE:
        return f
    
    current_score = _score_continuous(f)
    improved = True
    flip_count = 0
    
    while improved and flip_count < max_flips:
        improved = False
        
        # Try single flips
        for i in range(0, N, 2):  # Sample every other position for speed
            if time.time() - START_TIME > DEADLINE:
                return f
            
            f_new = f.copy()
            f_new[i] = -f_new[i]
            new_score = _score_continuous(f_new)
            
            if new_score < current_score - 1e-10:
                f = f_new
                current_score = new_score
                improved = True
                flip_count += 1
                if flip_count >= max_flips:
                    break
        
        if not improved:
            # Try double flips (sample pairs)
            for _ in range(100):
                if time.time() - START_TIME > DEADLINE:
                    return f
                
                i, j = np.random.choice(N, 2, replace=False)
                f_new = f.copy()
                f_new[i] = -f_new[i]
                f_new[j] = -f_new[j]
                new_score = _score_continuous(f_new)
                
                if new_score < current_score - 1e-10:
                    f = f_new
                    current_score = new_score
                    improved = True
                    flip_count += 1
                    if flip_count >= max_flips:
                        break
    
    return f


def construct():
    """
    Construct signed step function minimizing max|autoconv|/(sum)^2.
    
    Strategy:
    1. Start with constant sequence (known lower bound)
    2. Try structured patterns
    3. Run continuous relaxation optimization
    4. Local refinement on best candidate
    """
    # Baseline: all ones
    f_baseline = np.ones(N)
    baseline_score = _score_continuous(f_baseline)
    
    best_f = f_baseline.copy()
    best_score = baseline_score
    
    # Try structured patterns
    f_struct, score_struct = _try_structured_patterns()
    if f_struct is not None and score_struct < best_score:
        best_f = f_struct.copy()
        best_score = score_struct
    
    # Try continuous optimization
    if time.time() - START_TIME < DEADLINE - 20:
        result = _try_continuous_optimization()
        if result is not None:
            f_cont, score_cont = result
            if f_cont is not None and score_cont < best_score:
                best_f = f_cont.copy()
                best_score = score_cont
    
    # Local refinement on best candidate
    if time.time() - START_TIME < DEADLINE - 10:
        best_f = _local_refinement(best_f)
        final_score = _score_continuous(best_f)
        if final_score < best_score:
            best_score = final_score
    
    # Return the best found
    return best_f
>>>>>>> REPLACE
