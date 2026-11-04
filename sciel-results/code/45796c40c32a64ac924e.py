"""Randomized Alternating Projection (POCS) with Simulated Annealing for C3' minimization."""
import numpy as np
import time

N = 400
_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2

def _autoconv_values(x):
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return conv[:2 * N - 1]

def _metric(x):
    s = float(np.sum(x))
    if abs(s) < 1e-12:
        return float('inf')
    conv = _autoconv_values(x)
    return float(np.max(np.abs(conv))) / (s * s)

def construct():
    deadline = time.time() + 145.0
    rng = np.random.default_rng(42)
    
    # Initial guess: Random walk with zero mean to ensure non-trivial integral? 
    # Actually, let's start with a simple structured signal.
    # A "bump" function often has lower autoconv peaks than flat.
    x_init = np.exp(-0.5 * (np.linspace(-1, 1, N))**2)
    x_init -= np.mean(x_init) # Remove DC to make it zero-sum? No, integral must be non-zero.
    # The metric divides by s^2. If s is small, metric explodes. 
    # We want s to be large and peaks small.
    # Let's start with ones, which has s=N, peaks ~N. Ratio ~1/N.
    # Wait, if x is constant c, s = c*N. conv[k] = c^2 * (N-|k|). Max is c^2*N.
    # Metric = c^2*N / (c^2*N^2) = 1/N.
    # So baseline score is 1/400 = 0.0025?
    # The task says "score is the upper bound on C3'". "lower is better".
    # The current program returns a sequence. The evaluator computes the score.
    # The current program's score is 1.2336. This suggests the "score" in the prompt 
    # "seed/value" implies the value returned by the evaluator is compared to seed.
    # If seed value is V_seed, and we return x, eval(x) = V. Score = V_seed / V.
    # If V is smaller, Score is larger.
    # So we want to MINIMIZE the metric value.
    
    best_x = np.ones(N)
    best_val = _metric(best_x)
    
    # Strategy: Alternating Projections in Frequency Domain
    # We want to minimize max(|conv|) / s^2.
    # This is equivalent to minimizing max(|conv|) subject to s=1 (normalization).
    # Let's normalize x such that sum(x) = 1.
    # Then metric = max(|conv|).
    
    # We will work with normalized x where sum(x) = 1.
    x = best_x.copy()
    x /= np.sum(x)
    best_val = _metric(x) # This will be max(|conv|) since s=1
    
    # POCS Loop:
    # 1. FFT x -> F
    # 2. Modify F to be "flatter" (reduce peaks in spectrum? No, we want flat magnitude to minimize time-domain peaks via uncertainty principle?)
    # Actually, flat magnitude spectrum corresponds to a "random-like" time domain signal, which tends to have low autoconvolution peaks (decorrelated).
    # So, we want |F[k]| to be constant.
    # 3. Inverse FFT -> x
    # 4. Project x to real, non-negative? No, signed step functions.
    # 5. Project x to have sum 1?
    
    # Let's try a different angle:
    # Minimize max(|conv|) is hard.
    # But we can use the fact that if x is "white noise" (flat spectrum), the autoconvolution is a spike at 0 and small noise elsewhere.
    # The peak at 0 is sum(x^2). The off-peak values are small.
    # For white noise with sum 1, x_i ~ 1/sqrt(N) * (random sign).
    # sum(x) = 0? No, we need sum(x)=1.
    # Let x = a + noise.
    
    # Let's use Simulated Annealing on the phases of the FFT, keeping magnitudes flat.
    # This is similar to the current program's "spectral design" but using SA instead of L-BFGS which was slow/stuck.
    
    # Initialize with flat magnitude spectrum
    target_mags = np.ones(_FFT_SIZE // 2 + 1)
    # DC component must be 1 (since sum(x)=1)
    target_mags[0] = 1.0
    
    # Random phases
    phases = rng.uniform(0, 2 * np.pi, size=len(target_mags))
    phases[0] = 0.0
    
    # Reconstruct x
    F = target_mags * np.exp(1j * phases)
    x = np.fft.irfft(F, _FFT_SIZE)[:N]
    
    # Normalize x to have sum 1
    s = np.sum(x)
    if abs(s) > 1e-12:
        x /= s
    
    current_val = _metric(x)
    best_x = x.copy()
    best_val = current_val
    
    # Simulated Annealing
    T = 1.0
    T_min = 1e-6
    alpha = 0.999
    
    steps = 0
    while time.time() < deadline - 5.0:
        # Perturb phases
        idx = rng.integers(1, len(phases))
        delta_phase = rng.normal(0, T)
        
        old_phase = phases[idx]
        phases[idx] += delta_phase
        
        # Reconstruct
        F = target_mags * np.exp(1j * phases)
        x_new = np.fft.irfft(F, _FFT_SIZE)[:N]
        s_new = np.sum(x_new)
        if abs(s_new) < 1e-12:
            phases[idx] = old_phase
            continue
        x_new /= s_new
        
        new_val = _metric(x_new)
        
        # Metropolis criterion
        if new_val < current_val or rng.random() < np.exp(-(new_val - current_val) / T):
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_x = x_new.copy()
        else:
            phases[idx] = old_phase
        
        T *= alpha
        if T < T_min:
            # Reheat
            T = 0.5
            # Randomize some phases to escape local minima
            num_reheat = 10
            reheat_idx = rng.choice(np.arange(1, len(phases)), size=num_reheat, replace=False)
            phases[reheat_idx] = rng.uniform(0, 2 * np.pi, size=num_reheat)
            # Recalculate current state
            F = target_mags * np.exp(1j * phases)
            x_curr = np.fft.irfft(F, _FFT_SIZE)[:N]
            s_curr = np.sum(x_curr)
            if abs(s_curr) > 1e-12:
                x_curr /= s_curr
                current_val = _metric(x_curr)
            else:
                # Reset
                phases = rng.uniform(0, 2 * np.pi, size=len(phases))
                phases[0] = 0
                F = target_mags * np.exp(1j * phases)
                x_curr = np.fft.irfft(F, _FFT_SIZE)[:N]
                s_curr = np.sum(x_curr)
                if abs(s_curr) > 1e-12:
                    x_curr /= s_curr
                    current_val = _metric(x_curr)
        
        steps += 1
        if steps % 1000 == 0:
            # Check time
            if time.time() > deadline - 10.0:
                break

    # Final Local Search to polish
    if time.time() < deadline - 2.0:
        # Coordinate descent on time domain
        x = best_x.copy()
        val = best_val
        no_improve = 0
        step_size = 0.01
        
        while time.time() < deadline - 1.0 and no_improve < 50:
            improved = False
            for i in range(N):
                if time.time() > deadline - 1.0:
                    break
                # Try small perturbations
                for delta in [-step_size, step_size]:
                    x_test = x.copy()
                    x_test[i] += delta
                    s_test = np.sum(x_test)
                    if abs(s_test) < 1e-12:
                        continue
                    x_test_norm = x_test / s_test
                    v_test = _metric(x_test_norm)
                    if v_test < val:
                        val = v_test
                        x = x_test_norm
                        improved = True
                        no_improve = 0
                        break
            if not improved:
                no_improve += 1
                step_size *= 0.8
                if step_size < 1e-5:
                    break
        
        if val < best_val:
            best_val = val
            best_x = x

    return best_x
