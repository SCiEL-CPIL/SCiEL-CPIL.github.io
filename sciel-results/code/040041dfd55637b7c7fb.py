import numpy as np
import time
from scipy.optimize import differential_evolution

N = 600
H = 0.5 / N
X = -0.25 + (np.arange(N) + 0.5) * H

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    def compute_score(heights):
        n_fft = 2048
        F = np.fft.rfft(heights, n=n_fft)
        conv = np.fft.irfft(F * F, n=n_fft) * H
        max_conv = np.max(conv)
        total_mass = np.sum(heights) * H
        return max_conv / (total_mass ** 2)

    # Strategy: Represent the step function as the product of a smooth, nonnegative envelope
    # (raised cosine) and a periodic square-wave-like modulation with a small number of adjustable harmonic phases.
    # This ensures non-negativity by construction and allows efficient optimization of the modulation parameters.
    
    # Envelope: raised cosine on [-0.25, 0.25]
    envelope = 0.5 * (1 + np.cos(2 * np.pi * X / 0.5))
    
    # Modulation: sum of a few harmonics with adjustable phases
    # We'll use 3 harmonics to keep the parameter space small for differential evolution
    def modulation(params):
        # params: [phase1, phase2, phase3, amp1, amp2, amp3]
        phase1, phase2, phase3, amp1, amp2, amp3 = params
        mod = 1.0
        mod += amp1 * np.cos(2 * np.pi * X / 0.5 + phase1)
        mod += amp2 * np.cos(4 * np.pi * X / 0.5 + phase2)
        mod += amp3 * np.cos(6 * np.pi * X / 0.5 + phase3)
        return mod
    
    def objective(params):
        mod = modulation(params)
        heights = envelope * mod
        # Ensure non-negativity
        heights = np.maximum(heights, 0)
        # Check if any negative values were clipped (if so, penalize)
        if np.any(heights < 0):
            return 1e6
        return compute_score(heights)
    
    # Define bounds for the parameters
    # Phases in [0, 2*pi], amplitudes in [-0.5, 0.5] to ensure modulation stays positive
    bounds = [(0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)]
    
    best_val = float('inf')
    best_heights = None
    
    # Run differential evolution
    try:
        result = differential_evolution(objective, bounds, maxiter=100, popsize=10, tol=1e-4, 
                                         seed=42, polish=True, workers=-1,
                                         callback=lambda x, convergence: None)
        
        if result.success and result.fun < best_val:
            best_val = result.fun
            mod = modulation(result.x)
            best_heights = envelope * mod
            best_heights = np.maximum(best_heights, 0)
    except Exception as e:
        pass
    
    # Fallback: try a few fixed phase configurations
    if best_heights is None or time.time() < deadline - 10:
        test_configs = [
            [0, 0, 0, 0.1, 0.05, 0.02],
            [np.pi/2, np.pi/2, np.pi/2, 0.1, 0.05, 0.02],
            [np.pi, np.pi, np.pi, 0.1, 0.05, 0.02],
            [0, np.pi, 0, 0.1, 0.05, 0.02],
            [np.pi/2, 0, np.pi/2, 0.1, 0.05, 0.02],
        ]
        for config in test_configs:
            if time.time() > deadline - 5:
                break
            mod = modulation(config)
            heights = envelope * mod
            heights = np.maximum(heights, 0)
            val = compute_score(heights)
            if val < best_val:
                best_val = val
                best_heights = heights.copy()
    
    # Local refinement if we have a good starting point
    if best_heights is not None and time.time() < deadline - 15:
        current = best_heights.copy()
        current_val = best_val
        np.random.seed(42)
        step_size = 0.005
        for _ in range(300):
            if time.time() > deadline - 5:
                break
            mask = np.random.random(N) < 0.05
            if mask.sum() == 0:
                continue
            pert = (np.random.random(mask.sum()) - 0.5) * step_size
            new_h = current.copy()
            new_h[mask] += pert
            new_h = np.maximum(new_h, 0)
            val = compute_score(new_h)
            if val < current_val:
                current = new_h
                current_val = val
                if current_val < best_val:
                    best_val = current_val
                    best_heights = current.copy()
    
    if best_heights is None:
        # Default: just the envelope
        best_heights = envelope.copy()
    
    return list(best_heights)
