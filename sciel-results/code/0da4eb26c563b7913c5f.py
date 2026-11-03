"""
Spectral Phase Design Strategy for Erdos Minimum Overlap.

Instead of optimizing the heights h directly, we assume the optimal solution
is binary (0 or 1) with exactly N/2 ones. We represent the binary sequence
as a sum of Fourier components. By fixing the amplitudes of the Fourier
components (based on a base pattern like a square wave or random binary)
and optimizing only the phases, we ensure the resulting signal has the
correct energy and structure, while the phase optimization minimizes the
peak sidelobe level (PSL) of the autocorrelation.

The objective is to minimize max_k overlap_k = N/2 - autocorr_k.
This is equivalent to maximizing min_k autocorr_k.
We use a smooth surrogate: -logsumexp(beta * autocorr) to approximate -min(autocorr).
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 100

# EVOLVE-BLOCK-START

def _autocorr_from_h(h):
    """Compute circular autocorrelation of h."""
    H = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(H * np.conj(H)))
    return autocorr

def _max_overlap_from_autocorr(autocorr):
    """Compute max overlap from autocorrelation."""
    return N / 2.0 - np.min(autocorr)

def _smooth_objective(autocorr, beta=100.0):
    """
    Smooth surrogate for -min(autocorr).
    We want to maximize min(autocorr), so we minimize -min(autocorr).
    Using log-sum-exp: -logsumexp(beta * autocorr) approximates -min(autocorr)
    as beta -> infinity.
    """
    max_ac = np.max(autocorr)
    # Shift for numerical stability
    shifted = autocorr - max_ac
    logsumexp = max_ac + np.log(np.sum(np.exp(beta * shifted))) / beta
    return -logsumexp

def _smooth_gradient(autocorr, beta=100.0):
    """
    Gradient of -logsumexp(beta * autocorr) with respect to autocorr.
    d(-LSE)/d(ac_k) = -exp(beta * (ac_k - max_ac)) / sum(exp(beta * (ac_j - max_ac)))
    """
    max_ac = np.max(autocorr)
    shifted = autocorr - max_ac
    exp_shifted = np.exp(beta * shifted)
    weights = exp_shifted / np.sum(exp_shifted)
    return -weights

def _h_from_phases(amps, phases):
    """
    Reconstruct h from fixed amplitudes and variable phases.
    h[n] = sum_k amps[k] * cos(2*pi*k*n/N + phases[k])
    We only use real-valued symmetric amplitudes to keep h real.
    """
    h = np.zeros(N)
    for k, amp in enumerate(amps):
        if k == 0:
            # DC component is fixed, no phase
            h += amp
        elif k == N // 2:
            # Nyquist frequency
            h += amp * np.cos(phases[k]) * np.cos(np.pi * np.arange(N))
        else:
            # Pair k and N-k
            # h[n] += 2 * amp * cos(2*pi*k*n/N + phase)
            # To maintain real h, we assume amplitude is for the pair.
            # Actually, if we define amps[k] as the magnitude of the complex coeff for k,
            # and we enforce symmetry, we just sum.
            # Let's use the standard real form:
            # h[n] = c0 + sum_{k=1}^{N/2-1} c_k cos(2*pi*k*n/N + phi_k) + c_{N/2} cos(pi*n)
            # Here amps[k] corresponds to c_k.
            angles = 2 * np.pi * k * np.arange(N) / N + phases[k]
            h += amp * np.cos(angles)
    return h

def _project_to_binary(h, num_ones=None):
    """
    Project h to a binary sequence with exactly num_ones ones.
    If num_ones is None, use round(N/2).
    """
    if num_ones is None:
        num_ones = N // 2
    
    # Sort indices by h value
    indices = np.argsort(h)[::-1]
    h_bin = np.zeros(N)
    h_bin[indices[:num_ones]] = 1.0
    return h_bin

def _initialize_spectral(N):
    """
    Initialize amplitudes and phases.
    We start with a random binary sequence, compute its FFT,
    and fix the amplitudes, keeping the phases as variables.
    """
    # Start with a random binary sequence
    h_init = np.zeros(N)
    h_init[np.random.choice(N, N//2, replace=False)] = 1.0
    
    H = np.fft.fft(h_init)
    
    # Extract amplitudes and phases
    # We will optimize phases for k = 1 to N/2 - 1.
    # k=0 is fixed (DC). k=N/2 is fixed (Nyquist, real).
    num_vars = N // 2 - 1
    amps = np.zeros(num_vars)
    phases_init = np.zeros(num_vars)
    
    for i in range(num_vars):
        k = i + 1
        amps[i] = np.abs(H[k])
        phases_init[i] = np.angle(H[k])
    
    # Also store the DC and Nyquist components to reconstruct properly
    c0 = np.real(H[0])
    c_nyq = np.real(H[N//2]) if N % 2 == 0 else 0.0
    
    return amps, phases_init, c0, c_nyq

def _reconstruct_h(amps, phases, c0, c_nyq):
    """Reconstruct h from spectral parameters."""
    h = np.full(N, c0)
    n = np.arange(N)
    
    for i, amp in enumerate(amps):
        k = i + 1
        h += 2.0 * amp * np.cos(2 * np.pi * k * n / N + phases[i])
        
    if N % 2 == 0:
        h += c_nyq * np.cos(np.pi * n)
        
    return h

def _spectral_optimize(amps, phases_init, c0, c_nyq, time_limit=30.0, beta=50.0):
    """
    Optimize phases to minimize smooth max overlap.
    """
    start_time = time.time()
    
    def objective(phases):
        if time.time() - start_time > time_limit:
            return 1e10
        h = _reconstruct_h(amps, phases, c0, c_nyq)
        # Project to binary to ensure we are comparing valid binary sequences?
        # No, the spectral representation might not be binary.
        # The task allows h in [0,1].
        # However, the "Spectral Phase Design" insight suggests that
        # minimizing PSL for the continuous h generated by fixed amps
        # will lead to a near-binary structure.
        # Let's evaluate the continuous h first.
        autocorr = _autocorr_from_h(h)
        return _smooth_objective(autocorr, beta)
        
    def gradient(phases):
        if time.time() - start_time > time_limit:
            return np.zeros(len(phases))
        h = _reconstruct_h(amps, phases, c0, c_nyq)
        autocorr = _autocorr_from_h(h)
        grad_ac = _smooth_gradient(autocorr, beta)
        
        # Chain rule: d(obj)/d(phi_i) = sum_k grad_ac[k] * d(autocorr_k)/d(phi_i)
        # autocorr_k = sum_n h[n] h[n-k]
        # h[n] = c0 + sum_j 2 amps_j cos(2 pi j n / N + phi_j) + c_nyq cos(pi n)
        # d(h[n])/d(phi_i) = -2 amps_i sin(2 pi i n / N + phi_i)
        
        # This gradient calculation is expensive if done naively for all k.
        # Let's use FFT properties or just numerical gradient if needed,
        # but SLSQP can handle it. Let's try to derive it.
        # Actually, for a 120s budget, a numerical gradient might be too slow
        # if we have many iterations.
        # Let's provide a numerical gradient to be safe and simple, 
        # or use a finite difference step.
        # Given the complexity, let's use a simple finite difference gradient.
        
        eps = 1e-6
        grad = np.zeros(len(phases))
        for i in range(len(phases)):
            phases_plus = phases.copy()
            phases_plus[i] += eps
            h_plus = _reconstruct_h(amps, phases_plus, c0, c_nyq)
            ac_plus = _autocorr_from_h(h_plus)
            obj_plus = _smooth_objective(ac_plus, beta)
            
            phases_minus = phases.copy()
            phases_minus[i] -= eps
            h_minus = _reconstruct_h(amps, phases_minus, c0, c_nyq)
            ac_minus = _autocorr_from_h(h_minus)
            obj_minus = _smooth_objective(ac_minus, beta)
            
            grad[i] = (obj_plus - obj_minus) / (2 * eps)
            
        return grad

    try:
        result = minimize(
            objective, phases_init,
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 100, 'maxfun': 500, 'maxls': 50}
        )
        if result.success:
            return result.x
    except Exception:
        pass
    
    return phases_init

def construct():
    start_time = time.time()
    deadline = 150.0 - 5.0
    np.random.seed(42)
    
    best_h = np.full(N, 0.5)
    best_score = _max_overlap_from_autocorr(_autocorr_from_h(best_h))
    
    # Phase 1: Spectral Phase Optimization from multiple random binary starts
    num_starts = 20
    for i in range(num_starts):
        if time.time() > deadline - 20:
            break
            
        # Initialize from a random binary sequence
        amps, phases_init, c0, c_nyq = _initialize_spectral(N)
        
        # Optimize phases
        time_per_start = min(10.0, (deadline - 20 - (time.time() - start_time)) / max(1, num_starts - i))
        phases_opt = _spectral_optimize(amps, phases_init, c0, c_nyq, time_limit=time_per_start, beta=50.0)
        
        # Reconstruct h
        h_opt = _reconstruct_h(amps, phases_opt, c0, c_nyq)
        
        # Clip to [0,1] and normalize sum?
        # The spectral reconstruction might not have sum exactly N/2 or be in [0,1].
        # We need to project to the feasible set.
        # However, the spectral method preserves the "shape".
        # Let's project to [0,1] and fix sum.
        
        # A better projection for spectral results:
        # 1. Clip to [0,1]
        h_opt = np.clip(h_opt, 0, 1)
        
        # 2. Adjust sum to N/2
        current_sum = np.sum(h_opt)
        target_sum = N / 2.0
        if abs(current_sum - target_sum) > 1e-10:
            diff = target_sum - current_sum
            if diff > 0:
                room = 1.0 - h_opt
                total_room = np.sum(room)
                if total_room > 1e-10:
                    h_opt += diff * room / total_room
            else:
                total_val = np.sum(h_opt)
                if total_val > 1e-10:
                    h_opt += diff * h_opt / total_val
            h_opt = np.clip(h_opt, 0, 1)
            
        # 3. Final sum correction (exact)
        current_sum = np.sum(h_opt)
        if abs(current_sum - target_sum) > 1e-12:
            residual = target_sum - current_sum
            if residual > 0:
                mask = h_opt < 1.0
                if np.sum(mask) > 0:
                    h_opt[mask] += residual / np.sum(mask)
            else:
                mask = h_opt > 0.0
                if np.sum(mask) > 0:
                    h_opt[mask] += residual / np.sum(mask)
            h_opt = np.clip(h_opt, 0, 1)
            
        # Evaluate
        score = _max_overlap_from_autocorr(_autocorr_from_h(h_opt))
        
        if score < best_score:
            best_score = score
            best_h = h_opt.copy()
            
        # Also try the binary version of this h
        h_bin = _project_to_binary(h_opt)
        score_bin = _max_overlap_from_autocorr(_autocorr_from_h(h_bin))
        if score_bin < best_score:
            best_score = score_bin
            best_h = h_bin.copy()

    # Phase 2: Local Search Polish on the best continuous h
    # Use a simple coordinate descent to refine the continuous h
    if time.time() < deadline - 10:
        h = best_h.copy()
        best_score_local = best_score
        step = 0.05
        for _ in range(500):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in np.random.permutation(N):
                for delta in [step, -step]:
                    new_h = h.copy()
                    new_h[i] = np.clip(new_h[i] + delta, 0, 1)
                    # Quick sum check
                    if abs(np.sum(new_h) - N/2.0) > 1e-8:
                        # Simple projection
                        diff = N/2.0 - np.sum(new_h)
                        if diff > 0:
                            mask = new_h < 1.0
                            if np.sum(mask) > 0:
                                new_h[mask] += diff / np.sum(mask)
                        else:
                            mask = new_h > 0.0
                            if np.sum(mask) > 0:
                                new_h[mask] += diff / np.sum(mask)
                        new_h = np.clip(new_h, 0, 1)
                        
                    score = _max_overlap_from_autocorr(_autocorr_from_h(new_h))
                    if score < best_score_local - 1e-8:
                        h = new_h
                        best_score_local = score
                        improved = True
                        break
            if not improved:
                step *= 0.8
                if step < 0.001:
                    break
                
        if best_score_local < best_score:
            best_score = best_score_local
            best_h = h

    return best_h

# EVOLVE-BLOCK-END
