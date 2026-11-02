import numpy as np
import time
import scipy.optimize as opt

N = 100
W = 2.0 / N

def _compute_overlaps(h):
    """
    Compute the overlap for all shifts k.
    overlap[k] = W * (sum_h - autocorr[k])
    """
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    sum_h = np.sum(h)
    overlaps = W * (sum_h - autocorr)
    return overlaps

def _project_to_simplex_box(h, target_sum):
    """
    Project h onto the intersection of the box [0,1] and the hyperplane sum(h) = target_sum.
    """
    lo = np.min(h) - 1.0
    hi = np.max(h) + 1.0
    
    for _ in range(50):
        mid = (lo + hi) / 2.0
        proj = np.clip(h - mid, 0.0, 1.0)
        s = np.sum(proj)
        if s > target_sum:
            lo = mid
        else:
            hi = mid
            
    lambda_opt = (lo + hi) / 2.0
    proj = np.clip(h - lambda_opt, 0.0, 1.0)
    return proj

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Strategy: Spectral Shaping of the zero-mean component u
    # h = 0.5 + u, where sum(u) = 0, -0.5 <= u <= 0.5
    # Overlap[k] = W * (N/2 - autocorr_h[k])
    # autocorr_h[k] = sum_j h_j h_{j+k} = sum_j (0.5+u_j)(0.5+u_{j+k})
    # = N*0.25 + 0.5*sum u_j + 0.5*sum u_{j+k} + sum u_j u_{j+k}
    # Since sum u = 0, autocorr_h[k] = 0.25*N + autocorr_u[k]
    # Overlap[k] = W * (N/2 - (0.25*N + autocorr_u[k]))
    # = W * (0.25*N - autocorr_u[k])
    # We want to minimize max Overlap[k] -> maximize min autocorr_u[k]
    # Note: autocorr_u[0] is fixed by the norm of u.
    # The overlaps are symmetric. overlap[0] = W * (N/2 - sum h^2).
    # To minimize max overlap, we want autocorr_u[k] to be as large as possible for all k != 0.
    
    # We optimize the power spectrum S_u (non-negative) such that the inverse FFT gives u.
    # Let U_k = FFT(u). Then u = IFFT(U).
    # autocorr_u[k] = IFFT(|U|^2)_k.
    # We want to maximize min_k (IFFT(|U|^2)_k) for k != 0, subject to constraints on u.
    
    # Variables: log_power = log(|U_k|^2 + eps) to ensure positivity.
    # Actually, we can just optimize the real and imaginary parts of U, but that's harder to constrain.
    # Let's optimize the power spectrum P_k = |U_k|^2 directly.
    # Constraint: u = IFFT(sqrt(P) * phase). We can fix the phase to be zero (symmetric u) or optimize it?
    # For a real u, U_k must satisfy conjugate symmetry.
    # Let's assume u is real and symmetric? No, u doesn't have to be symmetric.
    # However, the autocorrelation only depends on the power spectrum.
    # So we can choose the phase of U arbitrarily as long as it yields a real u.
    # A simple choice: Phase = 0 for all k? No, that makes u symmetric and even.
    # If we set U_k = sqrt(P_k), then u is real and even.
    # Does restricting to even u lose optimality?
    # The objective depends only on autocorrelation, which depends only on P.
    # The constraints are on u (bounds).
    # If the optimal P yields a u that violates bounds, we need to adjust.
    # Let's try optimizing P_k directly.
    
    # Initial guess: White noise spectrum (constant power) -> u is random-like.
    # Or, start with a specific structure.
    
    # Let's use scipy.optimize.minimize on the log-power spectrum.
    # Objective: -min(autocorr_u[1:]) (we want to maximize the minimum autocorrelation for k!=0)
    # Wait, overlap[0] is also a constraint.
    # overlap[0] = W * (N/2 - sum h^2).
    # sum h^2 = sum (0.5+u)^2 = N/4 + sum u^2.
    # overlap[0] = W * (N/2 - N/4 - sum u^2) = W * (N/4 - sum u^2).
    # To minimize overlap[0], we want to MAXIMIZE sum u^2.
    # To minimize overlap[k] for k!=0, we want to MAXIMIZE autocorr_u[k].
    # There is a trade-off.
    
    # Let's define the objective as the maximum overlap directly, using the spectral representation.
    # But computing u from P requires IFFT.
    
    def objective_and_grad(log_P):
        # P = exp(log_P)
        # Symmetry: P[k] = P[N-k]
        # We only optimize P[0:N//2]
        # Actually, let's just work with the full vector and enforce symmetry in the gradient or param.
        
        # Simpler: Optimize the vector u directly but with a spectral regularizer?
        # No, let's stick to the plan: Optimize P.
        
        # To ensure real u, we set U_k = sqrt(P_k) * exp(i * theta_k).
        # If we set theta_k = 0, u is real and even.
        # Let's try this first. If it fails, we can add phase optimization.
        
        P = np.exp(log_P)
        # Enforce symmetry: P[k] = P[-k]
        # The input log_P should be symmetric?
        # Let's define the parameter vector as log_P_sym (length N//2 + 1)
        
        # For now, let's just use the full N vector and average P[k] and P[N-k] to enforce symmetry.
        P_sym = np.zeros(N)
        for k in range(N // 2 + 1):
            if k == 0 or k == N // 2:
                P_sym[k] = P[k]
            else:
                P_sym[k] = P_sym[N-k] = (P[k] + P[N-k]) / 2
        
        # U = sqrt(P_sym) (real, so phase 0)
        U = np.sqrt(P_sym)
        u = np.fft.ifft(U).real
        
        # h = 0.5 + u
        h = 0.5 + u
        
        # Project to feasible region?
        # The spectral method doesn't guarantee bounds [0,1].
        # We need to project u to [-0.5, 0.5].
        # This breaks the spectral property.
        # Alternative: Use a penalty for bound violations.
        
        # Let's compute overlaps for the projected h to get a valid score.
        h_proj = np.clip(h, 0.0, 1.0)
        # Re-normalize sum to N/2
        h_proj = _project_to_simplex_box(h_proj, N / 2.0)
        
        overlaps = _compute_overlaps(h_proj)
        max_ov = np.max(overlaps)
        
        # Penalty for bound violations before projection?
        # The projection is a hard step, non-differentiable.
        # This makes gradient descent difficult.
        
        # Alternative approach:
        # Use the spectral representation to generate candidate h, project them, and use a local search on the spectral parameters?
        # Or, use the fact that the optimal u for maximizing min autocorr is often related to a "flat" spectrum with some structure.
        
        # Let's try a different structural change:
        # Instead of optimizing P, let's optimize the coefficients of a polynomial that defines the spectrum?
        
        # Actually, the inspiration mentioned "Spectral shaping ... Measured: 1 -> 1.051".
        # Let's implement a simpler version:
        # 1. Start with a random u.
        # 2. Compute P = |FFT(u)|^2.
        # 3. Modify P to increase the minimum autocorrelation.
        #    autocorr = IFFT(P).
        #    We want to increase min(autocorr[1:]).
        #    Gradient of min(autocorr[1:]) w.r.t P is tricky (min is non-smooth).
        #    Use soft-min: -log(sum(exp(-beta * autocorr[1:]))).
        
        # Let's optimize log_P using gradient descent on the soft-min objective.
        
        # Objective: minimize max_overlap(h_proj)
        # This is still non-smooth due to projection.
        
        # Let's try to enforce bounds via the spectrum?
        # If u is a sum of cosines, can we bound it?
        
        # Let's go back to the current program's success: it used gradient descent on h.
        # The new idea is to use the spectral domain to escape local minima.
        
        # Strategy:
        # 1. Optimize h using gradient descent (as in current).
        # 2. Periodically, transform h to spectral domain, perturb the spectrum (e.g., flatten it, or add noise to log-power), transform back, project, and continue.
        
        # This is a hybrid approach.
        
        # Let's implement the hybrid.
        pass

    # Hybrid Strategy Implementation
    best_h = np.full(N, 0.5)
    best_score = np.inf
    
    np.random.seed(42)
    
    # Initial h
    h = np.full(N, 0.5) + np.random.normal(0, 0.01, N)
    h = _project_to_simplex_box(h, N / 2.0)
    
    beta = 10.0
    beta_max = 100.0
    learning_rate = 0.05
    noise_scale = 0.1
    noise_decay = 0.995
    max_iters = 3000
    
    # Spectral perturbation parameters
    spectral_perturb_prob = 0.05
    spectral_noise_scale = 0.5
    
    for i in range(max_iters):
        if time.time() > deadline - 5:
            break
        
        # Compute overlaps
        overlaps = _compute_overlaps(h)
        current_val = np.max(overlaps)
        
        if current_val < best_score:
            best_score = current_val
            best_h = h.copy()
        
        # Gradient of soft-max overlap
        max_ov = np.max(overlaps)
        shifted = beta * (overlaps - max_ov)
        exp_vals = np.exp(shifted)
        sum_exp = np.sum(exp_vals)
        weights = exp_vals / sum_exp
        
        # Gradient w.r.t h
        H = np.fft.fft(h)
        Wk = np.fft.fft(weights)
        conv = np.fft.ifft(H * Wk).real
        grad_obj = -W * 2.0 * conv
        
        # Step
        noise = np.random.normal(0, noise_scale, N)
        h_new = h - learning_rate * grad_obj + noise
        h_new = _project_to_simplex_box(h_new, N / 2.0)
        
        # Spectral Perturbation
        if np.random.random() < spectral_perturb_prob:
            # Transform h to spectral domain
            H_spec = np.fft.fft(h_new)
            # Perturb the power spectrum
            P = np.abs(H_spec)**2
            # Add noise to log-power
            log_P = np.log(P + 1e-9)
            log_P += np.random.normal(0, spectral_noise_scale, N)
            # Enforce symmetry
            log_P_sym = np.zeros(N)
            for k in range(N // 2 + 1):
                if k == 0 or k == N // 2:
                    log_P_sym[k] = log_P[k]
                else:
                    log_P_sym[k] = log_P_sym[N-k] = (log_P[k] + log_P[N-k]) / 2
            P_new = np.exp(log_P_sym)
            # Reconstruct H_spec with new power, keep phase
            phase = np.angle(H_spec)
            H_spec_new = np.sqrt(P_new) * np.exp(1j * phase)
            # Ensure symmetry of H_spec for real signal
            H_spec_new_sym = np.zeros(N, dtype=complex)
            for k in range(N // 2 + 1):
                if k == 0 or k == N // 2:
                    H_spec_new_sym[k] = H_spec_new[k]
                else:
                    # Average with conjugate symmetric counterpart
                    H_spec_new_sym[k] = (H_spec_new[k] + np.conj(H_spec_new[N-k])) / 2
                    H_spec_new_sym[N-k] = np.conj(H_spec_new_sym[k])
            
            h_spec = np.fft.ifft(H_spec_new_sym).real
            h_new = 0.5 + h_spec
            h_new = _project_to_simplex_box(h_new, N / 2.0)
        
        # Accept/Reject
        overlaps_new = _compute_overlaps(h_new)
        current_val_new = np.max(overlaps_new)
        
        if current_val_new < current_val:
            h = h_new
        else:
            # Annealing
            if np.random.random() < 0.1 and current_val_new < current_val + 0.01:
                h = h_new
        
        # Anneal
        beta = min(beta * 1.01, beta_max)
        noise_scale *= noise_decay
    
    best_h = _project_to_simplex_box(best_h, N / 2.0)
    if not np.isclose(np.sum(best_h), N/2.0, atol=1e-5):
        best_h = np.full(N, 0.5)
        
    return best_h
