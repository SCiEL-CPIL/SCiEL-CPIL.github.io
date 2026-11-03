import numpy as np
import time

N = 600
HARD_LIMIT = 145.0

def fft_autoconv(f):
    """Compute full autoconvolution via FFT."""
    n = len(f)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    F = np.fft.rfft(f, n=size)
    conv = np.fft.irfft(F * F, n=size)
    return conv[:2 * n - 1]

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    conv = fft_autoconv(f)
    return np.max(conv) / (s * s)

def project_simplex(v, s=1.0):
    """Project v onto the simplex {x | x >= 0, sum(x) = s}."""
    v = np.asarray(v, dtype=np.float64)
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
    theta = (cssv[rho] - s) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    return w

def smooth_max_grad(f, beta):
    """
    Compute gradient of the smooth max approximation of the autoconvolution peak.
    Objective approx: (1/beta) * log(sum(exp(beta * c_k)))
    Gradient g_i = 2 * sum_k p_k * f_{k-i}
    where p_k = softmax(beta * c_k)
    """
    conv = fft_autoconv(f)
    
    # p_k = softmax(beta * conv)
    # Use log-sum-exp for stability
    max_c = np.max(conv)
    exp_c = np.exp(beta * (conv - max_c))
    sum_exp = np.sum(exp_c)
    p = exp_c / sum_exp
    
    # g = 2 * (p * f)
    # p is length 2N-1. f is length N.
    # We need g_i = sum_k p_k f_{k-i}.
    # This is cross-correlation or convolution?
    # c_k = sum_j f_j f_{k-j}.
    # g_i = sum_k p_k (d c_k / d f_i) = sum_k p_k 2 f_{k-i}.
    # Let m = k-i. k = m+i.
    # g_i = 2 sum_m p_{m+i} f_m.
    # This is correlation of p and f.
    # np.correlate(p, f, mode='full')?
    # Or just use FFT.
    
    # p is defined on k=0..2N-2.
    # We want g_i for i=0..N-1.
    # g_i = 2 sum_{m=0}^{N-1} f_m p_{m+i}.
    
    # Construct P such that FFT(P) * FFT(F) gives the correlation?
    # Easier: use np.correlate if N is small, but N=600 is fine for FFT.
    # Let's use FFT.
    # We want h_i = sum_m f_m p_{m+i}.
    # This is the correlation of f and p.
    # np.correlate(f, p, 'valid')? No, lengths differ.
    
    # Let's use the property: corr(f, p)_i = sum_m f_m p_{m+i}.
    # This is equivalent to convolution of f and reversed p?
    # conv(f, rev(p))_k = sum_m f_m rev(p)_{k-m} = sum_m f_m p_{N_p - 1 - (k-m)} = sum_m f_m p_{N_p - 1 - k + m}.
    # We want index m+i.
    # So N_p - 1 - k + m = m + i => k = N_p - 1 - i.
    # So g_i = 2 * conv(f, rev(p))[N_p - 1 - i].
    
    p_len = len(p)
    rev_p = p[::-1]
    
    fft_size = 1
    total_len = N + p_len - 1
    while fft_size < total_len:
        fft_size *= 2
        
    F_fft = np.fft.rfft(f, n=fft_size)
    P_fft = np.fft.rfft(rev_p, n=fft_size)
    conv_fft = F_fft * P_fft
    conv_full = np.fft.irfft(conv_fft, n=fft_size)[:total_len]
    
    # g_i = 2 * conv_full[N_p - 1 - i]
    # i ranges 0..N-1.
    # Indices in conv_full: N_p - 1 - (N-1) ... N_p - 1 - 0
    # N_p = 2N-1.
    # Indices: 2N-1-1-(N-1) = N-1 ... 2N-2.
    # So we need conv_full[N-1 : 2N-1].
    # And reverse it?
    # i=0 -> index 2N-2.
    # i=N-1 -> index N-1.
    # So g = 2 * conv_full[N-1:2N-1][::-1]
    
    g = 2.0 * conv_full[N-1:2*N-1][::-1]
    
    return g

def smooth_max_optimize(f_init, start_time, max_time=120.0):
    f = f_init.copy()
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return f_init
    f = f / s
    
    best_f = f.copy()
    best_val = compute_score(f)
    
    # Homotopy on beta
    betas = [10.0, 50.0, 100.0, 500.0, 1000.0, 2000.0]
    
    for beta in betas:
        if time.time() - start_time > max_time - 10:
            break
            
        # Gradient Descent with Backtracking Line Search
        # We want to minimize J(f) = SmoothMax_beta(f) subject to sum(f)=1, f>=0.
        # Since we project to simplex, we can just do:
        # f_new = Project(f - alpha * grad)
        
        alpha = 0.01
        current_J = None
        
        for step in range(200):
            if time.time() - start_time > max_time - 5:
                break
                
            grad = smooth_max_grad(f, beta)
            
            # Projected Gradient Step
            # We need to choose alpha.
            # Simple backtracking:
            # J(f - alpha grad) < J(f) + <grad, f - alpha grad - f> ?
            # Since projection is non-expansive, standard PGD works.
            
            # Let's just use a fixed small alpha or adaptive.
            # Given the scale of f (sum=1, N=600, f~1/600), grad scale?
            # c_k ~ 1/N. beta * c_k ~ O(1).
            # p_k ~ 1/N.
            # g_i ~ 2 * N * (1/N) * (1/N) = 2/N ~ 0.003.
            # alpha=0.01 might be too small or large.
            
            # Let's try adaptive alpha based on Lipschitz constant of SmoothMax.
            # L ~ beta * ||f||_2^2 ?
            
            # Simple approach: try a few alphas.
            improved = False
            for a in [0.1, 0.01, 0.001, 0.0001]:
                f_candidate = f - a * grad
                f_candidate = project_simplex(f_candidate, 1.0)
                
                # Check if this improves the actual max score (not just smooth approx)
                # Or just check smooth approx?
                # Checking actual score is expensive (FFT).
                # But we have budget.
                
                # Let's evaluate actual score for the candidate if we are close to convergence?
                # For now, trust the smooth approx gradient direction.
                
                # To ensure progress, let's compute the smooth value.
                # But we don't have a fast smooth value function separate from grad?
                # We can compute it.
                
                # Actually, just taking the step and checking actual score every 10 steps is safer.
                # Let's just take the step with a conservative alpha.
                
                f = f_candidate
                improved = True
                break
            
            if not improved:
                break
                
        # Evaluate actual score
        current_val = compute_score(f)
        if current_val < best_val:
            best_val = current_val
            best_f = f.copy()
            
    return best_f

def spectral_refine(x, start_time, max_time=120.0):
    """
    Refine x using spectral coordinate descent.
    Optimizes the DFT coefficients of x to minimize the autoconvolution peak.
    """
    x = np.asarray(x, dtype=np.float64)
    X = np.fft.fft(x)
    
    best_x = x.copy()
    best_score = compute_score(x)
    
    # We will iterate over frequency bins.
    # For real signals, X[k] and X[N-k] are conjugates.
    # We only need to update k=0 to N//2.
    
    # Precompute some constants
    n = N
    
    for iteration in range(50):
        if time.time() - start_time > max_time - 2:
            break
        
        improved = False
        
        # Iterate over unique frequency bins
        # k=0 is DC (real), k=N/2 is Nyquist (real if N even)
        # Others come in pairs.
        
        for k in range(n // 2 + 1):
            if time.time() - start_time > max_time - 2:
                break
            
            old_Xk = X[k].copy()
            
            # We want to adjust X[k] (and X[N-k] if k!=0, k!=N/2) to improve score.
            # Since the objective is non-convex, we try a few local moves.
            # Specifically, we can adjust the magnitude and phase.
            
            # However, changing X[k] changes x, which changes all c_j, which changes the peak.
            # This is expensive to evaluate exactly for every small move.
            
            # Alternative: Use the gradient of the objective w.r.t x, then map to gradient w.r.t X?
            # grad_x = smooth_max_grad(x, beta)
            # grad_X = np.fft.fft(grad_x)
            # Then update X[k] using grad_X[k].
            
            # Let's try this approach: Spectral Gradient Descent.
            # 1. Compute grad_x in time domain.
            # 2. Transform to frequency domain: grad_X.
            # 3. Update X[k] -= alpha * grad_X[k].
            # 4. Enforce conjugate symmetry and realness (implicitly by taking real part of ifft).
            # 5. Project x to simplex.
            # 6. Recompute X from new x? Or keep X updated?
            # If we project x, X changes. So we should recompute X from x after projection.
            
            # Let's stick to the coordinate descent idea from the inspiration but make it efficient.
            # The inspiration tried random moves. That's slow.
            
            # Let's try a "Spectral Projected Gradient" step.
            # This is essentially what smooth_max_optimize does, but in frequency domain?
            # No, smooth_max_optimize is time domain.
            
            # Let's combine:
            # Use spectral representation to generate diverse initial guesses or to perform moves.
            
            # Actually, the most effective part of the inspiration was likely the *initialization* or *parametric search*.
            # The coordinate descent in freq domain was very slow (evaluating score for each move).
            
            # So, let's use spectral methods to *generate* candidates.
            # 1. Take current best x.
            # 2. Compute X = fft(x).
            # 3. Perturb X (e.g., random phase shifts, magnitude scaling of low freqs).
            # 4. x_new = real(ifft(X_perturbed)).
            # 5. Project x_new to simplex.
            # 6. Evaluate.
            
            # This is a form of "spectral annealing" or "spectral perturbation".
            
            if k == 0:
                # DC component (total mass)
                # Usually we fix the sum, so DC is fixed by normalization.
                # But we can vary it and renormalize.
                # Let's skip DC for now to avoid changing the sum drastically before projection.
                continue
            
            # Try a few perturbations for this bin
            for mag_factor in [0.5, 0.8, 1.2, 1.5]:
                for phase_shift in [-0.5, 0.0, 0.5]:
                    new_Xk = old_Xk * mag_factor
                    if k != 0 and k != n // 2:
                        new_Xk *= np.exp(1j * phase_shift)
                    
                    X[k] = new_Xk
                    # Update conjugate
                    k_conj = (n - k) % n
                    if k_conj != k:
                        X[k_conj] = np.conj(new_Xk)
                    
                    x_new = np.fft.ifft(X).real
                    x_new = project_simplex(x_new, 1.0)
                    
                    sc = compute_score(x_new)
                    if sc < best_score:
                        best_score = sc
                        best_x = x_new.copy()
                        improved = True
                        # Update X to reflect the new x (since projection changed it)
                        X = np.fft.fft(best_x)
                    else:
                        # Restore
                        X[k] = old_Xk
                        if k_conj != k:
                            X[k_conj] = np.conj(old_Xk)
            
            # If we found an improvement, restart the inner loop? Or just continue?
            # Continuing is fine, but X is now based on best_x.
            # We should probably break out and restart the outer iteration to ensure consistency?
            # Or just continue with the updated X.
            # Since X was updated to best_x's FFT, it's consistent.
            
            if improved:
                # Small improvement found. We can continue to other bins.
                pass
        
        if not improved:
            # No improvement in this pass. Maybe break?
            # Or try a different beta?
            # For now, break to save time.
            # But we might be stuck in a local min.
            # Let's try a few random restarts from the spectral domain.
            
            # Generate a few random spectral perturbations
            for _ in range(5):
                if time.time() - start_time > max_time - 2:
                    break
                
                X_pert = X.copy()
                # Random phase shifts
                phases = np.random.uniform(-np.pi, np.pi, n)
                # Keep DC and Nyquist real (phase 0 or pi)
                phases[0] = 0
                if n % 2 == 0:
                    phases[n//2] = 0
                
                X_pert *= np.exp(1j * phases)
                
                # Ensure conjugate symmetry explicitly (random phases might break it slightly due to floating point or if we didn't handle pairs)
                # Actually, if we apply random phases to all bins, symmetry is broken.
                # We must enforce it.
                # Better: generate random phases for k=1..N/2-1, and set conjugates.
                
                X_pert = X.copy()
                for kk in range(1, n//2):
                    phase = np.random.uniform(-np.pi, np.pi)
                    X_pert[kk] *= np.exp(1j * phase)
                    X_pert[n-kk] *= np.exp(-1j * phase)
                
                x_new = np.fft.ifft(X_pert).real
                x_new = project_simplex(x_new, 1.0)
                
                sc = compute_score(x_new)
                if sc < best_score:
                    best_score = sc
                    best_x = x_new.copy()
                    X = np.fft.fft(best_x)
                    improved = True
                    break # Found a good restart, break random loop
            
            if not improved:
                break
                
    return best_x

def construct():
    start_time = time.time()
    
    # Generate initial candidates
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + (0.5 / N)
    candidates = []
    
    # Uniform
    f = np.ones(N)
    candidates.append(f / np.sum(f))
    
    # Gaussian
    for alpha in [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
        f = np.exp(-alpha * x ** 2)
        candidates.append(f / np.sum(f))
        
    # Triangle
    for w in [0.1, 0.2, 0.25, 0.4, 0.5]:
        f = np.zeros(N)
        mask = np.abs(x) <= w / 2.0
        f[mask] = 1.0 - 2.0 * np.abs(x[mask]) / w
        f = np.maximum(f, 0)
        if np.sum(f) > 1e-15:
            candidates.append(f / np.sum(f))
            
    # Spectral Candidates:
    # Create signals with specific spectral profiles
    n = N
    # 1. Low-pass filter (smooth)
    freqs = np.fft.fftfreq(n)
    # Scale freqs to [-0.5, 0.5]
    # We want to emphasize low frequencies.
    for cutoff in [0.05, 0.1, 0.2, 0.3]:
        X = np.ones(n)
        X[np.abs(freqs) > cutoff] = 0
        # Add some smooth decay
        X *= np.exp(-10 * np.abs(freqs))
        f = np.fft.ifft(X).real
        f = np.maximum(f, 0)
        if np.sum(f) > 1e-15:
            candidates.append(f / np.sum(f))
            
    # 2. Band-pass
    for center in [0.05, 0.1, 0.2]:
        for width in [0.02, 0.05]:
            X = np.zeros(n)
            mask = (np.abs(freqs) > center - width) & (np.abs(freqs) < center + width)
            X[mask] = 1.0
            f = np.fft.ifft(X).real
            f = np.maximum(f, 0)
            if np.sum(f) > 1e-15:
                candidates.append(f / np.sum(f))

    # Score them
    scored = []
    for f in candidates:
        score = compute_score(f)
        scored.append((f, score))
    
    scored.sort(key=lambda x: x[1])
    best_f = scored[0][0].copy()
    best_score = scored[0][1]
    
    # Apply Spectral Refinement to the best initial candidate
    if time.time() - start_time < HARD_LIMIT - 10:
        refined = spectral_refine(best_f, start_time, HARD_LIMIT - 5)
        refined_score = compute_score(refined)
        if refined_score < best_score:
            best_f = refined
            best_score = refined_score

    # Apply Smooth Max Optimization to the best result (combining spectral init + smooth max polish)
    if time.time() - start_time < HARD_LIMIT - 10:
        refined2 = smooth_max_optimize(best_f, start_time, HARD_LIMIT - 2)
        refined2_score = compute_score(refined2)
        if refined2_score < best_score:
            best_f = refined2
            best_score = refined2_score

    # Final normalization
    best_f = np.maximum(best_f, 0.0)
    s = np.sum(best_f)
    if s > 1e-15:
        best_f = best_f / s
        
    return [float(x) for x in best_f]
