import numpy as np
import time
import sys

N = 600
FFT_SIZE = 2048
TIME_LIMIT = 145.0 # Safety margin before 150s kill

# Precompute grid
x_edges = np.linspace(-0.25, 0.25, N + 1)
x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0

def compute_score(f):
    """
    Compute the normalized maximum of the autoconvolution.
    f: array of shape (N,)
    Returns: max(conv) / (sum(f)^2)
    """
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    
    # Use FFT for fast convolution
    F1 = np.fft.rfft(f, n=FFT_SIZE)
    conv = np.fft.irfft(F1 * F1, n=FFT_SIZE)
    
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
        
    max_conv = np.max(conv)
    return max_conv / (s * s)

def construct():
    start_time = time.time()
    
    # 1. Initialize Frequency Spectrum
    # We want a time-domain signal that is non-negative.
    # A Fejer kernel in time corresponds to a triangular magnitude spectrum in frequency.
    # However, we are working in frequency domain. 
    # Let's initialize the magnitude spectrum M[k] to be a smooth, decaying function.
    # A common choice for a "bump" in time is a sinc-like or Gaussian in freq.
    # Actually, if we want f(t) >= 0, it's easier to start with a known good f(t) 
    # and get its spectrum.
    
    # Let's start with a Gaussian in time domain, which is positive.
    sigma_t = 0.1
    f_init_time = np.exp(-0.5 * (x_centers / sigma_t) ** 2)
    
    # Compute spectrum
    F_init = np.fft.rfft(f_init_time, n=FFT_SIZE)
    
    # Extract magnitude and phase
    M = np.abs(F_init)
    # Ensure M is strictly positive to avoid log(0) issues if we were optimizing log-mag, 
    # but we are optimizing phase. M is fixed.
    
    # If M has zeros, phase is undefined. We can set phase to 0 there.
    phase = np.angle(F_init)
    
    # Optimization Variables: phase angles theta[k] for k=0..FFT_SIZE//2
    # Note: k=0 and k=FFT_SIZE//2 (if even) are real, so phase is 0 or pi. 
    # For rfft, the last bin is real. We only optimize the interior bins.
    n_freq = FFT_SIZE // 2 + 1
    # Indices 1 to n_freq-2 are complex. Index 0 and n_freq-1 are real.
    
    # We will optimize the phase of the complex bins.
    # Let's define the vector of phases for the complex bins.
    # Actually, to keep it simple and differentiable, we can optimize the full complex spectrum 
    # but constrain it to have fixed magnitude M.
    
    # Let theta be the phase vector of length n_freq.
    # F[k] = M[k] * exp(1j * theta[k])
    # Constraint: time signal f = ifft(F) must be >= 0.
    
    # We use a smooth approximation for the non-negativity constraint and the max objective.
    
    def get_time_signal(theta):
        """Reconstruct time signal from magnitude M and phase theta."""
        # Ensure theta is real
        F_complex = M * np.exp(1j * theta)
        # Enforce Hermitian symmetry? 
        # rfft/irfft handles the half-spectrum. 
        # irfft expects the half-spectrum. 
        # The input to irfft is the complex array of length n_freq.
        # It implicitly assumes conjugate symmetry for the missing half.
        f = np.fft.irfft(F_complex, n=FFT_SIZE)
        return f[:N]
    
    def objective_and_grad(theta, beta):
        """
        Compute objective value and gradient w.r.t theta.
        Objective: smooth_max(conv) / (sum(f))^2
        We use log-sum-exp for smooth_max.
        """
        f = get_time_signal(theta)
        
        # Softplus projection to enforce non-negativity?
        # The strategy says "optimize phase ... while maintaining non-negativity".
        # If we just project f = max(f, 0), the gradient is 0 where f<0, which stops learning there.
        # Better: Use a soft constraint or just project and accept that gradient is zero there?
        # The inspiration suggests "softplus projection maintains non-negativity while preserving smoothness".
        # Let's use a softplus based "effective f" for the objective calculation?
        # No, the output must be the step heights. If we use softplus in objective, 
        # we should return the softplus values? Or project?
        # The task asks for nonnegative step heights. 
        # If we minimize the objective using a soft approximation of the constraint,
        # we should return the projected (or softplus) values.
        # Let's try: f_pos = softplus(f, temperature)
        # Then compute conv of f_pos.
        # Gradient of conv w.r.t f_pos is standard.
        # Gradient of f_pos w.r.t f is sigmoid(f/T).
        # Gradient of f w.r.t theta is ifft of (1j * M * exp(1j*theta) * dtheta).
        
        # Let's stick to a simpler approach first: Hard projection in the forward pass,
        # but use a smoothed max for the objective.
        # The gradient of the hard projection is 0 for f<0. This might get stuck.
        # Let's use the "softplus" idea from the knowledge base.
        
        T_soft = 0.01 # Temperature for softplus
        # f_soft = T_soft * log(1 + exp(f/T_soft))
        # To avoid overflow:
        f_soft = np.where(f > 0, f + T_soft * np.log1p(np.exp(-f/T_soft)), T_soft * np.log1p(np.exp(f/T_soft)))
        # Actually, standard softplus: x + log(1+exp(-x)) for x>0, log(1+exp(x)) for x<0.
        # Let's just use np.maximum(f, 0) for the return value, but for gradient calculation,
        # we might need to be careful.
        
        # Let's try a different angle: 
        # Minimize the peak of the autoconvolution of the *positive part* of f.
        # f_pos = np.maximum(f, 0)
        
        f_pos = np.maximum(f, 0.0)
        
        s = np.sum(f_pos)
        if s < 1e-15:
            return np.inf, np.zeros_like(theta)
        
        F_pos = np.fft.rfft(f_pos, n=FFT_SIZE)
        conv = np.fft.irfft(F_pos * F_pos, n=FFT_SIZE)
        
        # Smooth max
        # beta controls smoothness. Higher beta -> closer to true max.
        max_conv_smooth = (1.0/beta) * np.log(np.sum(np.exp(beta * conv)))
        
        obj = max_conv_smooth / (s * s)
        
        # Gradient calculation
        # d obj / d theta = d obj / d conv * d conv / d f_pos * d f_pos / d f * d f / d theta
        # d f_pos / d f = 1 if f > 0 else 0. (Hard projection gradient)
        # This might be too sparse.
        
        # Let's use the softplus gradient for f_pos -> f.
        # Let f_soft be the softplus of f.
        # f_soft = T * log(1 + exp(f/T))
        # d f_soft / d f = sigmoid(f/T)
        # We minimize objective of f_soft.
        # f_soft is always positive.
        
        # Recalculate with softplus
        T_soft = 0.05
        # Stable softplus
        f_soft = T_soft * np.log1p(np.exp(np.clip(f / T_soft, -50, 50)))
        
        s_soft = np.sum(f_soft)
        if s_soft < 1e-15:
            return np.inf, np.zeros_like(theta)
            
        F_soft = np.fft.rfft(f_soft, n=FFT_SIZE)
        conv_soft = np.fft.irfft(F_soft * F_soft, n=FFT_SIZE)
        
        # Smooth max of conv_soft
        w = np.exp(beta * (conv_soft - np.max(conv_soft))) # Stability shift
        w /= np.sum(w)
        max_conv_soft = (1.0/beta) * np.log(np.sum(w) * np.exp(beta * np.max(conv_soft))) # This is just max + log(sum(w))/beta? No.
        # smooth_max(x) = (1/beta) log(sum(exp(beta x)))
        # Let m = max(conv). sum(exp(beta x)) = exp(beta m) sum(exp(beta(x-m)))
        # log(...) = beta m + log(sum(exp(beta(x-m))))
        # smooth_max = m + (1/beta) log(sum(exp(beta(x-m))))
        
        m_conv = np.max(conv_soft)
        exp_shifted = np.exp(beta * (conv_soft - m_conv))
        sum_exp = np.sum(exp_shifted)
        max_conv_smooth_val = m_conv + (1.0/beta) * np.log(sum_exp)
        
        obj_val = max_conv_smooth_val / (s_soft * s_soft)
        
        # Gradient of obj_val w.r.t conv_soft
        # d/dconv_j [ (1/beta) log(sum exp(beta conv)) / S^2 ]
        # = (1/beta) * (1/sum_exp) * exp(beta conv_j) * beta / S^2
        # = w_j / S^2
        # where w_j = exp(beta(conv_j - m)) / sum(exp(...))
        
        grad_conv = w / (s_soft * s_soft)
        
        # Gradient of conv_soft w.r.t f_soft
        # conv_soft = f_soft * f_soft
        # d/d f_i (conv)_j = f_{j-i} + f_{j-i} (symmetric) ?
        # conv = ifft(fft(f)^2)
        # Let G = fft(f). conv = ifft(G^2).
        # d conv / d f = ifft( 2 * G * dG/d f )
        # dG_k / d f_i = exp(-2j pi k i / FFT_SIZE)
        # So d conv / d f = ifft( 2 * G * exp(-2j pi k i / FFT_SIZE) ) ... this is getting complex to vectorize directly.
        
        # Easier: Use the property that gradient of ||f*f||_inf (or smooth max) 
        # is approximately the cross-correlation of f with the "peak weight" vector.
        # Specifically, if we define v = w (the weights from the smooth max),
        # then grad_f = 2 * ifft(fft(f) * fft(v)) ?
        # Let's check: 
        # J = sum_j w_j (f*f)_j
        # d J / d f_i = sum_j w_j (f_{j-i} + f_{j-i}) = 2 sum_j w_j f_{j-i}
        # This is 2 * (w * f)_i (convolution).
        # So grad_f = 2 * np.convolve(w, f, mode='same')? 
        # Note: w is defined on the full FFT_SIZE grid? 
        # conv_soft is length FFT_SIZE. w is length FFT_SIZE.
        # f_soft is length N, padded to FFT_SIZE.
        # Let's pad f_soft and w to FFT_SIZE.
        
        f_soft_pad = np.zeros(FFT_SIZE)
        f_soft_pad[:N] = f_soft
        
        # grad_f_soft = 2 * (w * f_soft_pad) linear convolution, truncated to N?
        # The gradient w.r.t f_soft[i] is 2 * sum_j w[j] f_soft_pad[j-i]
        # This is the cross-correlation of w and f_soft_pad, or convolution of w and flipped f.
        # Since f is real, and we are dealing with autoconvolution, the gradient is symmetric.
        # Let's compute via FFT for speed.
        # grad_f_soft = 2 * ifft( fft(w) * conj(fft(f_soft_pad)) ) ?
        # Let's verify:
        # sum_j w_j f_{j-i} = (w * f_rev)_i ?
        # In Fourier domain: Ifft( Fw * Ff_conj ) gives correlation.
        # Correlation C[i] = sum_j w[j] f[j-i].
        # Yes.
        
        F_w = np.fft.rfft(w, n=FFT_SIZE)
        F_f = np.fft.rfft(f_soft_pad, n=FFT_SIZE)
        
        # Correlation = ifft( F_w * conj(F_f) )
        # But F_f is real FFT. 
        # Let's just do:
        grad_f_soft_full = np.fft.irfft(F_w * np.conj(np.fft.rfft(f_soft_pad, n=FFT_SIZE)), n=FFT_SIZE)
        # Wait, rfft returns half spectrum. To multiply in freq domain for correlation:
        # corr = ifft( fft(w) * conj(fft(f)) )
        # For real signals, we can use rfft.
        # Let's just use the full complex FFT to be safe with the correlation identity.
        F_w_c = np.fft.fft(w, n=FFT_SIZE)
        F_f_c = np.fft.fft(f_soft_pad, n=FFT_SIZE)
        grad_f_soft_full = np.fft.ifft(F_w_c * np.conj(F_f_c)).real
        grad_f_soft = grad_f_soft_full[:N] * 2.0
        
        # Gradient w.r.t f (before softplus)
        # d f_soft / d f = sigmoid(f/T)
        sigmoid_f = 1.0 / (1.0 + np.exp(-f / T_soft))
        grad_f = grad_f_soft * sigmoid_f
        
        # Gradient w.r.t theta
        # f = ifft( M * exp(1j theta) )
        # d f / d theta_k = ifft( dF_k / d theta_k )
        # dF_k / d theta_k = 1j * M[k] * exp(1j theta[k]) = 1j * F_k
        # So d f / d theta is the ifft of a spectrum where only bin k is non-zero?
        # No, we need the gradient for all theta_k simultaneously.
        # grad_theta_k = sum_i grad_f[i] * (d f_i / d theta_k)
        # d f_i / d theta_k = (1/FFT_SIZE) * 1j * M[k] * exp(1j theta[k]) * exp(2j pi k i / FFT_SIZE)
        # This is (1/FFT_SIZE) * 1j * F_k * exp(2j pi k i / FFT_SIZE)
        
        # Vectorized approach:
        # grad_theta = (1/FFT_SIZE) * 1j * M * exp(1j theta) * fft(grad_f_pad) ?
        # Let's check indices.
        # f_i = sum_k (1/FFT_SIZE) F_k exp(2j pi k i / N_fft)
        # d f_i / d theta_k = (1/FFT_SIZE) (1j F_k) exp(2j pi k i / N_fft)
        # grad_theta_k = sum_i grad_f_i (1/FFT_SIZE) (1j F_k) exp(2j pi k i / N_fft)
        # grad_theta_k = (1/FFT_SIZE) (1j F_k) sum_i grad_f_i exp(2j pi k i / N_fft)
        # sum_i grad_f_i exp(2j pi k i / N_fft) is the DFT of grad_f at frequency k.
        # Note: standard DFT is sum x_i exp(-2j pi k i / N). 
        # Here we have positive sign. So it's the DFT of grad_f reversed? Or conj?
        # DFT(x)_k = sum x_i exp(-2j pi k i / N).
        # Our sum is sum x_i exp(+2j pi k i / N) = DFT(x)_-k = conj(DFT(x)_k) if x is real.
        # grad_f is real.
        # So sum_i grad_f_i exp(2j pi k i / N) = conj( fft(grad_f)_k )
        
        grad_f_pad = np.zeros(FFT_SIZE)
        grad_f_pad[:N] = grad_f
        
        G_grad_f = np.fft.fft(grad_f_pad)
        
        # F_k = M_k exp(1j theta_k)
        # grad_theta_k = (1/FFT_SIZE) * 1j * F_k * conj(G_grad_f_k)
        
        F_curr = M * np.exp(1j * theta)
        
        grad_theta_complex = (1.0 / FFT_SIZE) * 1j * F_curr * np.conj(G_grad_f)
        
        # theta is real. The gradient must be real.
        # Is grad_theta_complex real?
        # F_curr is complex. G_grad_f is complex.
        # Let's check symmetry.
        # grad_f is real. G_grad_f has Hermitian symmetry.
        # F_curr has Hermitian symmetry (since f is real).
        # Product of two Hermitian symmetric vectors?
        # A_k = F_k. B_k = conj(G_k).
        # If G is Hermitian, G_{N-k} = conj(G_k).
        # B_{N-k} = conj(G_{N-k}) = conj(conj(G_k)) = G_k.
        # So B is not Hermitian, it's "anti-Hermitian" in a sense? Or just symmetric?
        # B_k = conj(G_k). B_{N-k} = G_k.
        # A_k = F_k. A_{N-k} = conj(F_k).
        # C_k = 1j A_k B_k.
        # C_{N-k} = 1j A_{N-k} B_{N-k} = 1j conj(F_k) G_k = -1j F_k conj(G_k) = - C_k.
        # So C is anti-Hermitian (skew-symmetric in frequency).
        # The ifft of an anti-Hermitian spectrum is purely imaginary.
        # Wait, we want grad_theta to be real.
        # Let's re-evaluate.
        # theta_k is the phase of F_k.
        # For real f, F_{N-k} = conj(F_k). So theta_{N-k} = -theta_k.
        # We only optimize independent variables? 
        # In rfft, we have k=0..N/2. 
        # The variables are theta[1..N/2-1]. theta[0] and theta[N/2] are fixed to 0 or pi.
        # If we optimize the full vector theta[0..N/2], we must maintain symmetry.
        # It is easier to optimize only the free phases and reconstruct.
        
        # Let's restrict optimization to k=1 to n_freq-2.
        # free_idx = np.arange(1, n_freq-1)
        
        # However, the gradient calculation above gave a complex vector.
        # The real part of the gradient w.r.t the real variable theta?
        # If we parameterize F_k = M_k (cos theta_k + i sin theta_k).
        # d F_k / d theta_k = M_k (-sin theta_k + i cos theta_k) = 1j M_k e^{i theta_k} = 1j F_k.
        # This holds for the complex representation.
        # Since theta_k is real, the gradient of the real objective w.r.t theta_k is:
        # Re( d_obj/dF_k * dF_k/dtheta_k ) * 2 ? No.
        # obj is real. F is complex.
        # d obj / d theta_k = Re( (d obj / d F_k)^* * d F_k / d theta_k ) ?
        # Actually, for real objective J(F, F*), dJ = sum (dJ/dF_k dF_k + dJ/dF*_k dF*_k).
        # dJ/dF*_k = conj(dJ/dF_k).
        # dF_k = 1j F_k dtheta_k. dF*_k = -1j F*_k dtheta_k.
        # dJ = sum (dJ/dF_k 1j F_k - dJ/dF*_k 1j F*_k) dtheta_k
        # = sum (1j (dJ/dF_k F_k - dJ/dF*_k F*_k)) dtheta_k
        # Since dJ/dF*_k = conj(dJ/dF_k), let A_k = dJ/dF_k.
        # Term = 1j (A_k F_k - conj(A_k) F*_k) = 1j (A_k F_k - conj(A_k F_k)) = 1j (2j Im(A_k F_k)) = -2 Im(A_k F_k).
        
        # So grad_theta_k = -2 * Im( A_k * F_k )
        # where A_k = d obj / d F_k.
        
        # We calculated grad_f. We need A_k = d obj / d F_k.
        # obj = f(conv(f))
        # d obj / d F_k = sum_i (d obj / d f_i) (d f_i / d F_k)
        # d f_i / d F_k = (1/FFT_SIZE) exp(2j pi k i / FFT_SIZE)
        # So A_k = (1/FFT_SIZE) sum_i grad_f_i exp(2j pi k i / FFT_SIZE)
        # This is (1/FFT_SIZE) * DFT_plus(grad_f)_k
        # Which is (1/FFT_SIZE) * conj(fft(grad_f)_k) for real grad_f.
        
        A_k = (1.0 / FFT_SIZE) * np.conj(G_grad_f)
        
        # grad_theta_k = -2 * Im( A_k * F_k )
        # F_k = M_k exp(1j theta_k)
        
        grad_theta_full = -2.0 * np.imag(A_k * F_curr)
        
        # Extract gradient for free variables
        # We will optimize the full theta vector but mask out fixed bins in the update?
        # Or just optimize the free ones.
        
        return obj_val, grad_theta_full

    # Optimization Loop
    # Initialize theta
    theta = np.zeros(n_freq)
    
    # We only update indices 1 to n_freq-2
    free_idx = np.arange(1, n_freq - 1)
    
    # Adam Optimizer
    m = np.zeros(n_freq)
    v = np.zeros(n_freq)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    lr = 0.01
    
    best_score = np.inf
    best_theta = theta.copy()
    
    # Adaptive beta for smooth max
    beta = 50.0 # Start with some smoothing
    
    steps = 0
    max_steps = 2000
    
    while time.time() - start_time < TIME_LIMIT - 10:
        obj_val, grad_theta = objective_and_grad(theta, beta)
        
        if np.isnan(obj_val) or np.isinf(obj_val):
            # If objective blows up, reduce learning rate or break
            lr *= 0.5
            if lr < 1e-6:
                break
            continue
        
        if obj_val < best_score:
            best_score = obj_val
            best_theta = theta.copy()
            
        # Adam update
        steps += 1
        m[free_idx] = beta1 * m[free_idx] + (1 - beta1) * grad_theta[free_idx]
        v[free_idx] = beta2 * v[free_idx] + (1 - beta2) * (grad_theta[free_idx] ** 2)
        
        m_hat = m[free_idx] / (1 - beta1 ** steps)
        v_hat = v[free_idx] / (1 - beta2 ** steps)
        
        theta[free_idx] -= lr * m_hat / (np.sqrt(v_hat) + eps)
        
        # Anneal beta (increase smoothness sharpness)
        # beta = min(beta * 1.01, 500.0)
        
        # Occasionally check the hard score
        if steps % 100 == 0:
            f_check = get_time_signal(theta)
            f_check = np.maximum(f_check, 0.0)
            hard_score = compute_score(f_check)
            if hard_score < best_score:
                best_score = hard_score
                best_theta = theta.copy()

    # Final refinement: Use the best theta found
    f_best = get_time_signal(best_theta)
    f_best = np.maximum(f_best, 0.0)
    
    # One last check with Nelder-Mead on a few parameters? 
    # No, the phase optimization should be sufficient.
    # Let's just return the best f.
    
    # Ensure non-negative
    f_best = np.maximum(f_best, 0.0)
    
    return [float(x) for x in f_best]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
