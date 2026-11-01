import numpy as np
from scipy.optimize import minimize
import time

N = 400
DT = 0.5 / N

def _compute_autoconv(signed_heights):
    n = 1
    while n < 2 * N:
        n <<= 1
    F = np.fft.rfft(signed_heights, n)
    R = np.fft.irfft(F * np.conj(F), n)
    return R[:N] * DT

def _gumbel_softmax_loss(logits, beta=1.0, tau=1.0):
    # logits: shape (N,)
    # gumbel noise
    g = -np.log(-np.log(np.random.rand(N) + 1e-10) + 1e-10)
    # gumbel sigmoid
    # s = 2 * sigmoid((logits + g) / tau) - 1
    # To make it differentiable and bounded, we use the straight-through estimator or just the soft version for the loss calculation
    # For the loss, we use the soft values
    x = (logits + g) / tau
    # sigmoid
    s_soft = 1.0 / (1.0 + np.exp(-x))
    s = 2.0 * s_soft - 1.0
    
    R = _compute_autoconv(s)
    R_sidelobes = R[1:]
    abs_sidelobes = np.abs(R_sidelobes)
    
    # Soft max of the absolute sidelobes
    m = np.max(abs_sidelobes)
    if m < 1e-8:
        return 0.0
    # Use a small temperature for the soft max to approximate max
    loss = m + tau * np.log(np.sum(np.exp((abs_sidelobes - m) / tau)))
    return loss

def _gumbel_softmax_grad(logits, beta=1.0, tau=1.0):
    # Re-sample gumbel noise? No, for gradient we need the same noise.
    # This is tricky with standard minimize.
    # Alternative: Use the reparameterization trick properly.
    # But scipy.optimize.minimize doesn't support stochastic gradients easily.
    # We can use a fixed set of gumbel samples or average over a few.
    # For simplicity and stability, let's use a deterministic approximation:
    # Minimize the expected loss.
    # E[soft_max(|R|)]
    # This is hard to differentiate analytically without sampling.
    
    # Let's stick to the tanh relaxation which is deterministic, but improve the optimization.
    # The previous attempt used tanh.
    # Let's use a better initialization and more iterations.
    pass

def construct():
    start_time = time.time()
    timeout = 110.0
    
    # Seed: all ones
    seed_signs = np.ones(N)
    R_seed = _compute_autoconv(seed_signs)
    val_seed = np.max(np.abs(R_seed[1:]))
    
    # Random initialization
    np.random.seed(42)
    best_signs = np.random.choice([-1, 1], N)
    R_best = _compute_autoconv(best_signs)
    val_best = np.max(np.abs(R_best[1:]))
    
    # Try to improve with gradient descent on tanh relaxation
    # Loss: soft max of squared sidelobes
    def loss_and_grad(logits):
        s = np.tanh(logits)
        R = _compute_autoconv(s)
        R_sidelobes = R[1:]
        squared = R_sidelobes ** 2
        
        m = np.max(squared)
        if m < 1e-8:
            return 0.0, np.zeros(N)
        
        tau = 0.1
        exp_s = np.exp((squared - m) / tau)
        sum_exp = np.sum(exp_s)
        weights = exp_s / sum_exp
        
        loss = m + tau * np.log(sum_exp)
        
        # Gradient
        A = np.zeros(N)
        A[1:] = 2.0 * weights * R_sidelobes
        
        V = np.zeros(N)
        for k in range(1, N):
            if A[k] == 0.0:
                continue
            V[:N-k] += A[k] * s[k:]
            V[k:] += A[k] * s[:N-k]
        V *= DT
        
        ds = 1.0 - s**2
        grad = V * ds
        return loss, grad

    # Run optimization
    if time.time() - start_time < timeout - 10:
        init_logits = np.zeros(N)
        # Add small noise to break symmetry
        init_logits += np.random.randn(N) * 0.1
        
        try:
            res = minimize(
                loss_and_grad,
                init_logits,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 100, 'ftol': 1e-8}
            )
            if res.success:
                opt_signs = np.sign(res.x)
                opt_signs[opt_signs == 0] = 1.0
                R_opt = _compute_autoconv(opt_signs)
                val_opt = np.max(np.abs(R_opt[1:]))
                
                if val_opt < val_best:
                    val_best = val_opt
                    best_signs = opt_signs
        except Exception:
            pass

    # Compare with seed
    if val_seed < val_best:
        best_signs = seed_signs

    return best_signs
