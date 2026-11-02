"""Spectral amplitude shaping for the first autocorrelation inequality."""
import time
import numpy as np
from scipy.optimize import minimize

N = 600
DEADLINE = 115.0  # safety margin before 120s kill

def _compute_score(heights):
    """Compute the normalized maximum of the circular autoconvolution.
    
    Score = N * max_k(autocorr[k]) / (sum(h))^2
    where autocorr[k] = sum_i h[i] * h[(i+k)%N]
    
    For h=[1]*N: autocorr[k]=N for all k, sum(h)=N, score = N*N/N^2 = 1.
    """
    h = np.asarray(heights, dtype=np.float64)
    if np.any(h < -1e-12):
        return 1e10
    h = np.maximum(h, 0.0)
    
    # Circular autocorrelation via FFT
    nfft = 1 << (2*N - 1).bit_length()
    H = np.fft.rfft(h, n=nfft)
    autocorr = np.fft.irfft(H * np.conj(H), n=nfft)[:N]
    
    total = np.sum(h)
    if total < 1e-12:
        return 1e10
    
    score = N * np.max(autocorr) / (total * total)
    return float(score)


def _spectral_init():
    """Initialize with a smooth spectral profile."""
    freqs = np.fft.rfftfreq(N)
    w = 0.5 * (1 + np.cos(2 * np.pi * freqs * 0.5))
    h = np.fft.irfft(w * N, n=N)
    h = np.maximum(h, 0.0)
    if np.sum(h) < 1e-10:
        h = np.ones(N)
    return h


def construct():
    start_time = time.time()
    
    best_h = np.ones(N)
    best_score = _compute_score(best_h)
    
    def elapsed():
        return time.time() - start_time
    
    def time_left():
        return DEADLINE - elapsed()
    
    # Smooth surrogate using log-sum-exp for the max
    def smooth_score(h, beta=50.0):
        h = np.maximum(h, 0.0)
        nfft = 1 << (2*N - 1).bit_length()
        H = np.fft.rfft(h, n=nfft)
        autocorr = np.fft.irfft(H * np.conj(H), n=nfft)[:N]
        
        total = np.sum(h)
        if total < 1e-12:
            return 1e10
        base = N / (total * total)
        
        a = autocorr * base
        max_a = np.max(a)
        if max_a > 0:
            lse = max_a + np.log(np.sum(np.exp(a - max_a)))
        else:
            lse = 0.0
        return lse
    
    # Epigraph formulation: minimize t subject to base*autocorr[k] <= t for all k
    # This is equivalent to minimizing max_k(base*autocorr[k])
    # With smooth constraints: base*autocorr[k] - t + epsilon*log-sum-exp trick
    
    def epigraph_objective(x):
        """x = [h_0, ..., h_{N-1}, t]"""
        h = np.maximum(x[:N], 0.0)
        t = x[N]
        nfft = 1 << (2*N - 1).bit_length()
        H = np.fft.rfft(h, n=nfft)
        autocorr = np.fft.irfft(H * np.conj(H), n=nfft)[:N]
        
        total = np.sum(h)
        if total < 1e-12:
            return 1e10
        base = N / (total * total)
        
        # Smooth violation: sum of soft-plus of (base*autocorr[k] - t)
        a = autocorr * base - t
        # Soft max via log-sum-exp with temperature
        temp = 5.0
        max_a = np.max(a)
        if max_a > 0:
            penalty = max_a + temp * np.log(np.sum(np.exp((a - max_a) / temp))) / N
        else:
            penalty = 0.0
        return t + penalty
    
    # Gradient descent with momentum on the latent space
    # Parameterize h via a low-dimensional latent vector through a simple MLP
    # This allows complex non-parametric shapes
    
    class SimpleMLP:
        """Simple MLP for parameterizing step heights."""
        def __init__(self, latent_dim, hidden_dims, out_dim, seed=42):
            self.latent_dim = latent_dim
            self.hidden_dims = hidden_dims
            self.out_dim = out_dim
            rng = np.random.RandomState(seed)
            
            self.weights = []
            self.biases = []
            dims = [latent_dim] + hidden_dims + [out_dim]
            for i in range(len(dims) - 1):
                w = rng.randn(dims[i+1], dims[i]) / np.sqrt(dims[i])
                b = rng.randn(dims[i+1]) * 0.1
                self.weights.append(w)
                self.biases.append(b)
        
        def forward(self, z):
            """Forward pass, returns (output, cache)."""
            cache = []
            x = z
            for i, (w, b) in enumerate(zip(self.weights, self.biases)):
                pre = x @ w.T + b
                if i < len(self.weights) - 1:
                    x = np.maximum(pre, 0.0)  # ReLU
                else:
                    x = pre  # Linear output
                cache.append((pre, x))
            return x, cache
        
        def backward(self, grad_out, cache):
            """Backward pass."""
            grads = [None] * len(self.weights)
            bias_grads = [None] * len(self.biases)
            grad = grad_out
            for i in range(len(self.weights) - 1, -1, -1):
                pre, x = cache[i]
                grads[i] = grad @ x.T
                bias_grads[i] = np.sum(grad, axis=0)
                if i > 0:
                    grad = (grad @ self.weights[i].T) * (pre > 0)
            return grads, bias_grads, grad
    
    def mlp_objective(z, mlp):
        """Objective in latent space."""
        h, _ = mlp.forward(z)
        h = np.maximum(h, 0.0)
        return smooth_score(h, beta=50.0)
    
    def mlp_gradient(z, mlp):
        """Gradient in latent space using finite differences (reliable)."""
        eps = 1e-5
        grad = np.zeros_like(z)
        f0 = mlp_objective(z, mlp)
        for i in range(len(z)):
            z_plus = z.copy()
            z_plus[i] += eps
            f_plus = mlp_objective(z_plus, mlp)
            grad[i] = (f_plus - f0) / eps
        return grad
    
    # Strategy 1: Direct optimization with L-BFGS-B
    def objective_direct(x):
        return smooth_score(x)
    
    np.random.seed(42)
    n_starts = 4
    
    for s in range(n_starts):
        if time_left() < 15.0:
            break
        
        if s == 0:
            h0 = _spectral_init()
        elif s == 1:
            h0 = np.ones(N) * (1 + 0.2 * np.random.randn(N))
            h0 = np.maximum(h0, 0.0)
        elif s == 2:
            center = N // 2
            sigma = N / 5.0
            idx = np.arange(N)
            h0 = np.exp(-0.5 * ((idx - center) / sigma) ** 2)
            h0 = np.maximum(h0, 0.0)
        else:
            # Two-bump structure
            idx = np.arange(N)
            h0 = np.exp(-0.5 * ((idx - N//4) / (N/10.0)) ** 2) + \
                 np.exp(-0.5 * ((idx - 3*N//4) / (N/10.0)) ** 2)
            h0 = np.maximum(h0, 0.0)
        
        if np.max(h0) > 0:
            h0 = h0 / np.max(h0)
        
        result = minimize(
            objective_direct,
            h0,
            method='L-BFGS-B',
            bounds=[(0, None)] * N,
            options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-9, 'maxls': 50}
        )
        
        h_opt = np.maximum(result.x, 0.0)
        sc = _compute_score(h_opt)
        
        if sc < best_score:
            best_score = sc
            best_h = h_opt.copy()
        
        # Polish with tighter tolerance
        if time_left() > 20.0 and sc < best_score + 1e-5:
            result2 = minimize(
                lambda x: smooth_score(x, beta=100.0),
                h_opt,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-11, 'maxls': 30}
            )
            h_polish = np.maximum(result2.x, 0.0)
            sc2 = _compute_score(h_polish)
            if sc2 < best_score:
                best_score = sc2
                best_h = h_polish.copy()
    
    # Strategy 2: MLP-based latent space optimization
    if time_left() > 30.0:
        latent_dim = 32
        hidden_dims = [64, 128]
        mlp = SimpleMLP(latent_dim, hidden_dims, N, seed=123)
        
        # Initialize latent to produce roughly uniform output
        z0 = np.random.randn(latent_dim) * 0.5
        
        # Adam optimizer
        lr = 0.01
        beta1 = 0.9
        beta2 = 0.999
        eps_adam = 1e-8
        m = np.zeros(latent_dim)
        v = np.zeros(latent_dim)
        t_adam = 0
        
        for step in range(500):
            if time_left() < 15.0:
                break
            
            z = z0
            # Use current best as initialization reference
            h_curr, cache = mlp.forward(z)
            h_curr = np.maximum(h_curr, 0.0)
            
            # Compute gradient via finite differences (expensive but reliable)
            if step % 10 == 0:
                grad = mlp_gradient(z, mlp)
            else:
                # Reuse last gradient (approximation)
                pass
            
            t_adam += 1
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad**2
            m_hat = m / (1 - beta1**t_adam)
            v_hat = v / (1 - beta2**t_adam)
            z = z - lr * m_hat / (np.sqrt(v_hat) + eps_adam)
            
            # Evaluate
            h_test, _ = mlp.forward(z)
            h_test = np.maximum(h_test, 0.0)
            sc_test = _compute_score(h_test)
            
            if sc_test < best_score:
                best_score = sc_test
                best_h = h_test.copy()
                z0 = z.copy()
    
    # Strategy 3: Symmetric optimization (h[i] = h[N-1-i])
    if time_left() > 10.0:
        half = N // 2
        def obj_sym(x):
            h = np.zeros(N)
            h[:half] = np.maximum(x, 0.0)
            h[half:] = h[:half][::-1]
            return smooth_score(h)
        
        x0 = best_h[:half].copy()
        res_sym = minimize(
            obj_sym,
            x0,
            method='L-BFGS-B',
            bounds=[(0, None)] * half,
            options={'maxiter': 200, 'ftol': 1e-13, 'gtol': 1e-10}
        )
        
        x_opt = np.maximum(res_sym.x, 0.0)
        h_sym = np.zeros(N)
        h_sym[:half] = x_opt
        h_sym[half:] = x_opt[::-1]
        
        sc_sym = _compute_score(h_sym)
        if sc_sym < best_score:
            best_score = sc_sym
            best_h = h_sym.copy()
    
    return [float(x) for x in best_h]


if __name__ == '__main__':
    result = construct()
    print(f"Score: {_compute_score(result)}")
