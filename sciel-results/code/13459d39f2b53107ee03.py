import numpy as np
import time
import scipy.optimize as opt
from scipy.linalg import eigh

N = 50

def compute_ratio(h):
    """
    Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf) for h >= 0.
    """
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    
    l1 = np.sum(h)
    if l1 < 1e-300:
        return 0.0
    
    l2_sq = np.sum(h * h)
    
    # Efficient autoconvolution using FFT
    # Note: The problem defines h on [-1/4, 1/4] with 50 steps.
    # The autoconvolution h*h is typically defined on a larger domain or wrapped.
    # Given "equal intervals of [-1/4, 1/4]" and 50 heights, the step size is 1/50.
    # The convolution of two functions on [-1/4, 1/4] results in a function on [-1/2, 1/2].
    # The L-infinity norm is the max value of this convolution.
    # Standard linear convolution of length N and N has length 2N-1.
    
    n_fft = 1 << (2 * N - 1).bit_length()
    H = np.fft.rfft(h, n=n_fft)
    hh = np.fft.irfft(H * H, n=n_fft)
    # The linear convolution is in the first 2N-1 samples of the circular convolution of length >= 2N-1
    hh_max = np.max(hh[:2*N-1])
    
    if hh_max < 1e-300:
        return 0.0
        
    return l2_sq / (l1 * hh_max)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Initial seed: uniform
    best_h = np.ones(N) / N
    best_score = compute_ratio(best_h)
    
    # Strategy: 
    # The problem is maximizing R(h) = sum(h_i^2) / (sum(h_i) * max_k (sum_i h_i h_{i+k}))
    # This is a ratio of quadratic forms.
    # By fixing sum(h_i) = 1 (scale invariance), we maximize sum(h_i^2) / max_k (h * shift^k(h)).
    # Let C_k(h) = sum_i h_i h_{i+k}. We want to minimize max_k C_k(h) subject to sum(h_i)=1, h_i>=0.
    # And maximize sum(h_i^2).
    
    # Actually, the SDP approach is complex to implement from scratch in 150s without a solver like CVXPY.
    # However, we can use the fact that the optimal solution often lies in a low-dimensional subspace or has specific symmetries.
    
    # Alternative Structural Change: 
    # The current program uses Sidon sets (sparse). 
    # A different family is "Anti-Sidon" or "Flat" structures, or specifically, 
    # using **Eigenvalue Optimization of the Circulant Matrix**.
    
    # Let's try a different optimization landscape: 
    # Instead of sparse supports, we explore **Periodic Structures**.
    # If h is periodic with period P, the autoconvolution has specific properties.
    # Also, we can try **Gaussian-like** or **Cosine-like** envelopes which are dense.
    
    # But the directive asks for a materially different strategy.
    # Let's implement a **Simulated Annealing** on the full vector, but with a **spectral penalty** 
    # that encourages the autoconvolution to be flat (low peak).
    
    # Actually, a more powerful structural change is to use **Nelder-Mead** on the **Log-Spectrum** 
    # of the signal, which ensures positivity via the inverse transform if we handle the phase correctly, 
    # or simply use **Particle Swarm Optimization (PSO)** on the full vector, which is a global search method 
    # distinct from the local gradients and combinatorial sidon search.
    
    # Let's implement a robust **Particle Swarm Optimization (PSO)** for the full vector.
    # PSO is a population-based stochastic optimization technique. It is structurally different from 
    # L-BFGS (local gradient) and Sidon (combinatorial).
    
    def neg_ratio(h):
        h = np.maximum(h, 0)
        s = np.sum(h)
        if s < 1e-12: return 1e6
        return -compute_ratio(h/s)
    
    # PSO Parameters
    n_particles = 20
    n_iter = 100
    w = 0.5  # Inertia
    c1 = 1.5 # Cognitive
    c2 = 1.5 # Social
    
    # Initialize particles
    # We will use a mix of sparse and dense initializations
    particles = []
    p_bests = []
    p_best_scores = []
    
    global_best = best_h.copy()
    global_best_score = best_score
    
    # Initialize with diverse strategies
    for i in range(n_particles):
        if i % 3 == 0:
            # Sparse random
            h = np.zeros(N)
            k = np.random.randint(5, 15)
            idx = np.random.choice(N, k, replace=False)
            h[idx] = 1.0
        elif i % 3 == 1:
            # Gaussian envelope
            x = np.linspace(-1, 1, N)
            h = np.exp(-x**2 * np.random.uniform(1, 5))
        else:
            # Cosine bump
            x = np.linspace(0, np.pi, N)
            h = np.sin(x) ** np.random.uniform(1, 5)
        
        h = np.maximum(h, 0)
        s = np.sum(h)
        if s > 0:
            h /= s
        else:
            h = np.ones(N)/N
            
        score = -neg_ratio(h)
        particles.append(h)
        p_bests.append(h.copy())
        p_best_scores.append(score)
        
        if score > global_best_score:
            global_best_score = score
            global_best = h.copy()
            
    velocities = [np.zeros(N) for _ in range(n_particles)]
    
    for it in range(n_iter):
        if time.time() > deadline:
            break
            
        for i in range(n_particles):
            # Update velocity
            r1 = np.random.random(N)
            r2 = np.random.random(N)
            
            velocities[i] = w * velocities[i] + c1 * r1 * (p_bests[i] - particles[i]) + c2 * r2 * (global_best - particles[i])
            
            # Update position
            particles[i] = particles[i] + velocities[i]
            
            # Enforce non-negativity and normalize
            particles[i] = np.maximum(particles[i], 0)
            s = np.sum(particles[i])
            if s < 1e-12:
                particles[i] = np.ones(N)/N
                s = 1
            particles[i] /= s
            
            # Evaluate
            score = -neg_ratio(particles[i])
            
            # Update personal best
            if score > p_best_scores[i]:
                p_best_scores[i] = score
                p_bests[i] = particles[i].copy()
                
                # Update global best
                if score > global_best_score:
                    global_best_score = score
                    global_best = particles[i].copy()
                    
    if global_best_score > best_score:
        best_score = global_best_score
        best_h = global_best
        
    # Refine the best solution found by PSO using a local optimizer (Nelder-Mead)
    # This combines global search (PSO) with local refinement, which is structurally different
    # from the current program's "Combinatorial Search + Local Refinement on Sparse Support".
    # Here, the local refinement is on the FULL vector, not just sparse weights.
    
    if time.time() < deadline - 10:
        def neg_ratio_full(w):
            w = np.maximum(w, 0)
            s = np.sum(w)
            if s < 1e-12: return 1e6
            return -compute_ratio(w/s)
        
        try:
            res = opt.minimize(neg_ratio_full, best_h, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < -best_score:
                w_opt = np.maximum(res.x, 0)
                s = np.sum(w_opt)
                if s > 1e-12:
                    w_opt /= s
                    score = compute_ratio(w_opt)
                    if score > best_score:
                        best_score = score
                        best_h = w_opt
        except Exception:
            pass
            
    return list(best_h)

if __name__ == "__main__":
    h = construct()
    print(compute_ratio(h))
