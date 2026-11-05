import numpy as np
import time
from scipy.optimize import minimize

N = 400
_DEADLINE = 145.0

def _evaluate(h):
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _objective_and_grad(z):
    """
    z: complex vector of length N.
    h = real(z).
    We optimize in the complex domain to allow phase shifts, 
    but the final h must be real. 
    Actually, the task requires real h. 
    So we should optimize real h directly, or use a parametrization that maps to real h.
    The prompt suggests "sum of two orthogonal, time-shifted Zadoff-Chu sequences".
    Zadoff-Chu sequences are complex. 
    If we take the real part of a Zadoff-Chu sequence, we get a cosine-like sequence.
    Let's stick to real h optimization but use structured initializations 
    and potentially a Fourier-based gradient descent or just L-BFGS-B on real h.
    
    Wait, the directive says: "Represent the step heights as a sum of two orthogonal, time-shifted Zadoff-Chu sequences with independently optimized relative phase and amplitude".
    Since h must be real, we can define:
    h[n] = A1 * Re( z1[n] * exp(i*phi1) ) + A2 * Re( z2[n+tau] * exp(i*phi2) )
    where z1, z2 are Zadoff-Chu sequences.
    But optimizing 400 variables with L-BFGS-B is what the current program does.
    The "materially different" part is the representation/initialization and potentially the search space.
    
    If I just use L-BFGS-B on real h, it's the same as before.
    To be materially different, I should optimize the parameters of this specific structure, 
    or use a different optimization method like Gradient Descent with momentum 
    or Simulated Annealing on the full 400-dim space but with a different objective handling?
    
    Actually, the current program uses L-BFGS-B. 
    A different optimization method could be:
    1. Coordinate Descent (Inspiration 1 used this).
    2. Fourier Domain Optimization (Inspiration 2/3 used this).
    3. Structured Parametric Optimization (The directive's suggestion).
    
    Let's try the Structured Parametric Optimization as requested by the directive, 
    but extended to allow some flexibility or just stick to the structure.
    If the structure is too rigid, it might not beat the seed.
    However, the directive says: "Idea target: design the new strategy so that its result answers this open question...".
    And "Make a STRUCTURAL change".
    
    Let's implement the parametric strategy:
    h[n] = c1 * ZC1[n] + c2 * ZC2[n+tau]
    where ZC1, ZC2 are Zadoff-Chu sequences (real parts).
    Parameters: c1, c2, tau, and maybe phases if we allow complex scaling? 
    If we restrict to real h, we can use:
    h[n] = A1 * cos(2*pi*u1*n/N + p1) + A2 * cos(2*pi*u2*(n+tau)/N + p2)
    This is similar to the sinusoidal family in the current program but with two terms and specific frequencies.
    
    Actually, Zadoff-Chu sequences have constant amplitude and zero autocorrelation (in complex domain).
    The real part of a Zadoff-Chu sequence is a cosine wave with specific frequency.
    For N=400, we need u such that gcd(u, N) = 1.
    
    Let's try a hybrid:
    1. Generate candidate h using the parametric structure.
    2. Refine using a few steps of coordinate descent or gradient descent?
    Or just optimize the parameters of the structure.
    
    If I only optimize the parameters (A1, A2, p1, p2, u1, u2, tau), the dimension is very low (~7).
    This is a very different search space.
    
    Let's implement this.
    """
    # Parse parameters
    # p = [A1, A2, p1, p2, u1, u2, tau]
    A1, A2, p1, p2, u1, u2, tau = z
    
    # Ensure u1, u2 are valid frequencies for Zadoff-Chu (or just sinusoids)
    # For real cosine, any frequency works.
    # Let's use the cosine representation of the real part of Zadoff-Chu.
    # ZC[n] = exp(-j * pi * u * n^2 / N)
    # Re(ZC[n]) = cos(-pi * u * n^2 / N)
    # This is a chirp, not a simple sinusoid!
    # Chirps have excellent autocorrelation properties.
    
    n = np.arange(N)
    # Chirp 1
    # u1 should be an integer. Let's fix u1=1, u2=3 for simplicity, or optimize them?
    # Optimizing non-integer u might break the "Zadoff-Chu" property but allow more flexibility.
    # Let's fix u1=1, u2=2 (or other coprime integers) and optimize amplitudes, phases, and shift.
    # But the prompt says "independently optimized relative phase and amplitude".
    # It doesn't explicitly say optimize frequency.
    # Let's fix frequencies to 1 and 3 (coprime with 400? 1 is coprime, 3 is coprime).
    
    u1 = 1
    u2 = 3
    
    # Phases
    # h[n] = A1 * cos(pi * u1 * n^2 / N + p1) + A2 * cos(pi * u2 * (n+tau)^2 / N + p2)
    # Note: (n+tau)^2 = n^2 + 2*n*tau + tau^2
    # So the second term is a shifted chirp.
    
    term1 = np.cos(np.pi * u1 * n**2 / N + p1)
    # Shifted chirp
    n_shifted = (n + tau) % N  # Circular shift? Or linear?
    # Zadoff-Chu is defined on a group. Circular shift is natural.
    # But convolve is linear.
    # Let's use linear shift for the argument.
    arg2 = np.pi * u2 * (n + tau)**2 / N + p2
    term2 = np.cos(arg2)
    
    h = A1 * term1 + A2 * term2
    
    s = np.sum(h)
    if abs(s) < 1e-6:
        return 1e10, np.zeros(7)
        
    conv = np.convolve(h, h)
    max_val = np.max(conv)
    val = abs(2.0 * N * max_val / (s * s))
    
    # Gradient calculation for this parametric form is complex.
    # We can use numerical gradient or just use Nelder-Mead (derivative-free) for the 7 params.
    # Nelder-Mead is robust for low-dimensional problems.
    
    return val, None

def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    best_h = np.ones(N)
    best_val = 2.0
    
    rng = np.random.default_rng(42)
    
    # Parameters: A1, A2, p1, p2, tau
    # We fix u1=1, u2=3.
    # We can also try different u pairs.
    
    u_pairs = [(1, 3), (1, 5), (3, 5), (1, 7), (3, 7)]
    
    for u1, u2 in u_pairs:
        if time.time() > deadline - 10:
            break
            
        # Initial guess
        x0 = np.array([1.0, 1.0, rng.uniform(0, 2*np.pi), rng.uniform(0, 2*np.pi), rng.uniform(0, N)])
        
        # Define objective for Nelder-Mead
        def obj(x):
            A1, A2, p1, p2, tau = x
            n = np.arange(N)
            term1 = np.cos(np.pi * u1 * n**2 / N + p1)
            arg2 = np.pi * u2 * (n + tau)**2 / N + p2
            term2 = np.cos(arg2)
            h = A1 * term1 + A2 * term2
            
            s = np.sum(h)
            if abs(s) < 1e-6:
                return 1e10
            conv = np.convolve(h, h)
            return abs(2.0 * N * np.max(conv) / (s * s))
        
        try:
            res = minimize(obj, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-6})
            
            if res.fun < best_val:
                best_val = res.fun
                A1, A2, p1, p2, tau = res.x
                n = np.arange(N)
                term1 = np.cos(np.pi * u1 * n**2 / N + p1)
                arg2 = np.pi * u2 * (n + tau)**2 / N + p2
                term2 = np.cos(arg2)
                best_h = A1 * term1 + A2 * term2
                
        except Exception:
            continue
            
    # If the parametric search didn't beat the seed, we might want to try a simple random restart with L-BFGS-B 
    # as a fallback? No, the directive asks for a structurally different strategy.
    # But if it fails, we return the seed.
    
    final_val = _evaluate(best_h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        best_h = np.ones(N)
        
    return best_h
