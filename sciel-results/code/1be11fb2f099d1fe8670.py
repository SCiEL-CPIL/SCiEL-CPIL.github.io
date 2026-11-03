"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 140.0  # safety margin before 150s kill
_START = time.time()


def _elapsed():
    return time.time() - _START


def _autoconv(x):
    """Full autoconvolution of x with x (linear, not circular)."""
    n = len(x)
    size = 2 * n - 1
    f = np.fft.rfft(x, size=size)
    ac = np.fft.irfft(f * f, n=size)
    return ac


def _l2_energy_and_grad(x):
    """Compute L2 autoconvolution energy and its gradient via FFT."""
    n = len(x)
    size = 2 * n - 1
    f = np.fft.rfft(x, size=size)
    ac = np.fft.irfft(f * f, n=size)
    energy = np.sum(ac * ac)
    # gradient = 4 * (ac convolved with x), but we only need the part overlapping x
    # triple = ac * f in freq? No: grad = 4 * ifft( |f|^2 * f ) restricted to first n
    # Actually: dE/df = 4 * ifft(f * f * f) [first n samples]
    triple = np.fft.irfft(f * f * f, n=size)
    grad = 4.0 * triple[:n]
    return energy, grad


def _metric(x):
    """Maximum absolute autoconvolution divided by squared integral."""
    ac = _autoconv(x)
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return np.max(np.abs(ac)) * 1e12
    return np.max(np.abs(ac)) / (integral * integral)


def _flip_polish(x, max_iters=5000):
    """Local search: flip bits to reduce max |autoconv| / integral^2."""
    x = x.copy()
    n = len(x)
    current = _metric(x)
    best = x.copy()
    best_val = current
    rng = np.random.default_rng(42)
    
    for it in range(max_iters):
        if _elapsed() > _DEADLINE:
            break
        # Try flipping a random position
        i = rng.integers(n)
        x[i] = -x[i]
        val = _metric(x)
        if val < best_val:
            best_val = val
            best = x.copy()
        else:
            x[i] = -x[i]  # revert
    return best, best_val


def construct():
    global _START
    _START = time.time()
    
    # Phase 1: Continuous relaxation via gradient descent on L2 autoconv energy
    # Initialize with random ±1
    rng = np.random.default_rng(123)
    x = rng.choice([-1.0, 1.0], size=N)
    
    # Run gradient descent on continuous relaxation
    lr = 0.01
    best_continuous = None
    best_energy = np.inf
    
    for step in range(20000):
        if _elapsed() > _DEADLINE - 20:  # reserve 20s for projection+polish
            break
        
        energy, grad = _l2_energy_and_grad(x)
        
        if energy < best_energy:
            best_energy = energy
            best_continuous = x.copy()
        
        # Adaptive step size
        x -= lr * grad
        
        # Occasionally re-normalize magnitude to keep values reasonable
        if step % 500 == 0:
            norm = np.sqrt(np.mean(x * x))
            if norm > 0:
                x = x / norm * 1.0
        
        # Decay learning rate
        lr *= 0.9999
    
    if best_continuous is not None:
        x = best_continuous
    
    # Phase 2: Project to ±1
    x_bin = np.sign(x)
    x_bin[x_bin == 0] = 1.0
    
    # Phase 3: Discrete polish
    x_polished, val = _flip_polish(x_bin, max_iters=30000)
    
    # Final check: also try the all-ones seed as fallback
    seed_val = _metric(np.ones(N))
    if val > seed_val:
        return np.ones(N)
    
    return x_polished
# EVOLVE-BLOCK-END
