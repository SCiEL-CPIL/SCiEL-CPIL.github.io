"""Find the optimal 24-coefficient +/-1 polynomial minimizing sup norm on |z|=1."""
import time
import numpy as np
from itertools import product

# Known optimal polynomials for degree 23 (24 coefficients) from Odlyzko's tables.
# The exhaustive optimum sup norm / sqrt(24) is 1.2583758...
# We need to find which specific coefficient sequences achieve this.

# The seed uses Golay-Rudin-Shapiro which gives a good but not optimal answer.
# Let's use a local search approach to find the true optimum.

def _supnorm(c):
    """Compute max |P(z)| for |z|=1 using dense sampling + golden section refinement."""
    n = len(c)
    # Dense sampling
    m = 2048
    angles = np.linspace(0, 2*np.pi, m, endpoint=False)
    z = np.exp(1j * angles)
    
    # Compute P(z) for all sample points using FFT-like approach
    # P(z) = sum c_k z^k, so P(exp(i*theta)) = sum c_k exp(i*k*theta)
    # This is essentially the DFT of the coefficient sequence
    vals = np.array([np.sum(c * (z[j] ** np.arange(n))) for j in range(m)])
    mag = np.abs(vals)
    
    # Find the top few peaks and refine with golden section
    top_k = 5
    peak_indices = np.argsort(mag)[-top_k:]
    
    best = 0.0
    for idx in peak_indices:
        # Golden section refinement around the peak
        left = (idx - 1) * (2*np.pi / m)
        right = (idx + 1) * (2*np.pi / m)
        
        # Make sure left < right and handle wrap-around
        if left < 0:
            left += 2*np.pi
        if right > 2*np.pi:
            right -= 2*np.pi
        
        def f(theta):
            z_val = np.exp(1j * theta)
            return np.abs(np.sum(c * (z_val ** np.arange(n))))
        
        # Golden section search
        gr = (np.sqrt(5) - 1) / 2  # ~0.618
        a, b = left, right
        c1 = b - gr * (b - a)
        c2 = a + gr * (b - a)
        f1, f2 = f(c1), f(c2)
        
        for _ in range(50):
            if f1 < f2:
                b = c2
                c2 = c1
                f2 = f1
                c1 = b - gr * (b - a)
                f1 = f(c1)
            else:
                a = c1
                c1 = c2
                f1 = f2
                c2 = a + gr * (b - a)
                f2 = f(c2)
        
        peak_val = max(f1, f2, f((a+b)/2))
        if peak_val > best:
            best = peak_val
    
    return best

def _fast_supnorm(c, m=4096):
    """Faster sup norm estimate using numpy vectorization."""
    n = len(c)
    angles = np.linspace(0, 2*np.pi, m, endpoint=False)
    # Compute z^k for all k and all angles using broadcasting
    # Shape: (m, n)
    exponents = np.arange(n)
    z_powers = np.exp(1j * np.outer(angles, exponents))
    vals = z_powers @ c
    mag = np.abs(vals)
    return np.max(mag)

def _local_search(c, max_iter=2000, timeout=100):
    """Local search: flip individual coefficients to reduce sup norm."""
    c = list(c)
    start_time = time.time()
    current_score = _fast_supnorm(c)
    best_c = c[:]
    best_score = current_score
    
    n = len(c)
    
    for iteration in range(max_iter):
        if time.time() - start_time > timeout:
            break
        
        improved = False
        for i in range(n):
            # Try flipping coefficient i
            c[i] = -c[i]
            new_score = _fast_supnorm(c)
            if new_score < current_score:
                current_score = new_score
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_c = c[:]
            else:
                c[i] = -c[i]  # revert
        
        if not improved:
            break
    
    return best_c, best_score

def _multi_start_local_search(num_starts=50, timeout=100):
    """Run local search from multiple random starts."""
    start_time = time.time()
    best_c = None
    best_score = float('inf')
    
    n = 24
    
    # Start with the Golay-Rudin-Shapiro polynomial
    p, q = [1], [1]
    while len(p) < n:
        p, q = p + q, p + [-c for c in q]
    grs = p[:n]
    
    # First try local search from GRS
    c, score = _local_search(grs, max_iter=500, timeout=10)
    if score < best_score:
        best_score = score
        best_c = c
    
    if time.time() - start_time > timeout:
        return best_c, best_score
    
    # Random starts
    rng = np.random.RandomState(42)
    for start in range(num_starts):
        if time.time() - start_time > timeout - 5:
            break
        
        c_init = list(rng.choice([-1, 1], size=n))
        c, score = _local_search(c_init, max_iter=300, timeout=15)
        if score < best_score:
            best_score = score
            best_c = c
    
    return best_c, best_score

def construct():
    """Find the 24 coefficients minimizing sup norm / sqrt(24)."""
    start_time = time.time()
    deadline = 110  # Safety margin before 120s limit
    
    # Try multi-start local search
    best_c, best_score = _multi_start_local_search(num_starts=80, timeout=deadline - 5)
    
    if best_c is None:
        # Fallback: Golay-Rudin-Shapiro
        p, q = [1], [1]
        n = 24
        while len(p) < n:
            p, q = p + q, p + [-c for c in q]
        return p[:n]
    
    return best_c
