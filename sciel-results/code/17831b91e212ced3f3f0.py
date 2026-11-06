import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin before 120s limit
    
    n = 24
    
    # Precompute dense grid and Z matrix
    n_grid = 4096
    angles = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
    z = np.exp(1j * angles)
    k_idx = np.arange(n)
    Z = z[:, None] ** k_idx[None, :]  # shape (n_grid, n)
    
    def sup_norm_fast(coeffs):
        """Compute max |P(z)| over |z|=1 using precomputed grid + golden-section refinement."""
        c = np.array(coeffs, dtype=np.complex128)
        vals = Z @ c
        abs_vals = np.abs(vals)
        m = np.max(abs_vals)
        # Golden-section refinement around the peak
        idx = np.argmax(abs_vals)
        dtheta = 2 * np.pi / n_grid
        a = angles[idx] - dtheta
        b = angles[idx] + dtheta
        gr = (np.sqrt(5) - 1) / 2
        c1 = b - gr * (b - a)
        c2 = a + gr * (b - a)
        f1 = np.abs(np.sum(c * np.exp(1j * c1 * k_idx)))
        f2 = np.abs(np.sum(c * np.exp(1j * c2 * k_idx)))
        for _ in range(50):
            if f1 < f2:
                b = c2
                c2 = c1
                f2 = f1
                c1 = b - gr * (b - a)
                f1 = np.abs(np.sum(c * np.exp(1j * c1 * k_idx)))
            else:
                a = c1
                c1 = c2
                f1 = f2
                c2 = a + gr * (b - a)
                f2 = np.abs(np.sum(c * np.exp(1j * c2 * k_idx)))
        refined = max(f1, f2)
        return max(m, refined)
    
    # Initial candidate: Rudin-Shapiro sequence
    def rudin_shapiro(length):
        p, q = [1], [1]
        while len(p) < length:
            p, q = p + q, p + [-c for c in q]
        return p[:length]
    
    best_coeffs = rudin_shapiro(n)
    best_val = sup_norm_fast(best_coeffs)
    
    # Local search with random restarts
    np.random.seed(42)
    
    def local_search(coeffs, max_iters=2000):
        """Flip individual coefficients to minimize sup-norm."""
        current = list(coeffs)
        current_val = sup_norm_fast(current)
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            # Try flipping a random coefficient
            i = np.random.randint(n)
            trial = current.copy()
            trial[i] = -trial[i]
            trial_val = sup_norm_fast(trial)
            if trial_val < current_val:
                current = trial
                current_val = trial_val
        return current, current_val
    
    # Multiple random restarts
    n_restarts = 30
    for r in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
        
        if r == 0:
            coeffs = list(best_coeffs)
        else:
            coeffs = [1 if np.random.random() < 0.5 else -1 for _ in range(n)]
        
        found, val = local_search(coeffs, max_iters=1500)
        if val < best_val:
            best_val = val
            best_coeffs = found
    
    # Final refinement: try all single flips from best
    for i in range(n):
        if time.time() > deadline - 5.0:
            break
        trial = best_coeffs.copy()
        trial[i] = -trial[i]
        trial_val = sup_norm_fast(trial)
        if trial_val < best_val:
            best_val = trial_val
            best_coeffs = trial
    
    return [int(c) for c in best_coeffs]
