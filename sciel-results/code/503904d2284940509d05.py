import numpy as np
import time

N = 100

def construct():
    """
    Minimize the maximum overlap of h with shifted (1-h).
    
    Uses the insight that uniform h=0.5 achieves the theoretical lower bound.
    Employs a robust optimization approach with feasibility projection
    and time management to ensure we return a valid solution.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Start with the known optimal/uniform solution
    h = np.full(N, 0.5)
    
    def compute_overlap(h_vec):
        """Compute overlap values C(k) for all non-zero circular shifts."""
        N_local = len(h_vec)
        s = h_vec.sum()
        # Circular autocorrelation via FFT
        F = np.fft.fft(h_vec)
        ac = np.real(np.fft.ifft(F * np.conj(F)))[:N_local]
        # Overlap at shift k: sum_i h[i] * (1 - h[i+k]) = s - ac[k]
        # For non-zero shifts
        C_vals = s - ac[1:]
        return C_vals
    
    def max_overlap(h_vec):
        """Maximum overlap over non-zero shifts."""
        C_vals = compute_overlap(h_vec)
        return np.max(C_vals)
    
    def project_to_feasible(h_vec):
        """Project h to satisfy 0 <= h[i] <= 1 and sum(h) = N/2."""
        h_vec = np.clip(h_vec, 0.0, 1.0)
        # Iterative projection to fix sum
        for _ in range(20):
            diff = N / 2.0 - h_vec.sum()
            if abs(diff) < 1e-12:
                break
            # Add/subtract uniformly, respecting bounds
            if diff > 0:
                # Need to increase sum: add to elements < 1
                mask = h_vec < 1.0
                if not np.any(mask):
                    break
                add_per = diff / np.sum(mask)
                h_vec[mask] = np.minimum(1.0, h_vec[mask] + add_per)
            else:
                # Need to decrease sum: subtract from elements > 0
                mask = h_vec > 0.0
                if not np.any(mask):
                    break
                sub_per = (-diff) / np.sum(mask)
                h_vec[mask] = np.maximum(0.0, h_vec[mask] - sub_per)
        # Final exact fix: distribute residual proportionally
        residual = N / 2.0 - h_vec.sum()
        if abs(residual) > 1e-15:
            # Add residual to a single element that has room
            if residual > 0:
                for i in range(N):
                    if h_vec[i] < 1.0:
                        add = min(residual, 1.0 - h_vec[i])
                        h_vec[i] += add
                        residual -= add
                        if abs(residual) < 1e-15:
                            break
            else:
                for i in range(N):
                    if h_vec[i] > 0.0:
                        sub = min(-residual, h_vec[i])
                        h_vec[i] -= sub
                        residual += sub
                        if abs(residual) < 1e-15:
                            break
        return h_vec
    
    # Verify uniform solution
    best_h = h.copy()
    best_val = max_overlap(h)
    
    # Try a few perturbations to see if we can improve (unlikely, but robust)
    rng = np.random.default_rng(12345)
    
    for restart in range(3):
        if time.time() > deadline:
            break
        
        # Create a perturbed initial guess
        h_trial = h + 0.05 * rng.standard_normal(N)
        h_trial = project_to_feasible(h_trial)
        
        trial_val = max_overlap(h_trial)
        if trial_val < best_val:
            best_val = trial_val
            best_h = h_trial.copy()
    
    # Local search refinement around the best solution
    if time.time() < deadline - 30:
        h = best_h.copy()
        for iteration in range(200):
            if time.time() > deadline:
                break
            
            # Random pairwise perturbation
            i, j = rng.integers(0, N, size=2)
            if i == j:
                continue
            
            delta = rng.uniform(-0.01, 0.01)
            h_trial = h.copy()
            h_trial[i] += delta
            h_trial[j] -= delta
            
            if np.all(h_trial >= -1e-10) and np.all(h_trial <= 1 + 1e-10):
                h_trial = project_to_feasible(h_trial)
                val = max_overlap(h_trial)
                if val < best_val - 1e-12:
                    best_val = val
                    best_h = h_trial.copy()
                    h = h_trial.copy()
    
    # Final validation and return
    best_h = project_to_feasible(best_h)
    
    # Ensure exact sum
    final_diff = N / 2.0 - best_h.sum()
    if abs(final_diff) > 1e-15:
        # Add to first element with room
        if final_diff > 0:
            for i in range(N):
                if best_h[i] < 1.0:
                    add = min(final_diff, 1.0 - best_h[i])
                    best_h[i] += add
                    final_diff -= add
                    if abs(final_diff) < 1e-15:
                        break
        else:
            for i in range(N):
                if best_h[i] > 0.0:
                    sub = min(-final_diff, best_h[i])
                    best_h[i] -= sub
                    final_diff += sub
                    if abs(final_diff) < 1e-15:
                        break
    
    return best_h
