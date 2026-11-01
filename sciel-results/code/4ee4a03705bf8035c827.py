"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_c3_value(h):
    """
    Compute the exact C3 upper bound for a given height vector h.
    Metric: max_k |conv(h, h)[k]| / (sum(|h|))^2
    """
    n_fft = 2**int(np.ceil(np.log2(2*N-1)))
    H = np.fft.rfft(h, n=n_fft)
    C = np.fft.irfft(H * H, n=n_fft)
    conv = C[:2*N-1]
    max_abs = np.max(np.abs(conv))
    sum_abs_h = np.sum(np.abs(h))
    if sum_abs_h < 1e-15:
        return float('inf')
    return max_abs / (sum_abs_h**2)


def _c3_upper_bound(h):
    """
    Compute the upper bound on C3 for a signed step function with heights h.
    Uses discrete binary search with simulated annealing and local refinement.
    """
    start_time = time.time()
    deadline = 115.0  # Leave 35s margin for safety
    
    def _evaluate_binary(h):
        """Evaluate a binary (+1/-1) height vector."""
        return _compute_c3_value(h)
    
    def _local_refine(h, max_iter=100):
        """Local refinement: try flipping each position and keep improvements."""
        h = h.copy()
        best_val = _evaluate_binary(h)
        
        for _ in range(max_iter):
            if time.time() - start_time > deadline:
                break
            
            improved = False
            # Try flipping each position
            for i in range(N):
                h_old = h[i]
                h[i] = -h[i]
                val = _evaluate_binary(h)
                if val < best_val:
                    best_val = val
                    improved = True
                else:
                    h[i] = h_old
            
            if not improved:
                break
        
        return h, best_val
    
    def _simulated_annealing(h_init, max_iter=5000, T_start=1.0, T_end=0.01):
        """Simulated annealing on binary states."""
        h = h_init.copy()
        current_val = _evaluate_binary(h)
        best_h = h.copy()
        best_val = current_val
        
        # Precompute positions for efficiency
        for i in range(max_iter):
            if time.time() - start_time > deadline:
                break
            
            T = T_start * (T_end / T_start) ** (i / max_iter)
            
            # Random flip
            pos = np.random.randint(N)
            h_old = h[pos]
            h[pos] = -h[pos]
            new_val = _evaluate_binary(h)
            
            delta = new_val - current_val
            if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-10)):
                current_val = new_val
                if new_val < best_val:
                    best_val = new_val
                    best_h = h.copy()
            else:
                h[pos] = h_old
        
        return best_h, best_val
    
    # Start with all +1
    h_current = np.ones(N)
    best_h, best_val = _local_refine(h_current, max_iter=50)
    
    # Simulated annealing from multiple restarts
    rng = np.random.default_rng(42)
    
    # Initial diverse starting points
    starting_points = [
        np.ones(N),
        np.ones(N) * -1,
        np.where(rng.random(N) > 0.5, 1.0, -1.0),
        np.where(rng.random(N) > 0.3, 1.0, -1.0),
        np.where(rng.random(N) > 0.7, 1.0, -1.0),
    ]
    
    for sp in starting_points:
        if time.time() - start_time > deadline:
            break
        
        # Local refine first
        h_refined, val_refined = _local_refine(sp, max_iter=30)
        if val_refined < best_val:
            best_val = val_refined
            best_h = h_refined.copy()
        
        # Simulated annealing
        if time.time() - start_time < deadline - 5.0:
            h_sa, val_sa = _simulated_annealing(h_refined, max_iter=3000)
            if val_sa < best_val:
                best_val = val_sa
                best_h = h_sa.copy()
            
            # Final local refinement
            if time.time() - start_time < deadline - 2.0:
                h_final, val_final = _local_refine(h_sa, max_iter=100)
                if val_final < best_val:
                    best_val = val_final
                    best_h = h_final.copy()
    
    # Additional random restarts if time permits
    if time.time() - start_time < deadline - 10.0:
        for _ in range(10):
            if time.time() - start_time > deadline - 5.0:
                break
            
            h_rand = np.where(rng.random(N) > 0.5, 1.0, -1.0)
            h_refined, val_refined = _local_refine(h_rand, max_iter=20)
            
            if val_refined < best_val:
                best_val = val_refined
                best_h = h_refined.copy()
            
            if time.time() - start_time < deadline - 5.0:
                h_sa, val_sa = _simulated_annealing(h_refined, max_iter=1000)
                if val_sa < best_val:
                    best_val = val_sa
                    best_h = h_sa.copy()
                
                if time.time() - start_time < deadline - 2.0:
                    h_final, val_final = _local_refine(h_sa, max_iter=50)
                    if val_final < best_val:
                        best_val = val_final
                        best_h = h_final.copy()
    
    return best_h


def construct():
    return _c3_upper_bound(np.ones(N))
# EVOLVE-BLOCK-END
