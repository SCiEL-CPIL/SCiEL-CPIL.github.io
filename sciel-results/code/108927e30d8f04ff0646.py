"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time

def _log_energy_and_grad(pts):
    """
    Compute logarithmic energy and gradient for points on the unit sphere.
    Energy: sum_{i<j} log(1 / |x_i - x_j|) = - sum_{i<j} log(|x_i - x_j|)
    Gradient w.r.t. x_i: sum_{j != i} (x_i - x_j) / |x_i - x_j|^2  (for the term -log|d|)
    Wait, let's derive carefully.
    E = sum_{i<j} -log(||x_i - x_j||)
    dE/dx_i = sum_{j != i} - (1/||x_i - x_j||) * (x_i - x_j) / ||x_i - x_j||
           = - sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
           = sum_{j != i} (x_j - x_i) / ||x_i - x_j||^2
    """
    # pts shape (N, 3)
    # Compute pairwise squared distances
    # d2[i,j] = ||x_i - x_j||^2
    # Using broadcasting
    # (N, 1, 3) - (1, N, 3) -> (N, N, 3)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero (diagonal)
    # Set diagonal to 1 to avoid inf, but we will mask it out
    np.fill_diagonal(d2, 1.0)
    
    # d = sqrt(d2)
    d = np.sqrt(d2)
    
    # Energy: sum_{i<j} -log(d_ij)
    # We can sum the upper triangle of -log(d)
    # But -log(d) is symmetric.
    # Sum over all i != j is 2 * sum_{i<j}.
    # So Energy = 0.5 * sum_{i != j} (-log(d_ij))
    # Let's compute the full matrix sum and divide by 2.
    # log_d = log(d)
    # We need to avoid log(0) on diagonal, but d_diag is 1 so log(1)=0.
    log_d = np.log(d)
    # Mask diagonal to 0
    np.fill_diagonal(log_d, 0.0)
    energy = -0.5 * np.sum(log_d)
    
    # Gradient:
    # grad_i = sum_{j != i} (x_j - x_i) / d_ij^2
    # Let's compute a matrix G where G[i,j] = (x_j - x_i) / d_ij^2
    # diff is (x_i - x_j). So (x_j - x_i) = -diff.
    # term = -diff / d2
    # diff shape (N, N, 3), d2 shape (N, N)
    # We need to divide diff by d2 along the last axis.
    # d2[:, :, np.newaxis]
    # But diagonal d2 is 1, diff is 0, so term is 0. Good.
    term = -diff / d2[:, :, np.newaxis]
    
    # grad_i = sum_{j} term[i,j]
    grad = np.sum(term, axis=1)
    
    return energy, grad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial points: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize just in case
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    
    best_pts = pts.copy()
    best_energy, _ = _log_energy_and_grad(best_pts)
    
    # Momentum Gradient Descent
    # v = 0
    v = np.zeros_like(pts)
    mu = 0.9  # momentum
    
    # Learning rate schedule or fixed?
    # Start with a reasonable LR.
    # Energy magnitude is ~ -2300. Distances ~ 0.2.
    # Gradient magnitude?
    # Let's try adaptive or fixed with decay.
    lr = 0.001
    
    # Frenzy parameters
    # If energy doesn't improve for X steps, restart with noise.
    # Or just always add small noise?
    # Let's do: run for T steps. If no improvement, perturb.
    
    step_count = 0
    no_improve_count = 0
    patience = 50
    
    while time.time() < deadline:
        # Compute gradient
        energy, grad = _log_energy_and_grad(pts)
        
        # Update velocity
        v = mu * v - lr * grad
        
        # Tentative new points
        new_pts = pts + v
        
        # Project to sphere
        norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
        # Avoid zero norm
        norms[norms < 1e-10] = 1.0
        new_pts = new_pts / norms
        
        # Check validity (min distance)
        # Quick check: min pairwise distance
        # We can do this efficiently or just trust the optimizer mostly.
        # But the task says "Points closer than 1e-9 ... invalid".
        # Let's check min distance occasionally or every step if cheap enough.
        # For N=150, pairwise distance is 22500 checks. Very fast.
        # Let's do it.
        # Actually, _log_energy_and_grad already computes d2.
        # We can extract min d2 from there? No, it's local.
        # Let's just compute min d2.
        diff = new_pts[:, np.newaxis, :] - new_pts[np.newaxis, :, :]
        d2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(d2, 1.0)
        min_d2 = np.min(d2)
        
        if min_d2 < 1e-18: # (1e-9)^2
            # Invalid, revert and maybe reduce step
            v *= 0.5
            continue
            
        # Evaluate energy of new points
        new_energy, _ = _log_energy_and_grad(new_pts)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pts = new_pts.copy()
            no_improve_count = 0
            # Maybe increase LR slightly? Or keep.
            # Let's keep LR constant for stability, or decay slowly.
            # lr *= 0.999
        else:
            no_improve_count += 1
            # If stagnating, perturb
            if no_improve_count > patience:
                # Add noise
                noise = np.random.normal(0, 0.01, size=new_pts.shape)
                new_pts = new_pts + noise
                new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
                # Reset momentum
                v = np.zeros_like(pts)
                no_improve_count = 0
                # Maybe reduce LR?
                # lr = max(1e-5, lr * 0.5)
                
        pts = new_pts
        step_count += 1
        
        # Decay learning rate slowly
        lr = max(1e-6, lr * 0.999)
        
        # If we hit a very good energy, we might want to do a finer local search
        # But for now, keep going.
        
        # Periodically check time
        if time.time() > deadline:
            break

    # Final normalization
    best_pts /= np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    return best_pts
# EVOLVE-BLOCK-END
