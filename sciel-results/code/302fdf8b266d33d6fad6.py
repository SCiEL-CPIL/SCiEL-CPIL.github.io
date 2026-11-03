"""Alternative strategy: Stochastic Coordinate Descent on Sphere."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import sys

# Global deadline
_deadline = 0.0

def _log_energy_fast(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(pts)
    # dist^2 = 2 - 2*dot
    dot = pts @ pts.T
    dist_sq = np.maximum(2.0 - 2.0 * dot, 1e-15)
    np.fill_diagonal(dist_sq, 1.0)
    # We need sum of log(1/sqrt(dist_sq)) = -0.5 * sum(log(dist_sq))
    # Only upper triangle
    iu = np.triu_indices(n, k=1)
    d_sq = dist_sq[iu]
    # Check for invalid
    if np.any(d_sq < 1e-18):
        return float('inf')
    terms = -0.5 * np.log(d_sq)
    return math.fsum(terms.tolist())

def _fibonacci_sphere(n):
    """Golden-angle Fibonacci spiral points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _optimize_coordinate_descent(pts, max_sweeps=1000, time_limit=None):
    """
    Optimize points on sphere using stochastic coordinate descent (pairwise rotations).
    Instead of computing full gradients and Hessians, we pick a pair (i, j),
    and optimize their positions relative to each other and the rest of the points.
    
    For a pair (i, j), let u = pts[i], v = pts[j].
    The energy contribution involving i or j is:
    E_pair = -log|u-v| - log|u-w_k| - log|v-w_k| ...
    
    We can project u and v onto the plane spanned by them (or tangent space) and rotate.
    Actually, a simpler and very effective method for spherical codes is to
    treat the pair as a 2D problem in the plane spanned by u and v, 
    but they are constrained to the sphere.
    
    A more robust local move:
    For each point i, compute the gradient g_i.
    Then, instead of a full Newton step, we can try moving i in the direction of -g_i 
    projected to the tangent space, but we do it one point at a time (Gauss-Seidel style)
    with a line search. This is often faster than full-batch GD because we can 
    accept/reject quickly.
    
    However, the "pairwise rotation" is even better for local minima escape.
    Let's implement a "simulated annealing" style random pair rotation.
    Pick two random points i, j.
    Define the plane spanned by pts[i] and pts[j].
    Rotate them by a small angle theta in that plane.
    Compute the change in energy.
    If energy decreases, accept.
    If energy increases, accept with probability exp(-delta_E / T).
    
    This is essentially a specialized Metropolis algorithm on the sphere.
    """
    pts = pts.copy().astype(np.float64)
    n = len(pts)
    
    # Precompute current energy
    curr_energy = _log_energy_fast(pts)
    best_energy = curr_energy
    best_pts = pts.copy()
    
    # Temperature schedule
    T = 1e-2
    T_min = 1e-8
    cooling = 0.995
    
    # We will iterate through random pairs
    # To speed up energy evaluation, we can update only the terms involving i and j.
    # E = sum_{k<l, k,l not in {i,j}} log(1/d_kl) + terms involving i or j
    # Let's precompute the "background" energy? No, that's complex to maintain.
    # Given N=150, N^2=22500. 
    # _log_energy_fast takes ~5-10ms in Python/NumPy?
    # 1000 sweeps * 150 pairs * 10ms = 1500 seconds. Too slow.
    
    # We need a faster energy update.
    # Let's use the gradient-based Gauss-Seidel but with a very aggressive 
    # "local" line search that only evaluates the energy change for the moved point.
    
    # Actually, let's stick to the full energy evaluation but reduce the number of 
    # expensive evaluations by using a smart step size.
    
    # Alternative: Use the "Fast Multipole Method" approximation? No, standard libs only.
    
    # Let's try a hybrid: 
    # 1. Compute full gradient once.
    # 2. For each point i (Gauss-Seidel), compute the step direction.
    # 3. Do a line search for point i ONLY, updating the energy incrementally?
    #    Incremental update:
    #    E_new - E_old = sum_{j!=i} [ log(1/|x_i' - x_j|) - log(1/|x_i - x_j|) ]
    #    This is O(N) per line search step.
    #    Line search ~10 steps. O(10N) per point.
    #    Total per sweep: N * 10N = 10N^2 = 225,000 ops.
    #    This is ~20x faster than full O(N^2) log evaluation if log is expensive, 
    #    but here log is vectorized.
    #    However, full _log_energy_fast is O(N^2) but highly vectorized (BLAS).
    #    Incremental is Python loop or small vector op.
    #    Let's test if incremental is faster.
    
    # Let's implement the incremental energy update for Gauss-Seidel.
    
    # Precompute distances? No, they change.
    
    # Let's use the gradient-based approach but with a better line search 
    # and maybe a "momentum" term to escape shallow local minima.
    
    # Actually, the previous program used full-batch GD. 
    # Let's use Stochastic Gradient Descent (SGD) with mini-batches?
    # Or just full-batch GD but with a better optimizer like Adam?
    # Adam is good for sparse gradients, but here gradients are dense.
    
    # Let's try "Nesterov Accelerated Gradient" (NAG) on the sphere.
    # v_{t+1} = beta * v_t + grad(x_t)
    # x_{t+1} = project(x_t - lr * (beta * v_{t+1} + (1-beta) * grad(x_{t+1})))
    # This requires 2 gradient evaluations per step.
    
    # Let's try a simpler "Adaptive Learning Rate" GD.
    
    # Re-evaluating the "Pairwise Rotation" idea with O(N) energy update.
    # If we pick pair (i, j), and rotate them in the plane they span, 
    # the change in energy is:
    # Delta E = sum_{k!=i,j} [ log(1/|x_i' - x_k|) + log(1/|x_j' - x_k|) - log(1/|x_i - x_k|) - log(1/|x_j - x_k|) ]
    #          + log(1/|x_i' - x_j'|) - log(1/|x_i - x_j|)
    # Note: |x_i' - x_j'| is constant if we rotate in the plane spanned by them?
    # No, if we rotate i and j by opposite angles, the distance changes.
    # If we rotate both by the same angle, distance is constant.
    # Let's rotate i by +theta and j by -theta.
    
    # This O(N) update is feasible.
    # Let's implement this.
    
    def _energy_delta_pair(i, j, theta):
        """
        Compute energy change if we rotate pts[i] by +theta and pts[j] by -theta
        in the plane spanned by pts[i] and pts[j].
        """
        u = pts[i]
        v = pts[j]
        
        # Normal of the plane
        n = np.cross(u, v)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            # Points are parallel or anti-parallel. Choose an arbitrary perpendicular.
            # If parallel, we can't rotate in the plane. 
            # We can rotate in a plane defined by u and some other vector.
            # For simplicity, return large energy (don't move).
            return float('inf')
        n = n / norm_n
        
        # Rotation matrices
        # R(theta) on u: u' = cos(theta)u + sin(theta)(n x u)
        # n x u is perpendicular to u and in the plane.
        # Let w = n x u. |w| = 1 (since |u|=1, |n|=1, n perp u).
        w_u = np.cross(n, u)
        w_v = np.cross(n, v)
        
        c = np.cos(theta)
        s = np.sin(theta)
        
        u_new = c * u + s * w_u
        v_new = c * v - s * w_v  # Rotate by -theta
        
        # Check if they are still on sphere (they should be)
        
        # Compute Delta E
        # Terms involving i or j with k not in {i, j}
        # d_old_k = |u - x_k|, d_new_k = |u_new - x_k|
        # Sum over k != i, j of log(d_old_k) - log(d_new_k) ... wait, energy is -log(d)
        # Delta E = sum_{k!=i,j} [ -log|u_new - x_k| - log|v_new - x_k| - (-log|u - x_k| - log|v - x_k|) ]
        #         = sum_{k!=i,j} [ log|u - x_k| + log|v - x_k| - log|u_new - x_k| - log|v_new - x_k| ]
        
        # Vectorized computation
        # x_k for k != i, j
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        others = pts[mask]
        
        # Distances old
        d_u_old = np.linalg.norm(u[None, :] - others, axis=1)
        d_v_old = np.linalg.norm(v[None, :] - others, axis=1)
        
        # Distances new
        d_u_new = np.linalg.norm(u_new[None, :] - others, axis=1)
        d_v_new = np.linalg.norm(v_new[None, :] - others, axis=1)
        
        # Avoid log(0)
        d_u_old = np.maximum(d_u_old, 1e-15)
        d_v_old = np.maximum(d_v_old, 1e-15)
        d_u_new = np.maximum(d_u_new, 1e-15)
        d_v_new = np.maximum(d_v_new, 1e-15)
        
        delta_others = np.sum(np.log(d_u_old) + np.log(d_v_old) - np.log(d_u_new) - np.log(d_v_new))
        
        # Term between i and j
        d_ij_old = np.linalg.norm(u - v)
        d_ij_new = np.linalg.norm(u_new - v_new)
        d_ij_old = max(d_ij_old, 1e-15)
        d_ij_new = max(d_ij_new, 1e-15)
        
        delta_ij = np.log(d_ij_old) - np.log(d_ij_new)
        
        return delta_others + delta_ij

    # Main loop
    # We will do sweeps. In each sweep, we pick random pairs.
    # To ensure convergence, we can use a cooling schedule for the step size theta.
    
    rng = np.random.default_rng(42)
    
    for sweep in range(max_sweeps):
        if time_limit and time.time() > time_limit:
            break
        
        # Update temperature
        T *= cooling
        
        # Pick random pairs
        # To be efficient, we can pick N random pairs per sweep
        for _ in range(n):
            i = rng.integers(0, n)
            j = rng.integers(0, n)
            if i == j:
                continue
            
            # Try a few small rotations
            # Step size decreases with T
            max_theta = 0.1 * (T / 1e-2) # Scale step with T
            if max_theta < 1e-4:
                max_theta = 1e-4
                
            # Try random theta
            theta = rng.uniform(-max_theta, max_theta)
            
            dE = _energy_delta_pair(i, j, theta)
            
            # Metropolis criterion
            if dE < 0:
                # Accept
                # Apply rotation
                u = pts[i]
                v = pts[j]
                n_vec = np.cross(u, v)
                norm_n = np.linalg.norm(n_vec)
                if norm_n > 1e-10:
                    n_vec = n_vec / norm_n
                    w_u = np.cross(n_vec, u)
                    w_v = np.cross(n_vec, v)
                    c = np.cos(theta)
                    s = np.sin(theta)
                    pts[i] = c * u + s * w_u
                    pts[j] = c * v - s * w_v
                    curr_energy += dE
                    if curr_energy < best_energy:
                        best_energy = curr_energy
                        best_pts = pts.copy()
            else:
                # Accept with probability exp(-dE / T)
                if rng.random() < np.exp(-dE / T):
                    u = pts[i]
                    v = pts[j]
                    n_vec = np.cross(u, v)
                    norm_n = np.linalg.norm(n_vec)
                    if norm_n > 1e-10:
                        n_vec = n_vec / norm_n
                        w_u = np.cross(n_vec, u)
                        w_v = np.cross(n_vec, v)
                        c = np.cos(theta)
                        s = np.sin(theta)
                        pts[i] = c * u + s * w_u
                        pts[j] = c * v - s * w_v
                        curr_energy += dE
                        if curr_energy < best_energy:
                            best_energy = curr_energy
                            best_pts = pts.copy()
                            
    return best_pts, best_energy

def construct():
    global _deadline
    _deadline = time.time() + 115.0  # Safety margin
    
    # 1. Initialize with Fibonacci
    best_pts = _fibonacci_sphere(N)
    best_energy = _log_energy_fast(best_pts)
    
    # 2. Optimize using Coordinate Descent / Simulated Annealing
    opt_pts, opt_energy = _optimize_coordinate_descent(best_pts, max_sweeps=2000, time_limit=_deadline - 5.0)
    if opt_energy < best_energy:
        best_energy = opt_energy
        best_pts = opt_pts
        
    # 3. Final polish with a few more sweeps at very low T
    if time.time() < _deadline - 2.0:
        opt_pts2, opt_energy2 = _optimize_coordinate_descent(best_pts, max_sweeps=500, time_limit=_deadline - 1.0)
        if opt_energy2 < best_energy:
            best_energy = opt_energy2
            best_pts = opt_pts2

    # Ensure valid output
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    best_pts = best_pts / norms
    
    return best_pts
# EVOLVE-BLOCK-END
