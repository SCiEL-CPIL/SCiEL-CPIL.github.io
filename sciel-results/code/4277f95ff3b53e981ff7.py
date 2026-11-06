import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_radius(points):
    """
    Computes the maximum radius for the given points.
    r = min(min_pairwise_dist / 2, min_dist_to_boundary)
    """
    # Distance to boundary
    norms = np.linalg.norm(points, axis=1)
    min_boundary = np.min(1.0 - norms)
    
    # Pairwise distances
    # Use broadcasting for efficiency
    # points shape (N, 2)
    # diff shape (N, N, 2)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_boundary, min_pair)

def energy_and_grad(x):
    """
    x: flat array of shape (N, 2)
    Returns energy and gradient.
    Energy = sum(1/d_ij^2) + sum(1/(1-r_i)^2)
    """
    points = x.reshape(N, 2)
    
    # Pairwise repulsion
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    
    # Avoid division by zero
    # We add a small epsilon to prevent inf if points coincide
    dists_sq_safe = np.maximum(dists_sq, 1e-8)
    
    # Energy term: 1/d^2
    # E_pair = sum_{i<j} 1/d_ij^2
    # Sum over upper triangle
    iu = np.triu_indices(N, k=1)
    e_pair = np.sum(1.0 / dists_sq_safe[iu])
    
    # Gradient of pair term
    # d/dx_i (1/||x_i - x_j||^2) = -2 (x_i - x_j) / ||x_i - x_j||^4
    # We accumulate gradients
    grad = np.zeros_like(points)
    
    # Vectorized gradient accumulation for pairs is tricky, 
    # but for N=50, a loop over pairs is fast enough (1225 pairs)
    # Let's use a loop for clarity and correctness, or vectorize carefully.
    
    # Vectorized approach for gradient:
    # grad_i = sum_{j != i} -2 * (x_i - x_j) / dists_sq_ij^2
    # Let's compute the full matrix of forces
    # Force matrix F[i,j] = -2 * (x_i - x_j) / dists_sq_ij^2
    # Note: dists_sq is symmetric.
    
    # To avoid 0 division in gradient, use safe dists_sq
    # F = -2 * diff / (dists_sq_safe**2)[:,:,np.newaxis]
    # But we only want i != j
    mask = ~np.eye(N, dtype=bool)
    F = np.zeros_like(diff)
    F[mask] = -2.0 * diff[mask] / (dists_sq_safe[mask, np.newaxis] ** 2)
    
    # Sum over j for each i
    grad_pair = np.sum(F, axis=1)
    grad += grad_pair
    
    # Boundary repulsion
    # E_bound = sum 1/(1-r_i)^2
    norms = np.linalg.norm(points, axis=1)
    norms_safe = np.maximum(norms, 1e-8)
    # Clip norms to be < 1 to avoid negative denominator or singularity at 1
    # If point is outside, (1-r) is negative, square is positive. 
    # But we want to push it in. 
    # The term 1/(1-r)^2 is symmetric around r=1. 
    # To strictly penalize outside, we might need a different potential.
    # However, L-BFGS-B allows bounds. We can set bounds [ -1, 1 ] for coords?
    # No, better to use the potential that blows up at boundary.
    # If r > 1, 1-r is negative. (1-r)^2 is positive. 
    # The gradient will point inwards if r < 1, and outwards if r > 1?
    # d/dr (1/(1-r)^2) = 2/(1-r)^3.
    # If r < 1, 1-r > 0, grad > 0. Force is +x/r. This pushes OUT?
    # Wait. Energy should be high when close to boundary.
    # If r=0.9, E=100. If r=0.99, E=10000.
    # We want to minimize E. So we want to move away from r=1.
    # If r < 1, we want to move to smaller r? No, we want to move to center?
    # Actually, for packing, we want to fill the space.
    # The repulsion from other circles will push them to the boundary.
    # The boundary term should prevent them from crossing.
    # If we use 1/(1-r)^2, the minimum is at r=0.
    # This will collapse all circles to the center!
    # We need a potential that is low when r < 1 and high when r >= 1.
    # Or, we rely on the pairwise repulsion to spread them, and the boundary term to keep them in.
    # If we use a potential that penalizes r > 1, e.g., max(0, r-1)^2, it's non-smooth.
    # Let's use 1/(1-r)^2 but only for r < 1? No.
    
    # Alternative: Use a soft boundary.
    # E_bound = sum (1 - r_i)^2 for r_i < 1? No, that minimizes distance to boundary (pushes to boundary).
    # We want to keep them INSIDE.
    # The pairwise repulsion will push them to the boundary.
    # We need a term that stops them from going outside.
    # A common trick: E_bound = sum max(0, r_i - 1 + margin)^2.
    # This is non-smooth at r=1-margin.
    
    # Let's stick to the simple 1/(1-r)^2 but realize that without a centering force, they might collapse.
    # However, the pairwise repulsion 1/d^2 is very strong.
    # If they collapse to center, d->0, E_pair -> inf.
    # So they will spread out.
    # The boundary term 1/(1-r)^2 will be small if they are near center, and large if near boundary.
    # This will actually pull them IN from the boundary, fighting the pairwise repulsion.
    # This might result in a configuration that is not fully expanded to the boundary.
    
    # To fix this, we can use a potential that is 0 for r < 1 - epsilon and rises for r > 1 - epsilon.
    # Or, simply use bounds in L-BFGS-B?
    # L-BFGS-B supports bounds.
    # But the objective function needs to be smooth.
    
    # Let's try a different energy:
    # Minimize sum_{i<j} max(0, 1 - d_ij)^2 + sum_i max(0, r_i - 1)^2
    # This is the standard "overlap" energy.
    # It is non-smooth at d=1 and r=1.
    # We can smooth it:
    # E = sum_{i<j} (1 - d_ij)_+^2 ...
    
    # Given the complexity, let's use the **Repulsion** model but add a **Centering** force?
    # No, the container is the constraint.
    
    # Let's try minimizing:
    # E = sum_{i<j} 1/d_ij^2 + lambda * sum_i r_i^2
    # The second term pulls them to the center.
    # The first term pushes them apart.
    # This will create a stable cluster in the center.
    # We want them to fill the circle.
    
    # Actually, the standard way to pack circles is to maximize the minimum distance.
    # Minimizing sum 1/d^2 is a good proxy.
    # If we just minimize sum 1/d^2, they will spread out as much as possible.
    # Will they hit the boundary?
    # Yes, because if they are all in the center, d is small, E is high.
    # If they move to the boundary, d increases, E decreases.
    # So the minimum of sum 1/d^2 will be on the boundary.
    # We just need to ensure they don't go OUTSIDE.
    # We can add a penalty for being outside:
    # E_out = sum_i max(0, r_i - 1)^2
    # This is non-smooth.
    # We can approximate max(0, x)^2 with a smooth function, or just use the fact that L-BFGS-B can handle bounds?
    # If we use bounds, the objective function doesn't need to penalize out-of-bounds, the optimizer handles it.
    # But L-BFGS-B bounds are on the variables x_i, y_i.
    # We can set bounds such that x^2 + y^2 <= 1? No, bounds are rectangular.
    # So we must handle the circular boundary in the objective.
    
    # Let's use a smooth approximation for the boundary constraint:
    # E_bound = sum_i softplus(r_i - 1)^2 ?
    # Or simply: if r_i > 1, add a large penalty.
    
    # Let's use the energy:
    # E = sum_{i<j} 1/d_ij^2 + 100 * sum_i max(0, r_i - 1)^2
    # And use a smooth approx for max:
    # max(0, x) approx x * sigmoid(k*x) ?
    
    # Simpler: Just use the 1/(1-r)^2 term but with a large weight, and hope the pairwise repulsion dominates?
    # No, 1/(1-r)^2 pulls to center.
    
    # Let's try this:
    # Minimize sum_{i<j} 1/d_ij^2
    # Subject to r_i <= 1.
    # We can use `scipy.optimize.minimize` with `L-BFGS-B` and no boundary term, but we must initialize inside and hope it stays inside?
    # No, it will go outside if it minimizes energy.
    
    # Correct approach:
    # E = sum_{i<j} 1/d_ij^2 + sum_i (r_i - 1)^2 * (r_i > 1)
    # To make it smooth, use:
    # E_bound = sum_i exp(r_i - 1) for r_i > 1?
    
    # Let's use the **Penalty Method** with a smooth penalty:
    # P(r) = (r - 1)^2 if r > 1, else 0.
    # Smooth approx: P(r) = (r - 1 + sqrt((r-1)^2 + eps))^2 / 4 ?
    
    # Given time constraints, let's use a simpler, robust local search:
    # **Simulated Annealing** on the exact score.
    # It's slower per step but doesn't require gradient derivation.
    # With N=50, and vectorized score calculation, we can do ~1000-5000 steps/second.
    # 100s * 5000 = 500,000 steps. This should be enough to find a good local optimum.
    
    # Let's implement Simulated Annealing.
    
    return e_pair, grad_pair # Placeholder, will implement SA instead

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_points = None
    best_score = -1.0
    
    # Seeds
    seeds = []
    
    # 1. Hexagonal
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            hex_seed = points[np.argsort(radius, kind="stable")[:N]]
            seeds.append(hex_seed)
            break
        spacing *= 0.98
        
    # 2. Rings
    # 1 center, 6, 12, 18, 13?
    ring_radii = [0, 0.3, 0.6, 0.85, 1.0]
    ring_counts = [1, 6, 12, 18, 13]
    pts = []
    for r, count in zip(ring_radii, ring_counts):
        if count == 0: continue
        angles = np.linspace(0, 2*np.pi, count, endpoint=False)
        for a in angles:
            pts.append([r * np.cos(a), r * np.sin(a)])
    ring_seed = np.array(pts[:N])
    if len(ring_seed) == N:
        seeds.append(ring_seed)
        
    # 3. Random
    np.random.seed(42)
    for _ in range(3):
        angles = np.random.uniform(0, 2 * np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 1, N))
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        seeds.append(pts)
        
    for seed in seeds:
        if time.time() > deadline:
            break
            
        current_points = seed.copy()
        current_score = compute_radius(current_points)
        
        if current_score > best_score:
            best_score = current_score
            best_points = current_points.copy()
            
        # Simulated Annealing
        T = 0.05
        cooling = 0.99
        step_size = 0.05
        
        while time.time() < deadline and T > 1e-6:
            # Pick random index
            i = np.random.randint(N)
            
            # Generate move
            delta = np.random.normal(0, step_size, 2)
            new_point = current_points[i] + delta
            
            # Check boundary
            if np.linalg.norm(new_point) > 1.0:
                # Reflect or reject
                # Reflect: project back
                norm = np.linalg.norm(new_point)
                new_point = new_point / norm * 1.0
                # Small nudge in
                new_point *= 0.99
                
            old_points = current_points.copy()
            current_points[i] = new_point
            new_score = compute_radius(current_points)
            
            # Acceptance
            if new_score > current_score:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
            else:
                # Accept with probability exp((new - old)/T)
                delta_E = current_score - new_score
                if np.random.rand() < np.exp(-delta_E / T):
                    current_score = new_score
                else:
                    current_points[i] = old_points[i]
            
            T *= cooling
            step_size *= 0.999 # Slowly decrease step size
            
    return best_points
