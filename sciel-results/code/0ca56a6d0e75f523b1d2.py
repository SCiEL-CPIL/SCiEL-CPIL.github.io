import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    max_time = 115.0  # Safety margin for 120s limit
    
    # 1. Initialization: Hexagonal Lattice
    # This provides a good, dense starting point to avoid chaotic early energy
    s_init = np.sqrt(2 * np.pi / (N * np.sqrt(3)))
    points = []
    R = 1.2
    k = int(R / s_init) + 2
    for j in range(-k, k + 1):
        y = j * s_init * np.sqrt(3) / 2
        x_offset = (j % 2) * s_init / 2
        for i in range(-k, k + 1):
            x = i * s_init + x_offset
            if np.hypot(x, y) <= 1.0:
                points.append([x, y])
    
    points = np.array(points)
    if len(points) < N:
        points = np.random.randn(N, 2) * 0.5
    else:
        radii = np.hypot(points[:, 0], points[:, 1])
        indices = np.argsort(radii)
        points = points[indices[:N]]
        
    x = points.astype(np.float64).copy()
    
    # Precompute indices for pair interactions
    i_idx, j_idx = np.triu_indices(N, k=1)
    num_pairs = len(i_idx)
    
    def calc_score(x):
        """
        Compute the exact score: min(min_pair_dist/2, min_bound_dist)
        """
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        min_bound = np.min(1.0 - np.linalg.norm(x, axis=1))
        return min(min_pair, min_bound)
    
    # Initial score
    best_x = x.copy()
    best_r = calc_score(best_x)
    
    # Simulated Annealing Parameters
    # We minimize Energy = sum of penalties for overlaps and boundary violations
    # Penalty is quadratic for smoothness
    # E = sum_{i<j} max(0, 2r - d_ij)^2 + sum_i max(0, r - (1 - |c_i|))^2
    # Wait, the objective is to MAXIMIZE r.
    # It is easier to think of it as: Find configuration that maximizes r.
    # Equivalent to: Minimize "violations" at a target r.
    # But r is the variable we want to find.
    
    # Alternative: Directly maximize the minimum distance.
    # Let's use a potential energy landscape that pushes points apart.
    # E = sum_{i<j} (1/d_ij)^k + sum_i (1/(1-|c_i|))^k
    # Minimizing this pushes points to equilibrium where distances are maximized.
    # This is a standard "repulsive potential" approach.
    
    # Let's use a simpler energy function that directly relates to the score.
    # Score r is the bottleneck.
    # If we define E(r) = sum_{i<j} max(0, 2r - d_ij)^2 + sum_i max(0, r - (1 - |c_i|))^2
    # Then the maximum r such that E(r) = 0 is the optimal packing radius.
    # We can perform a binary search on r, or simply anneal the system while slowly increasing a "target radius" R_target.
    
    # Strategy:
    # 1. Start with a small R_target.
    # 2. Anneal the system to satisfy constraints for R_target.
    # 3. Slowly increase R_target.
    # 4. The final R_target achieved is the score.
    
    # However, calculating the exact "max feasible R" during annealing is tricky.
    # Instead, let's use the "Repulsive Potential" method which is robust.
    # Minimize U(x) = sum_{i<j} exp(-d_ij / sigma) + sum_i exp(- (1-|c_i|) / sigma)
    # As sigma -> 0, this forces d_ij to be large and |c_i| to be small.
    # But we want to maximize the MINIMUM distance.
    
    # Let's stick to the "Target Radius" approach as it is more direct for the score.
    # We will try to find the largest R such that we can place points.
    # We will use a "soft" constraint solver.
    
    # Energy Function:
    # We want to push points so that d_ij >= 2R and 1-|c_i| >= R.
    # E(R, x) = sum_{i<j} max(0, 2R - d_ij)^2 + sum_i max(0, R - (1 - |c_i|))^2
    # If we can make E(R, x) close to 0, then R is a feasible radius.
    
    # Algorithm:
    # 1. Initialize x.
    # 2. Start R = 0.01.
    # 3. Loop:
    #    a. Perform Simulated Annealing steps to minimize E(R, x).
    #    b. If E < threshold, increase R slightly.
    #    c. If time runs out, stop.
    
    # To make this fast, we need efficient gradient of E.
    # dE/dx_i = sum_{j != i} 2 * max(0, 2R - d_ij) * (-(x_i - x_j)/d_ij) * 2  (chain rule for dist)
    # Actually:
    # Let g_ij = max(0, 2R - d_ij).
    # term = g_ij^2.
    # d(term)/d(x_i) = 2 * g_ij * d(g_ij)/d(x_i)
    # d(g_ij)/d(x_i) = -d(d_ij)/d(x_i) = -(x_i - x_j)/d_ij
    # So dE/dx_i (pair part) = sum_{j != i} -2 * g_ij * (x_i - x_j)/d_ij
    
    # Boundary:
    # h_i = max(0, R - (1 - |c_i|)).
    # term = h_i^2.
    # d(term)/d(c_i) = 2 * h_i * d(h_i)/d(c_i)
    # d(h_i)/d(c_i) = -d(1-|c_i|)/d(c_i) = -(-c_i/|c_i|) = c_i/|c_i|
    # So dE/dx_i (bound part) = 2 * h_i * (x_i / |x_i|)
    
    def compute_energy_grad(x, R):
        """
        Computes gradient of E(R, x).
        Returns gradient array of shape (N, 2) and the energy value.
        """
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        dists = np.sqrt(dists_sq)
        
        # Pair interactions
        # We only care about pairs where d_ij < 2R
        # To be efficient, we can compute all, but mask the non-violating ones.
        # Given N=300, N^2 is 90k, which is fine.
        
        # g_ij = max(0, 2R - d_ij)
        # We use the upper triangle to avoid double counting in the sum, but the gradient
        # accumulation needs to handle both i and j.
        # It's easier to compute the full matrix gradient contribution and sum.
        
        # Let's use the vectorized approach for the full matrix, then sum.
        # But we only need the diagonal sum for the gradient of x.
        
        # g_mat[i, j] = max(0, 2R - dists[i, j])
        g_mat = np.maximum(0, 2 * R - dists)
        # Avoid division by zero
        safe_dists = np.where(dists > 1e-8, dists, 1e-8)
        
        # grad_pair[i, j] = -2 * g_mat[i, j] * (x_i - x_j) / dists[i, j]
        # Note: (x_i - x_j) is diff[i, j]
        # So grad_pair[i, j] = -2 * g_mat[i, j] / safe_dists[i, j] * diff[i, j]
        
        # The gradient of E with respect to x_i is the sum over j of grad_pair[i, j]
        # (since the term involving x_i in the sum over pairs (i,j) is the one where i is the first index,
        # but wait, the pair (j,i) is the same pair.
        # In the sum E = sum_{i<j} ..., the derivative w.r.t x_i gets contributions from all j != i.
        # So yes, summing the row i of the "force matrix" gives the force on i.
        
        # force_mat[i, j] is the force on i due to j?
        # d/dx_i (g_ij^2) = -2 g_ij (x_i - x_j) / d_ij.
        # This is the force on i.
        # So grad_x_i = sum_{j != i} force_mat[i, j]
        
        force_mat = -2.0 * g_mat / safe_dists * diff
        # Set diagonal to 0 (it's already 0 because g_mat[i,i]=0 and diff[i,i]=0, but safe_dists handles div0)
        np.fill_diagonal(force_mat, 0)
        
        grad_pairs = np.sum(force_mat, axis=1)
        
        # Boundary interactions
        norms = np.linalg.norm(x, axis=1)
        safe_norms = np.where(norms > 1e-8, norms, 1e-8)
        
        # h_i = max(0, R - (1 - |c_i|))
        h_vec = np.maximum(0, R - (1.0 - norms))
        
        # grad_bound[i] = 2 * h_i * (x_i / |x_i|)
        grad_bounds = 2.0 * h_vec * (x / safe_norms[:, np.newaxis])
        
        total_grad = grad_pairs + grad_bounds
        
        # Calculate Energy for tracking
        # E = 0.5 * sum_{i,j} g_ij^2 (since sum_{i<j} counts each pair once, but my matrix sums over all i,j, 
        # so sum_{i,j} g_ij^2 = 2 * sum_{i<j} g_ij^2.
        # The energy defined in the objective was sum_{i<j} g_ij^2.
        # So E_pairs = 0.5 * np.sum(g_mat**2)
        E_pairs = 0.5 * np.sum(g_mat**2)
        E_bounds = np.sum(h_vec**2)
        E = E_pairs + E_bounds
        
        return total_grad, E

    # Annealing Loop
    T = 1.0  # Initial temperature
    T_min = 0.01
    cooling_rate = 0.995
    
    # We want to increase R as the system stabilizes.
    # Start with a small R.
    R_current = 0.02
    R_target_increment = 0.0005
    
    steps_per_R = 100  # How many SA steps to take for each R increment
    
    current_E = 1e10
    best_E_for_R = 1e10
    
    while time.time() - start_time < max_time:
        # Check if we should increase R
        # If the energy is low enough, the constraints are satisfied for R_current.
        # Threshold for "satisfied" depends on R.
        # Let's say if E < 1e-4 * R^2, it's good enough to push R up.
        threshold = 1e-5
        if current_E < threshold:
            # Try to increase R
            R_new = R_current + R_target_increment
            # Quick check: is R_new feasible?
            # We don't need to check explicitly, we just set R_current = R_new and let SA try to satisfy it.
            # If it can't, E will rise, and we might need to cool down or accept the lower R.
            R_current = R_new
            # Reset temperature slightly to allow rearrangement?
            # Or just continue.
            # Let's reset T to a small value to refine, or keep it.
            # Keeping it is fine.
            
        # Compute Gradient and Energy
        grad, current_E = compute_energy_grad(x, R_current)
        
        # If this is a good configuration for this R, update best
        # The score is R_current if E is small.
        # Actually, the true score is the max R such that E=0.
        # If E is small, R_current is a valid lower bound on the score.
        # We track the highest R_current for which we achieved low energy.
        
        if current_E < threshold:
            # This R_current is feasible (approximately)
            # Update best_r
            # The actual score might be slightly higher, but R_current is a safe bet.
            # To be precise, we could do a binary search at the end, but let's just track R_current.
            if R_current > best_r:
                best_r = R_current
                best_x = x.copy()
        
        # Simulated Annealing Step
        # Propose new position: x_new = x + grad_step + noise
        # Step size should be related to T and gradient magnitude.
        g_norm = np.linalg.norm(grad)
        if g_norm > 1e-8:
            grad_normed = grad / g_norm
        else:
            grad_normed = np.zeros_like(grad)
            
        # Step size: balance between following gradient and thermal noise
        # If E is high, we need to move a lot.
        step_size = 0.01 * T
        
        noise = np.random.randn(N, 2) * step_size
        x_new = x - grad_normed * step_size * 2.0 + noise # Factor 2 for stability
        
        # Project to unit circle
        norms = np.linalg.norm(x_new, axis=1)
        out_mask = norms > 1.0
        if np.any(out_mask):
            x_new[out_mask] = x_new[out_mask] / norms[out_mask, np.newaxis]
        
        # Evaluate new energy
        grad_new, E_new = compute_energy_grad(x_new, R_current)
        
        # Metropolis Acceptance
        delta_E = E_new - current_E
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            x = x_new
            current_E = E_new
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            # If we are stuck at low T and low E, but R hasn't increased, 
            # it means we are at a local minimum for this R.
            # The logic above increases R if E < threshold.
            # If E is NOT < threshold, we are stuck.
            # We should reheat to escape local minima.
            T = 0.1
    
    # Final Polish
    # Use the best_x found.
    # We can do a few steps of gradient descent on the "exact" score to squeeze out a bit more.
    # But the SA with R increase should have found a good configuration.
    
    # One last check: calculate the exact score of best_x to ensure validity.
    # The score returned by the evaluator is min(min_pair/2, min_bound).
    # Our best_r is an approximation (the R we successfully annealed to).
    # The actual score of best_x might be slightly different.
    # We should return best_x, and the evaluator will calculate the true score.
    
    return best_x

if __name__ == "__main__":
    # Quick self-test
    pts = construct()
    print(f"Shape: {pts.shape}")
    # Calculate score
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    min_bound = np.min(1.0 - np.linalg.norm(pts, axis=1))
    score = min(min_pair, min_bound)
    print(f"Score: {score}")
