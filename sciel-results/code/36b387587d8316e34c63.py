"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_score(pts):
    """Exact packing score: min of pairwise distances/2 and container margins."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    radii = np.linalg.norm(pts, axis=1)
    min_margin = np.min(1.0 - radii)
    return min(min_pair, min_margin)


def _soft_min_obj_and_grad(pts, beta=50.0):
    """
    Soft-min objective and analytic gradient.
    Objective = -soft_min(constraints) where constraints = [pairwise_dists/2, container_margins].
    soft_min(c) = m - (1/beta)*log(sum(exp(-beta*(c_i - m)))) where m = max(c).
    We minimize -soft_min = -m + (1/beta)*log(sum(exp(-beta*(c_i - m)))).
    """
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0  # (N, N)
    
    # Container margins
    radii = np.linalg.norm(pts, axis=1)
    margins = 1.0 - radii  # (N,)
    
    # Flatten all constraint values
    # Exclude diagonal from pair_vals
    mask = ~np.eye(N, dtype=bool)
    pair_flat = pair_vals[mask]  # (N*(N-1),)
    all_vals = np.concatenate([pair_flat, margins])  # (N*(N-1)+N,)
    
    # Soft-min computation
    max_val = np.max(all_vals)
    shifted = all_vals - max_val  # <= 0
    exp_terms = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_terms)
    log_sum = np.log(sum_exp)
    obj = -max_val + (1.0 / beta) * log_sum
    
    # Gradient
    # d/dx_j of -soft_min = (1/beta) * d/dx_j log(sum exp(-beta*(c_i - m)))
    # = (1/beta) * sum_i [exp(-beta*(c_i - m)) / sum_exp * (-beta) * dc_i/dx_j]
    # = -sum_i [exp(-beta*(c_i - m)) / sum_exp * dc_i/dx_j]
    # = -sum_i w_i * dc_i/dx_j  where w_i = exp(-beta*(c_i - m)) / sum_exp
    
    weights = exp_terms / sum_exp  # (N*(N-1)+N,)
    
    # Split weights back
    n_pairs = N * (N-1)
    pair_w = weights[:n_pairs].reshape(N, N)  # (N, N)
    margin_w = weights[n_pairs:]  # (N,)
    
    # Gradient of pairwise constraint c_ij = |p_i - p_j| / 2 with respect to p_i:
    # dc_ij/dp_i = (p_i - p_j) / (2 * |p_i - p_j|) = (p_i - p_j) / (4 * c_ij)
    # dc_ij/dp_j = (p_j - p_i) / (4 * c_ij)
    # But c_ij = c_ji, so we accumulate contributions.
    
    grad = np.zeros_like(pts)  # (N, 2)
    
    # For each pair (i,j), i != j:
    # contribution to grad[i] from c_ij: -w_ij * (p_i - p_j) / (2 * dist_ij)
    # contribution to grad[j] from c_ij: -w_ij * (p_j - p_i) / (2 * dist_ij)
    # Note: w_ij = w_ji (symmetric), and the pair appears twice in the flattened array.
    # Actually pair_vals is (N,N) with diagonal inf, and we flattened with mask.
    # The mask selects all off-diagonal elements. So pair (i,j) and (j,i) both appear.
    # pair_w[i,j] is the weight for constraint c_ij = dist(i,j)/2.
    
    # Vectorized: for each i, grad[i] -= sum_j w_ij * (p_i - p_j) / (2*dist_ij)
    # And grad[i] += sum_j w_ji * (p_i - p_j) / (2*dist_ij)  [from the j->i direction]
    
    # Let's do it more carefully:
    # For constraint c_ij (i != j): dc_ij/dp_i = (p_i - p_j)/(2*|p_i-p_j|)
    # The total gradient contribution to p_i from all constraints:
    #   - sum_{j!=i} w_ij * (p_i - p_j)/(2*|p_i-p_j|)   [from c_ij]
    #   - sum_{j!=i} w_ji * (p_j - p_i)/(2*|p_j-p_i|)   [from c_ji, since dc_ji/dp_i = (p_j-p_i)/(2|p_j-p_i|)]
    # Note (p_j-p_i)/(2|p_j-p_i|) = -(p_i-p_j)/(2|p_i-p_j|)
    # So: -w_ji * (p_j-p_i)/(2|p_j-p_i|) = w_ji * (p_i-p_j)/(2|p_i-p_j|)
    # Total for p_i: sum_{j!=i} (w_ji - w_ij) * (p_i - p_j)/(2*|p_i-p_j|)
    # Since w_ij = w_ji (symmetric weights for same distance), this would be zero?
    # Wait, no. w_ij is the weight for the constraint value c_ij. Since c_ij = c_ji, w_ij = w_ji.
    # Hmm, but they are separate entries in the all_vals array.
    # Actually, since c_ij = c_ji exactly, w_ij = w_ji.
    # So the total gradient for p_i from pairwise terms:
    #   -sum_j w_ij * (p_i-p_j)/(2*dist_ij) - sum_j w_ji * (p_j-p_i)/(2*dist_ji)
    #   = -sum_j w_ij * (p_i-p_j)/(2*dist_ij) + sum_j w_ij * (p_i-p_j)/(2*dist_ij)
    #   = 0?? That can't be right.
    
    # Let me re-derive. The objective is f = -soft_min(c_1, ..., c_M).
    # f = -[m - (1/beta) log(sum_k exp(-beta(c_k - m)))]
    #   = -m + (1/beta) log(sum_k exp(-beta(c_k - m)))
    # df/dx_j = -dm/dx_j + (1/beta) * [sum_k exp(-beta(c_k-m)) * (-beta) dc_k/dx_j] / sum_k exp(-beta(c_k-m))
    #          = -dm/dx_j - sum_k w_k * dc_k/dx_j
    # where w_k = exp(-beta(c_k-m)) / sum_k exp(-beta(c_k-m))
    # And dm/dx_j = dc_m/dx_j where m is the index of the max (but with soft-min, the "max" contribution is handled).
    # Actually for the soft-min, the derivative of m = max(c_k) is dc_m/dx_j at the argmax,
    # but the soft version smooths this. The correct derivative of -soft_min is:
    # df/dx_j = -sum_k w_k * dc_k/dx_j  (the -dm/dx_j is absorbed because w_k already accounts for it)
    # Wait, let me be more careful.
    # soft_min(c) = m - (1/beta) log(sum_k exp(-beta(c_k - m)))
    # d(soft_min)/dx_j = dm/dx_j - (1/beta) * [sum_k exp(-beta(c_k-m)) * (-beta) dc_k/dx_j] / sum_k exp(-beta(c_k-m))
    #                  = dm/dx_j + sum_k w_k * dc_k/dx_j
    # where w_k = exp(-beta(c_k-m))/sum_k exp(-beta(c_k-m))
    # Note: dm/dx_j is the derivative of the max, which in the soft version is approximated.
    # Actually, for the exact soft-min formula, the derivative is:
    # d(soft_min)/dx_j = sum_k v_k * dc_k/dx_j  where v_k = w_k * (1 - delta_{k,m}) + ... 
    # This is getting complicated. Let me use a simpler formulation.
    
    # Alternative: use the fact that soft_min(c) = -1/beta * log(sum_k exp(-beta*c_k))
    # (without the max shift, but we add it back for stability)
    # d(soft_min)/dx_j = (1/beta) * sum_k exp(-beta*c_k) * (-beta) dc_k/dx_j / sum_k exp(-beta*c_k)
    #                  = -sum_k w_k * dc_k/dx_j
    # where w_k = exp(-beta*c_k) / sum_k exp(-beta*c_k)
    # So d(-soft_min)/dx_j = sum_k w_k * dc_k/dx_j
    
    # OK so with the shifted version (for numerical stability):
    # soft_min = m - (1/beta)*log(sum exp(-beta*(c_k - m)))
    # The weights are w_k = exp(-beta*(c_k - m)) / sum exp(-beta*(c_k - m))
    # And d(soft_min)/dx_j = sum_k v_k * dc_k/dx_j where v_k are the "soft" max weights
    # Actually: d(soft_min)/dx_j = dm/dx_j - (1/beta)*d/dx_j[log(sum exp(-beta(c_k-m)))]
    # = dm/dx_j - (1/beta) * [sum exp(-beta(c_k-m)) * (-beta)(dc_k/dx_j - dm/dx_j)] / sum exp(-beta(c_k-m))
    # = dm/dx_j + sum_k w_k * (dc_k/dx_j - dm/dx_j)
    # = dm/dx_j + sum_k w_k * dc_k/dx_j - dm/dx_j * sum_k w_k
    # = dm/dx_j + sum_k w_k * dc_k/dx_j - dm/dx_j
    # = sum_k w_k * dc_k/dx_j
    
    # Great! So d(soft_min)/dx_j = sum_k w_k * dc_k/dx_j
    # Therefore d(-soft_min)/dx_j = -sum_k w_k * dc_k/dx_j
    
    # Now compute:
    # For pairwise constraint c_ij = |p_i - p_j|/2:
    # dc_ij/dp_i = (p_i - p_j) / (2 * |p_i - p_j|)
    # dc_ij/dp_j = (p_j - p_i) / (2 * |p_i - p_j|)
    
    # For margin constraint c_i = 1 - |p_i|:
    # dc_i/dp_i = -p_i / |p_i|  (for |p_i| > 0)
    
    # grad[i] = -sum_{j!=i} w_ij * (p_i - p_j)/(2*|p_i-p_j|) 
    #           -sum_{j!=i} w_ji * (p_j - p_i)/(2*|p_j-p_i|)  [wait, this is wrong]
    
    # Let me re-index. The constraints are:
    # c_1, ..., c_{N(N-1)}: pairwise, c_{N(N-1)+1}, ..., c_{N(N-1)+N}: margins
    # For pairwise constraint at position (i,j) in the N x N matrix (i != j):
    # dc_{ij}/dp_i = (p_i - p_j)/(2*dist_ij)
    # dc_{ij}/dp_j = (p_j - p_i)/(2*dist_ij)
    
    # grad[p_i] = -sum over all constraints k that involve p_i of w_k * dc_k/dp_i
    # For p_i, the relevant pairwise constraints are:
    #   c_{ij} for all j != i: dc_{ij}/dp_i = (p_i - p_j)/(2*dist_ij)
    #   c_{ji} for all j != i: dc_{ji}/dp_i = (p_j - p_i)/(2*dist_ji) = -(p_i - p_j)/(2*dist_ij)
    # So contribution from pairwise: 
    #   -sum_{j!=i} [w_ij * (p_i-p_j)/(2*dist_ij) + w_ji * (-(p_i-p_j)/(2*dist_ij))]
    #   = -sum_{j!=i} (w_ij - w_ji) * (p_i-p_j)/(2*dist_ij)
    # Since c_ij = c_ji, we have w_ij = w_ji, so this is 0!
    
    # That seems wrong. The issue is that I'm double-counting. In the flattened array,
    # both (i,j) and (j,i) are separate constraints with the same value.
    # The gradient of c_{ij} with respect to p_i is (p_i-p_j)/(2*dist).
    # The gradient of c_{ji} with respect to p_i is (p_j-p_i)/(2*dist) = -(p_i-p_j)/(2*dist).
    # So total: -w_ij*(p_i-p_j)/(2*dist) - w_ji*(-(p_i-p_j)/(2*dist)) = -(w_ij - w_ji)*(p_i-p_j)/(2*dist) = 0.
    
    # This means the pairwise constraints as formulated (both directions) give zero net gradient!
    # This is a bug in my formulation. I should only include each pair once.
    
    # Let me redo: include only upper triangle pairs.
    # Redo the computation with only N*(N-1)/2 unique pairs.
    
    # Recompute with unique pairs only
    triu_idx = np.triu_indices(N, k=1)  # (N*(N-1)/2,)
    pair_dists = dists[triu_idx]  # unique pair distances
    pair_vals_u = pair_dists / 2.0
    
    all_vals_u = np.concatenate([pair_vals_u, margins])
    M = len(all_vals_u)
    
    max_val_u = np.max(all_vals_u)
    shifted_u = all_vals_u - max_val_u
    exp_u = np.exp(-beta * shifted_u)
    sum_exp_u = np.sum(exp_u)
    log_sum_u = np.log(sum_exp_u)
    obj_u = -max_val_u + (1.0 / beta) * log_sum_u
    
    weights_u = exp_u / sum_exp_u
    n_pairs_u = len(pair_vals_u)
    pair_w_u = weights_u[:n_pairs_u]
    margin_w_u = weights_u[n_pairs_u:]
    
    # Gradient
    grad_u = np.zeros_like(pts)
    
    # For each unique pair (i, j) with i < j:
    # c = dist(i,j)/2
    # dc/dp_i = (p_i - p_j) / (2*dist)
    # dc/dp_j = (p_j - p_i) / (2*dist)
    # Contribution to grad: -w * dc/dp_i for p_i, -w * dc/dp_j for p_j
    
    ii, jj = triu_idx
    # Vectorized computation
    # For each pair (i,j): direction = (p_i - p_j) / (2*dist_ij)
    # grad[i] -= w_ij * direction
    # grad[j] += w_ij * direction  (since dc/dp_j = -direction)
    
    pair_diff = pts[ii] - pts[jj]  # (n_pairs_u, 2)
    pair_dists_vec = pair_dists  # (n_pairs_u,)
    # Avoid division by zero
    safe_dists = np.where(pair_dists_vec > 1e-10, pair_dists_vec, 1.0)
    directions = pair_diff / (2.0 * safe_dists_vec[:, np.newaxis])  # (n_pairs_u, 2)
    
    # Accumulate gradients
    # grad[i] -= w_i * direction_i
    # grad[j] += w_i * direction_i
    np.add.at(grad_u, ii, -pair_w_u[:, np.newaxis] * directions)
    np.add.at(grad_u, jj, pair_w_u[:, np.newaxis] * directions)
    
    # Margin constraints: c_i = 1 - |p_i|
    # dc_i/dp_i = -p_i / |p_i|
    radii_vec = np.linalg.norm(pts, axis=1)
    safe_radii = np.where(radii_vec > 1e-10, radii_vec, 1.0)
    margin_dirs = -pts / safe_radii[:, np.newaxis]  # (N, 2)
    
    grad_u += -margin_w_u[:, np.newaxis] * margin_dirs
    
    return obj_u, grad_u


def _hex_lattice_init():
    """Generate initial hexagonal lattice points inside unit circle."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
        spacing *= 0.98


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # 1. Generate initial hexagonal lattice
    init_pts = _hex_lattice_init()
    
    best_pts = init_pts.copy()
    best_score = _compute_score(best_pts)
    
    # 2. Optimization with L-BFGS-B using analytic gradient
    # Pass 1: beta=50 (coarse)
    # Pass 2: beta=100 (fine)
    # Pass 3: beta=200 (polish)
    
    def _obj_grad_flat(x, beta):
        pts = x.reshape(-1, 2)
        obj, grad = _soft_min_obj_and_grad(pts, beta)
        return obj, grad.flatten()
    
    for beta, maxiter in [(50.0, 300), (100.0, 200), (200.0, 100)]:
        if time.time() > deadline - 10:
            break
        
        try:
            res = minimize(
                lambda x: _obj_grad_flat(x, beta)[0],
                lambda x: _obj_grad_flat(x, beta)[1],
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 50}
            )
            
            if time.time() < deadline:
                new_pts = res.x.reshape(-1, 2)
                new_score = _compute_score(new_pts)
                if new_score > best_score:
                    best_pts = new_pts
                    best_score = new_score
                    
        except Exception:
            pass
    
    # 3. Final polish: local optimization on tightest constraints
    if time.time() < deadline - 5:
        try:
            # Find the tightest constraints and do a focused optimization
            pts = best_pts
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=-1)
            np.fill_diagonal(dists, np.inf)
            min_pair = np.min(dists) / 2.0
            radii = np.linalg.norm(pts, axis=1)
            min_margin = np.min(1.0 - radii)
            
            # Try a small perturbation + re-optimize with high beta
            rng = np.random.RandomState(42)
            for _ in range(3):
                if time.time() > deadline - 5:
                    break
                perturbed = best_pts + rng.normal(0, 0.005, best_pts.shape)
                # Clip to stay in container
                r = np.linalg.norm(perturbed, axis=1)
                for i in range(N):
                    if r[i] > 0.999:
                        perturbed[i] *= 0.999 / r[i]
                
                try:
                    res_polish = minimize(
                        lambda x: _obj_grad_flat(x, 200.0)[0],
                        lambda x: _obj_grad_flat(x, 200.0)[1],
                        perturbed.flatten(),
                        method='L-BFGS-B',
                        options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-7, 'maxls': 30}
                    )
                    new_pts = res_polish.x.reshape(-1, 2)
                    new_score = _compute_score(new_pts)
                    if new_score > best_score:
                        best_pts = new_pts
                        best_score = new_score
                except Exception:
                    pass
        except Exception:
            pass
    
    return best_pts
# EVOLVE-BLOCK-END
