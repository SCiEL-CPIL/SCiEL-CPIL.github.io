import numpy as np
from scipy.optimize import minimize

N = 50

def construct():
    def soft_min(x, beta):
        # Stable soft-min
        m = np.max(x)
        return m - (1.0 / beta) * np.log(np.sum(np.exp(beta * (x - m))))
    
    def clearances_and_grads(params, beta, gamma_wall=50.0):
        centers = params.reshape(N, 3)
        
        # 1. Pairwise clearances
        # diff[i, j, k] = centers[i, k] - centers[j, k]
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Mask diagonal
        iu = np.triu_indices(N, k=1)
        pair_dists = dists[iu]
        pair_clears = pair_dists / 2.0
        
        # Gradients for pairwise clearances
        # grad_pair[i, j] = 0.5 * (c_i - c_j) / dist_ij
        # We need to accumulate into grad_centers
        grad_centers = np.zeros_like(centers)
        
        # Vectorized gradient for pairs
        # dists[i, j] is the distance.
        # We only care about upper triangle.
        # Let's compute the full gradient matrix and then sum.
        
        # To avoid division by zero, add small epsilon
        safe_dists = np.where(dists > 1e-8, dists, 1.0)
        
        # Unit vectors
        # u[i, j, k] = (c_i - c_j) / dist
        unit_vecs = diff / safe_dists[:, :, np.newaxis]
        
        # Gradient contribution for pair (i, j) to center i is 0.5 * u[i, j]
        # Gradient contribution for pair (i, j) to center j is -0.5 * u[i, j]
        
        # We need weights w_pair for each pair.
        # The soft-min is over ALL clearances (pairs + walls).
        # So we first need the wall clearances to compute the total weights.
        
        # 2. Wall clearances
        abs_c = np.abs(centers)
        max_abs = np.max(abs_c, axis=1, keepdims=True)
        
        # Soft max of |c|
        exp_terms_wall = np.exp(gamma_wall * (abs_c - max_abs))
        sum_exp_wall = np.sum(exp_terms_wall, axis=1, keepdims=True)
        soft_max_wall = (max_abs + (1.0 / gamma_wall) * np.log(sum_exp_wall)).flatten()
        wall_clears = 0.5 - soft_max_wall
        
        # Combine clearances
        n_pairs = len(pair_clears)
        all_clears = np.concatenate([pair_clears, wall_clears])
        
        # Compute weights for soft-min
        # w_k = exp(-beta * C_k) / sum(exp(-beta * C_j))
        # Stable computation
        m_clears = np.max(all_clears)
        exp_terms = np.exp(-beta * (all_clears - m_clears))
        sum_exp = np.sum(exp_terms)
        weights = exp_terms / sum_exp
        
        # Objective value: -soft_min
        # soft_min = -1/beta * log(sum(exp(-beta C)))
        # = -1/beta * log(sum(exp(-beta (C - m)) * exp(-beta m)))
        # = -1/beta * (log(sum(exp(-beta (C - m)))) - beta m)
        # = m - 1/beta * log(sum(exp(-beta (C - m))))
        # Wait, soft_min(x) = -1/beta log sum exp(-beta x).
        # Let's re-verify.
        # If x_i are all equal to x, sum = n exp(-beta x). log = log n - beta x.
        # -1/beta (log n - beta x) = x - log n / beta.
        # As beta -> inf, log n / beta -> 0. Result x. Correct.
        
        soft_min_val = -m_clears - (1.0 / beta) * np.log(sum_exp)
        # Wait.
        # log(sum exp(-beta (C - m))) = log(sum exp(-beta C + beta m)) = log(exp(beta m) sum exp(-beta C)) = beta m + log(sum exp(-beta C)).
        # So -1/beta log(sum exp(-beta C)) = -1/beta (log(sum exp(-beta (C-m))) - beta m) = -1/beta log(sum...) + m.
        # So soft_min = m - 1/beta log(sum exp(-beta (C-m))).
        
        val = -soft_min_val
        
        # Gradient
        # grad = - sum w_k grad C_k
        # Pairwise part
        w_pairs = weights[:n_pairs]
        w_walls = weights[n_pairs:]
        
        # We need to map w_pairs back to the pair indices
        # iu[0] and iu[1] give indices
        i_idx, j_idx = iu
        
        # For each pair (i, j), weight w_p
        # grad_centers[i] += w_p * 0.5 * unit_vecs[i, j]
        # grad_centers[j] -= w_p * 0.5 * unit_vecs[i, j]
        
        # Vectorized update
        # contrib[i] = sum_{j > i} w_{ij} * 0.5 * u_{ij} - sum_{j < i} w_{ji} * 0.5 * u_{ji}
        # Note u_{ji} = -u_{ij}. So -w_{ji} * 0.5 * (-u_{ij}) = w_{ji} * 0.5 * u_{ij}.
        # So grad_centers[i] = 0.5 * sum_{j != i} w_{ij} * u_{ij}
        
        # Create a matrix W of weights
        W = np.zeros((N, N))
        W[i_idx, j_idx] = w_pairs
        # Symmetric? No, w_{ij} is the weight for the clearance C_{ij}.
        # The clearance C_{ij} is the same for (i,j) and (j,i).
        # The weight is the same.
        # So W should be symmetric?
        # The soft-min is over the set of clearances. The set contains C_{ij} once.
        # So the weight w_{ij} applies to the gradient of C_{ij}.
        # grad C_{ij} w.r.t c_i is 0.5 u_{ij}.
        # grad C_{ij} w.r.t c_j is -0.5 u_{ij}.
        
        # So:
        # grad_centers[i] += sum_{j != i} w_{ij} * 0.5 * u_{ij}
        
        # We can compute this by multiplying the weight matrix by the unit vector matrix?
        # grad_centers[i, k] = 0.5 * sum_j W[i, j] * unit_vecs[i, j, k]
        
        # We need W to be symmetric if we want to use matrix multiplication easily?
        # W[i, j] = w_{ij}.
        # The term for center i involves sum over j of w_{ij} u_{ij}.
        # Let's construct W such that W[i, j] = w_{ij}.
        # Then grad_centers = 0.5 * (W @ unit_vecs) ?
        # (W @ unit_vecs)[i, k] = sum_j W[i, j] unit_vecs[i, j, k].
        # Yes.
        
        # But we only have weights for i < j.
        # w_{ij} = w_{ji}.
        # So W[i, j] = w_{ij} for i<j, and W[j, i] = w_{ij} for i<j.
        # So W is symmetric.
        
        W_full = np.zeros((N, N))
        W_full[i_idx, j_idx] = w_pairs
        W_full[j_idx, i_idx] = w_pairs
        
        # unit_vecs[i, j, k] is (c_i - c_j) / dist.
        # grad_centers[i] = 0.5 * sum_j W[i, j] * unit_vecs[i, j]
        
        grad_pairs = 0.5 * np.einsum('ij,ijk->ik', W_full, unit_vecs)
        
        # Wall part
        # grad_centers[k] -= w_wall_k * grad(soft_max_wall_k)
        # grad(soft_max_wall_k) = sign(c_k) * softmax(|c_k|)
        
        # softmax(|c_k|) is the vector of probabilities p_k
        p_walls = exp_terms_wall / sum_exp_wall
        signs = np.sign(centers)
        grad_walls = signs * p_walls
        
        # w_walls is shape (N,)
        # grad_centers[i] -= w_walls[i] * grad_walls[i]
        grad_centers -= (w_walls[:, np.newaxis] * grad_walls)
        
        # Total gradient
        grad_total = grad_pairs - grad_walls # Wait, objective is -soft_min.
        # d/dx (-soft_min) = - d/dx (soft_min) = - sum w_k grad C_k.
        # grad_pairs computed above is sum w_k grad C_k (positive).
        # So we subtract it.
        
        # Let's re-verify signs.
        # f = -m_beta.
        # grad f = - sum w_k grad C_k.
        # grad_pairs = sum w_k grad C_k_pair.
        # grad_walls_term = sum w_k grad C_k_wall.
        # So grad f = - (grad_pairs + grad_walls_term).
        
        # In my code:
        # grad_pairs = 0.5 * ... (This is sum w grad C_pair)
        # grad_walls_vec = signs * p (This is grad soft_max).
        # grad C_wall = - grad soft_max.
        # So sum w grad C_wall = - sum w * grad_walls_vec.
        # Total sum w grad C = grad_pairs - sum w * grad_walls_vec.
        # grad f = - (grad_pairs - sum w * grad_walls_vec) = -grad_pairs + sum w * grad_walls_vec.
        
        # Let's check the code above:
        # grad_centers = grad_pairs - (w_walls[:, np.newaxis] * grad_walls)
        # This is grad_pairs - sum w * grad_walls_vec.
        # This is sum w grad C.
        # We want - sum w grad C.
        # So we should return -grad_centers?
        
        # Let's re-evaluate.
        # val = -soft_min.
        # We want to minimize val.
        # So we return grad val.
        # grad val = - grad soft_min = - sum w grad C.
        
        # My calculated `grad_centers` variable currently holds `sum w grad C`.
        # So I should return `-grad_centers.flatten()`.
        
        return val, -grad_centers.flatten()

    def optimize(centers_init, max_iters=1000):
        betas = [10, 50, 100, 200, 500]
        current_centers = centers_init.copy()
        
        for beta in betas:
            res = minimize(
                lambda p: clearances_and_grads(p, beta)[0],
                current_centers.flatten(),
                jac=lambda p: clearances_and_grads(p, beta)[1],
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': max_iters // len(betas), 'gtol': 1e-8}
            )
            current_centers = res.x.reshape(N, 3)
            
        return current_centers

    def score(centers):
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        max_c = np.max(np.abs(centers), axis=1)
        min_wall = 0.5 - np.min(max_c)
        
        return min(min_pair, min_wall)

    # Initializations
    best_centers = None
    best_score = -np.inf
    
    # Try a few seeds
    for seed in range(20):
        np.random.seed(seed)
        
        # Strategy 1: Random
        if seed % 2 == 0:
            c_init = np.random.uniform(-0.4, 0.4, (N, 3))
        else:
            # Strategy 2: Perturbed Lattice
            # 3x4x4 = 48 points. Add 2 random?
            # Or 5x5x2 = 50.
            # Let's use a 5x5x2 grid centered.
            x = np.linspace(-0.4, 0.4, 5)
            y = np.linspace(-0.4, 0.4, 5)
            z = np.linspace(-0.4, 0.4, 2)
            
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            pts = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
            
            # Shuffle to break symmetry
            idx = np.random.permutation(N)
            pts = pts[idx]
            
            # Add noise
            pts += np.random.normal(0, 0.05, (N, 3))
            c_init = np.clip(pts, -0.45, 0.45)
            
        c_opt = optimize(c_init, max_iters=200)
        s = score(c_opt)
        
        if s > best_score:
            best_score = s
            best_centers = c_opt

    return best_centers
