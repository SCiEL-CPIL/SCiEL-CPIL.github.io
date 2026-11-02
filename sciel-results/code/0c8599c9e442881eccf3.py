import numpy as np
import time
import sys

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s safety margin

    # 1. Initialization: Hexagonal lattice
    # Generate a hexagonal grid and select the N points closest to the origin
    # that fit within the unit circle (with some margin).
    
    # Estimate spacing
    # Area of circle = pi. N circles of area pi*r^2.
    # r ~ sqrt(pi/N) / 2 ? No, packing density ~ 0.9.
    # N * pi * r^2 ~ 0.9 * pi => r ~ sqrt(0.9/N).
    # For N=300, r ~ sqrt(0.003) ~ 0.055.
    # Spacing d = 2r ~ 0.11.
    
    points = []
    spacing = 0.1
    # Generate grid points
    # We need enough points to pick N from.
    # Range: -1.1 to 1.1
    # steps: 2.2 / 0.1 = 22.
    # Hex grid:
    # row j: y = j * spacing * sqrt(3)/2
    # col i: x = (i + 0.5*(j%2)) * spacing
    
    while len(points) < N * 1.5:
        k = int(1.2 / spacing) + 2
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r < 1.0 - spacing/2: # Ensure initial points are strictly inside
                    rows.append([x, y])
        points = np.array(rows)
        if len(points) >= N:
            # Sort by radius and take the first N
            radii = np.hypot(points[:, 0], points[:, 1])
            indices = np.argsort(radii)[:N]
            c = points[indices].copy()
            break
        spacing *= 0.95
        
    if len(c) < N:
        # Fallback: random
        c = np.random.rand(N, 2) * 2 - 1
        # Project to disk
        r = np.hypot(c[:, 0], c[:, 1])
        c = c / r[:, np.newaxis] * 0.9

    # 2. Optimization Loop with Annealing
    # Objective: Minimize -soft_min(distances)
    # soft_min(V) = -1/beta * log(sum(exp(-beta * V)))
    # We want to maximize soft_min(V).
    # So we minimize f = 1/beta * log(sum(exp(-beta * V)))
    
    # V contains:
    # - Pairwise distances d_ij
    # - 2 * (1 - |c_i|)  (scaled so that r = min(V)/2)
    
    def objective(x, beta):
        # x is shape (N, 2)
        # Compute pairwise distances
        # c = x.reshape(N, 2)
        c = x
        # Pairwise distances
        # diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        # dist = np.linalg.norm(diff, axis=2)
        # This creates a 300x300 matrix. 90k elements. Fast.
        
        # To save memory/time, we can just compute the upper triangle
        # But vectorized norm is fast enough.
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        
        # Boundary terms: 2 * (1 - norm(c))
        norms = np.linalg.norm(c, axis=1)
        boundary = 2.0 * (1.0 - norms)
        
        # Combine into one array
        # We only need unique pairs for dist to avoid doubling the weight, 
        # but the log-sum-exp is invariant to scaling of the sum if we are consistent?
        # No, sum(exp(-beta*v)) depends on the number of terms.
        # If we include both d_ij and d_ji, the term is 2*exp(-beta*d_ij).
        # log(2*exp(...)) = log(2) - beta*d_ij.
        # The constant log(2) shifts the objective but doesn't change the gradient wrt x?
        # d/dx log(sum exp) = -beta * sum(exp * dv/dx) / sum(exp).
        # The constant log(2) in the numerator sum (if factored out) cancels?
        # Actually, it's easier to just use the upper triangle to be clean.
        
        iu = np.triu_indices(N, k=1)
        pair_dists = dist[iu]
        
        # V = [pair_dists, boundary]
        # Note: boundary is already scaled by 2.
        # So min(V) corresponds to 2*r.
        
        V = np.concatenate([pair_dists, boundary])
        
        # Soft min
        # To avoid overflow/underflow in exp, subtract max?
        # We are computing log(sum(exp(-beta * V))).
        # Let m = min(V). Then -beta * V = -beta * (V - m) - beta * m.
        # exp(-beta * V) = exp(-beta * (V - m)) * exp(-beta * m).
        # sum = exp(-beta * m) * sum(exp(-beta * (V - m))).
        # log(sum) = -beta * m + log(sum(exp(-beta * (V - m)))).
        # This is numerically stable.
        
        m = np.min(V)
        exp_terms = np.exp(-beta * (V - m))
        log_sum = np.log(np.sum(exp_terms))
        
        soft_min_val = m + log_sum / beta
        # We want to MAXIMIZE soft_min_val.
        # So we return -soft_min_val for minimization.
        return -soft_min_val

    def gradient(x, beta):
        # Analytic gradient of the objective
        c = x
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        # Set diagonal to 1 to avoid div by zero in gradient calc if needed, 
        # but we don't use diagonal.
        
        iu = np.triu_indices(N, k=1)
        pair_dists = dist[iu]
        
        norms = np.linalg.norm(c, axis=1)
        boundary = 2.0 * (1.0 - norms)
        
        V = np.concatenate([pair_dists, boundary])
        
        m = np.min(V)
        exp_terms = np.exp(-beta * (V - m))
        sum_exp = np.sum(exp_terms)
        
        # We are minimizing f = 1/beta * log(sum_exp) - m
        # Wait, the objective function defined above returns -soft_min.
        # soft_min = m + (1/beta) log(sum_exp)
        # f = -m - (1/beta) log(sum_exp)
        
        # Gradient of f wrt x:
        # df/dx = -dm/dx - (1/beta) * (1/sum_exp) * d(sum_exp)/dx
        # d(sum_exp)/dx = sum( exp(-beta(V_i-m)) * (-beta) * dV_i/dx )
        # dV_i/dx for pair dist d_ij = |c_i - c_j|:
        # d(d_ij)/dc_i = (c_i - c_j) / d_ij
        # d(d_ij)/dc_j = (c_j - c_i) / d_ij
        # dV_i/dx for boundary 2(1-|c_i|):
        # d/d c_i (2 - 2|c_i|) = -2 * c_i / |c_i|
        
        # Let's compute the weights w_i = exp(-beta(V_i-m)) / sum_exp
        w = exp_terms / sum_exp
        
        # Gradient of -m:
        # m = min(V). The gradient of min is the gradient of the argmin element.
        # If there are multiple minima, it's a convex combination, but usually one dominates.
        # In the soft-min context, the term -m is actually not differentiable at ties.
        # However, the term (1/beta) log(sum_exp) is smooth.
        # As beta -> inf, the soft-min approaches the min.
        # The gradient of the soft-min is:
        # d(soft_min)/dx = sum_i w_i * dV_i/dx
        # where w_i = exp(-beta(V_i-m)) / sum_exp.
        # Note: this is the gradient of the soft-min value.
        # Since we are minimizing -soft_min, the gradient is -sum_i w_i * dV_i/dx.
        
        grad = np.zeros_like(c)
        
        # 1. Pairwise terms
        # V_pair = pair_dists
        # w_pair = w[:N_pairs]
        # dV_pair/dx contributes to c_i and c_j
        
        n_pairs = len(pair_dists)
        w_pair = w[:n_pairs]
        
        # For each pair (i, j), contribution to grad is:
        # - w_pair_k * ( (c_i - c_j)/d_ij ) for c_i
        # - w_pair_k * ( (c_j - c_i)/d_ij ) for c_j
        
        # We can use np.add.at or scatter.
        # diff is (N, N, 2).
        # dist is (N, N).
        # We need (c_i - c_j) / dist_ij.
        
        # Avoid division by zero
        dist_safe = np.where(dist > 1e-10, dist, 1.0)
        unit_vecs = diff / dist_safe[:, :, np.newaxis]
        
        # We only care about upper triangle pairs.
        # Let's extract the unit vectors for the pairs.
        # u_ij = unit_vecs[i, j] = (c_i - c_j) / d_ij
        
        # The term in the sum for pair k (corresponding to i, j) is:
        # w_k * u_ij added to grad[i]
        # -w_k * u_ij added to grad[j]  (since u_ji = -u_ij)
        
        # Vectorized approach:
        # Create a matrix W of weights for all pairs, 0 for non-pairs.
        # Actually, just iterate or use fancy indexing.
        # N=300, pairs=45k. Looping in Python is slow.
        
        # Let's use the full matrix approach.
        # W_matrix is zero everywhere except upper triangle where it is w_pair.
        W_mat = np.zeros((N, N))
        W_mat[iu] = w_pair
        
        # grad_pair = - ( W_mat * unit_vecs ).sum(axis=0) ?
        # Let's check signs.
        # dV_ij/dc_i = u_ij.
        # dV_ij/dc_j = -u_ij.
        # Sum over j for fixed i: sum_j w_ij u_ij.
        # Sum over i for fixed j: sum_i w_ij (-u_ij) = - sum_i w_ij u_ij.
        # Note w_ij is non-zero only if i<j.
        
        # So grad[i] gets - sum_j w_ij u_ij.
        # grad[j] gets - sum_i w_ij u_ij (where i<j).
        
        # Let S = W_mat * unit_vecs.
        # S[i, j] = w_ij * u_ij.
        # sum_j S[i, j] is the contribution to c_i from pairs where i is the first index.
        # But for pairs where i is the second index (i > j), the term is w_ji * u_ji = w_ji * (-u_ij).
        # So the total gradient contribution to c_i is:
        # - ( sum_{j>i} w_ij u_ij + sum_{j<i} w_ji (-u_ij) )
        # = - ( sum_{j>i} w_ij u_ij - sum_{j<i} w_ji u_ij )
        
        # This is getting complicated to vectorize simply with just one matrix multiplication.
        # Let's use the property that the gradient is symmetric.
        
        # Alternative:
        # grad = - ( W_mat * unit_vecs ).sum(axis=1) - ( W_mat.T * unit_vecs ).sum(axis=1) ?
        # Let's test.
        # Term for c_i:
        # From W_mat * unit_vecs: sum_j W[i,j] u[i,j]. This includes j>i terms (w_ij u_ij).
        # From W_mat.T * unit_vecs: sum_j W.T[i,j] u[i,j] = sum_j W[j,i] u[i,j].
        # For j < i, W[j,i] = w_ji. u[i,j] = -u[j,i].
        # So term is w_ji (-u[j,i]).
        # The formula for grad[i] should be - ( sum_{j>i} w_ij u_ij + sum_{j<i} w_ji (-u_ij) )?
        # Wait, dV_ji/dc_i = u_ji = -u_ij.
        # So contribution is w_ji * (-u_ij).
        # So total deriv of V wrt c_i is sum_{j>i} w_ij u_ij + sum_{j<i} w_ji (-u_ij).
        # We want to minimize -V_soft, so grad = - (that).
        
        # Let A = W_mat * unit_vecs. A[i,j] = w_ij u_ij.
        # Sum over j: A_sum[i] = sum_j w_ij u_ij.
        # Let B = W_mat.T * unit_vecs. B[i,j] = w_ji u_ij.
        # Sum over j: B_sum[i] = sum_j w_ji u_ij.
        
        # We want: - ( sum_{j>i} w_ij u_ij + sum_{j<i} w_ji (-u_ij) )
        # Note: In A_sum[i], the term for j<i is 0 because W_mat is upper triangular (w_ji=0 if j<i? No, W_mat[i,j] is 0 if i>j).
        # So A_sum[i] = sum_{j>i} w_ij u_ij.
        # In B_sum[i], B[i,j] = W_mat[j,i] u_ij.
        # W_mat[j,i] is non-zero if j<i.
        # So B_sum[i] = sum_{j<i} w_ji u_ij.
        
        # So the derivative of V wrt c_i is A_sum[i] - B_sum[i].
        # Gradient of objective (-V_soft) is -(A_sum[i] - B_sum[i]) = B_sum[i] - A_sum[i].
        
        grad_pair = B_sum - A_sum
        
        # 2. Boundary terms
        # V_bound_i = 2(1 - |c_i|)
        # dV_bound_i/dc_i = -2 * c_i / |c_i|
        # w_bound_i = w[n_pairs + i]
        # Contribution to grad[i] = - w_bound_i * (-2 * c_i / |c_i|) = 2 * w_bound_i * c_i / |c_i|
        
        w_bound = w[n_pairs:]
        norms_safe = np.where(norms > 1e-10, norms, 1.0)
        grad_bound = 2.0 * w_bound[:, np.newaxis] * c / norms_safe[:, np.newaxis]
        
        total_grad = grad_pair + grad_bound
        
        return total_grad

    # Optimization stages
    betas = [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    
    best_c = c
    best_score = -1e10
    
    for beta in betas:
        if time.time() > deadline - 20:
            break
            
        try:
            res = minimize(
                objective, 
                best_c, 
                jac=gradient, 
                method='L-BFGS-B', 
                args=(beta,),
                options={'maxiter': 100, 'gtol': 1e-5, 'ftol': 1e-8}
            )
            if res.success or res.fun < best_score:
                best_c = res.x
                best_score = res.fun
        except Exception as e:
            print(f"Optimization failed at beta={beta}: {e}")
            pass

    # Final calculation of the actual score
    c = best_c
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    min_dist = np.min(dist[iu])
    
    norms = np.linalg.norm(c, axis=1)
    min_boundary = np.min(1.0 - norms)
    
    r = min(min_dist / 2.0, min_boundary)
    
    return best_c

from scipy.optimize import minimize
