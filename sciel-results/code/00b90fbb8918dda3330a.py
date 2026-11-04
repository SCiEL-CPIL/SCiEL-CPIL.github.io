"""
Circle packing in unit square using Voronoi-based inradius optimization.
Each circle's radius is the inradius of its Voronoi cell.
Optimize centers via L-BFGS-B on smooth sum of inradii.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Voronoi, Delaunay
import time

N = 26
DEADLINE = 145.0  # Safety margin from 150s limit


def construct():
    start_time = time.time()

    def voronoi_inradii(x, y):
        """Compute inradii using Voronoi diagram.
        Returns (radii, neighbors) where neighbors[i] is list of (point_idx, constraint_type, value).
        """
        pts = np.column_stack([x, y])
        # Add boundary points for proper Voronoi clipping
        # We'll use a simpler approach: compute distance to all other points and to boundaries
        # This is equivalent to the inradius of the Voronoi cell
        
        radii = np.zeros(N)
        
        for i in range(N):
            # Distance to boundaries
            d_bound = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
            
            # Distance to other circles (half the distance)
            dx = x[i] - x
            dy = y[i] - y
            dist = np.sqrt(dx**2 + dy**2)
            dist[i] = np.inf
            d_neighbor = 0.5 * np.min(dist)
            
            radii[i] = min(d_bound, d_neighbor)
        
        return radii

    def voronoi_inradii_vectorized(x, y):
        """Vectorized computation of inradii."""
        r_bound = np.minimum(
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        )
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)
        r_neighbor = 0.5 * np.min(dist, axis=1)
        r = np.minimum(r_bound, r_neighbor)
        r = np.maximum(r, 1e-10)
        return r

    def compute_objective_and_grad(params):
        """Compute -sum(inradii) and its gradient.
        Uses subgradient of the min operation (pick the active constraint).
        """
        x = params[:N].copy()
        y = params[N:2*N].copy()
        
        # Compute radii and identify active constraints
        r_bound = np.minimum(
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        )
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2 + 1e-30)
        np.fill_diagonal(dist, np.inf)
        r_neighbor = 0.5 * np.min(dist, axis=1)
        
        r = np.minimum(r_bound, r_neighbor)
        r = np.maximum(r, 1e-10)
        
        obj_val = -np.sum(r)
        
        # Gradient: for each circle, the gradient depends on which constraint is active
        grad = np.zeros(2 * N)
        
        for i in range(N):
            # Determine if boundary or neighbor is active
            if r_bound[i] <= r_neighbor[i] + 1e-12:
                # Boundary constraint active
                # Find which boundary
                bounds = [x[i], 1.0 - x[i], y[i], 1.0 - y[i]]
                min_idx = np.argmin(bounds)
                if min_idx == 0:  # x is min -> dr/dx = +1
                    grad[i] = -1.0
                elif min_idx == 1:  # 1-x is min -> dr/dx = -1
                    grad[i] = 1.0
                elif min_idx == 2:  # y is min -> dr/dy = +1
                    grad[N + i] = -1.0
                else:  # 1-y is min -> dr/dy = -1
                    grad[N + i] = 1.0
            else:
                # Neighbor constraint active
                # Find the nearest neighbor
                dx_i = x[i] - x
                dy_i = y[i] - y
                dist_i = np.sqrt(dx_i**2 + dy_i**2 + 1e-30)
                dist_i[i] = np.inf
                j = np.argmin(dist_i)
                
                # r_i = 0.5 * dist(i,j)
                # dr_i/dx_i = 0.5 * (x_i - x_j) / dist(i,j)
                # dr_i/dx_j = 0.5 * (x_j - x_i) / dist(i,j)
                
                d = dist_i[j]
                if d > 1e-15:
                    gx = 0.5 * (x[i] - x[j]) / d
                    gy = 0.5 * (y[i] - y[j]) / d
                    grad[i] += -gx
                    grad[N + i] += -gy
                    grad[j] += gx
                    grad[N + j] += gy
        
        return obj_val, grad

    def compute_objective_and_grad_smooth(params, beta=100.0):
        """Smooth version using log-sum-exp for the min operation."""
        x = params[:N].copy()
        y = params[N:2*N].copy()
        
        # Build constraint matrix: for each i, constraints are
        # [x_i, 1-x_i, y_i, 1-y_i, 0.5*d(i,j) for all j != i]
        # r_i = min of these constraints
        
        # For efficiency, use the soft-min approach
        r_bound_arr = np.stack([x, 1.0 - x, y, 1.0 - y], axis=1)  # (N, 4)
        
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2 + 1e-30)
        np.fill_diagonal(dist, np.inf)
        half_dist = 0.5 * dist  # (N, N)
        
        # For each i, the constraints are: 4 boundary + (N-1) neighbor distances
        # Total: N+3 constraints per circle
        # Build full constraint matrix (N, N+3)
        # Columns 0-3: boundary constraints
        # Columns 4-4+N-1: neighbor constraints (skip self)
        
        # Actually, let's use a simpler formulation:
        # r_i = min(r_bound_i, min_j 0.5*d(i,j))
        # = min over all constraints
        
        # Soft-min: r_i ≈ -(1/beta) * log(sum_k exp(-beta * c_ik))
        
        # For each i, constraints:
        # c_i0 = x_i, c_i1 = 1-x_i, c_i2 = y_i, c_i3 = 1-y_i
        # c_i(4+j) = 0.5*d(i, j) for j != i
        
        # Let's compute this efficiently
        # neg_beta_c for boundaries: (N, 4)
        nb_bound = -beta * r_bound_arr  # (N, 4)
        
        # neg_beta_c for neighbors: (N, N) with diagonal set to -inf
        nb_neigh = -beta * half_dist
        nb_neigh = np.where(np.eye(N, dtype=bool), -np.inf, nb_neigh)
        
        # Combine: for each i, we have 4 + (N-1) = N+3 terms
        # But it's easier to just compute log-sum-exp over all N+3 terms
        # Let's stack them: (N, N+3)
        # Actually, let's just do it row by row for clarity, but vectorized
        
        # For each row i:
        # log_sum_i = log(exp(-beta*x_i) + exp(-beta*(1-x_i)) + exp(-beta*y_i) + exp(-beta*(1-y_i)) + sum_{j!=i} exp(-beta*0.5*d(i,j)))
        # r_i = -(1/beta) * (log_sum_i + beta * max_c) ... no
        # r_i = -(1/beta) * log(sum_k exp(-beta * c_ik))
        # = -(1/beta) * (max_neg + log(sum exp(neg_c - max_neg)))
        
        # max_neg for each row: max of all -beta*c_ik
        # For boundaries: max of nb_bound[i]
        # For neighbors: max of nb_neigh[i] (excluding -inf)
        
        max_bound = np.max(nb_bound, axis=1)  # (N,)
        max_neigh = np.max(nb_neigh, axis=1)  # (N,) - should be fine since we set diag to -inf
        
        # Overall max for each row
        max_all = np.maximum(max_bound, max_neigh)  # (N,)
        
        # Compute sum of exp(neg_c - max_all) for each row
        # Boundaries contribution: sum_k exp(nb_bound[i,k] - max_all[i])
        exp_bound = np.exp(nb_bound - max_all[:, None])  # (N, 4)
        sum_bound = np.sum(exp_bound, axis=1)  # (N,)
        
        # Neighbors contribution: sum_j exp(nb_neigh[i,j] - max_all[i])
        exp_neigh = np.exp(nb_neigh - max_all[:, None])  # (N, N)
        sum_neigh = np.sum(exp_neigh, axis=1)  # (N,)
        
        total_sum = sum_bound + sum_neigh  # (N,)
        
        # r_i = -(1/beta) * (max_all_i + log(total_sum_i))
        r = -(max_all + np.log(total_sum + 1e-30)) / beta  # (N,)
        r = np.maximum(r, 1e-10)
        
        obj_val = -np.sum(r)
        
        # Gradient computation
        # dr_i/dp = sum_k w_ik * dc_ik/dp
        # where w_ik = exp(-beta*c_ik) / sum_k' exp(-beta*c_ik')
        
        # Weights for boundaries: (N, 4)
        w_bound = exp_bound / (total_sum[:, None] + 1e-30)
        # Weights for neighbors: (N, N)
        w_neigh = exp_neigh / (total_sum[:, None] + 1e-30)
        
        grad = np.zeros(2 * N)
        
        # Boundary contributions to gradient
        # c_i0 = x_i: dc/dx_i = 1, dc/dy_i = 0
        # c_i1 = 1-x_i: dc/dx_i = -1, dc/dy_i = 0
        # c_i2 = y_i: dc/dx_i = 0, dc/dy_i = 1
        # c_i3 = 1-y_i: dc/dx_i = 0, dc/dy_i = -1
        
        # dr_i/dx_i from boundaries = w_bound[i,0]*1 + w_bound[i,1]*(-1) + w_bound[i,2]*0 + w_bound[i,3]*0
        # dr_i/dy_i from boundaries = w_bound[i,0]*0 + w_bound[i,1]*0 + w_bound[i,2]*1 + w_bound[i,3]*(-1)
        
        dr_dx_bound = w_bound[:, 0] - w_bound[:, 1]  # (N,)
        dr_dy_bound = w_bound[:, 2] - w_bound[:, 3]  # (N,)
        
        # Neighbor contributions
        # c_ij = 0.5 * dist(i,j) for j != i
        # dc_ij/dx_i = 0.5 * (x_i - x_j) / dist(i,j)
        # dc_ij/dx_j = 0.5 * (x_j - x_i) / dist(i,j)
        # dc_ij/dy_i = 0.5 * (y_i - y_j) / dist(i,j)
        # dc_ij/dy_j = 0.5 * (y_j - y_i) / dist(i,j)
        
        dist_safe = dist.copy()
        np.fill_diagonal(dist_safe, 1.0)  # avoid div by zero
        
        # For each pair (i,j), the contribution to dr_i/dx_i is w_neigh[i,j] * 0.5*dx[i,j]/dist[i,j]
        # And to dr_j/dx_i is w_neigh[j,i] * 0.5*dx[j,i]/dist[j,i] = w_neigh[j,i] * (-0.5*dx[i,j])/dist[i,j]
        
        # Let's compute the full gradient contribution from neighbors
        # d_c_dx_i[i,j] = 0.5 * dx[i,j] / dist_safe[i,j] for j != i, 0 for j == i
        d_c_dx_i = 0.5 * dx / dist_safe
        np.fill_diagonal(d_c_dx_i, 0)
        d_c_dy_i = 0.5 * dy / dist_safe
        np.fill_diagonal(d_c_dy_i, 0)
        d_c_dx_j = -0.5 * dx / dist_safe
        np.fill_diagonal(d_c_dx_j, 0)
        d_c_dy_j = -0.5 * dy / dist_safe
        np.fill_diagonal(d_c_dy_j, 0)
        
        # dr_i/dx_i from neighbors = sum_j w_neigh[i,j] * d_c_dx_i[i,j]
        dr_dx_neigh_row = np.sum(w_neigh * d_c_dx_i, axis=1)  # (N,)
        dr_dy_neigh_row = np.sum(w_neigh * d_c_dy_i, axis=1)  # (N,)
        
        # dr_i/dx_j from neighbors (for j != i) = w_neigh[i,j] * d_c_dx_j[i,j]
        # This contributes to the gradient w.r.t. x_j from the perspective of circle i
        # Total gradient w.r.t. x_k = sum_i dr_i/dx_k
        # = sum_i [from row i: if k==i, contribution from d_c_dx_i[i,k]; if k!=i, contribution from d_c_dx_j[i,k]]
        # = dr_dx_neigh_row[k] + sum_{i != k} w_neigh[i,k] * d_c_dx_j[i,k]
        
        # The second term: for fixed k, sum over i of w_neigh[i,k] * d_c_dx_j[i,k]
        # This is the k-th column of (w_neigh * d_c_dx_j)
        dr_dx_neigh_col = np.sum(w_neigh * d_c_dx_j, axis=0)  # (N,)
        dr_dy_neigh_col = np.sum(w_neigh * d_c_dy_j, axis=0)  # (N,)
        
        # Total gradient w.r.t. x:
        # grad_x[k] = -dr_k/dx_k (from objective = -sum(r))
        # dr_k/dx_k = dr_dx_bound[k] + dr_dx_neigh_row[k] + dr_dx_neigh_col[k]
        # Wait, dr_dx_neigh_col[k] already includes the contribution from all i to x_k
        # But dr_dx_neigh_row[k] is the contribution to dr_k/dx_k from row k
        # And dr_dx_neigh_col[k] is the contribution to dr_k/dx_k from other rows? No...
        
        # Let me re-derive:
        # sum(r) = sum_i r_i
        # d(sum(r))/dx_k = sum_i dr_i/dx_k
        # dr_i/dx_k = w_neigh[i,k] * dc_ik/dx_k (from neighbor constraint) + boundary terms if k==i
        # dc_ik/dx_k: if k==i, it's d_c_dx_i[i,i]=0 (diagonal). If k!=i, it's d_c_dx_j[i,k]
        # So dr_i/dx_k = w_neigh[i,k] * d_c_dx_j[i,k] for k != i, and 0 for k==i (from neighbor part)
        # Plus boundary: if k==i, dr_i/dx_k includes boundary gradient
        
        # So: d(sum(r))/dx_k = sum_{i != k} w_neigh[i,k] * d_c_dx_j[i,k] + (boundary gradient for circle k)
        # = dr_dx_neigh_col[k] + dr_dx_bound[k]
        
        # Wait, that's not right either. Let me be very careful.
        
        # r_i = sum_j w_ij * c_ij  (soft-min interpretation)
        # d(r_i)/dx_k = sum_j w_ij * dc_ij/dx_k
        # For j == k: dc_ik/dx_k. If k == i: dc_ii/dx_i = 0 (diagonal of neighbor part). 
        #   If k != i: dc_ik/dx_k = d_c_dx_j[i,k]
        # For j != k: dc_ij/dx_k = 0 (since c_ij only depends on x_i and x_j)
        
        # So d(r_i)/dx_k = w_ik * dc_ik/dx_k
        # If k == i: d(r_i)/dx_i = w_ii * 0 + boundary_grad_i = boundary_grad_i
        #   (w_ii is the weight for the diagonal, which we set to 0 in w_neigh)
        #   Actually, in our formulation, the boundary constraints are separate from neighbor constraints.
        #   So: d(r_i)/dx_i = boundary_grad_x[i] + sum_{j!=i} w_neigh[i,j] * dc_ij/dx_i
        #   = dr_dx_bound[i] + sum_{j!=i} w_neigh[i,j] * d_c_dx_i[i,j]
        #   = dr_dx_bound[i] + dr_dx_neigh_row[i]
        
        #   d(r_i)/dx_k for k != i: = w_neigh[i,k] * dc_ik/dx_k = w_neigh[i,k] * d_c_dx_j[i,k]
        
        # Therefore:
        # d(sum(r))/dx_k = d(r_k)/dx_k + sum_{i != k} d(r_i)/dx_k
        # = (dr_dx_bound[k] + dr_dx_neigh_row[k]) + sum_{i != k} w_neigh[i,k] * d_c_dx_j[i,k]
        
        # The sum_{i != k} w_neigh[i,k] * d_c_dx_j[i,k] is the k-th element of:
        # sum_i w_neigh[i,k] * d_c_dx_j[i,k] = (w_neigh * d_c_dx_j).sum(axis=0)[k]
        # But this includes i==k term: w_neigh[k,k] * d_c_dx_j[k,k] = 0 * 0 = 0
        # So it's correct.
        
        # Total:
        # d(sum(r))/dx_k = dr_dx_bound[k] + dr_dx_neigh_row[k] + dr_dx_neigh_col[k]
        # Similarly for y.
        
        d_sum_r_dx = dr_dx_bound + dr_dx_neigh_row + dr_dx_neigh_col
        d_sum_r_dy = dr_dy_bound + dr_dy_neigh_row + dr_dy_neigh_col
        
        # Objective is -sum(r), so gradient is -d(sum(r))/dparams
        grad[:N] = -d_sum_r_dx
        grad[N:] = -d_sum_r_dy
        
        return obj_val, grad

    def check_validity(x, y, r):
        if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9):
            return False
        if np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
            return False
        if np.any(r < -1e-9):
            return False

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        np.fill_diagonal(dist, np.inf)
        np.fill_diagonal(min_dist, 0)

        if np.any(dist < min_dist - 1e-9):
            return False
        return True

    def make_initial_guess(seed, mode='grid'):
        rng = np.random.RandomState(seed)
        if mode == 'grid':
            side = int(np.ceil(np.sqrt(N)))
            x = np.zeros(N)
            y = np.zeros(N)
            idx = 0
            for i in range(side):
                for j in range(side):
                    if idx >= N:
                        break
                    x[idx] = (i + 0.5) / side
                    y[idx] = (j + 0.5) / side
                    idx += 1
            x += rng.uniform(-0.08, 0.08, N)
            y += rng.uniform(-0.08, 0.08, N)
        elif mode == 'hex':
            side = int(np.ceil(np.sqrt(N)))
            x = np.zeros(N)
            y = np.zeros(N)
            idx = 0
            for i in range(side):
                for j in range(side):
                    if idx >= N:
                        break
                    x[idx] = (i + 0.5) / side + (0.5 / side) * (j % 2)
                    y[idx] = (j + 0.5) / side
                    idx += 1
            x += rng.uniform(-0.05, 0.05, N)
            y += rng.uniform(-0.05, 0.05, N)
        elif mode == 'fibonacci':
            # Fibonacci spiral for better space-filling
            golden_angle = np.pi * (3.0 - np.sqrt(5.0))
            x = np.zeros(N)
            y = np.zeros(N)
            for i in range(N):
                r = 0.45 * np.sqrt((i + 0.5) / N)
                theta = i * golden_angle
                x[i] = 0.5 + r * np.cos(theta)
                y[i] = 0.5 + r * np.sin(theta)
            x += rng.uniform(-0.02, 0.02, N)
            y += rng.uniform(-0.02, 0.02, N)
        else:  # random
            x = rng.uniform(0.1, 0.9, N)
            y = rng.uniform(0.1, 0.9, N)

        x = np.clip(x, 0.05, 0.95)
        y = np.clip(y, 0.05, 0.95)
        return x, y

    def gradient_optimize(x0, y0, beta_schedule=None, max_iter_total=500):
        """Multi-stage gradient optimization with increasing beta."""
        if beta_schedule is None:
            beta_schedule = [20, 50, 100, 200, 500, 1000]
        
        params0 = np.concatenate([x0, y0])
        bounds = [(0.01, 0.99)] * (2 * N)
        
        current_params = params0.copy()
        best_params = params0.copy()
        best_val = -np.inf
        
        def get_val(params):
            x = params[:N]
            y = params[N:2*N]
            r = voronoi_inradii_vectorized(x, y)
            if check_validity(x, y, r):
                return np.sum(r)
            for scale in [0.999, 0.99, 0.95]:
                r_test = r * scale
                if check_validity(x, y, r_test):
                    return np.sum(r_test)
            return -np.inf
        
        init_val = get_val(current_params)
        if init_val > best_val:
            best_val = init_val
            best_params = current_params.copy()
        
        iters_per_beta = max(1, max_iter_total // len(beta_schedule))
        
        for beta in beta_schedule:
            if time.time() - start_time > DEADLINE - 5:
                break
            
            try:
                res = minimize(compute_objective_and_grad_smooth, current_params,
                              method='L-BFGS-B', bounds=bounds,
                              args=(beta,),
                              jac=True,
                              options={'maxiter': iters_per_beta, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
                current_params = res.x
                val = get_val(current_params)
                if val > best_val:
                    best_val = val
                    best_params = current_params.copy()
            except Exception:
                pass
        
        return best_params, best_val

    def local_search(x, y, step_sizes=None, max_rounds=30):
        """Local search with multiple step sizes."""
        if step_sizes is None:
            step_sizes = [0.01, 0.005, 0.002, 0.001, 0.0005]
        
        directions = [(1,0), (-1,0), (0,1), (0,-1),
                      (1/np.sqrt(2), 1/np.sqrt(2)), (1/np.sqrt(2), -1/np.sqrt(2)),
                      (-1/np.sqrt(2), 1/np.sqrt(2)), (-1/np.sqrt(2), -1/np.sqrt(2))]
        
        x = x.copy()
        y = y.copy()
        r = voronoi_inradii_vectorized(x, y)
        curr_val = np.sum(r)
        
        for rnd in range(max_rounds):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            improved = False
            for step in step_sizes:
                for i in range(N):
                    for dx_dir, dy_dir in directions:
                        x_new = x.copy()
                        y_new = y.copy()
                        x_new[i] += dx_dir * step
                        y_new[i] += dy_dir * step
                        
                        if x_new[i] < 0.01 or x_new[i] > 0.99:
                            continue
                        if y_new[i] < 0.01 or y_new[i] > 0.99:
                            continue
                        
                        r_new = voronoi_inradii_vectorized(x_new, y_new)
                        new_val = np.sum(r_new)
                        
                        if new_val > curr_val + 1e-10:
                            x = x_new
                            y = y_new
                            r = r_new
                            curr_val = new_val
                            improved = True
            
            if not improved:
                break
        
        return x, y, voronoi_inradii_vectorized(x, y), np.sum(voronoi_inradii_vectorized(x, y))

    # Initial best: grid configuration
    side = int(np.ceil(np.sqrt(N)))
    best_x = np.zeros(N)
    best_y = np.zeros(N)
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= N:
                break
            best_x[idx] = (i + 0.5) / side
            best_y[idx] = (j + 0.5) / side
            idx += 1
    best_r = voronoi_inradii_vectorized(best_x, best_y)
    best_val = np.sum(best_r)

    # Multi-start with gradient optimization + local search
    modes = ['grid', 'hex', 'fibonacci', 'random']
    num_starts = 60
    for i in range(num_starts):
        if time.time() - start_time > DEADLINE - 15:
            break

        mode = modes[i % len(modes)]
        x0, y0 = make_initial_guess(i, mode)

        # Gradient optimization
        if time.time() - start_time < DEADLINE - 10:
            params_opt, val_opt = gradient_optimize(x0, y0)
            x_opt = params_opt[:N]
            y_opt = params_opt[N:]
            r_opt = voronoi_inradii_vectorized(x_opt, y_opt)
            
            if not check_validity(x_opt, y_opt, r_opt):
                for scale in [0.999, 0.99, 0.95]:
                    r_test = r_opt * scale
                    if check_validity(x_opt, y_opt, r_test):
                        r_opt = r_test
                        break
                else:
                    continue
            
            val_opt = np.sum(r_opt)
        else:
            x_opt, y_opt = x0, y0
            r_opt = voronoi_inradii_vectorized(x_opt, y_opt)
            val_opt = np.sum(r_opt)

        # Local search
        if time.time() - start_time < DEADLINE - 5:
            x_ls, y_ls, r_ls, val_ls = local_search(x_opt, y_opt)
            if val_ls > val_opt:
                x_opt, y_opt, r_opt, val_opt = x_ls, y_ls, r_ls, val_ls

        # Update best
        if val_opt > best_val:
            best_val = val_opt
            best_x = x_opt.copy()
            best_y = y_opt.copy()
            best_r = r_opt.copy()

    # Final polish on the best solution
    if time.time() - start_time < DEADLINE - 5:
        params_final, val_final = gradient_optimize(best_x, best_y, 
                                                    beta_schedule=[50, 100, 200, 500, 1000, 2000],
                                                    max_iter_total=300)
        x_final = params_final[:N]
        y_final = params_final[N:]
        r_final = voronoi_inradii_vectorized(x_final, y_final)
        
        if not check_validity(x_final, y_final, r_final):
            for scale in [0.999, 0.99, 0.95]:
                r_test = r_final * scale
                if check_validity(x_final, y_final, r_test):
                    r_final = r_test
                    break
            else:
                x_final, y_final, r_final = best_x, best_y, best_r
                val_final = best_val
        
        val_final = np.sum(r_final)
        
        if val_final > best_val:
            best_val = val_final
            best_x = x_final.copy()
            best_y = y_final.copy()
            best_r = r_final.copy()
        
        # Final local search
        if time.time() - start_time < DEADLINE - 2:
            x_ls, y_ls, r_ls, val_ls = local_search(best_x, best_y, 
                                                     step_sizes=[0.005, 0.002, 0.001, 0.0005],
                                                     max_rounds=20)
            if val_ls > best_val:
                best_val = val_ls
                best_x = x_ls.copy()
                best_y = y_ls.copy()
                best_r = r_ls.copy()

    # Final validity check
    if not check_validity(best_x, best_y, best_r):
        for scale in np.linspace(0.99, 0.90, 20):
            r_test = best_r * scale
            if check_validity(best_x, best_y, r_test):
                best_r = r_test
                break
        else:
            best_r = voronoi_inradii_vectorized(best_x, best_y)

    return np.column_stack((best_x, best_y, best_r))
