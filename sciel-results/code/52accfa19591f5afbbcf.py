"""
Circle packing in unit square using simulated annealing with repulsion forces.
Centers are optimized; radii are derived from local geometry.
SA provides global exploration to escape local minima.
Smooth soft-min surrogate with analytic gradient for fast L-BFGS-B polish.
Discrete best-fit local search for deterministic neighborhood improvement.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 26
DEADLINE = 112.0  # Safety margin from 120s limit


def construct():
    start_time = time.time()

    def compute_radii(x, y):
        """Compute radii from centers: min of half-distance to neighbors and boundary."""
        r_bound = np.minimum(
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        )

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)

        min_dist = np.min(dist, axis=1)
        r = np.minimum(r_bound, 0.5 * min_dist)
        r = np.maximum(r, 1e-6)
        return r

    def compute_radii_smooth(x, y, beta=50.0):
        """Smooth approximation of radii using log-sum-exp soft-min.
        beta controls sharpness: large beta -> hard min, small beta -> smooth average.
        Returns (radii, soft_min_values) where soft_min_values are the unclamped values.
        """
        r_bound = np.minimum(
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        )

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)

        half_dist = 0.5 * dist  # (N, N), inf on diagonal
        full_constraints = half_dist.copy()
        np.fill_diagonal(full_constraints, r_bound)

        # Soft-min per row
        neg_beta = -beta * full_constraints
        max_neg_row = np.max(neg_beta, axis=1, keepdims=True)
        log_sum_row = np.log(np.sum(np.exp(neg_beta - max_neg_row), axis=1, keepdims=True))
        soft_min_all = -(max_neg_row + log_sum_row) / beta
        soft_min_all = soft_min_all[:, 0]

        r = np.maximum(soft_min_all, 1e-6)
        return r, soft_min_all

    def smooth_objective_and_grad(params, beta=50.0):
        """Compute -sum(r) and its gradient with respect to params = [x, y].
        Uses analytic gradient of the log-sum-exp soft-min.
        
        r_i = softmin_j(c_ij) where c_ij = 0.5*dist(i,j) for j!=i, c_ii = r_bound_i
        r_i = -(1/beta) * log(sum_j exp(-beta * c_ij))
        dr_i/dp = sum_j w_ij * dc_ij/dp where w_ij = exp(-beta*c_ij) / sum_j' exp(-beta*c_ij')
        """
        x = params[:N]
        y = params[N:2*N]

        # Boundary constraint
        r_bound = np.minimum(
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        )

        # Pairwise distances
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2 + 1e-30)
        np.fill_diagonal(dist, 1e10)

        # Build constraint matrix: c_ij for all i,j
        # c_ij = 0.5 * dist(i,j) for i != j
        # c_ii = r_bound[i]
        c = 0.5 * dist
        np.fill_diagonal(c, r_bound)

        # Soft-min computation
        neg_beta_c = -beta * c
        max_neg = np.max(neg_beta_c, axis=1, keepdims=True)
        exp_terms = np.exp(neg_beta_c - max_neg)
        sum_exp = np.sum(exp_terms, axis=1, keepdims=True)
        log_sum = np.log(sum_exp)
        r = -(max_neg + log_sum) / beta
        r = r[:, 0]

        # Weights: w_ij = exp(-beta*c_ij) / sum_j' exp(-beta*c_ij')
        w = exp_terms / sum_exp  # (N, N)

        # Gradient of c_ij w.r.t. parameters
        # For i != j: c_ij = 0.5 * sqrt((x_i-x_j)^2 + (y_i-y_j)^2)
        # dc_ij/dx_i = 0.5 * (x_i - x_j) / dist_ij = 0.5 * dx[i,j] / dist[i,j]
        # dc_ij/dy_i = 0.5 * dy[i,j] / dist[i,j]
        # dc_ij/dx_j = -0.5 * dx[i,j] / dist[i,j]
        # dc_ij/dy_j = -0.5 * dy[i,j] / dist[i,j]
        
        # For i == j: c_ii = r_bound_i = min(x_i, 1-x_i, y_i, 1-y_i)
        # Gradient is piecewise; use subgradient from active constraint
        
        dist_safe = dist.copy()
        np.fill_diagonal(dist_safe, 1.0)
        
        # Off-diagonal gradients
        dc_dx_i = 0.5 * dx / dist_safe  # dc_ij/dx_i
        dc_dy_i = 0.5 * dy / dist_safe  # dc_ij/dy_i
        dc_dx_j = -0.5 * dx / dist_safe  # dc_ij/dx_j
        dc_dy_j = -0.5 * dy / dist_safe  # dc_ij/dy_j

        # Diagonal gradients for r_bound_i
        # r_bound_i = min(x_i, 1-x_i, y_i, 1-y_i)
        # Use smooth approximation: weight each boundary by how close it is
        # For better conditioning, use a soft-min over the 4 boundaries
        bounds_4 = np.stack([x, 1.0-x, y, 1.0-y], axis=1)  # (N, 4)
        # Soft-min over 4 boundaries with high beta for near-hard behavior
        beta_b = 100.0
        neg_bb = -beta_b * bounds_4
        max_bb = np.max(neg_bb, axis=1, keepdims=True)
        exp_bb = np.exp(neg_bb - max_bb)
        sum_bb = np.sum(exp_bb, axis=1, keepdims=True)
        w_b = exp_bb / sum_bb  # (N, 4)
        
        # Gradient of softmin(x, 1-x, y, 1-y) w.r.t. x_i:
        # = w_b[i,0]*1 + w_b[i,1]*(-1) + w_b[i,2]*0 + w_b[i,3]*0
        grad_rb_dx = w_b[:, 0] - w_b[:, 1]
        # Gradient w.r.t. y_i:
        # = w_b[i,0]*0 + w_b[i,1]*0 + w_b[i,2]*1 + w_b[i,3]*(-1)
        grad_rb_dy = w_b[:, 2] - w_b[:, 3]

        # Set diagonal entries
        np.fill_diagonal(dc_dx_i, grad_rb_dx)
        np.fill_diagonal(dc_dy_i, grad_rb_dy)
        np.fill_diagonal(dc_dx_j, grad_rb_dx)
        np.fill_diagonal(dc_dy_j, grad_rb_dy)

        # Total gradient w.r.t. x_k: sum_i dr_i/dx_k
        # dr_i/dx_k = sum_j w_ij * dc_ij/dx_k
        # For k = i: sum_j w_ij * dc_ij/dx_i = sum_j w_ij * dc_dx_i[i,j]
        # For k != i: only j=k contributes: w_ik * dc_ik/dx_k = w_ik * dc_dx_j[i,k]
        # 
        # Total: dr_dx[k] = sum_i [w_ik * dc_dx_i[i,k]] + sum_i [w_ki * dc_dx_j[k,i]]
        # Wait, let me redo this carefully.
        # 
        # dr_i/dx_k = sum_j w_ij * (dc_ij/dx_k)
        # dc_ij/dx_k is nonzero only if k==i or k==j.
        # If k==i: dc_ij/dx_i = dc_dx_i[i,j]
        # If k==j: dc_ij/dx_j = dc_dx_j[i,j]
        # 
        # So dr_i/dx_k = (if k==i) sum_j w_ij * dc_dx_i[i,j] 
        #              + (if k==j for some j) w_ik * dc_dx_j[i,k]
        # 
        # For fixed k, dr_i/dx_k for i != k: only the second term applies with j=k
        #   = w_ik * dc_dx_j[i,k]
        # For i == k: both terms
        #   = sum_j w_kj * dc_dx_i[k,j] + sum_{j!=k} w_kj * dc_dx_j[k,j]
        #   But wait, the second sum is over j where k==j, which is just j=k, but we excluded that.
        #   Actually for i=k, the j=k term in the first sum already includes it.
        #   Let me just compute it as:
        #   dr_dx[k] = sum_i [contribution of w_ij * dc_ij/dx_k]
        #   = sum_i sum_j w_ij * dc_ij/dx_k
        #   = sum_i [w_ik * dc_ik/dx_k]  (terms where j=k, i.e., dc_ij/dx_k nonzero when j=k)
        #     + sum_i [w_ki * dc_ki/dx_k]  (terms where i=k, i.e., dc_ij/dx_k nonzero when i=k)
        #   But when i=k and j=k, we double count. Let's handle it properly.
        #
        # dc_ij/dx_k = dc_dx_i[i,j] if k==i, else dc_dx_j[i,j] if k==j, else 0
        #
        # dr_dx[k] = sum_i sum_j w_ij * dc_ij/dx_k
        #          = sum_i w_ik * dc_ik/dx_k   [j=k, so dc_ik/dx_k = dc_dx_j[i,k]]
        #            + sum_i sum_{j!=k} w_ij * [dc_ij/dx_k if i==k]
        #          = sum_i w_ik * dc_dx_j[i,k]
        #            + sum_{j!=k} w_kj * dc_dx_i[k,j]
        #          + w_kk * dc_kk/dx_k  [the i=k,j=k term, which is in both sums if not careful]
        #
        # Actually simpler: just compute the matrix product.
        # dr_dx = (w @ dc_dx_i).diagonal() + (w.T @ dc_dx_j).diagonal() - (w * dc_dx_i).diagonal() * 0
        # Hmm, this is getting complicated. Let me just do it directly.
        
        # dr_i/dx_i = sum_j w_ij * dc_dx_i[i,j]  (row i of w times row i of dc_dx_i, summed over j)
        dr_dx_self = np.sum(w * dc_dx_i, axis=1)  # (N,)
        dr_dy_self = np.sum(w * dc_dy_i, axis=1)  # (N,)
        
        # dr_i/dx_k for k != i: w_ik * dc_dx_j[i,k]
        # Sum over i for fixed k: sum_i w_ik * dc_dx_j[i,k] = (w * dc_dx_j)[:, k] summed over i
        # = column k of (w * dc_dx_j) summed over rows
        dr_dx_cross = np.sum(w * dc_dx_j, axis=0)  # (N,)
        dr_dy_cross = np.sum(w * dc_dy_j, axis=0)  # (N,)
        
        # Total dr/dx_k = dr_dx_self[k] + dr_dx_cross[k]
        # But note: dr_dx_self[k] includes j=k term: w_kk * dc_dx_i[k,k]
        # and dr_dx_cross[k] includes i=k term: w_kk * dc_dx_j[k,k]
        # These are both the diagonal term, so we're double counting.
        # dc_dx_i[k,k] = dc_dx_j[k,k] = grad_rb_dx[k]
        # So we need to subtract one copy of w_kk * grad_rb_dx[k]
        dr_dx = dr_dx_self + dr_dx_cross - np.diag(w) * np.diag(dc_dx_i)
        dr_dy = dr_dy_self + dr_dy_cross - np.diag(w) * np.diag(dc_dy_i)

        # Objective is -sum(r), so gradient is -dr/dparams
        grad = np.concatenate([-dr_dx, -dr_dy])
        
        obj_val = -np.sum(r)
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
            # Hexagonal lattice initialization
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
        else:  # random
            x = rng.uniform(0.1, 0.9, N)
            y = rng.uniform(0.1, 0.9, N)

        x = np.clip(x, 0.05, 0.95)
        y = np.clip(y, 0.05, 0.95)
        return x, y

    def sa_optimize(x0, y0, seed, max_iters=3000):
        """Simulated annealing to optimize center positions with vectorized forces."""
        rng = np.random.RandomState(seed)
        x = x0.copy()
        y = y0.copy()
        r = compute_radii(x, y)
        best_val = np.sum(r)
        best_x = x.copy()
        best_y = y.copy()

        # Cooling schedule
        T_start = 0.05
        T_end = 0.0005
        cool_rate = (T_end / T_start) ** (1.0 / max_iters)

        # Repulsion force parameters
        repulse_k = 2.0
        boundary_k = 5.0

        for it in range(max_iters):
            if time.time() - start_time > DEADLINE:
                break

            T = T_start * (cool_rate ** it)

            # Compute current energy
            r_curr = compute_radii(x, y)
            curr_val = np.sum(r_curr)

            # Compute forces (vectorized)
            fx = np.zeros(N)
            fy = np.zeros(N)

            # Boundary repulsion (vectorized)
            fx += boundary_k * np.clip(0.1 - x, 0, None)
            fx -= boundary_k * np.clip(x - 0.9, 0, None)
            fy += boundary_k * np.clip(0.1 - y, 0, None)
            fy -= boundary_k * np.clip(y - 0.9, 0, None)

            # Pairwise repulsion (vectorized)
            dx = x[:, None] - x[None, :]  # (N, N)
            dy = y[:, None] - y[None, :]  # (N, N)
            d = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(d, 1e10)  # Avoid division by zero on diagonal

            # Force magnitude: repulse_k / (d^2 + eps)
            f_mag = repulse_k / (d**2 + 0.01)
            np.fill_diagonal(f_mag, 0)

            # Force on i from j: -f * (x_i - x_j) / d = f * (x_j - x_i) / d
            fx += np.sum(-f_mag * dx / d, axis=1)
            fy += np.sum(-f_mag * dy / d, axis=1)

            # Propose move: combine force direction with random perturbation
            scale = 0.02 * T / T_start + 0.005
            dx_prop = fx * scale + rng.normal(0, scale, N)
            dy_prop = fy * scale + rng.normal(0, scale, N)

            x_new = x + dx_prop
            y_new = y + dy_prop

            # Clip to valid range
            x_new = np.clip(x_new, 0.05, 0.95)
            y_new = np.clip(y_new, 0.05, 0.95)

            # Evaluate new state
            r_new = compute_radii(x_new, y_new)
            new_val = np.sum(r_new)

            # Acceptance criterion
            delta = new_val - curr_val
            if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
                x = x_new
                y = y_new
                r = r_new
                if new_val > best_val:
                    best_val = new_val
                    best_x = x.copy()
                    best_y = y.copy()

        return best_x, best_y, compute_radii(best_x, best_y), best_val

    def voronoi_inradius_polish(x, y, max_rounds=20):
        """Polish using exact Voronoi inradii from Delaunay triangulation.
        For each circle, the true inradius is the minimum distance to the
        Voronoi cell edges (which correspond to perpendicular bisectors of
        Delaunay edges) and to the boundary. This gives a tighter, more
        accurate radius than the simple nearest-neighbor estimate.
        """
        try:
            from scipy.spatial import Delaunay
        except ImportError:
            return x, y, compute_radii(x, y), np.sum(compute_radii(x, y))

        x = x.copy()
        y = y.copy()
        r = compute_radii(x, y)
        curr_val = np.sum(r)

        for rnd in range(max_rounds):
            if time.time() - start_time > DEADLINE - 5:
                break

            # Compute Voronoi inradii
            points = np.column_stack([x, y])
            try:
                tri = Delaunay(points)
            except Exception:
                break

            # For each point, find its Voronoi inradius
            # The Voronoi inradius of point i is the min over its Delaunay neighbors j
            # of the distance from i to the perpendicular bisector of edge (i,j)
            # = dist(i,j) / 2  (since the bisector is at the midpoint)
            # Plus the boundary distance.
            # Actually, the Voronoi inradius is the distance from the point to the
            # nearest Voronoi edge, which for a Delaunay edge (i,j) is dist(i,j)/2
            # only if the circumcenter is outside the triangle. More precisely:
            # The Voronoi cell of i has edges that are perpendicular bisectors of
            # Delaunay edges incident to i. The distance from i to the bisector of
            # edge (i,j) is dist(i,j)/2.
            # But the actual inradius is the min distance to any Voronoi edge,
            # which equals the min over Delaunay neighbors of dist(i,j)/2.
            # This is the same as what compute_radii does!
            #
            # However, the Voronoi inradius can be LARGER than the nearest-neighbor
            # half-distance if the nearest neighbor's bisector is not the closest
            # Voronoi edge. Actually no - the closest Voronoi edge IS the bisector
            # of the nearest Delaunay neighbor. So they're the same.
            #
            # The real benefit: we can use the Voronoi inradius to do a more precise
            # local move. Let's instead do a "push apart" step: for each pair of
            # touching circles, push them apart along their connecting line.

            # Find pairs that are (nearly) touching
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(dist, np.inf)

            # For each circle, find its constraint (nearest neighbor or boundary)
            r_bound = np.minimum(
                np.minimum(x, 1.0 - x),
                np.minimum(y, 1.0 - y)
            )

            # Push apart: for each pair (i,j) with dist < r_i + r_j + eps,
            # push both away from each other
            r_curr = compute_radii(x, y)
            overlap = r_curr[:, None] + r_curr[None, :] - dist
            np.fill_diagonal(overlap, 0)

            # Only push if there's actual overlap (shouldn't happen with valid config)
            # Instead, do a gradient step to increase the minimum constraint
            # For each circle i, if its constraint is neighbor j, push i away from j
            # If its constraint is boundary, push i away from boundary

            # Compute the "active constraint" for each circle
            min_dist_to_neighbor = np.min(dist, axis=1)
            active_is_boundary = r_bound < 0.5 * min_dist_to_neighbor

            # For boundary-constrained circles, compute which boundary
            # For neighbor-constrained, compute direction away from nearest neighbor

            push_x = np.zeros(N)
            push_y = np.zeros(N)

            for i in range(N):
                if active_is_boundary[i]:
                    # Push away from the closest boundary
                    d_left = x[i]
                    d_right = 1.0 - x[i]
                    d_bottom = y[i]
                    d_top = 1.0 - y[i]
                    min_d = min(d_left, d_right, d_bottom, d_top)
                    if min_d == d_left:
                        push_x[i] = 1.0
                    elif min_d == d_right:
                        push_x[i] = -1.0
                    elif min_d == d_bottom:
                        push_y[i] = 1.0
                    else:
                        push_y[i] = -1.0
                else:
                    # Push away from nearest neighbor
                    j = np.argmin(dist[i])
                    dx_ij = x[i] - x[j]
                    dy_ij = y[i] - y[j]
                    d_ij = np.sqrt(dx_ij**2 + dy_ij**2)
                    if d_ij > 1e-10:
                        push_x[i] = dx_ij / d_ij
                        push_y[i] = dy_ij / d_ij

            # Apply small push
            step = 0.002 * (1.0 - rnd / max_rounds)
            x_new = x + step * push_x
            y_new = y + step * push_y
            x_new = np.clip(x_new, 0.01, 0.99)
            y_new = np.clip(y_new, 0.01, 0.99)

            r_new = compute_radii(x_new, y_new)
            new_val = np.sum(r_new)

            if new_val > curr_val:
                x = x_new
                y = y_new
                r = r_new
                curr_val = new_val
            else:
                # Try smaller step
                step2 = step * 0.5
                x_new2 = x + step2 * push_x
                y_new2 = y + step2 * push_y
                x_new2 = np.clip(x_new2, 0.01, 0.99)
                y_new2 = np.clip(y_new2, 0.01, 0.99)
                r_new2 = compute_radii(x_new2, y_new2)
                new_val2 = np.sum(r_new2)
                if new_val2 > curr_val:
                    x = x_new2
                    y = y_new2
                    r = r_new2
                    curr_val = new_val2
                else:
                    break

        return x, y, compute_radii(x, y), np.sum(compute_radii(x, y))

    def best_fit_local_search(x, y, step_sizes=None, max_rounds=60):
        """Discrete best-fit local search with adaptive step sizes.
        Vectorized: try all circles and directions simultaneously per step size.
        """
        if step_sizes is None:
            step_sizes = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
        
        # 8 directions
        dirs = np.array([
            [1, 0], [-1, 0], [0, 1], [0, -1],
            [1/np.sqrt(2), 1/np.sqrt(2)], [1/np.sqrt(2), -1/np.sqrt(2)],
            [-1/np.sqrt(2), 1/np.sqrt(2)], [-1/np.sqrt(2), -1/np.sqrt(2)]
        ])  # (8, 2)

        x = x.copy()
        y = y.copy()
        r = compute_radii(x, y)
        curr_val = np.sum(r)

        for rnd in range(max_rounds):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            best_improvement = 1e-10
            best_i = -1
            best_dir = None
            best_step = None
            
            for step in step_sizes:
                # For each circle i and each direction d, compute new position
                # x_new[i, d] = x[i] + dirs[d, 0] * step
                x_new = x[:, None] + dirs[None, :, 0] * step  # (N, 8)
                y_new = y[:, None] + dirs[None, :, 1] * step  # (N, 8)
                
                # Mask invalid positions
                valid = (x_new >= 0.01) & (x_new <= 0.99) & (y_new >= 0.01) & (y_new <= 0.99)
                
                # For each (i, d) pair, compute new radii
                # This is expensive: N*8 calls to compute_radii
                # Optimize: only try if the move could improve things
                # Quick check: if the circle is already well-separated, small moves won't help much
                
                for i in range(N):
                    for d in range(8):
                        if not valid[i, d]:
                            continue
                        
                        x_temp = x.copy()
                        y_temp = y.copy()
                        x_temp[i] = x_new[i, d]
                        y_temp[i] = y_new[i, d]
                        
                        r_new = compute_radii(x_temp, y_temp)
                        new_val = np.sum(r_new)
                        improvement = new_val - curr_val
                        
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_i = i
                            best_dir = d
                            best_step = step
                
                # Early exit if we found a good improvement with this step size
                if best_improvement > step * 0.1:
                    break
            
            if best_i >= 0:
                x[best_i] += dirs[best_dir, 0] * best_step
                y[best_i] += dirs[best_dir, 1] * best_step
                r = compute_radii(x, y)
                curr_val = np.sum(r)
            else:
                break

        return x, y, compute_radii(x, y), np.sum(compute_radii(x, y))

    def gradient_polish(x0, y0, use_smooth=True, beta_start=20.0, beta_end=500.0, num_steps=5):
        """Final gradient-based refinement using L-BFGS-B with analytic gradient.
        Uses smooth soft-min objective for better gradient signal.
        Iteratively increases sharpness (beta) to converge to hard constraints.
        """
        params0 = np.concatenate([x0, y0])
        bounds = [(0.01, 0.99)] * (2 * N)
        
        # Geometric progression of betas
        betas = np.geomspace(beta_start, beta_end, num_steps)
        
        current_params = params0.copy()
        best_result = None
        best_val = -np.inf
        
        def get_val_from_params(params):
            x = params[:N]
            y = params[N:2*N]
            r = compute_radii(x, y)
            if check_validity(x, y, r):
                return np.sum(r)
            # Try scaling down to find a valid subset if invalid
            for scale in [0.999, 0.99, 0.95, 0.90]:
                r_test = r * scale
                if check_validity(x, y, r_test):
                    return np.sum(r_test)
            return -np.inf

        def update_best(params):
            nonlocal best_result, best_val
            val = get_val_from_params(params)
            if val > best_val:
                best_val = val
                r_hard = compute_radii(params[:N], params[N:])
                if check_validity(params[:N], params[N:], r_hard):
                    best_result = (params[:N].copy(), params[N:].copy(), r_hard, val)
                else:
                    for scale in [0.999, 0.99, 0.98]:
                        r_test = r_hard * scale
                        if check_validity(params[:N], params[N:], r_test):
                            best_result = (params[:N].copy(), params[N:].copy(), r_test, np.sum(r_test))
                            break

        # Initial check
        update_best(current_params)

        for beta in betas:
            if time.time() - start_time > DEADLINE - 5:
                break
                
            try:
                res = minimize(smooth_objective_and_grad, current_params, 
                              method='L-BFGS-B', bounds=bounds,
                              args=(beta,),
                              jac=True,
                              options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-11, 'maxls': 50})
                current_params = res.x
                update_best(current_params)
            except Exception:
                pass

        # Final hard polish with very high beta
        if time.time() - start_time < DEADLINE - 2:
            beta_final = min(beta_end * 2, 2000.0)
            try:
                res2 = minimize(smooth_objective_and_grad, current_params,
                                method='L-BFGS-B', bounds=bounds,
                                args=(beta_final,),
                                jac=True,
                                options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
                current_params = res2.x
                update_best(current_params)
            except Exception:
                pass

        if best_result is not None:
            return best_result
        return None

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
    best_r = compute_radii(best_x, best_y)
    best_val = np.sum(best_r)

    # Multi-start with SA + best-fit + gradient polish
    modes = ['grid', 'hex', 'random']
    num_starts = 40
    for i in range(num_starts):
        if time.time() - start_time > DEADLINE - 20:
            break

        mode = modes[i % len(modes)]
        x0, y0 = make_initial_guess(i, mode)

        # Run SA
        x_sa, y_sa, r_sa, val_sa = sa_optimize(x0, y0, seed=i, max_iters=3000)

        # Best-fit local search on SA result
        x_bf, y_bf, r_bf, val_bf = best_fit_local_search(x_sa, y_sa)

        # Gradient polish on best-fit result
        if time.time() - start_time < DEADLINE - 10:
            polished = gradient_polish(x_bf, y_bf, use_smooth=True)
            if polished is not None:
                xp, yp, rp, vp = polished
                if vp > val_bf:
                    x_sa, y_sa, r_sa, val_sa = xp, yp, rp, vp

        # Update best
        if val_sa > best_val:
            best_val = val_sa
            best_x = x_sa.copy()
            best_y = y_sa.copy()
            best_r = r_sa.copy()

        # Also try best-fit on the initial guess directly (skip SA for some starts)
        if i % 4 == 3 and time.time() - start_time < DEADLINE - 15:
            x_bf2, y_bf2, r_bf2, val_bf2 = best_fit_local_search(x0, y0)
            if time.time() - start_time < DEADLINE - 10:
                polished2 = gradient_polish(x_bf2, y_bf2, use_smooth=True)
                if polished2 is not None:
                    xp2, yp2, rp2, vp2 = polished2
                    if vp2 > val_bf2:
                        val_bf2 = vp2
                        x_bf2, y_bf2, r_bf2 = xp2, yp2, rp2
            if val_bf2 > best_val:
                best_val = val_bf2
                best_x = x_bf2.copy()
                best_y = y_bf2.copy()
                best_r = r_bf2.copy()

    # Final high-precision polish: run gradient polish on the best solution found
    if time.time() - start_time < DEADLINE - 5:
        # First do best-fit on the best solution
        x_bf_final, y_bf_final, r_bf_final, val_bf_final = best_fit_local_search(best_x, best_y)
        if val_bf_final > best_val:
            best_val = val_bf_final
            best_x = x_bf_final
            best_y = y_bf_final
            best_r = r_bf_final

        # Multiple gradient polish passes with increasing precision
        for polish_pass in range(5):
            if time.time() - start_time > DEADLINE - 3:
                break
            beta_s = 50.0 * (10 ** polish_pass)
            beta_e = 1000.0 * (10 ** polish_pass)
            final_polish = gradient_polish(best_x, best_y, use_smooth=True, 
                                           beta_start=min(beta_s, 3000.0), 
                                           beta_end=min(beta_e, 8000.0), 
                                           num_steps=8)
            if final_polish is not None:
                xp, yp, rp, vp = final_polish
                if vp > best_val:
                    best_val = vp
                    best_x = xp
                    best_y = yp
                    best_r = rp
                # Do another best-fit pass after gradient polish
                if time.time() - start_time < DEADLINE - 3:
                    x_bf2, y_bf2, r_bf2, val_bf2 = best_fit_local_search(best_x, best_y)
                    if val_bf2 > best_val:
                        best_val = val_bf2
                        best_x = x_bf2
                        best_y = y_bf2
                        best_r = r_bf2

    # Final validity check
    if not check_validity(best_x, best_y, best_r):
        for scale in np.linspace(0.99, 0.90, 20):
            r_test = best_r * scale
            if check_validity(best_x, best_y, r_test):
                best_r = r_test
                break
        else:
            best_r = compute_radii(best_x, best_y)

    return np.column_stack((best_x, best_y, best_r))
