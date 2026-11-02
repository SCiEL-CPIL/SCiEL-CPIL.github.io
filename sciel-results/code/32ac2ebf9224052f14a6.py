import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import time
import random

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def check_time():
        return time.time() > deadline

    def pack_hexagonal(rows):
        """
        Generate a hexagonal packing configuration.
        Returns: w, h, r, xs, ys
        """
        if sum(rows) != N:
            return None

        pts = []
        for k, n_circles in enumerate(rows):
            y_k = k * np.sqrt(3)
            offset = 1.0 if k % 2 == 1 else 0.0
            for i in range(n_circles):
                x_k = i * 2.0 + offset
                pts.append((x_k, y_k))
        
        pts = np.array(pts)
        if len(pts) != N:
            return None

        center = pts.mean(axis=0)
        pts -= center
        
        xs = pts[:, 0]
        ys = pts[:, 1]
        w_centers = xs.max() - xs.min()
        h_centers = ys.max() - ys.min()
        
        w_total = w_centers + 2.0
        h_total = h_centers + 2.0
        
        scale = 2.0 / (w_total + h_total)
        r = scale
        w = scale * w_total
        h = scale * h_total
        
        xs_final = pts[:, 0] * scale + w/2.0
        ys_final = pts[:, 1] * scale + h/2.0
        
        return w, h, r, xs_final, ys_final

    def pack_grid(rows, cols):
        """
        Generate a grid packing configuration.
        """
        pts = []
        count = 0
        for i in range(cols):
            for j in range(rows):
                if count >= N:
                    break
                pts.append((i * 2.0, j * 2.0))
                count += 1
            if count >= N:
                break
        
        pts = np.array(pts[:N])
        
        center = pts.mean(axis=0)
        pts -= center
        
        xs = pts[:, 0]
        ys = pts[:, 1]
        w_centers = xs.max() - xs.min()
        h_centers = ys.max() - ys.min()
        
        w_total = w_centers + 2.0
        h_total = h_centers + 2.0
        
        scale = 2.0 / (w_total + h_total)
        r = scale
        w = scale * w_total
        h = scale * h_total
        
        xs_final = pts[:, 0] * scale + w/2.0
        ys_final = pts[:, 1] * scale + h/2.0
        
        return w, h, r, xs_final, ys_final

    def physics_simulate(w, h, r, xs, ys, max_iter=200):
        """
        Physics-based refinement using repulsion and soft boundaries.
        Returns improved (w, h, r, xs, ys) or original if no improvement.
        """
        if check_time():
            return w, h, r, xs, ys

        x = xs.copy()
        y = ys.copy()
        
        current_r = r
        current_w = w
        current_h = h
        
        # Relaxation function
        def relax(x, y, r_val, w_val, h_val, steps=50):
            for _ in range(steps):
                if check_time():
                    break
                fx = np.zeros(N)
                fy = np.zeros(N)
                
                # Pairwise repulsion
                for i in range(N):
                    for j in range(i+1, N):
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        dist_sq = dx*dx + dy*dy
                        min_dist = 2.0 * r_val
                        min_dist_sq = min_dist * min_dist
                        if dist_sq < min_dist_sq:
                            dist = np.sqrt(dist_sq)
                            if dist > 1e-6:
                                overlap = min_dist - dist
                                force = 10.0 * overlap
                                fx[i] += (dx/dist) * force
                                fy[i] += (dy/dist) * force
                                fx[j] -= (dx/dist) * force
                                fy[j] -= (dy/dist) * force
                            else:
                                fx[i] += random.uniform(-1, 1)
                                fy[i] += random.uniform(-1, 1)
                                fx[j] -= random.uniform(-1, 1)
                                fy[j] -= random.uniform(-1, 1)
                
                # Boundary repulsion
                for i in range(N):
                    if x[i] < r_val:
                        fx[i] += 10.0 * (r_val - x[i])
                    if x[i] > w_val - r_val:
                        fx[i] -= 10.0 * (x[i] - (w_val - r_val))
                    if y[i] < r_val:
                        fy[i] += 10.0 * (r_val - y[i])
                    if y[i] > h_val - r_val:
                        fy[i] -= 10.0 * (y[i] - (h_val - r_val))
                
                x += 0.01 * fx
                y += 0.01 * fy
                
                x = np.clip(x, r_val, w_val - r_val)
                y = np.clip(y, r_val, h_val - r_val)
                
            return x, y

        def is_valid(x, y, r_val, w_val, h_val):
            for i in range(N):
                if x[i] < r_val - 1e-7 or x[i] > w_val - r_val + 1e-7:
                    return False
                if y[i] < r_val - 1e-7 or y[i] > h_val - r_val + 1e-7:
                    return False
            for i in range(N):
                for j in range(i+1, N):
                    dist_sq = (x[i]-x[j])**2 + (y[i]-y[j])**2
                    if dist_sq < 4*(r_val**2) - 1e-7:
                        return False
            return True

        if not is_valid(x, y, current_r, current_w, current_h):
            x, y = relax(x, y, current_r, current_w, current_h, steps=100)
            if not is_valid(x, y, current_r, current_w, current_h):
                return w, h, r, xs, ys

        max_r_possible = min(current_w, current_h) / 2.0
        r_try = current_r
        delta_r = 0.001
        improved = False
        
        while r_try < max_r_possible - 1e-4 and not check_time():
            x_new, y_new = relax(x, y, r_try + delta_r, current_w, current_h, steps=20)
            
            if is_valid(x_new, y_new, r_try + delta_r, current_w, current_h):
                r_try += delta_r
                x, y = x_new, y_new
                improved = True
            else:
                break
        
        if improved:
            return current_w, current_h, r_try, x, y
        else:
            return current_w, current_h, current_r, x, y

    def get_contact_graph(xs, ys, r, w, h, tolerance=0.05):
        """
        Build contact graph using Delaunay triangulation.
        Returns list of (i, j) pairs that are in contact or near-contact.
        """
        try:
            pts = np.column_stack([xs, ys])
            tri = Delaunay(pts)
            edges = set()
            for simplex in tri.simplices:
                for a in range(3):
                    for b in range(a+1, 3):
                        i, j = simplex[a], simplex[b]
                        if i > j:
                            i, j = j, i
                        edges.add((i, j))
            
            # Filter: keep only edges where circles are actually close
            contact_pairs = []
            for i, j in edges:
                dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                if dist <= 2.0 * r * (1.0 + tolerance):
                    contact_pairs.append((i, j))
            
            if len(contact_pairs) == 0:
                # Fallback: use all pairs within 2r*(1+tolerance)
                for i in range(N):
                    for j in range(i+1, N):
                        dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                        if dist <= 2.0 * r * (1.0 + tolerance):
                            contact_pairs.append((i, j))
            
            return contact_pairs
        except Exception:
            # Fallback: brute force
            contact_pairs = []
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                    if dist <= 2.0 * r * (1.0 + tolerance):
                        contact_pairs.append((i, j))
            return contact_pairs

    def refine_contact(w, h, r, xs, ys, is_polish=False):
        """
        Contact-graph-driven optimization:
        1. Build Delaunay contact graph
        2. Use only contact edges in penalty (not all O(n²))
        3. Add contact-stiffening force toward exact tangency
        4. Optimize w, r, xs, ys with L-BFGS-B
        """
        if check_time():
            return w, h, r, xs, ys

        # Pre-relax with physics
        w, h, r, xs, ys = physics_simulate(w, h, r, xs, ys)
        
        v0 = np.zeros(2 + 2*N)
        v0[0] = w
        v0[1] = r
        v0[2:2+N] = xs
        v0[2+N:2+2*N] = ys

        beta = 80.0 if not is_polish else 200.0
        penalty_weight = 15.0 if not is_polish else 60.0
        contact_weight = 5.0 if not is_polish else 20.0

        def softplus(x, beta_val=80.0):
            return np.where(x > 1.0/beta_val, x, np.log1p(np.exp(beta_val * x)) / beta_val)

        def d_softplus(x, beta_val=80.0):
            z = beta_val * x
            sig = np.where(z > 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
            return sig * beta_val

        def objective(v):
            w_val = v[0]
            r_val = v[1]
            h_val = 2.0 - w_val
            
            if w_val < 0.1 or w_val > 1.9 or r_val < 0.01 or r_val > 1.0:
                return 1e6

            xs_v = v[2:2+N]
            ys_v = v[2+N:2+2*N]
            
            penalty = 0.0
            
            # Boundary violations (all N circles)
            viol_x_left = r_val - xs_v
            viol_x_right = xs_v - (w_val - r_val)
            viol_y_bottom = r_val - ys_v
            viol_y_top = ys_v - (h_val - r_val)
            
            penalty += np.sum(softplus(viol_x_left, beta))
            penalty += np.sum(softplus(viol_x_right, beta))
            penalty += np.sum(softplus(viol_y_bottom, beta))
            penalty += np.sum(softplus(viol_y_top, beta))
            
            # Contact-graph-based overlap penalty
            # Use Delaunay to get relevant pairs
            try:
                pts = np.column_stack([xs_v, ys_v])
                tri = Delaunay(pts)
                edge_set = set()
                for simplex in tri.simplices:
                    for a in range(3):
                        for b in range(a+1, 3):
                            i, j = simplex[a], simplex[b]
                            if i > j:
                                i, j = j, i
                            edge_set.add((i, j))
                
                # Also add boundary-adjacent pairs
                for i in range(N):
                    for j in range(i+1, N):
                        dist = np.sqrt((xs_v[i]-xs_v[j])**2 + (ys_v[i]-ys_v[j])**2)
                        if dist <= 2.0 * r_val * 1.15:
                            if i > j:
                                i, j = j, i
                            edge_set.add((i, j))
                
                edges = list(edge_set)
            except Exception:
                # Fallback: all pairs
                edges = [(i, j) for i in range(N) for j in range(i+1, N)]
            
            if len(edges) > 0:
                idx_i = np.array([e[0] for e in edges])
                idx_j = np.array([e[1] for e in edges])
                dx = xs_v[idx_i] - xs_v[idx_j]
                dy = ys_v[idx_i] - ys_v[idx_j]
                dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                viol = 2.0 * r_val - dist
                penalty += np.sum(softplus(viol, beta))
            
            # Contact-stiffening: reward being exactly at 2r (not just >= 2r)
            # This pushes toward rigid contact networks
            if len(edges) > 0:
                idx_i = np.array([e[0] for e in edges])
                idx_j = np.array([e[1] for e in edges])
                dx = xs_v[idx_i] - xs_v[idx_j]
                dy = ys_v[idx_i] - ys_v[idx_j]
                dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                # Soft penalty for being too far from 2r (only for near-contact pairs)
                near_mask = dist < 2.0 * r_val * 1.05
                if np.any(near_mask):
                    diff = dist[near_mask] - 2.0 * r_val
                    penalty += contact_weight * np.sum(softplus(np.abs(diff) - 0.01, 100.0))
            
            return -r_val + penalty_weight * penalty

        def grad_objective(v):
            w_val = v[0]
            r_val = v[1]
            h_val = 2.0 - w_val
            xs_v = v[2:2+N]
            ys_v = v[2+N:2+2*N]
            
            grad = np.zeros_like(v)
            grad[1] = -1.0
            
            # Boundary gradients
            viol_x_left = r_val - xs_v
            viol_x_right = xs_v - (w_val - r_val)
            viol_y_bottom = r_val - ys_v
            viol_y_top = ys_v - (h_val - r_val)
            
            ds_xl = d_softplus(viol_x_left, beta)
            ds_xr = d_softplus(viol_x_right, beta)
            ds_yb = d_softplus(viol_y_bottom, beta)
            ds_yt = d_softplus(viol_y_top, beta)
            
            grad[2:2+N] -= penalty_weight * ds_xl
            grad[2:2+N] += penalty_weight * ds_xr
            grad[2+N:2+2*N] -= penalty_weight * ds_yb
            grad[2+N:2+2*N] += penalty_weight * ds_yt
            grad[1] += penalty_weight * (np.sum(ds_xl) + np.sum(ds_yb))
            grad[0] -= penalty_weight * np.sum(ds_xr)
            
            # Contact-graph-based overlap gradients
            try:
                pts = np.column_stack([xs_v, ys_v])
                tri = Delaunay(pts)
                edge_set = set()
                for simplex in tri.simplices:
                    for a in range(3):
                        for b in range(a+1, 3):
                            i, j = simplex[a], simplex[b]
                            if i > j:
                                i, j = j, i
                            edge_set.add((i, j))
                
                for i in range(N):
                    for j in range(i+1, N):
                        dist = np.sqrt((xs_v[i]-xs_v[j])**2 + (ys_v[i]-ys_v[j])**2)
                        if dist <= 2.0 * r_val * 1.15:
                            if i > j:
                                i, j = j, i
                            edge_set.add((i, j))
                
                edges = list(edge_set)
            except Exception:
                edges = [(i, j) for i in range(N) for j in range(i+1, N)]
            
            if len(edges) > 0:
                idx_i = np.array([e[0] for e in edges])
                idx_j = np.array([e[1] for e in edges])
                dx = xs_v[idx_i] - xs_v[idx_j]
                dy = ys_v[idx_i] - ys_v[idx_j]
                dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                viol = 2.0 * r_val - dist
                ds = d_softplus(viol, beta)
                
                grad_xi = penalty_weight * ds * (dx / dist)
                grad_xj = -penalty_weight * ds * (dx / dist)
                grad_yi = penalty_weight * ds * (dy / dist)
                grad_yj = -penalty_weight * ds * (dy / dist)
                
                np.add.at(grad[2:2+N], idx_i, -grad_xi)
                np.add.at(grad[2:2+N], idx_j, grad_xj)
                np.add.at(grad[2+N:2+2*N], idx_i, -grad_yi)
                np.add.at(grad[2+N:2+2*N], idx_j, grad_yj)
                grad[1] += penalty_weight * np.sum(ds * 2.0)
                
                # Contact-stiffening gradient
                near_mask = dist < 2.0 * r_val * 1.05
                if np.any(near_mask):
                    idx_i_n = idx_i[near_mask]
                    idx_j_n = idx_j[near_mask]
                    dx_n = dx[near_mask]
                    dy_n = dy[near_mask]
                    dist_n = dist[near_mask]
                    diff = dist_n - 2.0 * r_val
                    abs_diff = np.abs(diff)
                    viol_cs = abs_diff - 0.01
                    ds_cs = d_softplus(viol_cs, 100.0)
                    sign_diff = np.sign(diff)
                    
                    grad_xi_cs = contact_weight * ds_cs * sign_diff * (dx_n / dist_n)
                    grad_xj_cs = -contact_weight * ds_cs * sign_diff * (dx_n / dist_n)
                    grad_yi_cs = contact_weight * ds_cs * sign_diff * (dy_n / dist_n)
                    grad_yj_cs = -contact_weight * ds_cs * sign_diff * (dy_n / dist_n)
                    
                    np.add.at(grad[2:2+N], idx_i_n, -grad_xi_cs)
                    np.add.at(grad[2:2+N], idx_j_n, grad_xj_cs)
                    np.add.at(grad[2+N:2+2*N], idx_i_n, -grad_yi_cs)
                    np.add.at(grad[2+N:2+2*N], idx_j_n, grad_yj_cs)
                    # r gradient from contact stiffening: -2 * sign * ds_cs
                    grad[1] -= contact_weight * np.sum(ds_cs * sign_diff * 2.0)
            
            return grad

        bounds = [(0.2, 1.8), (0.01, 1.0)] + [(0.0, 2.0)] * (2*N)

        best_v = v0.copy()
        best_fun = objective(v0)

        try:
            if is_polish:
                options = {'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14, 'disp': False, 'maxls': 50}
            else:
                options = {'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-10, 'disp': False, 'maxls': 50}

            res = minimize(objective, v0, method='L-BFGS-B', jac=grad_objective, bounds=bounds, options=options)
            
            if res.fun < best_fun:
                best_v = res.x.copy()
                best_fun = res.fun

            # Second phase: re-optimize with updated contact graph
            if not check_time():
                # Extract current state
                w_cur = best_v[0]
                r_cur = best_v[1]
                xs_cur = best_v[2:2+N]
                ys_cur = best_v[2+N:2+2*N]
                
                # Rebuild contact graph with tighter tolerance
                contact_pairs = get_contact_graph(xs_cur, ys_cur, r_cur, w_cur, 2.0 - w_cur, tolerance=0.02)
                
                if len(contact_pairs) > 0:
                    idx_i2 = np.array([p[0] for p in contact_pairs])
                    idx_j2 = np.array([p[1] for p in contact_pairs])
                    
                    def objective2(v):
                        w_val = v[0]
                        r_val = v[1]
                        h_val = 2.0 - w_val
                        
                        if w_val < 0.1 or w_val > 1.9 or r_val < 0.01 or r_val > 1.0:
                            return 1e6

                        xs_v = v[2:2+N]
                        ys_v = v[2+N:2+2*N]
                        
                        penalty = 0.0
                        
                        # Boundary violations
                        viol_x_left = r_val - xs_v
                        viol_x_right = xs_v - (w_val - r_val)
                        viol_y_bottom = r_val - ys_v
                        viol_y_top = ys_v - (h_val - r_val)
                        
                        penalty += np.sum(softplus(viol_x_left, beta))
                        penalty += np.sum(softplus(viol_x_right, beta))
                        penalty += np.sum(softplus(viol_y_bottom, beta))
                        penalty += np.sum(softplus(viol_y_top, beta))
                        
                        # Fixed contact pair overlap
                        dx = xs_v[idx_i2] - xs_v[idx_j2]
                        dy = ys_v[idx_i2] - ys_v[idx_j2]
                        dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                        viol = 2.0 * r_val - dist
                        penalty += np.sum(softplus(viol, beta))
                        
                        # Contact stiffening
                        near_mask = dist < 2.0 * r_val * 1.05
                        if np.any(near_mask):
                            diff = dist[near_mask] - 2.0 * r_val
                            penalty += contact_weight * np.sum(softplus(np.abs(diff) - 0.01, 100.0))
                        
                        return -r_val + penalty_weight * penalty

                    def grad_objective2(v):
                        w_val = v[0]
                        r_val = v[1]
                        h_val = 2.0 - w_val
                        xs_v = v[2:2+N]
                        ys_v = v[2+N:2+2*N]
                        
                        grad = np.zeros_like(v)
                        grad[1] = -1.0
                        
                        viol_x_left = r_val - xs_v
                        viol_x_right = xs_v - (w_val - r_val)
                        viol_y_bottom = r_val - ys_v
                        viol_y_top = ys_v - (h_val - r_val)
                        
                        ds_xl = d_softplus(viol_x_left, beta)
                        ds_xr = d_softplus(viol_x_right, beta)
                        ds_yb = d_softplus(viol_y_bottom, beta)
                        ds_yt = d_softplus(viol_y_top, beta)
                        
                        grad[2:2+N] -= penalty_weight * ds_xl
                        grad[2:2+N] += penalty_weight * ds_xr
                        grad[2+N:2+2*N] -= penalty_weight * ds_yb
                        grad[2+N:2+2*N] += penalty_weight * ds_yt
                        grad[1] += penalty_weight * (np.sum(ds_xl) + np.sum(ds_yb))
                        grad[0] -= penalty_weight * np.sum(ds_xr)
                        
                        dx = xs_v[idx_i2] - xs_v[idx_j2]
                        dy = ys_v[idx_i2] - ys_v[idx_j2]
                        dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                        viol = 2.0 * r_val - dist
                        ds = d_softplus(viol, beta)
                        
                        grad_xi = penalty_weight * ds * (dx / dist)
                        grad_xj = -penalty_weight * ds * (dx / dist)
                        grad_yi = penalty_weight * ds * (dy / dist)
                        grad_yj = -penalty_weight * ds * (dy / dist)
                        
                        np.add.at(grad[2:2+N], idx_i2, -grad_xi)
                        np.add.at(grad[2:2+N], idx_j2, grad_xj)
                        np.add.at(grad[2+N:2+2*N], idx_i2, -grad_yi)
                        np.add.at(grad[2+N:2+2*N], idx_j2, grad_yj)
                        grad[1] += penalty_weight * np.sum(ds * 2.0)
                        
                        near_mask = dist < 2.0 * r_val * 1.05
                        if np.any(near_mask):
                            idx_i_n = idx_i2[near_mask]
                            idx_j_n = idx_j2[near_mask]
                            dx_n = dx[near_mask]
                            dy_n = dy[near_mask]
                            dist_n = dist[near_mask]
                            diff = dist_n - 2.0 * r_val
                            abs_diff = np.abs(diff)
                            viol_cs = abs_diff - 0.01
                            ds_cs = d_softplus(viol_cs, 100.0)
                            sign_diff = np.sign(diff)
                            
                            grad_xi_cs = contact_weight * ds_cs * sign_diff * (dx_n / dist_n)
                            grad_xj_cs = -contact_weight * ds_cs * sign_diff * (dx_n / dist_n)
                            grad_yi_cs = contact_weight * ds_cs * sign_diff * (dy_n / dist_n)
                            grad_yj_cs = -contact_weight * ds_cs * sign_diff * (dy_n / dist_n)
                            
                            np.add.at(grad[2:2+N], idx_i_n, -grad_xi_cs)
                            np.add.at(grad[2:2+N], idx_j_n, grad_xj_cs)
                            np.add.at(grad[2+N:2+2*N], idx_i_n, -grad_yi_cs)
                            np.add.at(grad[2+N:2+2*N], idx_j_n, grad_yj_cs)
                            grad[1] -= contact_weight * np.sum(ds_cs * sign_diff * 2.0)
                        
                        return grad

                    options2 = {'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-14, 'disp': False, 'maxls': 50}
                    res2 = minimize(objective2, best_v, method='L-BFGS-B', jac=grad_objective2, bounds=bounds, options=options2)
                    
                    if res2.fun < best_fun:
                        best_v = res2.x.copy()
                        best_fun = res2.fun

        except Exception:
            pass

        # Extract and validate result
        w_res = best_v[0]
        r_res = best_v[1]
        h_res = 2.0 - w_res
        xs_res = best_v[2:2+N]
        ys_res = best_v[2+N:2+2*N]
        
        valid = True
        if w_res <= 0 or h_res <= 0 or r_res <= 0:
            valid = False
        else:
            for i in range(N):
                if xs_res[i] < r_res - 1e-7 or xs_res[i] > w_res - r_res + 1e-7:
                    valid = False
                    break
                if ys_res[i] < r_res - 1e-7 or ys_res[i] > h_res - r_res + 1e-7:
                    valid = False
                    break
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        dist_sq = (xs_res[i]-xs_res[j])**2 + (ys_res[i]-ys_res[j])**2
                        if dist_sq < 4*(r_res**2) - 1e-7:
                            valid = False
                            break
                    if not valid:
                        break
        
        if valid and r_res > r:
            return w_res, h_res, r_res, xs_res, ys_res
        else:
            return w, h, r, xs, ys

    def refine_smooth(w, h, r, xs, ys, is_polish=False):
        """
        Refine the packing using L-BFGS-B with a smooth softplus penalty for constraints.
        Variables: w, r, xs, ys. h = 2 - w.
        Objective: -r + penalty
        Vectorized overlap computation for ~5x speedup.
        """
        if check_time():
            return w, h, r, xs, ys

        # Pre-relax with physics
        w, h, r, xs, ys = physics_simulate(w, h, r, xs, ys)
        
        v0 = np.zeros(2 + 2*N)
        v0[0] = w
        v0[1] = r
        v0[2:2+N] = xs
        v0[2+N:2+2*N] = ys

        # Softplus beta: larger is sharper (closer to hard constraint)
        # Start with moderate beta, increase during polish
        beta = 50.0 if not is_polish else 200.0
        penalty_weight = 10.0 if not is_polish else 50.0

        # Precompute index pairs for vectorized overlap
        idx_i, idx_j = np.triu_indices(N, k=1)

        def softplus(x, beta=50.0):
            # Numerically stable softplus
            return np.where(x > 1.0/beta, x, np.log1p(np.exp(beta * x)) / beta)

        def objective(v):
            w_val = v[0]
            r_val = v[1]
            h_val = 2.0 - w_val
            
            if w_val < 0.1 or w_val > 1.9 or r_val < 0.01 or r_val > 1.0:
                return 1e6

            xs = v[2:2+N]
            ys = v[2+N:2+2*N]
            
            penalty = 0.0
            
            # Boundary violations
            viol_x_left = r_val - xs
            viol_x_right = xs - (w_val - r_val)
            viol_y_bottom = r_val - ys
            viol_y_top = ys - (h_val - r_val)
            
            penalty += np.sum(softplus(viol_x_left, beta))
            penalty += np.sum(softplus(viol_x_right, beta))
            penalty += np.sum(softplus(viol_y_bottom, beta))
            penalty += np.sum(softplus(viol_y_top, beta))
            
            # Vectorized overlap violations
            dx = xs[idx_i] - xs[idx_j]
            dy = ys[idx_i] - ys[idx_j]
            dist = np.sqrt(dx*dx + dy*dy + 1e-12)
            viol = 2.0 * r_val - dist
            penalty += np.sum(softplus(viol, beta))
            
            return -r_val + penalty_weight * penalty

        def grad_objective(v):
            w_val = v[0]
            r_val = v[1]
            h_val = 2.0 - w_val
            xs = v[2:2+N]
            ys = v[2+N:2+2*N]
            
            grad = np.zeros_like(v)
            grad[1] = -1.0
            
            def d_softplus(x, beta=50.0):
                z = beta * x
                sig = np.where(z > 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
                return sig * beta

            # Boundary gradients
            viol_x_left = r_val - xs
            viol_x_right = xs - (w_val - r_val)
            viol_y_bottom = r_val - ys
            viol_y_top = ys - (h_val - r_val)
            
            ds_xl = d_softplus(viol_x_left, beta)
            ds_xr = d_softplus(viol_x_right, beta)
            ds_yb = d_softplus(viol_y_bottom, beta)
            ds_yt = d_softplus(viol_y_top, beta)
            
            grad[2:2+N] -= penalty_weight * ds_xl
            grad[2:2+N] += penalty_weight * ds_xr
            grad[2+N:2+2*N] -= penalty_weight * ds_yb
            grad[2+N:2+2*N] += penalty_weight * ds_yt
            grad[1] += penalty_weight * (np.sum(ds_xl) + np.sum(ds_yb))
            grad[0] -= penalty_weight * np.sum(ds_xr)
            
            # Vectorized overlap gradients
            dx = xs[idx_i] - xs[idx_j]
            dy = ys[idx_i] - ys[idx_j]
            dist = np.sqrt(dx*dx + dy*dy + 1e-12)
            viol = 2.0 * r_val - dist
            ds = d_softplus(viol, beta)
            
            # Accumulate gradients for each pair
            grad_xi = penalty_weight * ds * (dx / dist)
            grad_xj = -penalty_weight * ds * (dx / dist)
            grad_yi = penalty_weight * ds * (dy / dist)
            grad_yj = -penalty_weight * ds * (dy / dist)
            
            np.add.at(grad[2:2+N], idx_i, -grad_xi)
            np.add.at(grad[2:2+N], idx_j, grad_xj)
            np.add.at(grad[2+N:2+2*N], idx_i, -grad_yi)
            np.add.at(grad[2+N:2+2*N], idx_j, grad_yj)
            grad[1] += penalty_weight * np.sum(ds * 2.0)
            
            return grad

        bounds = [(0.2, 1.8), (0.01, 1.0)] + [(0.0, 2.0)] * (2*N)

        try:
            if is_polish:
                options = {'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-14, 'disp': False, 'maxls': 50}
            else:
                options = {'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-10, 'disp': False, 'maxls': 50}

            res = minimize(objective, v0, method='L-BFGS-B', jac=grad_objective, bounds=bounds, options=options)
            
            if res.fun < -r + 1e-6:
                w_res = res.x[0]
                r_res = res.x[1]
                h_res = 2.0 - w_res
                xs_res = res.x[2:2+N]
                ys_res = res.x[2+N:2+2*N]
                
                valid = True
                if w_res <= 0 or h_res <= 0 or r_res <= 0:
                    valid = False
                else:
                    for i in range(N):
                        if xs_res[i] < r_res - 1e-7 or xs_res[i] > w_res - r_res + 1e-7:
                            valid = False
                            break
                        if ys_res[i] < r_res - 1e-7 or ys_res[i] > h_res - r_res + 1e-7:
                            valid = False
                            break
                    if valid:
                        dx = xs_res[idx_i] - xs_res[idx_j]
                        dy = ys_res[idx_i] - ys_res[idx_j]
                        dist_sq = dx*dx + dy*dy
                        if np.any(dist_sq < 4*(r_res**2) - 1e-7):
                            valid = False
                
                if valid:
                    return w_res, h_res, r_res, xs_res, ys_res
        except Exception:
            pass
            
        return w, h, r, xs, ys

    best_r = 0
    best_config = None

    configs = []
    
    # Hexagonal variations
    configs.append(pack_hexagonal([7, 7, 7]))
    configs.append(pack_hexagonal([4, 5, 4, 4, 4]))
    configs.append(pack_hexagonal([3, 4, 4, 4, 4, 2]))
    configs.append(pack_hexagonal([5, 5, 5, 3, 3]))
    configs.append(pack_hexagonal([6, 6, 5, 4]))
    
    # Grid variations
    configs.append(pack_grid(5, 5))
    configs.append(pack_grid(3, 7))
    configs.append(pack_grid(4, 6))
    configs.append(pack_grid(7, 3))
    configs.append(pack_grid(6, 4))
    
    # Random restarts
    for _ in range(10):
        if check_time():
            break
        w = 1.0
        h = 1.0
        r = 0.08
        xs = np.random.uniform(r, w-r, N)
        ys = np.random.uniform(r, h-r, N)
        configs.append((w, h, r, xs, ys))

    valid_configs = [c for c in configs if c is not None]

    # First pass: Contact-graph optimization (primary method)
    for cfg in valid_configs:
        if check_time():
            break
        w, h, r, xs, ys = cfg
        w_new, h_new, r_new, xs_new, ys_new = refine_contact(w, h, r, xs, ys, is_polish=False)
        
        if r_new > best_r:
            best_r = r_new
            best_config = (w_new, h_new, r_new, xs_new, ys_new)

    # Second pass: Polish best with contact method
    if best_config is not None and not check_time():
        w, h, r, xs, ys = best_config
        w_new, h_new, r_new, xs_new, ys_new = refine_contact(w, h, r, xs, ys, is_polish=True)
        
        if r_new > best_r:
            best_r = r_new
            best_config = (w_new, h_new, r_new, xs_new, ys_new)

    # Third pass: Try smooth method on best as alternative
    if best_config is not None and not check_time():
        w, h, r, xs, ys = best_config
        w_new, h_new, r_new, xs_new, ys_new = refine_smooth(w, h, r, xs, ys, is_polish=True)
        
        if r_new > best_r:
            best_r = r_new
            best_config = (w_new, h_new, r_new, xs_new, ys_new)

    # Fourth pass: Final polish if time remains
    if best_config is not None and not check_time() and (deadline - time.time()) > 15.0:
        w, h, r, xs, ys = best_config
        w_new, h_new, r_new, xs_new, ys_new = refine_contact(w, h, r, xs, ys, is_polish=True)
        
        if r_new > best_r:
            best_r = r_new
            best_config = (w_new, h_new, r_new, xs_new, ys_new)

    if best_config is None:
        # Fallback
        w, h, r, xs, ys = valid_configs[0]
        return np.column_stack([xs, ys, np.full(N, r)])

    w, h, r, xs, ys = best_config
    
    # Final safety adjustment: tiny shrink to guarantee validity under floating-point
    r_final = r * (1 - 1e-6)
    
    return np.column_stack([xs, ys, np.full(N, r_final)])
