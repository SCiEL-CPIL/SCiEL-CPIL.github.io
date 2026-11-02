"""Program for packing 50 equal circles in a unit square; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve, lsqr
from scipy.optimize import minimize

def construct():
    """
    Rigid-body assembly approach: build contact graph, solve sparse linear systems
    to satisfy exact distance constraints, iterate to find optimal packing.
    """
    
    def initial_hex(n):
        """Hexagonal lattice initial guess."""
        pts = []
        for row in range(12):
            for col in range(12):
                x = col + 0.5 * (row % 2)
                y = row * np.sqrt(3) / 2.0
                pts.append([x, y])
        pts = np.array(pts[:n])
        pts -= pts.mean(axis=0)
        max_abs = np.max(np.abs(pts))
        if max_abs > 0:
            pts *= (0.45 / max_abs)
        return pts
    
    def initial_grid(n):
        """Grid initial guess."""
        m = int(np.ceil(np.sqrt(n)))
        xs, ys = np.meshgrid(np.linspace(-0.45, 0.45, m), np.linspace(-0.45, 0.45, m))
        pts = np.column_stack([xs.ravel(), ys.ravel()])[:n]
        return pts
    
    def compute_radius(pts):
        """Compute the packing radius."""
        dists = np.sqrt(((pts[:, np.newaxis] - pts[np.newaxis]) ** 2).sum(-1))
        np.fill_diagonal(dists, np.inf)
        r_pair = np.min(dists) / 2.0
        r_bound = 0.5 - np.max(np.abs(pts), axis=1)
        r_bound = np.min(r_bound)
        return min(r_pair, r_bound)
    
    def build_contact_graph(pts, r, threshold_factor=1.5):
        """
        Build contact graph: pairs of circles that are close (dist < threshold_factor * 2r)
        and boundary contacts. Returns list of tuples.
        """
        contacts = []
        threshold = threshold_factor * 2 * r
        
        # Circle-circle contacts
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < threshold:
                    contacts.append((i, j, 'cc'))
        
        # Boundary contacts
        for i in range(N):
            x, y = pts[i]
            d_left = x + 0.5
            d_right = 0.5 - x
            d_bottom = y + 0.5
            d_top = 0.5 - y
            
            if d_left < threshold:
                contacts.append((i, -1, 'bl'))
            if d_right < threshold:
                contacts.append((i, -2, 'br'))
            if d_bottom < threshold:
                contacts.append((i, -3, 'bb'))
            if d_top < threshold:
                contacts.append((i, -4, 'bt'))
        
        return contacts
    
    def solve_assembly(pts, r, max_iters=200, tol=1e-10):
        """
        Solve the rigid-body assembly problem: adjust positions to satisfy
        distance constraints for contact pairs using sparse linear solver.
        """
        pts = pts.copy()
        
        for iteration in range(max_iters):
            contacts = build_contact_graph(pts, r)
            
            if len(contacts) == 0:
                break
            
            n_vars = 2 * N
            A_rows = []
            A_cols = []
            A_vals = []
            b = np.zeros(len(contacts))
            
            for idx, (i, j, typ) in enumerate(contacts):
                if typ == 'cc':
                    v = pts[i] - pts[j]
                    d = np.linalg.norm(v)
                    if d < 1e-10:
                        continue
                    rhs = d * (2 * r - d)
                    A_rows.append(idx)
                    A_cols.append(2*i)
                    A_vals.append(v[0])
                    A_rows.append(idx)
                    A_cols.append(2*i+1)
                    A_vals.append(v[1])
                    A_rows.append(idx)
                    A_cols.append(2*j)
                    A_vals.append(-v[0])
                    A_rows.append(idx)
                    A_cols.append(2*j+1)
                    A_vals.append(-v[1])
                    b[idx] = rhs
                elif typ in ('bl', 'br', 'bb', 'bt'):
                    x, y = pts[i]
                    if typ == 'bl':
                        rhs = (-0.5 + r) - x
                        A_rows.append(idx)
                        A_cols.append(2*i)
                        A_vals.append(1.0)
                        b[idx] = rhs
                    elif typ == 'br':
                        rhs = (0.5 - r) - x
                        A_rows.append(idx)
                        A_cols.append(2*i)
                        A_vals.append(1.0)
                        b[idx] = rhs
                    elif typ == 'bb':
                        rhs = (-0.5 + r) - y
                        A_rows.append(idx)
                        A_cols.append(2*i+1)
                        A_vals.append(1.0)
                        b[idx] = rhs
                    elif typ == 'bt':
                        rhs = (0.5 - r) - y
                        A_rows.append(idx)
                        A_cols.append(2*i+1)
                        A_vals.append(1.0)
                        b[idx] = rhs
            
            if len(A_rows) == 0:
                break
            
            A = csr_matrix((A_vals, (A_rows, A_cols)), shape=(len(contacts), n_vars))
            
            try:
                dp, resid, iter_count, istop = lsqr(A, b, atol=1e-12, btol=1e-12, itermax=1000)
            except Exception:
                break
            
            step_size = 0.5
            pts += step_size * dp.reshape(N, 2)
            
            bound = 0.5 - r
            pts = np.clip(pts, -bound, bound)
            
            max_disp = np.max(np.abs(dp))
            if max_disp < tol:
                break
        
        return pts
    
    def local_optimize(pts, r, max_iters=100):
        """
        Local optimization: alternate between contact graph update and assembly solve.
        """
        pts = pts.copy()
        
        for _ in range(max_iters):
            new_pts = solve_assembly(pts, r, max_iters=50, tol=1e-9)
            
            current_r = compute_radius(new_pts)
            if current_r < r - 1e-8:
                r = current_r * 0.99
                break
            else:
                pts = new_pts
        
        return pts, compute_radius(pts)
    
    def binary_search_radius(pts_init, r_lo, r_hi, max_binary_iters=25):
        """Binary search for maximum feasible radius."""
        best_pts = pts_init.copy()
        best_r = r_lo
        
        for _ in range(max_binary_iters):
            r_mid = (r_lo + r_hi) / 2
            
            pts_candidate = solve_assembly(pts_init, r_mid, max_iters=100, tol=1e-9)
            pts_candidate = np.clip(pts_candidate, -(0.5 - r_mid), (0.5 - r_mid))
            
            dists = np.sqrt(((pts_candidate[:, np.newaxis] - pts_candidate[np.newaxis]) ** 2).sum(-1))
            np.fill_diagonal(dists, np.inf)
            min_dist = np.min(dists)
            min_bound = 0.5 - np.max(np.abs(pts_candidate), axis=1)
            min_bound = np.min(min_bound)
            
            if min_dist >= 2 * r_mid - 1e-7 and min_bound >= r_mid - 1e-7:
                r_lo = r_mid
                best_pts = pts_candidate.copy()
                best_r = r_mid
            else:
                r_hi = r_mid
            
            if r_hi - r_lo < 1e-6:
                break
        
        return best_pts, best_r
    
    configs = [
        initial_hex(N),
        initial_grid(N)
    ]
    
    rng = np.random.default_rng(42)
    for _ in range(4):
        base = initial_hex(N).copy()
        jitter = rng.normal(0, 0.03, size=(N, 2))
        cfg = base + jitter
        cfg = np.clip(cfg, -0.48, 0.48)
        configs.append(cfg)
    
    best_pos = None
    best_r = 0.0
    
    for cfg in configs:
        r0 = compute_radius(cfg)
        
        r_lo = r0 * 0.9
        r_hi = min(r0 * 1.15, 0.075)
        
        pts_lo = solve_assembly(cfg, r_lo, max_iters=50, tol=1e-8)
        pts_lo = np.clip(pts_lo, -(0.5 - r_lo), (0.5 - r_lo))
        
        dists = np.sqrt(((pts_lo[:, np.newaxis] - pts_lo[np.newaxis]) ** 2).sum(-1))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        min_bound = 0.5 - np.max(np.abs(pts_lo), axis=1)
        min_bound = np.min(min_bound)
        
        if min_dist < 2 * r_lo - 1e-6 or min_bound < r_lo - 1e-6:
            r_lo = r0 * 0.85
            pts_lo = solve_assembly(cfg, r_lo, max_iters=50, tol=1e-8)
            pts_lo = np.clip(pts_lo, -(0.5 - r_lo), (0.5 - r_lo))
        
        pos_candidate, r_candidate = binary_search_radius(pts_lo, r_lo, r_hi, max_binary_iters=20)
        
        if r_candidate > best_r:
            best_r = r_candidate
            best_pos = pos_candidate.copy()
    
    if best_pos is not None:
        refined_pts, refined_r = local_optimize(best_pos, best_r, max_iters=50)
        if refined_r > best_r:
            best_r = refined_r
            best_pos = refined_pts
    
    if best_pos is None or np.any(~np.isfinite(best_pos)):
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_pos = grid[:N]
    
    best_pos = np.clip(best_pos, -0.5, 0.5)
    
    dists = np.sqrt(((best_pos[:, np.newaxis] - best_pos[np.newaxis]) ** 2).sum(-1))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-10:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_pos = grid[:N]
    
    return best_pos
# EVOLVE-BLOCK-END
