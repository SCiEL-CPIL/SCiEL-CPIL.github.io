import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s

    def time_left():
        return deadline - time.time()

    best_value = -np.inf
    best_result = None
    best_wh = None

    def _evaluate(circles, w, h, tol=1e-7):
        """Return sum of radii if valid, else -inf."""
        n = len(circles)
        if n == 0:
            return -np.inf
        xs = circles[:, 0]
        ys = circles[:, 1]
        rs = circles[:, 2]
        if np.any(rs < -tol):
            return -np.inf
        if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
            return -np.inf
        if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
            return -np.inf
        for i in range(n):
            dx = xs[i] - xs[i + 1:]
            dy = ys[i] - ys[i + 1:]
            d2 = dx * dx + dy * dy
            need = rs[i] + rs[i + 1:] - tol
            if np.any(d2 < need * need):
                return -np.inf
        return float(np.sum(rs))

    def _softplus(x, beta=100.0):
        xb = beta * x
        return np.where(xb > 0,
                        x + np.log1p(np.exp(-xb)) / beta,
                        np.log1p(np.exp(xb)) / beta)

    def _softplus_grad(x, beta=100.0):
        return 1.0 / (1.0 + np.exp(-beta * x))

    def _hex_init(w, h, seed=0):
        """Generate hexagonal lattice initial packing."""
        rng = np.random.RandomState(seed)
        
        # Try different row configurations that sum to N
        row_configs = [
            [4, 4, 4, 4, 3, 2],
            [3, 4, 4, 4, 3, 3],
            [4, 4, 4, 4, 4, 1],
            [3, 3, 4, 4, 4, 3],
            [4, 3, 4, 4, 4, 2],
            [2, 4, 4, 4, 4, 3],
            [3, 4, 4, 4, 4, 2],
            [4, 4, 3, 4, 4, 2],
            [2, 4, 3, 4, 4, 4],
            [3, 4, 3, 4, 4, 3],
            [5, 4, 4, 4, 4],
            [4, 5, 4, 4, 4],
            [3, 5, 4, 4, 5],
            [4, 4, 5, 4, 4],
            [2, 5, 4, 5, 5],
            [5, 5, 5, 5, 1],
            [1, 5, 5, 5, 5],
            [3, 4, 5, 4, 5],
            [5, 4, 5, 4, 3],
            [4, 4, 4, 4, 4, 1],
            [1, 4, 4, 4, 4, 4],
            [3, 4, 4, 4, 4, 2],
            [2, 4, 4, 4, 4, 3],
            [4, 4, 4, 3, 4, 2],
            [3, 3, 3, 4, 4, 4],
            [4, 4, 4, 3, 3, 3],
        ]
        
        best = None
        best_val = -np.inf
        
        for row_counts in row_configs:
            if sum(row_counts) != N:
                continue
            
            n_rows = len(row_counts)
            row_h = h / n_rows
            
            circles = []
            for row_idx, count in enumerate(row_counts):
                # Offset alternate rows for hexagonal packing
                offset = 0.5 * (row_h * np.sqrt(3) / 2) * (row_idx % 2)
                cell_w = w / count
                for col in range(count):
                    cx = (col + 0.5) * cell_w + offset
                    cy = (row_idx + 0.5) * row_h
                    rad = 0.5 * min(cell_w, row_h) * 0.85 - 1e-4
                    if rad > 0:
                        circles.append([cx, cy, rad])
            
            if len(circles) != N:
                continue
            
            circles = np.array(circles)
            val = _evaluate(circles, w, h)
            if val > best_val:
                best_val = val
                best = circles.copy()
        
        if best is None:
            # Fallback: simple grid
            cols = int(np.ceil(np.sqrt(N * w / h)))
            rows = int(np.ceil(N / cols))
            cell_w = w / cols
            cell_h = h / rows
            circles = []
            for i in range(N):
                c = i % cols
                r = i // cols
                cx = (c + 0.5) * cell_w
                cy = (r + 0.5) * cell_h
                rad = 0.5 * min(cell_w, cell_h) - 1e-4
                circles.append([cx, cy, rad])
            circles = np.array(circles)
            best = circles
            best_val = _evaluate(circles, w, h)
        
        return best, best_val

    def _delaunay_neighbors(centers):
        """Get Delaunay neighbors for fast validity checking."""
        try:
            tri = Delaunay(centers)
            nbrs = [set() for _ in range(N)]
            for i, j, k in tri.simplices:
                nbrs[i].add(j); nbrs[i].add(k)
                nbrs[j].add(i); nbrs[j].add(k)
                nbrs[k].add(i); nbrs[k].add(j)
            return [sorted(s) for s in nbrs]
        except Exception:
            return [list(range(N)) for _ in range(N)]

    def _objective_joint(x, beta=100.0, eq_weight=0.02):
        """
        Objective with w as free variable; h = 2 - w.
        x = [w, x0, y0, r0, x1, y1, r1, ...]
        Total length = 1 + 3*N = 64
        """
        w = x[0]
        h = 2.0 - w
        
        # Ensure w and h are positive
        if w < 0.1 or h < 0.1:
            return 1e6, np.zeros_like(x)
        
        c = x[1:].reshape(-1, 3)
        xs = c[:, 0]
        ys = c[:, 1]
        rs = c[:, 2]

        obj = -np.sum(rs)
        grad = np.zeros_like(x)
        grad_w = 0.0

        # Equality constraint: encourage similar radii (optional, helps convergence)
        if eq_weight > 0:
            r_mean = np.mean(rs)
            r_dev = rs - r_mean
            obj += eq_weight * np.sum(r_dev * r_dev)
            grad[3::3] += -2.0 * eq_weight * r_dev

        # Boundary constraints
        v1 = rs - xs  # xi - ri >= 0
        obj += np.sum(_softplus(v1, beta))
        g1 = _softplus_grad(v1, beta)
        grad[1::3] -= g1
        grad[3::3] += g1

        v2 = xs + rs - w  # xi + ri <= w
        obj += np.sum(_softplus(v2, beta))
        g2 = _softplus_grad(v2, beta)
        grad[1::3] += g2
        grad[3::3] += g2
        grad_w -= np.sum(g2)

        v3 = rs - ys  # yi - ri >= 0
        obj += np.sum(_softplus(v3, beta))
        g3 = _softplus_grad(v3, beta)
        grad[2::3] -= g3
        grad[3::3] += g3

        v4 = ys + rs - h  # yi + ri <= h
        obj += np.sum(_softplus(v4, beta))
        g4 = _softplus_grad(v4, beta)
        grad[2::3] += g4
        grad[3::3] += g4
        grad_w += np.sum(g4)

        v5 = -rs  # ri >= 0
        obj += np.sum(_softplus(v5, beta))
        g5 = _softplus_grad(v5, beta)
        grad[3::3] -= g5

        # Pairwise non-overlap
        for i in range(N):
            dx = xs[i] - xs[i + 1:]
            dy = ys[i] - ys[i + 1:]
            d = np.sqrt(dx * dx + dy * dy + 1e-12)
            v = rs[i] + rs[i + 1:] - d
            obj += np.sum(_softplus(v, beta))
            g = _softplus_grad(v, beta)
            
            grad[1 + 3 * i] += -np.sum(g * dx / d)
            grad[2 + 3 * i] += -np.sum(g * dy / d)
            grad[3 + 3 * i] += np.sum(g)
            
            idx = np.arange(i + 1, N)
            grad[1 + 3 * idx] += g * dx / d
            grad[2 + 3 * idx] += g * dy / d
            grad[3 + 3 * idx] += g

        grad[0] = grad_w
        grad[3::3] += -1.0  # Gradient of -sum(ri)
        
        return obj, grad

    def _optimize_joint(circles0, w0, h0, max_iter=200, beta=100.0):
        """Optimize with joint aspect ratio."""
        x0 = np.concatenate([[w0], circles0.flatten()])
        
        result = minimize(_objective_joint, x0, method='L-BFGS-B', jac=True,
                         options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
        
        w_opt = result.x[0]
        h_opt = 2.0 - w_opt
        circles = result.x[1:].reshape(-1, 3)
        
        val = _evaluate(circles, w_opt, h_opt)
        return circles, w_opt, h_opt, val

    def _polish(circles0, w, h, max_iter=300, beta=200.0):
        """High-precision polish with fixed aspect ratio."""
        x0 = circles0.flatten()
        
        def obj_fixed(x):
            c = x.reshape(-1, 3)
            xs = c[:, 0]
            ys = c[:, 1]
            rs = c[:, 2]
            
            obj = -np.sum(rs)
            grad = np.zeros_like(x)
            
            v1 = rs - xs
            obj += np.sum(_softplus(v1, beta))
            g1 = _softplus_grad(v1, beta)
            grad[0::3] -= g1
            grad[2::3] += g1

            v2 = xs + rs - w
            obj += np.sum(_softplus(v2, beta))
            g2 = _softplus_grad(v2, beta)
            grad[0::3] += g2
            grad[2::3] += g2

            v3 = rs - ys
            obj += np.sum(_softplus(v3, beta))
            g3 = _softplus_grad(v3, beta)
            grad[1::3] -= g3
            grad[2::3] += g3

            v4 = ys + rs - h
            obj += np.sum(_softplus(v4, beta))
            g4 = _softplus_grad(v4, beta)
            grad[1::3] += g4
            grad[2::3] += g4

            v5 = -rs
            obj += np.sum(_softplus(v5, beta))
            g5 = _softplus_grad(v5, beta)
            grad[2::3] -= g5

            for i in range(N):
                dx = xs[i] - xs[i + 1:]
                dy = ys[i] - ys[i + 1:]
                d = np.sqrt(dx * dx + dy * dy + 1e-12)
                v = rs[i] + rs[i + 1:] - d
                obj += np.sum(_softplus(v, beta))
                g = _softplus_grad(v, beta)
                
                grad[3 * i] += -np.sum(g * dx / d)
                grad[3 * i + 1] += -np.sum(g * dy / d)
                grad[3 * i + 2] += np.sum(g)
                
                idx = np.arange(i + 1, N)
                grad[3 * idx] += g * dx / d
                grad[3 * idx + 1] += g * dy / d
                grad[3 * idx + 2] += g

            grad[2::3] += -1.0
            return obj, grad
        
        result = minimize(obj_fixed, x0, method='L-BFGS-B', jac=True,
                         options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
        
        circles = result.x.reshape(-1, 3)
        val = _evaluate(circles, w, h)
        return circles, val

    # Try multiple aspect ratios with joint optimization
    aspects = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.5, 0.5, 2.0, 0.4, 2.5, 0.3, 3.0, 4.0, 0.25]
    
    for aspect in aspects:
        if time_left() < 10:
            break
        
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 / (1.0 + aspect)
        
        circles0, val0 = _hex_init(w, h, seed=aspect * 100)
        
        if val0 > best_value:
            best_value = val0
            best_result = circles0.copy()
            best_wh = (w, h)
        
        # Joint optimization
        circles, w_opt, h_opt, val = _optimize_joint(circles0, w, h, max_iter=150, beta=100.0)
        
        if val > best_value:
            best_value = val
            best_result = circles.copy()
            best_wh = (w_opt, h_opt)
    
    # Polish the best result
    if best_result is not None and best_wh is not None and time_left() > 5:
        w, h = best_wh
        polished, val = _polish(best_result, w, h, max_iter=300, beta=200.0)
        if val > best_value:
            best_value = val
            best_result = polished.copy()
    
    # Final check
    if best_result is not None and best_wh is not None:
        val = _evaluate(best_result, best_wh[0], best_wh[1])
        if val > best_value:
            best_value = val
    
    if best_result is None:
        # Fallback
        circles, _, _, _ = _hex_init(1.0, 1.0, 0)
        best_result = circles
    
    return best_result
