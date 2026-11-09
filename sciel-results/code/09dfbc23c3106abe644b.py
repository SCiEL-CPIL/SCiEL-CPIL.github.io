import numpy as np
import time

class Optimizer:
    """Adaptive Grid + Quadratic Model optimizer for BBOB-style Sphere."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        n = self.dim
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(n)
        
        def eval_point(x):
            nonlocal calls, best_f, best_x
            if calls >= self.budget or time.time() > deadline:
                return None
            x_clipped = np.clip(x, -5, 5)
            f = func(x_clipped)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        def h_idx(i, j):
            if i > j: i, j = j, i
            return n + i * n - i * (i - 1) // 2 + (j - i)

        # Phase 1: Deterministic structured exploration
        # Use a stratified grid to find the basin quickly
        # Evaluate center and structured offsets
        
        x_center = np.zeros(n)
        
        # Evaluate origin first
        f0 = eval_point(x_center)
        if f0 is None:
            return best_x, best_f
        
        # Structured exploration: evaluate points at various scales
        # This gives early progress on AOCC
        scales = [3.0, 2.0, 1.5, 1.0, 0.5]
        for scale in scales:
            if calls >= self.budget or time.time() > deadline:
                break
            # Axis points at this scale
            for i in range(n):
                if calls >= self.budget or time.time() > deadline:
                    break
                x_plus = x_center.copy()
                x_plus[i] += scale
                eval_point(x_plus)
                x_minus = x_center.copy()
                x_minus[i] -= scale
                eval_point(x_minus)
        
        if calls >= self.budget or time.time() > deadline:
            return best_x, best_f
        
        # Phase 2: Adaptive grid refinement around best point
        # Iteratively refine the search region
        tr_radius = 1.0
        max_tr_radius = 3.0
        min_tr_radius = 1e-3
        max_grid_iters = 15
        
        for iter_idx in range(max_grid_iters):
            if time.time() > deadline:
                break
            remaining = self.budget - calls
            if remaining < 30:
                break
            
            # Build adaptive grid around current best
            # Use center + axis points + pair points for quadratic model
            points = [x_center.copy()]
            
            # Axis points
            for i in range(n):
                x_plus = x_center.copy()
                x_plus[i] += tr_radius
                points.append(x_plus)
                x_minus = x_center.copy()
                x_minus[i] -= tr_radius
                points.append(x_minus)
            
            # Pair points for Hessian off-diagonal
            for i in range(n):
                for j in range(i + 1, n):
                    x_pair = x_center.copy()
                    x_pair[i] += tr_radius / np.sqrt(2)
                    x_pair[j] += tr_radius / np.sqrt(2)
                    points.append(x_pair)
            
            # Evaluate all points
            fvals = []
            valid = True
            for p in points:
                f = eval_point(p)
                if f is None:
                    valid = False
                    break
                fvals.append(f)
                
            if not valid or len(fvals) < 2:
                break
            
            # Fit quadratic model
            m = len(points) - 1
            num_unknowns = n + n * (n + 1) // 2
            A = np.zeros((m, num_unknowns))
            y = np.array(fvals[1:]) - fvals[0]
            
            for k in range(m):
                d = points[k + 1] - x_center
                for i in range(n):
                    A[k, i] = d[i]
                for i in range(n):
                    for j in range(i, n):
                        if i == j:
                            coeff = 0.5 * d[i] * d[j]
                        else:
                            coeff = d[i] * d[j]
                        A[k, h_idx(i, j)] = coeff
                        
            try:
                theta, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
            except:
                break
                
            g = theta[:n]
            H = np.zeros((n, n))
            for i in range(n):
                for j in range(i, n):
                    val = theta[h_idx(i, j)]
                    H[i, j] = val
                    H[j, i] = val
            
            # Regularize Hessian
            try:
                eigvals = np.linalg.eigvalsh(H)
                if np.min(eigvals) < -1e-6:
                    H += (abs(np.min(eigvals)) + 1e-6) * np.eye(n)
            except:
                H = np.eye(n) * 0.1
            
            # Newton step
            try:
                step = np.linalg.solve(H, -g)
            except:
                step = -g * 0.1
                
            step_norm = np.linalg.norm(step)
            if step_norm > tr_radius:
                step = step * (tr_radius / step_norm)
            elif step_norm < 1e-12:
                # Converged or at minimum
                tr_radius *= 0.5
                if tr_radius < min_tr_radius:
                    break
                continue
                
            x_next = x_center + step
            x_next = np.clip(x_next, -5, 5)
            
            f_next = eval_point(x_next)
            if f_next is None:
                break
            
            # Accept if improvement
            if f_next < fvals[0]:
                x_center = x_next
                # Expand trust region if step was small
                if step_norm < tr_radius * 0.5:
                    tr_radius = min(tr_radius * 1.5, max_tr_radius)
                else:
                    tr_radius = min(tr_radius * 1.2, max_tr_radius)
            else:
                tr_radius *= 0.5
                
            if tr_radius < min_tr_radius:
                break
                
        # Phase 3: Final polishing with very small steps
        if time.time() < deadline and calls < self.budget - 10:
            tr_radius = min(tr_radius, 0.05)
            for _ in range(10):
                if time.time() > deadline or calls >= self.budget - 5:
                    break
                if tr_radius < 1e-6:
                    break
                    
                points = [x_center.copy()]
                for i in range(n):
                    x_plus = x_center.copy()
                    x_plus[i] += tr_radius
                    points.append(x_plus)
                    x_minus = x_center.copy()
                    x_minus[i] -= tr_radius
                    points.append(x_minus)
                for i in range(n):
                    for j in range(i + 1, n):
                        x_pair = x_center.copy()
                        x_pair[i] += tr_radius / np.sqrt(2)
                        x_pair[j] += tr_radius / np.sqrt(2)
                        points.append(x_pair)
                        
                fvals = []
                valid = True
                for p in points:
                    f = eval_point(p)
                    if f is None:
                        valid = False
                        break
                    fvals.append(f)
                    
                if not valid or len(fvals) < 2:
                    break
                    
                m = len(points) - 1
                num_unknowns = n + n * (n + 1) // 2
                A = np.zeros((m, num_unknowns))
                y = np.array(fvals[1:]) - fvals[0]
                
                for k in range(m):
                    d = points[k + 1] - x_center
                    for i in range(n):
                        A[k, i] = d[i]
                    for i in range(n):
                        for j in range(i, n):
                            if i == j:
                                coeff = 0.5 * d[i] * d[j]
                            else:
                                coeff = d[i] * d[j]
                            A[k, h_idx(i, j)] = coeff
                            
                try:
                    theta, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
                except:
                    break
                    
                g = theta[:n]
                H = np.zeros((n, n))
                for i in range(n):
                    for j in range(i, n):
                        val = theta[h_idx(i, j)]
                        H[i, j] = val
                        H[j, i] = val
                        
                try:
                    eigvals = np.linalg.eigvalsh(H)
                    if np.min(eigvals) < -1e-8:
                        H += (abs(np.min(eigvals)) + 1e-8) * np.eye(n)
                except:
                    H = np.eye(n) * 0.1
                    
                try:
                    step = np.linalg.solve(H, -g)
                except:
                    step = -g * 0.1
                    
                step_norm = np.linalg.norm(step)
                if step_norm > tr_radius:
                    step = step * (tr_radius / step_norm)
                elif step_norm < 1e-14:
                    tr_radius *= 0.5
                    continue
                    
                x_next = x_center + step
                x_next = np.clip(x_next, -5, 5)
                
                f_next = eval_point(x_next)
                if f_next is None:
                    break
                    
                if f_next < fvals[0]:
                    x_center = x_next
                    if step_norm < tr_radius * 0.5:
                        tr_radius = min(tr_radius * 1.3, 0.5)
                else:
                    tr_radius *= 0.5
                    
                if tr_radius < 1e-6:
                    break

        return best_x, best_f
