import numpy as np
import scipy.optimize as so
import time

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    delta = 1.0 / N
    deadline = time.time() + 590.0  # Safety margin
    
    # Exact volume computation using critical-point integration
    def compute_volume(X, Y):
        # Build typed endpoint lists for X and Y
        # Each endpoint: (A, B, is_right) where value at c is A + B*c
        eps_x = []
        for i in range(N):
            eps_x.append((X[i], i/N, 0))
            eps_x.append((X[i] + delta, i/N, 1))
        
        eps_y = []
        for j in range(N):
            eps_y.append((Y[j], j/N, 0))
            eps_y.append((Y[j] + delta, j/N, 1))
        
        # Find critical points where endpoint order changes
        def find_crit(eps):
            crit = {0.0, 1.0}
            for i in range(len(eps)):
                A1, B1, _ = eps[i]
                for j in range(i+1, len(eps)):
                    A2, B2, _ = eps[j]
                    if abs(B1 - B2) > 1e-15:
                        c = (A2 - A1) / (B1 - B2)
                        if 0.0 < c < 1.0:
                            crit.add(c)
            return crit
        
        crit_x = find_crit(eps_x)
        crit_y = find_crit(eps_y)
        all_crit = sorted(crit_x.union(crit_y))
        
        # Compute union length at a given c
        def union_length(eps, c):
            pts = []
            for A, B, is_r in eps:
                pts.append((A + B*c, is_r))
            pts.sort()
            total = 0.0
            count = 0
            start_pos = 0.0
            for val, is_r in pts:
                if is_r == 0:
                    if count == 0:
                        start_pos = val
                    count += 1
                else:
                    count -= 1
                    if count == 0:
                        total += val - start_pos
            return total
        
        # Integrate Lx(c)*Ly(c) over [0,1] using piecewise linear interpolation of the lengths
        # Note: Lx(c) and Ly(c) are piecewise linear in c.
        # Therefore Lx(c)*Ly(c) is piecewise quadratic.
        # We integrate exactly over each interval between critical points.
        total_vol = 0.0
        for k in range(len(all_crit) - 1):
            c0 = all_crit[k]
            c1 = all_crit[k+1]
            h = c1 - c0
            if h < 1e-12:
                continue
            
            Lx0 = union_length(eps_x, c0)
            Lx1 = union_length(eps_x, c1)
            Ly0 = union_length(eps_y, c0)
            Ly1 = union_length(eps_y, c1)
            
            dx = Lx1 - Lx0
            dy = Ly1 - Ly0
            
            # Lx(c) = Lx0 + (dx/h)*(c-c0)
            # Ly(c) = Ly0 + (dy/h)*(c-c0)
            # Integral of (Lx0 + u*dx)(Ly0 + u*dy) du from 0 to 1, where u = (c-c0)/h
            # = h * [ Lx0*Ly0 + 0.5*(Lx0*dy + Ly0*dx) + (1/3)*dx*dy ]
            
            term = Lx0 * Ly0
            term += 0.5 * (Lx0 * dy + Ly0 * dx)
            term += (1.0/3.0) * dx * dy
            
            total_vol += h * term
        
        return total_vol
    
    # Objective: minimize union volume
    def objective(params):
        X = params[:N]
        Y = params[N:]
        return compute_volume(X, Y)
    
    # Initial guess: use the Keich construction as starting point
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    X0 = keich.copy()
    Y0 = keich.copy()
    p0 = np.concatenate([X0, Y0])
    
    best_p = p0.copy()
    best_v = objective(p0)
    
    # Multi-start optimization
    rng = np.random.default_rng(42)
    
    # Generate diverse starts
    starts = [p0.copy()]
    
    # Halton sequence for better coverage of the 16D space
    def halton_sequence(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    # Generate 40 Halton points in [0, 1]^16 and scale to a reasonable range
    # The optimal positions are likely within [-1, 1] or so.
    num_halton = 40
    for i in range(1, num_halton + 1):
        if time.time() > deadline - 150:
            break
        p_h = np.array([halton_sequence(i, base) for base in range(2, 2 + 2*N)])
        # Scale to [-0.5, 0.5]
        p_h = p_h - 0.5
        starts.append(p_h)
    
    # Random perturbations around the initial guess
    for attempt in range(20):
        if time.time() > deadline - 150:
            break
        sigma = 0.1
        perturbation = rng.normal(0, sigma, size=2*N)
        p_try = p0 + perturbation
        starts.append(p_try)
    
    # Structured starts
    sym_x = np.linspace(-delta/2, -(N-1)*delta/2, N)
    sym_y = np.linspace(-delta/2, -(N-1)*delta/2, N)
    starts.append(np.concatenate([sym_x, sym_y]))
    
    anti_x = np.linspace(0, (N-1)*delta, N)
    anti_y = np.linspace(0, (N-1)*delta, N)
    starts.append(np.concatenate([anti_x, anti_y]))
    
    # Run optimization for each start
    # Strategy: Use Powell (robust to nonsmoothness) for the main search, 
    # then L-BFGS-B for polishing.
    
    for p_start in starts:
        if time.time() > deadline - 30:
            break
        
        try:
            # Phase 1: Powell for robust local search
            res = so.minimize(objective, p_start, method='Powell',
                            options={'maxiter': 100, 'xtol': 1e-6, 'ftol': 1e-6})
            if res.fun < best_v:
                best_v = res.fun
                best_p = res.x.copy()
                
            # Phase 2: L-BFGS-B for fast polishing from the Powell result
            if time.time() < deadline - 10:
                res2 = so.minimize(objective, best_p, method='L-BFGS-B',
                                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
                if res2.fun < best_v:
                    best_v = res2.fun
                    best_p = res2.x.copy()
                    
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        try:
            res = so.minimize(objective, best_p, method='Nelder-Mead',
                            options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.fun < best_v:
                best_v = res.fun
                best_p = res.x.copy()
        except Exception:
            pass

    X_opt = best_p[:N]
    Y_opt = best_p[N:]
    
    # Construct final positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = X_opt[i_flat]
    base_y = Y_opt[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions
