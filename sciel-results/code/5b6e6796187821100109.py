"""Coordinate-wise de-shifting + Nelder-Mead optimizer for BBOB f20 (Schwefel)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """
    Two-phase optimizer for BBOB f20 (Schwefel) on [-5,5]^5.
    Phase 1: Coordinate-wise 1-D ternary search to estimate the hidden shift center.
    Phase 2: Joint Nelder-Mead refinement around the estimated center.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def update_best(x, fx):
            nonlocal best_x, best_f
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        call_count = 0

        # Phase 1: Coordinate-wise 1-D ternary search to find the center
        # Schwefel f20 is roughly sum of independent unimodal components.
        # We estimate the center by minimizing along each axis.
        
        # Reserve budget for Phase 2 (Nelder-Mead)
        nm_budget = 300
        phase1_budget = self.budget - nm_budget
        
        # Allocate phase1 budget across dimensions
        # dim=5, so ~ (phase1_budget // dim) per dimension
        budget_per_dim = max(1, phase1_budget // self.dim)
        
        # Initial guess: center of the box (0,0,0,0,0)
        # But the hidden shift is in [-4,4]^5.
        # We'll use the result of 1-D search to refine the center.
        
        center = np.zeros(self.dim)
        
        # For each dimension, do a 1-D ternary search
        # We fix other dimensions at their current best estimate (initially 0)
        # and vary one dimension to find the minimum.
        
        for _iter in range(2):  # Two passes to refine
            for d in range(self.dim):
                if call_count >= phase1_budget:
                    break
                if time.time() > deadline:
                    break
                
                # 1-D ternary search on dimension d
                # Range: [-5, 5]
                a = -5.0
                b = 5.0
                # Number of steps for ternary search
                # Each step uses 2 function calls.
                # Let's use ~ budget_per_dim // 2 steps
                n_steps = min(budget_per_dim // 2, 30)  # Cap steps to avoid wasting too much
                
                for _s in range(n_steps):
                    if call_count >= phase1_budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    # Ternary search points
                    m1 = a + (b - a) / 3.0
                    m2 = b - (b - a) / 3.0
                    
                    # Construct test points
                    x1 = center.copy()
                    x1[d] = m1
                    x1 = np.clip(x1, -5, 5)
                    
                    x2 = center.copy()
                    x2[d] = m2
                    x2 = np.clip(x2, -5, 5)
                    
                    f1 = func(x1)
                    call_count += 1
                    update_best(x1, f1)
                    
                    f2 = func(x2)
                    call_count += 1
                    update_best(x2, f2)
                    
                    # Update interval
                    if f1 < f2:
                        b = m2
                    else:
                        a = m1
                
                # Update center for this dimension
                center[d] = (a + b) / 2.0

        # Ensure we have a valid best_x
        if best_x is None:
            # Fallback: evaluate center
            fx = func(center)
            call_count += 1
            update_best(center, fx)
            best_x = center.copy()
            best_f = fx

        # Phase 2: Nelder-Mead refinement
        # Start from the best point found in Phase 1
        x0 = best_x.copy()
        fx0 = best_f
        
        # Initialize simplex
        # Simplex size: adaptive based on remaining budget
        # Standard NM uses dim+1 points.
        # We have nm_budget left.
        # Each NM iteration uses ~dim+1 function calls (worst case) or ~1-2 on average if we reuse.
        # Let's limit the number of NM iterations.
        
        # Initial simplex: x0 and x0 + delta * e_i
        delta = 0.5  # Initial step size
        simplex = [x0.copy()]
        for i in range(self.dim):
            v = x0.copy()
            v[i] += delta
            simplex.append(np.clip(v, -5, 5))
        
        # Evaluate initial simplex
        simplexF = []
        for i, v in enumerate(simplex):
            if call_count >= self.budget:
                break
            if time.time() > deadline:
                break
            fv = func(v)
            call_count += 1
            update_best(v, fv)
            simplexF.append(fv)
        
        if len(simplexF) < self.dim + 1:
            # Not enough budget or time, return best
            return best_x, best_f
        
        # Nelder-Mead parameters
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink
        
        max_iter = 100  # Safety cap on iterations
        
        for _it in range(max_iter):
            if call_count >= self.budget:
                break
            if time.time() > deadline:
                break
            
            # Sort simplex by function value
            order = np.argsort(simplexF)
            simplex = [simplex[i] for i in order]
            simplexF = [simplexF[i] for i in order]
            
            # Check convergence
            if abs(simplexF[-1] - simplexF[0]) < 1e-8:
                break
            
            # Centroid of best points
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            call_count += 1
            update_best(xr, fr)
            
            if fr < simplexF[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                call_count += 1
                update_best(xe, fe)
                
                if fe < fr:
                    simplex[-1] = xe
                    simplexF[-1] = fe
                else:
                    simplex[-1] = xr
                    simplexF[-1] = fr
            elif fr < simplexF[-2]:
                # Accept reflection
                simplex[-1] = xr
                simplexF[-1] = fr
            else:
                # Contraction
                # Outside contraction if fr < simplexF[-1], else inside
                if fr < simplexF[-1]:
                    # Outside contraction
                    xc = centroid + 0.5 * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - 0.5 * (centroid - simplex[-1])
                
                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                call_count += 1
                update_best(xc, fc)
                
                if fc < simplexF[-1]:
                    simplex[-1] = xc
                    simplexF[-1] = fc
                else:
                    # Shrink
                    for i in range(1, self.dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fi = func(simplex[i])
                        call_count += 1
                        update_best(simplex[i], fi)
                        simplexF[i] = fi
        
        return best_x, best_f
# EVOLVE-BLOCK-END
