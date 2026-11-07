"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead simplex with deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        
        # Global best tracking
        best_x = None
        best_f = np.inf
        calls_made = 0

        def check_time():
            return time.time() > deadline

        def nelder_mead(x0, fx0, max_calls):
            """Run Nelder-Mead from x0 with value fx0, using at most max_calls additional evaluations."""
            nonlocal best_x, best_f, calls_made
            nonlocal deadline, check_time
            
            # Initialize simplex: x0 plus dim perturbed points
            alpha = 0.5
            simplex = [x0.copy()]
            values = [fx0]
            
            for i in range(dim):
                xi = x0.copy()
                xi[i] += alpha
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                calls_made += 1
                simplex.append(xi)
                values.append(fi)
                
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()
                
                if check_time() or calls_made >= max_calls:
                    return

            # Nelder-Mead iterations
            gamma = 2.0  # reflection
            rho = 0.5    # contraction
            sigma = 2.0  # expansion
            
            while not check_time() and calls_made < max_calls:
                # Sort simplex by value
                indices = np.argsort(values)
                simplex = [simplex[i] for i in indices]
                values = [values[i] for i in indices]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + gamma * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                calls_made += 1
                
                if fr < best_f:
                    best_f = fr
                    best_x = xr.copy()
                
                if check_time() or calls_made >= max_calls:
                    return
                
                if fr < values[0]:
                    # Expansion
                    xe = centroid + sigma * (xr - centroid)
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    calls_made += 1
                    
                    if fe < best_f:
                        best_f = fe
                        best_x = xe.copy()
                    
                    if check_time() or calls_made >= max_calls:
                        return
                    
                    if fe < fr:
                        simplex[-1] = xe
                        values[-1] = fe
                    else:
                        simplex[-1] = xr
                        values[-1] = fr
                elif fr < values[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    values[-1] = fr
                else:
                    # Contraction
                    if fr < values[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls_made += 1
                    
                    if fc < best_f:
                        best_f = fc
                        best_x = xc.copy()
                    
                    if check_time() or calls_made >= max_calls:
                        return
                    
                    if fc < min(fr, values[-1]):
                        simplex[-1] = xc
                        values[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fi = func(simplex[i])
                            calls_made += 1
                            
                            if fi < best_f:
                                best_f = fi
                                best_x = simplex[i].copy()
                            
                            if check_time() or calls_made >= max_calls:
                                return
                            values[i] = fi

        # Multi-start strategy
        # Start 1: center of the box
        x0 = np.zeros(dim)
        fx0 = func(x0)
        calls_made += 1
        best_f = fx0
        best_x = x0.copy()
        
        if not check_time():
            nelder_mead(x0, fx0, budget - 1)
        
        # If we still have budget and time, try additional starts
        remaining_calls = budget - calls_made
        if remaining_calls > 50 and not check_time():
            # Try a few random starts
            for _ in range(3):
                if remaining_calls < 50 or check_time():
                    break
                
                x_start = rng.uniform(-4, 4, dim)
                fx_start = func(x_start)
                calls_made += 1
                remaining_calls -= 1
                
                if fx_start < best_f:
                    best_f = fx_start
                    best_x = x_start.copy()
                
                if not check_time() and remaining_calls >= 50:
                    nelder_mead(x_start, fx_start, remaining_calls)
                    remaining_calls = budget - calls_made

        return best_x, best_f
# EVOLVE-BLOCK-END
