import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        N = self.budget
        
        # Nelder-Mead parameters
        # Initial simplex
        # Start at 0,0,0,0,0
        x0 = np.zeros(d)
        # Step size for initial simplex vertices
        # Box is [-5, 5]. Min is in [-4, 4].
        # Let's use a step of 1.0
        delta = 1.0
        
        # Simplex: d+1 points
        # P[0] is the best (initially)
        P = np.zeros((d + 1, d))
        P[0] = x0
        for i in range(1, d + 1):
            P[i] = x0
            P[i, i-1] = delta
            
        # Evaluate initial simplex
        F = np.array([func(p) for p in P])
        calls = d + 1
        
        best_x = P[0].copy()
        best_f = F[0]
        
        # Update best
        min_idx = np.argmin(F)
        best_f = F[min_idx]
        best_x = P[min_idx].copy()
        
        start_time = time.time()
        deadline = start_time + 290 # Safety margin for 300s limit
        
        alpha = 1.0 # Reflection
        gamma = 2.0 # Expansion
        rho = 0.5   # Contraction
        sigma = 0.5 # Shrinkage
        
        while calls < N and time.time() < deadline:
            # Sort simplex
            order = np.argsort(F)
            P = P[order]
            F = F[order]
            
            # Check for convergence (optional, but good for stopping early if stuck)
            # If range is very small, break?
            if F[-1] - F[0] < 1e-12:
                break
            
            # Centroid of all points except the worst
            x_c = np.mean(P[:-1], axis=0)
            
            # Reflection
            x_r = x_c + alpha * (x_c - P[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            calls += 1
            
            if f_r < F[0]:
                # Expansion
                x_e = x_c + gamma * (x_r - x_c)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                calls += 1
                
                if f_e < f_r:
                    P[-1] = x_e
                    F[-1] = f_e
                else:
                    P[-1] = x_r
                    F[-1] = f_r
            elif f_r < F[-2]:
                # Reflection is good
                P[-1] = x_r
                F[-1] = f_r
            else:
                # Contraction
                # Check if contraction towards reflection is better
                x_cr = x_c + rho * (x_r - x_c)
                x_cr = np.clip(x_cr, -5, 5)
                f_cr = func(x_cr)
                calls += 1
                
                if f_cr < f_r:
                    P[-1] = x_cr
                    F[-1] = f_cr
                else:
                    # Shrink
                    # Shrink all points except the best towards the best
                    for i in range(1, d + 1):
                        P[i] = P[0] + sigma * (P[i] - P[0])
                        P[i] = np.clip(P[i], -5, 5)
                        F[i] = func(P[i])
                        calls += 1
                        if calls >= N:
                            break
                    
                    if calls >= N:
                        break
            
            # Update global best
            min_idx = np.argmin(F)
            if F[min_idx] < best_f:
                best_f = F[min_idx]
                best_x = P[min_idx].copy()

        return None
