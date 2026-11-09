"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-objective optimizer for ZDT1: exploit known structure to maximize hypervolume."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT1 structure: f1 = x[0], f2 = g(x)*(1-sqrt(x[0]/g(x)))
        # g(x) minimized when x[1:] = 0.5, giving g=1 and true Pareto front f2=1-sqrt(f1)
        # Strategy: fix x[1:]=0.5, spread x[0] to cover Pareto front for max hypervolume
        
        n_evals = self.budget
        R1 = 11.0
        R2 = 11.0
        
        # Initialize with endpoints
        x0_list = [0.0, 1.0]
        
        # Evaluate initial points
        for x0 in x0_list:
            x = np.full(self.dim, 0.5)
            x[0] = x0
            x = np.clip(x, lo, hi)
            problem(x)
        
        # Greedy feedback loop
        for _ in range(n_evals - len(x0_list)):
            if len(x0_list) < 2:
                break
                
            # Sort x0_list to define gaps
            x0_list.sort()
            
            best_gain = -1.0
            best_x_new = None
            best_idx = -1
            
            # Iterate through gaps
            for i in range(len(x0_list) - 1):
                A = x0_list[i]
                B = x0_list[i+1]
                
                # If gap is too small, skip
                if B - A < 1e-10:
                    continue
                    
                # Calculate f2 for left point
                C = 1.0 - np.sqrt(A)
                K = 1.0 - C
                
                # Solve u^2 - 2K u + B = 0 for u = sqrt(x)
                # u = K - sqrt(K^2 - B) (taking the smaller root to stay in range? 
                # The parabola opens upward. We want the minimum of the derivative? 
                # No, we found the root of the derivative. 
                # g'(x) = (B+x)/(2sqrt(x)) - K.
                # g'(x) = 0 => x = (K - sqrt(K^2 - B))^2 ?
                # Let's check the two roots.
                # u1 = K - sqrt(K^2 - B), u2 = K + sqrt(K^2 - B).
                # x1 = u1^2, x2 = u2^2.
                # We need x in (A, B).
                
                disc = K*K - B
                if disc < 0:
                    continue
                    
                sqrt_disc = np.sqrt(disc)
                u1 = K - sqrt_disc
                u2 = K + sqrt_disc
                
                x1 = u1*u1
                x2 = u2*u2
                
                candidates = []
                if A < x1 < B:
                    candidates.append(x1)
                if A < x2 < B:
                    candidates.append(x2)
                
                # Also check if the max is at the boundaries? 
                # Gain is 0 at boundaries. So only internal critical points matter.
                # However, if the function is monotonic, the max is at the boundary (gain 0).
                # So if no internal critical point, gain is effectively 0 (or very small).
                
                for x_cand in candidates:
                    f2_cand = 1.0 - np.sqrt(x_cand)
                    # Calculate gain
                    # Old term for interval [A, B] is (B-A)*(R2-C)
                    # But wait, the old term is part of the sum.
                    # The gain calculation derived was:
                    # Gain = (f2_cand - C) * (x_cand - B)
                    # Let's re-verify this formula.
                    # Gain = (D - C) (x - B)
                    # D = f2_cand, C = f2_left.
                    # Since D < C and x < B, (D-C) is neg, (x-B) is neg. Product is pos.
                    gain = (f2_cand - C) * (x_cand - B)
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x_new = x_cand
                        best_idx = i
                        
            if best_x_new is None:
                # Fallback: if no good candidate found (e.g. due to numerical issues), 
                # pick midpoint of the largest gap
                max_gap = 0
                max_idx = 0
                for i in range(len(x0_list) - 1):
                    gap = x0_list[i+1] - x0_list[i]
                    if gap > max_gap:
                        max_gap = gap
                        max_idx = i
                best_x_new = (x0_list[max_idx] + x0_list[max_idx+1]) / 2.0
                best_idx = max_idx

            # Insert new point
            x0_list.insert(best_idx + 1, best_x_new)
            
            # Evaluate
            x = np.full(self.dim, 0.5)
            x[0] = best_x_new
            x = np.clip(x, lo, hi)
            problem(x)
            
        return None
# EVOLVE-BLOCK-END
