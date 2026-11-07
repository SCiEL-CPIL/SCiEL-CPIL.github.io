import numpy as np
import time

class Optimizer:
        def __init__(self, budget, dim):
            self.budget = budget
            self.dim = dim

        def __call__(self, func):
            start_time = time.time()
            deadline = start_time + 280.0
            
            d = self.dim
            lam = 1
            
            m = np.zeros(d)
            f_m = func(m)
            best_x = m.copy()
            best_f = f_m
            sigma = 2.0
            C = np.eye(d)
            Cinv_sqrt = np.eye(d)
            Cholesky = np.eye(d)
            
            pc = np.zeros(d)
            ps = np.zeros(d)
            
            mu = 1
            mueff = 1.0
            cc = (2 + 2/np.sqrt(d + 1.3)) / (d + 4 + 2/np.sqrt(d + 1.3))
            cs = (mueff + 2) / (d + mueff + 5)
            c1 = 2 / ((d + 1.7)**2 + mueff)
            cmu = 0.0
            dsig = 1 + 2 * max(0, np.sqrt((mueff - 1) / (d + 1)) - 1) + cs
            chiN = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))
            
            calls = 1
            rng = np.random.default_rng(42)
            iter_count = 0
            
            while calls < self.budget:
                if time.time() > deadline:
                    break
                
                # Generate
                z = rng.standard_normal(d)
                x = m + sigma * (Cholesky @ z)
                x = np.clip(x, -5, 5)
                
                fx = func(x)
                calls += 1
                iter_count += 1
                
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                
                # Update CMA-ES
                if fx < f_m:
                    # Success
                    y = x - m
                    # Update pc
                    # pc is in the space of C^{-1/2}
                    # Standard update: pc = (1-cc)pc + cc * y / (sigma * sqrt(diag(C)))? 
                    # No, for full CMA:
                    # The step is y = sigma * C^{1/2} z
                    # The "normalized" step is z = C^{-1/2} y / sigma
                    # pc update uses the normalized step?
                    # Hansen (2016) Eq 23: pc <- (1-cc)pc + cc * (x-m)/(sigma*sqrt(diag(C))) is for diagonal.
                    # For full: pc <- (1-cc)pc + cc * C^{-1/2} (x-m)/sigma
                    # Let's check.
                    z_step = Cinv_sqrt @ (y / sigma)
                    pc = (1 - cc) * pc + cc * z_step
                    
                    # Update ps
                    ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * z_step
                    
                    # Update sigma
                    norm_ps = np.linalg.norm(ps)
                    # Approximation for h_sigma
                    # h_sigma = 1 if norm_ps / chiN < (1.4 + 2/(d+1)) * (1 - (1-cs)**(1/iter_count))
                    # Simplified:
                    h_sigma = 1.0 if norm_ps / chiN < 1.4 + 2/(d+1) else 0.0
                    sigma *= np.exp(cs/dsig * (norm_ps/chiN - 1))
                    
                    # Update C
                    # C <- (1-c1)C + c1 (pc pc^T + (1-h_sigma)cc(2-cc)C)
                    # Note: pc is in the "z-space" (normalized), so we need to map back?
                    # No, the update formula for C in CMA-ES is:
                    # C <- (1 - c1 - cmu)C + c1 (pc pc^T + (1-h_c)c_c(2-c_c)C) + cmu * ...
                    # Where pc is the evolution path in the "search space" (i.e., C^{1/2} pc_z?)
                    # Actually, the path pc is usually defined such that the rank-one update is pc pc^T.
                    # If we define pc in the z-space, we need to transform it.
                    # Let's use the formulation where pc is in the "x-space" scaled by 1/sigma?
                    # Reference: 
                    # pc <- (1-cc)pc + cc * (x-m)/sigma
                    # C <- (1-c1)C + c1 (pc pc^T + (1-h_c)cc(2-cc)C)
                    # This is simpler. Let's use this.
                    
                    # Re-evaluating pc update with this simpler definition
                    pc = (1 - cc) * pc + cc * (y / sigma)
                    
                    # Recalculate ps with this definition?
                    # ps <- (1-cs)ps + sqrt(cs(2-cs)mueff) * (x-m)/sigma
                    ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (y / sigma)
                    
                    norm_ps = np.linalg.norm(ps)
                    h_sigma = 1.0 if norm_ps / chiN < 1.4 + 2/(d+1) else 0.0
                    sigma *= np.exp(cs/dsig * (norm_ps/chiN - 1))
                    
                    # Update C
                    # C <- (1 - c1)C + c1 (pc pc^T + (1-h_sigma)cc(2-cc)C)
                    C = (1 - c1) * C + c1 * (np.outer(pc, pc) + (1 - h_sigma) * cc * (2 - cc) * C)
                    
                    # Update Cholesky and Inverse Sqrt
                    # Eigen decomposition for stability
                    eigvals, eigvecs = np.linalg.eigh(C)
                    # Clip small eigenvalues
                    eigvals = np.clip(eigvals, 1e-12, None)
                    Cinv_sqrt = (eigvecs / np.sqrt(eigvals)) @ eigvecs.T
                    Cholesky = eigvecs @ np.diag(np.sqrt(eigvals))
                    
                    m = x
                    f_m = fx
                else:
                    # Failure
                    # Update ps
                    y = x - m
                    z_step = Cinv_sqrt @ (y / sigma)
                    ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * z_step
                    norm_ps = np.linalg.norm(ps)
                    h_sigma = 1.0 if norm_ps / chiN < 1.4 + 2/(d+1) else 0.0
                    sigma *= np.exp(cs/dsig * (norm_ps/chiN - 1))
                    
                    # Update pc
                    pc = (1 - cc) * pc
                    
                    # Update C
                    C = (1 - c1) * C + c1 * ((1 - h_sigma) * cc * (2 - cc) * C)
                    
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-12, None)
                    Cinv_sqrt = (eigvecs / np.sqrt(eigvals)) @ eigvecs.T
                    Cholesky = eigvecs @ np.diag(np.sqrt(eigvals))
            
            return best_x, best_f
