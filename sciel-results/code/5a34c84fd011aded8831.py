"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Adaptive coordinate descent with dense 1D golden-section search for Bent Cigar."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        # Track evaluations
        eval_count = 0
        best_x = None
        best_fx = float('inf')

        def evaluate(x):
            nonlocal eval_count, best_x, best_fx
            if eval_count >= budget:
                return float('inf')
            fx = func(x)
            eval_count += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Phase 1: Multi-start sampling to handle hidden shift/rotation
        # Sample several points to find a good starting region
        n_starts = 8
        candidates = []
        
        # Center point
        candidates.append(np.zeros(dim))
        
        # Random points in the box
        for _ in range(n_starts - 1):
            candidates.append(rng.uniform(-5, 5, size=dim))
        
        for x_init in candidates:
            if eval_count >= budget:
                break
            x_init = np.clip(x_init, -5, 5)
            evaluate(x_init)
        
        if best_x is None:
            # Should not happen, but safety net
            x_init = np.zeros(dim)
            best_fx = evaluate(x_init)
            best_x = x_init.copy()
        
        current_x = best_x.copy()
        current_fx = best_fx

        # Golden-section search on a single coordinate
        def golden_search_1d(coord_idx, x_base, max_iter):
            nonlocal eval_count, best_x, best_fx, current_fx
            
            if eval_count >= budget:
                return x_base, current_fx
            
            # Search range: full box for this coordinate
            a, b = -5.0, 5.0
            
            # Golden ratio constants
            inv_phi = 2.0 / (1.0 + np.sqrt(5.0))  # ~0.618
            inv_phi2 = 1.0 - inv_phi  # ~0.382
            
            # Initial interior points
            c = a + inv_phi2 * (b - a)
            d = a + inv_phi * (b - a)
            
            x_c = x_base.copy()
            x_c[coord_idx] = c
            x_d = x_base.copy()
            x_d[coord_idx] = d
            
            fc = evaluate(x_c)
            fd = evaluate(x_d)
            
            if eval_count >= budget:
                # Return best of the two
                if fc < fd:
                    return x_c, fc
                else:
                    return x_d, fd
            
            # Track local best
            local_best_x = x_base.copy()
            local_best_fx = current_fx
            if fc < local_best_fx:
                local_best_fx = fc
                local_best_x = x_c.copy()
            if fd < local_best_fx:
                local_best_fx = fd
                local_best_x = x_d.copy()
            
            # Iterative golden-section refinement
            for _ in range(max_iter):
                if eval_count >= budget:
                    break
                
                if fc < fd:
                    # Minimum is in [a, d]
                    b = d
                    d = c
                    fd = fc
                    c = a + inv_phi2 * (b - a)
                    x_c = local_best_x.copy()
                    x_c[coord_idx] = c
                    fc = evaluate(x_c)
                    if fc < local_best_fx:
                        local_best_fx = fc
                        local_best_x = x_c.copy()
                else:
                    # Minimum is in [c, b]
                    a = c
                    c = d
                    fc = fd
                    d = a + inv_phi * (b - a)
                    x_d = local_best_x.copy()
                    x_d[coord_idx] = d
                    fd = evaluate(x_d)
                    if fd < local_best_fx:
                        local_best_fx = fd
                        local_best_x = x_d.copy()
            
            return local_best_x, local_best_fx

        # Adaptive coordinate ordering: estimate gradient magnitude to prioritize narrow dimensions
        def estimate_coord_importance(x_base):
            """Estimate which coordinate has the steepest descent direction."""
            if eval_count >= budget - 2 * dim:
                return list(range(dim))
            
            eps = 0.01
            importances = []
            for i in range(dim):
                x_plus = x_base.copy()
                x_plus[i] = np.clip(x_plus[i] + eps, -5, 5)
                x_minus = x_base.copy()
                x_minus[i] = np.clip(x_minus[i] - eps, -5, 5)
                
                if eval_count >= budget - 1:
                    importances.append(0.0)
                    continue
                
                fp = evaluate(x_plus)
                if eval_count >= budget:
                    importances.append(0.0)
                    continue
                fm = evaluate(x_minus)
                grad = (fp - fm) / (2 * eps)
                importances.append(abs(grad))
            
            # Sort by importance (descending)
            order = list(range(dim))
            order.sort(key=lambda i: importances[i], reverse=True)
            return order

        # Main optimization loop: iterative coordinate descent
        # Use multiple passes with decreasing precision
        n_passes = 3
        
        for pass_num in range(n_passes):
            if eval_count >= budget:
                break
            
            remaining = budget - eval_count
            if remaining < 3 * dim:
                break
            
            # Estimate coordinate importance
            coord_order = estimate_coord_importance(current_x)
            
            # Number of golden-section iterations per coordinate
            # More iterations for higher-importance coordinates
            base_iter = max(5, remaining // (3 * dim))
            
            for i in coord_order:
                if eval_count >= budget:
                    break
                
                # Allocate more iterations to important coordinates
                if i == coord_order[0]:
                    max_iter = min(base_iter + 5, remaining // 2)
                else:
                    max_iter = min(base_iter, remaining // 2)
                
                if max_iter < 3:
                    max_iter = 3
                
                current_x, current_fx = golden_search_1d(i, current_x, max_iter)
            
            # Update current point to best found
            if best_x is not None and best_fx < current_fx:
                current_x = best_x.copy()
                current_fx = best_fx

        # Final refinement: use remaining budget for fine-tuning the best coordinate
        if eval_count < budget - 5:
            remaining = budget - eval_count
            if remaining > 10:
                # Do a few more fine-tuning passes on the most important coordinate
                for _ in range(2):
                    if eval_count >= budget:
                        break
                    coord_order = estimate_coord_importance(current_x)
                    if coord_order:
                        max_iter = min(10, remaining // 2)
                        current_x, current_fx = golden_search_1d(coord_order[0], current_x, max_iter)
                        if best_x is not None and best_fx < current_fx:
                            current_x = best_x.copy()
                            current_fx = best_fx

        return best_x, best_fx
# EVOLVE-BLOCK-END
