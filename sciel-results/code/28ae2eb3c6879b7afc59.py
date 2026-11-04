import time
import numpy as np

def construct(q, n, radius):
    t_start = time.time()
    deadline = 148.0
    
    space_size = q ** n
    q_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        q_powers[i] = q_powers[i - 1] * q
    
    def idx_to_word(idx):
        w = [0] * n
        for pos in range(n):
            w[pos] = (idx // q_powers[n - 1 - pos]) % q
        return w
    
    # Precompute neighbors (radius=1: self + all 1-change neighbors)
    nb_count = 1 + n * (q - 1)
    neighbors = np.empty((space_size, nb_count), dtype=np.int32)
    for idx in range(space_size):
        w = [0] * n
        for pos in range(n):
            w[pos] = (idx // q_powers[n - 1 - pos]) % q
        nb_list = [idx]
        for pos in range(n):
            for new_d in range(q):
                if new_d == w[pos]:
                    continue
                delta = (new_d - w[pos]) * q_powers[n - 1 - pos]
                nb_list.append(idx + delta)
        neighbors[idx] = nb_list
    
    # Build adjacency matrix (sparse representation using neighbor lists)
    # A[v][u] = 1 if u is in N(v) (including v itself)
    
    def build_cov(code_indices):
        cov = np.zeros(space_size, dtype=np.int32)
        for c in code_indices:
            cov[neighbors[c]] += 1
        return cov
    
    def lagrangian_relaxation(seed=0):
        """
        Solve min dominating set using Lagrangian relaxation.
        Primal: min sum(x_v) s.t. sum_{u in N(v)} x_u >= 1 for all v, x_v in {0,1}
        Lagrangian: L(x, lambda) = sum(x_v) + sum_v lambda_v * (1 - sum_{u in N(v)} x_u)
                   = sum_v lambda_v + sum_u x_u * (1 - sum_{v: u in N(v)} lambda_v)
        Reduced cost for node u: c_u = 1 - sum_{v: u in N(v)} lambda_v
        Greedy: pick nodes with most negative reduced cost (highest benefit)
        """
        rng = np.random.default_rng(seed)
        
        # Initialize dual variables
        lam = np.zeros(space_size)
        
        # Track best solution found
        best_code = None
        best_size = space_size + 1
        
        # Number of subgradient iterations
        max_iter = 50
        step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005]
        
        for it in range(max_iter):
            if (time.time() - t_start) > deadline - 30.0:
                break
            
            # Compute reduced costs: c_u = 1 - sum_{v: u in N(v)} lambda_v
            # Since the graph is symmetric (u in N(v) iff v in N(u)), 
            # sum_{v: u in N(v)} lambda_v = sum_{v in N(u)} lambda_v
            reduced_costs = 1.0 - np.sum(lam[neighbors], axis=1)
            
            # Greedy primal: select nodes with most negative reduced costs
            # until all nodes are covered
            uncovered = np.ones(space_size, dtype=bool)
            code = []
            
            while np.any(uncovered) and (time.time() - t_start) < deadline - 25.0:
                # Gain: number of uncovered nodes covered by each candidate
                gains = np.sum(uncovered[neighbors], axis=1)
                
                # Combine with reduced cost: prioritize low reduced cost (high benefit)
                # Scale reduced costs to be comparable to gains
                rc_scaled = -reduced_costs * 0.1  # negative because we want low rc
                combined = gains + rc_scaled
                
                # Add small random perturbation for diversity
                perturb = rng.random(space_size) * 0.001
                combined += perturb
                
                best = np.argmax(combined)
                if gains[best] == 0:
                    break
                
                code.append(int(best))
                uncovered[neighbors[best]] = False
            
            # Repair if needed
            cov = build_cov(code)
            if not np.all(cov > 0):
                uncovered_mask = cov == 0
                while np.any(uncovered_mask) and (time.time() - t_start) < deadline - 20.0:
                    gains = np.sum(uncovered_mask[neighbors], axis=1)
                    best = np.argmax(gains)
                    if gains[best] == 0:
                        break
                    code.append(int(best))
                    cov[neighbors[best]] += 1
                    uncovered_mask = cov == 0
            
            if len(code) < best_size:
                best_code = code[:]
                best_size = len(code)
            
            # Subgradient update: g_v = 1 - sum_{u in N(v)} x_u
            # where x_u = 1 if u is in code
            x = np.zeros(space_size, dtype=np.int32)
            for c in code:
                x[c] = 1
            
            # g_v = 1 - sum_{u in N(v)} x_u = 1 - cov_v
            # But we want the constraint violation: if sum_{u in N(v)} x_u < 1, violation
            # g_v = 1 - sum_{u in N(v)} x_u (positive if violated)
            cov_arr = np.zeros(space_size, dtype=np.int32)
            for c in code:
                cov_arr[neighbors[c]] += 1
            g = 1.0 - cov_arr.astype(np.float64)
            g = np.maximum(g, 0.0)  # Only positive violations matter
            
            # Subgradient step: lam_v += step * g_v
            step = step_sizes[min(it // 10, len(step_sizes) - 1)]
            lam += step * g
            
            # Ensure lam >= 0 (dual feasibility)
            lam = np.maximum(lam, 0.0)
        
        return best_code
    
    def greedy_cover(seed=0):
        """Standard greedy set cover for fallback/initialization."""
        rng = np.random.default_rng(seed)
        uncovered = np.ones(space_size, dtype=bool)
        code = []
        while np.any(uncovered) and (time.time() - t_start) < deadline - 15.0:
            gains = np.sum(uncovered[neighbors], axis=1)
            perturb = rng.random(space_size) * 0.001
            gains_pert = gains + perturb
            best = np.argmax(gains_pert)
            if gains[best] == 0:
                break
            code.append(int(best))
            uncovered[neighbors[best]] = False
        return code
    
    def greedy_repair(code_set, cov, deadline_time):
        code_list = list(code_set)
        uncovered_mask = cov == 0
        while np.any(uncovered_mask) and (time.time() - t_start) < deadline_time:
            gains = np.sum(uncovered_mask[neighbors], axis=1)
            best = np.argmax(gains)
            if gains[best] == 0:
                break
            code_list.append(int(best))
            cov[neighbors[best]] += 1
            uncovered_mask = cov == 0
        return code_list, cov
    
    def local_search_removal(code, cov, deadline_time):
        """Remove redundant codewords."""
        cur_code = code[:]
        cur_cov = cov.copy()
        
        for _ in range(100):
            if (time.time() - t_start) > deadline_time:
                break
            changed = False
            for i in range(len(cur_code)):
                c = cur_code[i]
                if np.min(cur_cov[neighbors[c]]) >= 2:
                    cur_code.pop(i)
                    cur_cov[neighbors[c]] -= 1
                    changed = True
                    break
            if not changed:
                break
        
        return cur_code, cur_cov
    
    # Phase 1: Run Lagrangian relaxation with multiple seeds
    best_code = None
    best_size = space_size + 1
    
    for seed in range(15):
        if (time.time() - t_start) > deadline - 40.0:
            break
        
        code = lagrangian_relaxation(seed)
        if code is None:
            continue
        
        cov = build_cov(code)
        if not np.all(cov > 0):
            code, cov = greedy_repair(code, cov, deadline - 30.0)
        
        # Local search removal
        code, cov = local_search_removal(code, cov, deadline - 25.0)
        
        if len(code) < best_size:
            best_code = code[:]
            best_size = len(code)
    
    # Phase 2: Also try standard greedy for comparison
    for seed in range(5):
        if (time.time() - t_start) > deadline - 25.0:
            break
        
        code = greedy_cover(seed)
        cov = build_cov(code)
        if not np.all(cov > 0):
            code, cov = greedy_repair(code, cov, deadline - 20.0)
        
        code, cov = local_search_removal(code, cov, deadline - 15.0)
        
        if len(code) < best_size:
            best_code = code[:]
            best_size = len(code)
    
    # Phase 3: Multi-kick local search on best solution
    if (time.time() - t_start) < deadline - 15.0 and best_code is not None:
        rng = np.random.default_rng(42)
        cur_code = best_code[:]
        cur_cov = build_cov(cur_code)
        
        for restart in range(5):
            if (time.time() - t_start) > deadline - 10.0:
                break
            
            for k in range(1, 4):
                if (time.time() - t_start) > deadline - 8.0:
                    break
                for trial in range(30):
                    if (time.time() - t_start) > deadline - 8.0:
                        break
                    if len(cur_code) <= k:
                        break
                    
                    remove_indices = rng.choice(len(cur_code), size=k, replace=False)
                    remove_set = set(int(ri) for ri in remove_indices)
                    
                    new_code = [c for i, c in enumerate(cur_code) if i not in remove_set]
                    new_cov = build_cov(new_code)
                    
                    new_code, new_cov = greedy_repair(new_code, new_cov, deadline - 12.0)
                    
                    if np.all(new_cov > 0) and len(new_code) < len(cur_code):
                        cur_code = new_code
                        cur_cov = new_cov
                        break
                
                # Local search removal after each k
                if (time.time() - t_start) < deadline - 8.0:
                    cur_code, cur_cov = local_search_removal(cur_code, cur_cov, deadline - 6.0)
            
            if len(cur_code) < best_size:
                best_code = cur_code[:]
                best_size = len(cur_code)
    
    # Final aggressive removal
    if (time.time() - t_start) < deadline - 3.0 and best_code is not None:
        cur_code = best_code[:]
        cur_cov = build_cov(cur_code)
        cur_code, cur_cov = local_search_removal(cur_code, cur_cov, deadline - 2.0)
        if len(cur_code) < best_size:
            best_code = cur_code[:]
            best_size = len(cur_code)
    
    # Final verification
    if best_code is None or len(best_code) == 0:
        best_code = greedy_cover(0)
    
    final_cov = build_cov(best_code)
    if not np.all(final_cov > 0):
        fc = greedy_cover(0)
        fcov = build_cov(fc)
        if np.all(fcov > 0):
            best_code = fc
        else:
            best_code = list(range(space_size))
    
    return [idx_to_word(c) for c in best_code]
