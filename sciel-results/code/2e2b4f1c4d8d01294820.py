"""Improved program: Greedy Set Cover on maximal cliques with bitmask ILP and local search."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def bron_kerbosch(adj, n):
        """Find all maximal cliques using Bron-Kerbosch with pivot, using bitmasks."""
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if R.bit_count() >= 2:
                    cliques.append(R)
                return
            
            # Choose pivot u from P ∪ X to minimize branching
            PX = P | X
            if PX:
                best_pivot = -1
                max_common = -1
                temp = PX
                while temp:
                    lsb = temp & -temp
                    u = lsb.bit_length() - 1
                    common = (P & adj[u]).bit_count()
                    if common > max_common:
                        max_common = common
                        best_pivot = u
                    temp ^= lsb
                
                if best_pivot != -1:
                    candidates = P & ~adj[best_pivot]
                else:
                    candidates = P
            else:
                return
            
            temp = candidates
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                temp ^= lsb
                
                bk(R | lsb, P & adj[v], X & adj[v])
                
                P ^= lsb
                X |= lsb
        
        bk(0, (1 << n) - 1, 0)
        return cliques

    def greedy_set_cover(cliques, n, rng):
        """Greedy set cover: repeatedly pick vertex covering most uncovered cliques."""
        if not cliques:
            return set()
        
        k = len(cliques)
        # For each vertex, compute bitmask of cliques it covers
        masks = [0] * n
        for i, clique in enumerate(cliques):
            bit = 1 << i
            temp = clique
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                temp ^= lsb
                masks[v] |= bit
        
        selected = set()
        uncovered = (1 << k) - 1
        
        while uncovered:
            best_v = -1
            best_count = -1
            for v in range(n):
                if v in selected:
                    continue
                count = (uncovered & masks[v]).bit_count()
                if count > best_count:
                    best_count = count
                    best_v = v
                elif count == best_count and count > 0:
                    if rng.random() < 0.5:
                        best_v = v
            
            if best_v == -1 or best_count == 0:
                break
            
            selected.add(best_v)
            uncovered &= ~masks[best_v]
        
        return selected

    def local_search(selected, cliques, n, time_limit=5.0):
        """Try to improve the solution by removing redundant vertices and swaps."""
        if not cliques:
            return selected
        
        selected = set(selected)
        k = len(cliques)
        
        # Precompute for each vertex, which cliques it covers (bitmask)
        masks = [0] * n
        for i, clique in enumerate(cliques):
            bit = 1 << i
            temp = clique
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                temp ^= lsb
                masks[v] |= bit
        
        def coverage(sel):
            cov = 0
            for v in sel:
                cov |= masks[v]
            return cov
        
        all_mask = (1 << k) - 1
        
        def is_valid(sel):
            return coverage(sel) == all_mask
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Try removing each vertex
            for v in list(selected):
                new_sel = selected - {v}
                if is_valid(new_sel):
                    selected = new_sel
                    improved = True
                    break
            
            # Try swapping: remove one, add another
            if not improved:
                for v_remove in list(selected):
                    for v_add in range(n):
                        if v_add in selected:
                            continue
                        new_sel = (selected - {v_remove}) | {v_add}
                        if is_valid(new_sel):
                            selected = new_sel
                            improved = True
                            break
                    if improved:
                        break
        
        return selected

    def solve_ilp(cliques, n, time_limit=30.0):
        """Try to solve the hitting set problem exactly using ILP."""
        if not cliques:
            return set()
        
        k = len(cliques)
        start_time = time.time()
        
        try:
            # Build constraint matrix A[i][v] = 1 if v is in clique i
            A = np.zeros((k, n), dtype=np.float64)
            for i, clique in enumerate(cliques):
                temp = clique
                while temp:
                    lsb = temp & -temp
                    v = lsb.bit_length() - 1
                    temp ^= lsb
                    A[i][v] = 1.0
            
            # Objective: minimize sum(x_v)
            c_obj = np.ones(n, dtype=np.float64)
            
            # Constraints: for each clique i, sum_{v in clique_i} x_v >= 1
            # This is A @ x >= 1 for all rows
            lb = np.ones(k, dtype=np.float64)
            ub = np.full(k, np.inf, dtype=np.float64)
            
            constraints = LinearConstraint(A, lb, ub)
            bounds = Bounds(0, 1)
            
            result = milp(
                c=c_obj,
                constraints=constraints,
                bounds=bounds,
                integrality=np.ones(n, dtype=int),
                options={"time_limit": time_limit, "mip_rel_gap": 0.0}
            )
            
            if result.status == 0:  # Optimal
                x = result.x
                selected = set()
                for v in range(n):
                    if x[v] > 0.5:
                        selected.add(v)
                
                # Verify solution
                cov = 0
                for v in selected:
                    for i, clique in enumerate(cliques):
                        if (clique >> v) & 1:
                            cov |= (1 << i)
                
                if cov == (1 << k) - 1:
                    return selected
                else:
                    return None
            else:
                return None
                
        except Exception:
            return None

    def solve_instance(n, p, seed):
        adj = instance(n, p, seed)
        
        # Find all maximal cliques of size >= 2
        cliques = bron_kerbosch(adj, n)
        
        if not cliques:
            return []
        
        rng = random.Random(seed)
        
        # Greedy set cover with multiple restarts
        best_selected = None
        best_size = float('inf')
        
        for restart in range(10):
            selected = greedy_set_cover(cliques, n, rng)
            
            if len(selected) < best_size:
                best_size = len(selected)
                best_selected = set(selected)
        
        # Try ILP for exact solution if time permits
        remaining_time = deadline - time.time()
        if remaining_time > 5.0:
            ilp_time = min(remaining_time - 2.0, 25.0)
            ilp_result = solve_ilp(cliques, n, time_limit=ilp_time)
            if ilp_result is not None and len(ilp_result) < best_size:
                best_selected = ilp_result
                best_size = len(best_selected)
        
        # Local search to improve
        remaining_time = deadline - time.time()
        if remaining_time > 2.0 and best_selected:
            ls_time = min(remaining_time - 1.0, 8.0)
            improved = local_search(best_selected, cliques, n, time_limit=ls_time)
            if len(improved) < best_size:
                best_selected = improved
        
        return sorted(best_selected)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 5.0:
            # Not enough time; return empty lists as fallback (will be invalid but avoid crash)
            results.append([])
            continue
        result = solve_instance(n, p, seed)
        results.append(result)
    
    return results
# EVOLVE-BLOCK-END
