import time
import random
from itertools import combinations

def construct():
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        return edges, adj

    def bron_kerbosch(adj, R, P, X):
        if not P and not X:
            if len(R) >= 2:
                yield tuple(sorted(R))
            return
        if P or X:
            u = max(P | X, key=lambda v: len(P & adj[v]))
            candidates = P - adj[u]
        else:
            candidates = set()
        for v in candidates:
            yield from bron_kerbosch(adj, R | {v}, P & adj[v], X & adj[v])
            P.remove(v)
            X.add(v)

    def get_maximal_cliques(adj):
        n = len(adj)
        return set(bron_kerbosch(adj, set(), set(range(n)), set()))

    def lagrangian_set_cover(cliques, n, rng, ordering_seed=0, max_subgrad_iters=200):
        """Lagrangian relaxation with subgradient optimization for set cover."""
        m = len(cliques)
        if m == 0:
            return set()
        
        # Precompute: for each vertex, which cliques it covers
        vertex_to_cliques = {}
        for i, c in enumerate(cliques):
            for v in c:
                if v not in vertex_to_cliques:
                    vertex_to_cliques[v] = set()
                vertex_to_cliques[v].add(i)
        
        # For each clique, which vertices can cover it
        clique_to_vertices = {}
        for i, c in enumerate(cliques):
            clique_to_vertices[i] = set(c)
        
        # Subgradient optimization for dual variables (lambda_i for each clique)
        # Dual: max sum_i lambda_i s.t. lambda_i >= 0, sum_{i: v in clique_i} lambda_i <= 1 for all v
        # This is a packing problem. We use subgradient method.
        
        # Initialize lambda to small positive values
        lam = [1.0 / max(1, len(clique_to_vertices[i])) for i in range(m)]
        
        best_primal = None
        best_primal_size = n + 1
        
        # Track best dual solution found
        best_dual_val = -1
        
        step_scale = 1.0
        alpha = 1.0  # Step size multiplier
        
        for it in range(max_subgrad_iters):
            # Primal heuristic: select vertices with highest reduced cost
            # Reduced cost for vertex v: 1 - sum_{i: v in clique_i} lambda_i
            # We want to select vertices where this is most negative (i.e., they cover many high-lambda cliques)
            
            # Compute for each vertex: sum of lambda over cliques it covers
            v_lambda_sum = [0.0] * n
            for i in range(m):
                if lam[i] <= 0:
                    continue
                for v in clique_to_vertices[i]:
                    v_lambda_sum[v] += lam[i]
            
            # Primal solution: greedily select vertices with highest v_lambda_sum
            # But we need to ensure all cliques are covered
            # Strategy: sort vertices by v_lambda_sum descending, add until all covered
            # Then polish
            
            uncovered = set(range(m))
            selected = set()
            
            # Sort vertices by their lambda sum (descending)
            v_order = sorted(range(n), key=lambda v: v_lambda_sum[v], reverse=True)
            
            for v in v_order:
                if not uncovered:
                    break
                # Check if v covers any uncovered clique
                covers_uncovered = False
                for i in uncovered:
                    if v in clique_to_vertices[i]:
                        covers_uncovered = True
                        break
                if covers_uncovered:
                    selected.add(v)
                    # Remove covered cliques
                    covered = vertex_to_cliques.get(v, set())
                    uncovered -= covered
            
            # If still uncovered, add vertices to cover remaining
            while uncovered:
                # Find vertex covering most uncovered cliques
                best_v = -1
                best_count = 0
                for v in range(n):
                    count = len(uncovered & vertex_to_cliques.get(v, set()))
                    if count > best_count:
                        best_count = count
                        best_v = v
                if best_v == -1 or best_count == 0:
                    break
                selected.add(best_v)
                covered = vertex_to_cliques.get(best_v, set())
                uncovered -= covered
            
            # Polish: try removing vertices
            changed = True
            while changed:
                changed = False
                for v in list(selected):
                    candidate = selected - {v}
                    valid = True
                    for i in range(m):
                        if not any(x in candidate for x in cliques[i]):
                            valid = False
                            break
                    if valid:
                        selected = candidate
                        changed = True
            
            if len(selected) < best_primal_size:
                best_primal_size = len(selected)
                best_primal = set(selected)
            
            # Subgradient update for lambda
            # g_i = 1 - sum_{v in clique_i} x_v  (where x_v = 1 if v selected)
            # This is the subgradient of the dual objective
            x = [0.0] * n
            for v in selected:
                x[v] = 1.0
            
            grad = [0.0] * m
            for i in range(m):
                s = 0.0
                for v in clique_to_vertices[i]:
                    s += x[v]
                grad[i] = 1.0 - s
            
            # Compute step size
            # ||grad||^2
            grad_sq = sum(g * g for g in grad)
            if grad_sq < 1e-12:
                break
            
            # Upper bound for dual: n (since each lambda_i <= 1 and sum constraint)
            # Actually, a simple upper bound: sum of max lambda_i over cliques
            ub = float(n)  # Crude upper bound
            
            dual_val = sum(lam[i] for i in range(m))
            if dual_val > best_dual_val:
                best_dual_val = dual_val
            
            # Step size: alpha * (ub - dual_val) / grad_sq
            if ub > dual_val:
                step = alpha * (ub - dual_val) / grad_sq
            else:
                step = alpha / grad_sq
            
            # Update lambda
            new_lam = []
            for i in range(m):
                new_val = max(0.0, lam[i] + step * grad[i])
                new_lam.append(new_val)
            
            # Also enforce the constraint: sum_{i: v in clique_i} lambda_i <= 1
            # Project by scaling down if violated
            v_lam_sum = [0.0] * n
            for i in range(m):
                for v in clique_to_vertices[i]:
                    v_lam_sum[v] += new_lam[i]
            
            max_violation = max(v_lam_sum) if v_lam_sum else 0
            if max_violation > 1.0:
                scale = 1.0 / max_violation
                new_lam = [l * scale for l in new_lam]
            
            lam = new_lam
            
            # Decay alpha occasionally
            if it % 50 == 49:
                alpha *= 0.8
        
        return best_primal if best_primal is not None else set(range(n))

    def local_search(selected, cliques, n, rng, max_iters=100):
        """Try to improve solution via single-vertex swaps and removals."""
        selected = set(selected)
        
        # Precompute for efficiency
        vertex_to_cliques = {}
        for i, c in enumerate(cliques):
            for v in c:
                if v not in vertex_to_cliques:
                    vertex_to_cliques[v] = set()
                vertex_to_cliques[v].add(i)
        
        def is_valid(sol):
            """Check if sol covers all cliques."""
            for c in cliques:
                if not any(v in sol for v in c):
                    return False
            return True
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            # Try removals
            for v in list(selected):
                candidate = selected - {v}
                if is_valid(candidate):
                    selected = candidate
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swaps: remove one, add one
            for v_out in list(selected):
                for v_in in range(n):
                    if v_in in selected:
                        continue
                    candidate = (selected - {v_out}) | {v_in}
                    if len(candidate) == len(selected):  # Same size, no improvement
                        continue
                    if is_valid(candidate):
                        selected = candidate
                        improved = True
                        break
                if improved:
                    break
        
        return selected

    def polish(selected, cliques, n):
        """Final polish: try all possible single removals in multiple passes."""
        selected = set(selected)
        
        changed = True
        while changed:
            changed = False
            for v in list(selected):
                candidate = selected - {v}
                valid = True
                for c in cliques:
                    if not any(x in candidate for x in c):
                        valid = False
                        break
                if valid:
                    selected = candidate
                    changed = True
        
        return selected

    out = []
    start_time = time.time()
    total_deadline = 110.0  # Leave 10s margin before 120s kill
    
    for idx, (n, p, seed) in enumerate(((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203))):
        # Allocate time proportional to remaining budget
        remaining = total_deadline - (time.time() - start_time)
        if remaining <= 0:
            out.append(sorted(range(n)))  # Fallback
            continue
        
        # Allocate ~1/3 of remaining for each instance, with minimum 10s
        inst_budget = max(10.0, remaining / (3 - idx))
        inst_start = time.time()
        
        edges, adj = random_graph(n, p, seed)
        cliques = get_maximal_cliques(adj)
        
        if not cliques:
            out.append([])
            continue
        
        cliques_list = list(cliques)
        
        best_set = None
        best_size = n + 1
        
        # Phase 1: Lagrangian relaxation runs with different initializations
        num_lagr = 0
        while time.time() - inst_start < inst_budget * 0.6:
            iter_seed = 1000 + num_lagr
            rng_iter = random.Random(iter_seed)
            order_seed = 5000 + num_lagr
            
            # Vary max iterations based on time remaining
            remaining_frac = (inst_budget * 0.6 - (time.time() - inst_start)) / max(0.1, inst_budget * 0.6)
            max_iters = max(50, int(200 * remaining_frac))
            
            candidate = lagrangian_set_cover(cliques_list, n, rng_iter, ordering_seed=order_seed, max_subgrad_iters=max_iters)
            candidate = polish(candidate, cliques_list, n)
            
            if len(candidate) < best_size:
                best_size = len(candidate)
                best_set = candidate
                
            num_lagr += 1
        
        # Phase 2: Local search on best solution
        if best_set is not None and time.time() - inst_start < inst_budget * 0.9:
            ls_rng = random.Random(99999)
            improved = local_search(best_set, cliques_list, n, ls_rng, max_iters=50)
            improved = polish(improved, cliques_list, n)
            if len(improved) < best_size:
                best_size = len(improved)
                best_set = improved
        
        # Phase 3: More Lagrangian runs if time remains
        while time.time() - inst_start < inst_budget * 0.95:
            iter_seed = 20000 + num_lagr
            rng_iter = random.Random(iter_seed)
            order_seed = 15000 + num_lagr
            
            remaining_frac = (inst_budget * 0.95 - (time.time() - inst_start)) / max(0.1, inst_budget * 0.95)
            max_iters = max(30, int(150 * remaining_frac))
            
            candidate = lagrangian_set_cover(cliques_list, n, rng_iter, ordering_seed=order_seed, max_subgrad_iters=max_iters)
            candidate = polish(candidate, cliques_list, n)
            
            if len(candidate) < best_size:
                best_size = len(candidate)
                best_set = candidate
                # Try local search on new best
                if time.time() - inst_start < inst_budget * 0.95:
                    ls_rng2 = random.Random(88888 + num_lagr)
                    improved = local_search(best_set, cliques_list, n, ls_rng2, max_iters=30)
                    improved = polish(improved, cliques_list, n)
                    if len(improved) < best_size:
                        best_size = len(improved)
                        best_set = improved
                        
            num_lagr += 1
        
        if best_set is None:
            best_set = set(range(n))
        
        out.append(sorted(best_set))
    
    return out
