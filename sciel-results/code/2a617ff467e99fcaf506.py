import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch(adj):
        """
        Yields all maximal cliques in the graph.
        Uses the pivot heuristic.
        """
        n = len(adj)
        
        def bk(R, P, X):
            if not P and not X:
                yield R
                return
            
            # Pivot selection: choose vertex in P ∪ X with max degree in P
            pivot_candidates = P | X
            if not pivot_candidates:
                return
            
            # Find pivot u in P ∪ X that maximizes |P ∩ N(u)|
            u = None
            max_inter = -1
            for v in pivot_candidates:
                inter = len(P & adj[v])
                if inter > max_inter:
                    max_inter = inter
                    u = v
            
            # Branch on vertices in P that are NOT neighbors of u
            candidates = P - adj[u]
            
            for v in list(candidates):
                new_R = R | {v}
                new_P = P & adj[v]
                new_X = X & adj[v]
                yield from bk(new_R, new_P, new_X)
                
                P.remove(v)
                X.add(v)
        
        yield from bk(set(), set(range(n)), set())

    def solve_hit_set(cliques, n):
        """
        Finds a minimum hitting set for the given list of cliques.
        Each clique is a frozenset of vertices.
        Returns the set of vertices.
        """
        if not cliques:
            return set()
        
        # Convert cliques to bitmasks for fast intersection checks
        clique_masks = []
        for c in cliques:
            mask = 0
            for v in c:
                mask |= (1 << v)
            clique_masks.append(mask)
        
        best_size = float('inf')
        best_mask = 0
        n_cliques = len(clique_masks)
        
        # Precompute coverage: for each vertex, which cliques it hits
        # coverage[v] = list of clique indices that contain v
        coverage = [[] for _ in range(n)]
        for i, m in enumerate(clique_masks):
            for v in range(n):
                if m & (1 << v):
                    coverage[v].append(i)
        
        def lower_bound(remaining_indices):
            """
            Simple lower bound: ceil(len(remaining) / max_coverage)
            """
            if not remaining_indices:
                return 0
            
            # Find max coverage among remaining
            max_cov = 0
            for v in range(n):
                bit = 1 << v
                cov = 0
                for idx in remaining_indices:
                    if clique_masks[idx] & bit:
                        cov += 1
                if cov > max_cov:
                    max_cov = cov
            
            if max_cov == 0:
                return float('inf')
                
            return (len(remaining_indices) + max_cov - 1) // max_cov

        def solve(remaining_indices, current_mask, current_size):
            nonlocal best_size, best_mask
            
            if time.time() > deadline:
                return
                
            if not remaining_indices:
                if current_size < best_size:
                    best_size = current_size
                    best_mask = current_mask
                return
            
            if current_size >= best_size:
                return
                
            lb = lower_bound(remaining_indices)
            if current_size + lb >= best_size:
                return
            
            # Select a clique to branch on.
            # Pick the smallest clique (fewest vertices) to reduce branching
            target_idx = remaining_indices[0]
            min_size = n
            for idx in remaining_indices:
                sz = bin(clique_masks[idx]).count('1')
                if sz < min_size:
                    min_size = sz
                    target_idx = idx
            
            target_mask = clique_masks[target_idx]
            
            # Get vertices in target_mask
            vertices = []
            for v in range(n):
                if target_mask & (1 << v):
                    vertices.append(v)
            
            # Order vertices by how many other remaining cliques they cover (most first)
            def coverage_score(v):
                bit = 1 << v
                cnt = 0
                for idx in remaining_indices:
                    if clique_masks[idx] & bit:
                        cnt += 1
                return cnt
            
            vertices.sort(key=coverage_score, reverse=True)
            
            for v in vertices:
                bit = 1 << v
                new_mask = current_mask | bit
                new_remaining = []
                for idx in remaining_indices:
                    if not (new_mask & clique_masks[idx]):
                        new_remaining.append(idx)
                
                solve(new_remaining, new_mask, current_size + 1)
                
                # Prune: if we found a solution of size 1, we can't do better
                if best_size == 1:
                    return

        # Start with all clique indices
        all_indices = list(range(n_cliques))
        solve(all_indices, 0, 0)
        
        # Convert best_mask back to set
        result = set()
        for v in range(n):
            if best_mask & (1 << v):
                result.add(v)
        return result

    def local_search_improve(solution, cliques, n, time_limit):
        """
        Try to improve the solution using local search:
        1. Remove any non-essential vertex
        2. Try 1-exchanges (swap one vertex for another)
        """
        current_sol = set(solution)
        improved = True
        
        while improved and time.time() < time_limit:
            improved = False
            
            # 1. Try to remove any vertex
            for v in list(current_sol):
                if time.time() > time_limit:
                    break
                # Check if current_sol - {v} is valid
                # i.e., does every clique intersect current_sol - {v}?
                # Equivalent: Is there a clique C such that C & current_sol == {v}?
                essential = False
                for c in cliques:
                    if c & current_sol == {v}:
                        essential = True
                        break
                if not essential:
                    current_sol.remove(v)
                    improved = True
                    # Restart loop to check for more removals
                    break 
            
            if improved:
                continue
            
            # 2. Try 1-exchange: swap v in sol with u not in sol
            sol_list = list(current_sol)
            for v in sol_list:
                if time.time() > time_limit:
                    break
                for u in range(n):
                    if u in current_sol:
                        continue
                    # Try swap
                    new_sol = set(current_sol)
                    new_sol.remove(v)
                    new_sol.add(u)
                    
                    # Check if new_sol is valid
                    valid = True
                    for c in cliques:
                        if not (c & new_sol):
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # If valid, check if we can remove any vertex from new_sol
                    # If yes, accept the swap and the removal
                    for w in list(new_sol):
                        essential = False
                        for c in cliques:
                            if c & new_sol == {w}:
                                essential = True
                                break
                        if not essential:
                            new_sol.remove(w)
                            current_sol = new_sol
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        return current_sol

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            # Fallback: fill remaining with all vertices
            while len(out) < 3:
                # Determine n for the next instance
                idx = len(out)
                next_n = (30, 35, 40)[idx]
                out.append(list(range(next_n)))
            break
            
        adj = random_graph(n, p, seed)
        
        # Get maximal cliques of size >= 2
        cliques = []
        for c in bron_kerbosch(adj):
            if len(c) >= 2:
                cliques.append(frozenset(c))
        
        # Solve hitting set using branch and bound
        solution = solve_hit_set(cliques, n)
        
        # Apply local search to potentially improve
        if time.time() < deadline - 5.0:
            solution = local_search_improve(solution, cliques, n, deadline - 2.0)
        
        out.append(sorted(solution))
        
        # Check time
        if time.time() > deadline:
             while len(out) < 3:
                 idx = len(out)
                 next_n = (30, 35, 40)[idx]
                 out.append(list(range(next_n)))
             break

    return out
