def construct():
    import random

    def get_random_maximal_independent_set(adj, n, rng):
        remaining_set = set(range(n))
        mis = set()
        while remaining_set:
            degs = {}
            for v in remaining_set:
                degs[v] = sum(1 for u in remaining_set if u != v and adj[v][u])
            min_deg = min(degs.values())
            candidates = [v for v in remaining_set if degs[v] == min_deg]
            v = rng.choice(candidates)
            mis.add(v)
            neighbors = {u for u in remaining_set if adj[v][u]}
            remaining_set.discard(v)
            remaining_set -= neighbors
        return mis

    def has_independent_set_of_size_k(adj, n, k):
        # Check if alpha >= k by finding a clique of size k in the complement
        # Build complement adjacency
        comp = [[(i != j and not adj[i][j]) for j in range(n)] for i in range(n)]
        
        def backtrack(current_size, candidates):
            if current_size == k:
                return True
            if not candidates:
                return False
            if current_size + len(candidates) < k:
                return False
            
            for idx, v in enumerate(candidates):
                # v is connected to all in current clique (by construction of candidates)
                # New candidates are those after v that are connected to v
                new_candidates = [u for u in candidates[idx+1:] if comp[v][u]]
                if backtrack(current_size + 1, new_candidates):
                    return True
            return False
            
        return backtrack(0, list(range(n)))

    # Try n from 39 down to 27
    for n in range(39, 26, -1):
        for attempt in range(100):
            rng = random.Random(attempt)
            adj = [[False]*n for _ in range(n)]
            for i in range(n):
                if i == 0:
                    continue
                mis = get_random_maximal_independent_set(adj, i, rng)
                for v in mis:
                    adj[i][v] = True
                    adj[v][i] = True
            
            # Check if valid: no independent set of size 10
            if not has_independent_set_of_size_k(adj, n, 10):
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if adj[i][j]:
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
            
    # Fallback to seed logic
    n = 26
    edges = []
    steps = [x for x in range(1, n) if x % 3 == 1]
    for i in range(n):
        for s in steps:
            j = (i + s) % n
            if i < j:
                edges.append([i, j])
    return {'n': n, 'edges': edges}
