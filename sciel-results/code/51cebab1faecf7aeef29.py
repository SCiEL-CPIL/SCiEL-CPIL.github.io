import time

def construct():
    start_time = time.time()
    deadline = 300.0

    def is_valid_graph(n, edges):
        adj = [[False] * n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        
        # Check for K5
        def has_clique(size):
            # Use bitmask for speed if n < 64
            # Here n is small, simple recursion is ok
            # But let's use a slightly optimized backtracking
            def backtrack(candidates, current):
                if len(current) == size:
                    return True
                if len(candidates) + len(current) < size:
                    return False
                for i in range(len(candidates)):
                    v = candidates[i]
                    # Check connectivity
                    ok = True
                    for u in current:
                        if not adj[v][u]:
                            ok = False
                            break
                    if ok:
                        if backtrack(candidates[i+1:], current + [v]):
                            return True
                return False
            for v in range(n):
                if backtrack(list(range(v+1, n)), [v]):
                    return True
            return False

        if has_clique(5):
            return False

        # Check for Independent Set of size 5
        def has_independent_set(size):
            def backtrack(candidates, current):
                if len(current) == size:
                    return True
                if len(candidates) + len(current) < size:
                    return False
                for i in range(len(candidates)):
                    v = candidates[i]
                    ok = True
                    for u in current:
                        if adj[v][u]:
                            ok = False
                            break
                    if ok:
                        if backtrack(candidates[i+1:], current + [v]):
                            return True
                return False
            for v in range(n):
                if backtrack(list(range(v+1, n)), [v]):
                    return True
            return False

        if has_independent_set(5):
            return False
        return True

    # Try Paley(41) first, then 37
    for q in [41, 37]:
        if time.time() > deadline - 10:
            break
        residues = {(x * x) % q for x in range(1, q)}
        edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
        
        if is_valid_graph(q, edges):
            return q, edges
            
    # Fallback to 37 if 41 failed (should not happen if logic is correct)
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    return q, [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
