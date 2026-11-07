import time
import sys

def construct():
    from random import Random
    
    # Keep the exact generator as required
    def instance(splits, seed, palette=5):
        rng = Random(seed)
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        edges = {(min(a, b), max(a, b)) for f in faces for a, b in zip(f, f[1:] + f[:1])}
        n = 8
        for _ in range(splits):
            face = faces.pop(rng.randrange(len(faces)))
            inner = list(range(n, n + 4))
            n += 4
            for i in range(4):
                a, b = face[i], face[(i + 1) % 4]
                edges.add((min(a, inner[i]), max(a, inner[i])))
                edges.add((min(inner[i], inner[(i + 1) % 4]), max(inner[i], inner[(i + 1) % 4])))
                faces.append([a, b, inner[(i + 1) % 4], inner[i]])
            faces.append(inner)
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        return n, sorted(edges), lists

    def solve_instance(n, edges, lists, deadline):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Initialize domains
        # domains[v] is a set of available colors
        domains = [set(l) for l in lists]
        color = [-1] * n
        uncolored = set(range(n))
        
        # MRV Backtracking with Forward Checking
        # To optimize, we can pre-sort vertices by degree or just use dynamic MRV
        # Dynamic MRV: at each step, find uncolored vertex with min len(domains[v])
        
        def backtrack():
            # Check deadline
            if time.time() > deadline:
                return False
            
            if not uncolored:
                return True
            
            # Select vertex with Minimum Remaining Values (MRV)
            # Tie-breaker: Highest Degree (Least Constraining Value is harder, so we pick most constrained variable)
            v = None
            min_size = float('inf')
            for u in uncolored:
                size = len(domains[u])
                if size == 0:
                    return False # Dead end
                if size < min_size:
                    min_size = size
                    v = u
                    if size == 1:
                        break # Can't get better than 1
            
            # Try each color in domain of v
            for c in list(domains[v]):
                # Assign c to v
                color[v] = c
                uncolored.remove(v)
                
                # Forward Checking: remove c from neighbors' domains
                # Record changes to undo
                changes = []
                valid = True
                for u in adj[v]:
                    if u in uncolored:
                        if c in domains[u]:
                            domains[u].discard(c)
                            changes.append((u, c))
                            if not domains[u]:
                                valid = False
                                break
                
                if valid:
                    if backtrack():
                        return True
                
                # Undo changes
                for u, c_removed in changes:
                    domains[u].add(c_removed)
                
                color[v] = -1
                uncolored.add(v)
            
            return False
        
        if backtrack():
            # Ensure types are Python int
            return [int(c) for c in color]
        else:
            # Should not happen if Alon-Tarsi holds, but return best effort or None
            # The problem states all are colourable.
            # If we fail due to time or bug, return a dummy valid structure to avoid crash, 
            # though it will score 0 for that instance.
            return [0] * n

    start_time = time.time()
    # 120s limit, keep 10s margin for safety and overhead
    deadline = start_time + 110.0
    
    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            # If we run out of time, we must return something valid structurally
            # to avoid "invalid result" which scores 0 for everything.
            # However, the prompt says "An instance whose colouring breaks a list or an edge scores 0... 
            # but a wrong length or type makes the whole result invalid."
            # So we must return 4 lists of correct length.
            # We can try to generate the instance and return a greedy solution or zeros.
            n, edges, lists = instance(splits, seed)
            # Fallback: Greedy
            adj = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            colour = [0] * n
            for v in range(n):
                taken = {colour[u] for u in adj[v]}
                free = [c for c in lists[v] if c not in taken]
                colour[v] = free[0] if free else lists[v][0]
            out.append([int(c) for c in colour])
            continue
            
        n, edges, lists = instance(splits, seed)
        # Pass deadline to solver
        res = solve_instance(n, edges, lists, deadline)
        # Ensure res is a list of ints
        if res is None:
            res = [0] * n
        out.append([int(c) for c in res])
        
    return out

if __name__ == "__main__":
    # Simple self-test
    res = construct()
    # Verify structure
    assert len(res) == 4
    for i, coloring in enumerate(res):
        assert isinstance(coloring, list)
        for c in coloring:
            assert isinstance(c, int)
            assert 0 <= c <= 4
    print("Structure valid")
