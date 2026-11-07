import time
import sys
from collections import deque

def construct():
    def instance(splits, seed, palette=5):
        from random import Random
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

    def solve(n, edges, lists, time_limit=10.0):
        start_time = time.time()
        
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Initial domains are the lists of available colors
        # Using sets for fast intersection and difference
        domains = [set(l) for l in lists]
        
        # Check for initial empty domains
        for i in range(n):
            if not domains[i]:
                return None

        # AC-3 style propagation to ensure arc consistency initially
        # This might prune some domains before search starts
        def propagate(queue):
            # queue contains (u, v) meaning check if domain[u] is consistent with domain[v]
            while queue:
                u, v = queue.popleft()
                if not domains[u]:
                    return False
                # Remove colors from domain[u] that are not in domain[v]
                # Since it's a bipartite graph and we want proper coloring, 
                # actually for list coloring on bipartite graphs, 
                # the constraint is just c_u != c_v.
                # So we remove from domain[u] any color that is the *only* choice for v? 
                # No, AC-3 for binary constraint c_u != c_v removes c from D(u) 
                # if D(v) == {c}.
                
                # Wait, standard AC-3 for != constraint:
                # If D(v) has only one color c, then c cannot be in D(u).
                if len(domains[v]) == 1:
                    c = next(iter(domains[v]))
                    if c in domains[u]:
                        domains[u].remove(c)
                        if not domains[u]:
                            return False
                        # If domain of u changed, we need to re-check neighbors of u
                        for w in adj[u]:
                            if w != v:
                                queue.append((w, u))
            return True

        # Initialize queue with all edges
        initial_queue = deque()
        for u, v in edges:
            initial_queue.append((u, v))
            initial_queue.append((v, u))
            
        if not propagate(initial_queue):
            return None

        # Backtracking search with MRV and Forward Checking
        # We use a copy of domains for the search state to allow backtracking
        # However, copying sets for every node is expensive. 
        # Given the small size (max ~ 8 + 4*100 = 408 vertices), and 3 colors,
        # we can afford a simple recursive backtracking with explicit domain tracking 
        # or just re-checking constraints. 
        # Let's use a simpler approach: assign colors, check consistency with neighbors.
        # But to be safe and fast, we'll use a depth-first search with MRV.
        
        # To avoid deep recursion limits and overhead, we implement an iterative DFS
        # or a recursive one with a small limit. n is ~400, recursion depth 400 is fine.
        
        color = [-1] * n
        # We need to track which vertices are unassigned
        unassigned = set(range(n))
        
        # Helper to find MRV vertex
        def get_mrv_vertex():
            min_size = float('inf')
            best_v = -1
            for v in unassigned:
                # Calculate remaining options considering already assigned neighbors
                # A color c is valid for v if c in lists[v] and c not in {color[u] for u in adj[v] if color[u] != -1}
                used_colors = set()
                for u in adj[v]:
                    if color[u] != -1:
                        used_colors.add(color[u])
                avail = [c for c in lists[v] if c not in used_colors]
                if len(avail) < min_size:
                    min_size = len(avail)
                    best_v = v
                    if min_size == 0:
                        return v, [] # Fail
                    if min_size == 1:
                        break # Can't get better than 1
            return best_v, [c for c in lists[best_v] if c not in set(color[u] for u in adj[best_v] if color[u] != -1)]

        def backtrack():
            if not unassigned:
                return True
            
            if time.time() - start_time > time_limit:
                return False # Timeout, treat as failure for this path
            
            v, avail = get_mrv_vertex()
            if not avail:
                return False
            
            unassigned.remove(v)
            
            for c in avail:
                color[v] = c
                if backtrack():
                    return True
                color[v] = -1
            
            unassigned.add(v)
            return False

        if backtrack():
            # Verify solution
            for u in range(n):
                if color[u] not in lists[u]:
                    return None
                for v in adj[u]:
                    if u < v and color[u] == color[v]:
                        return None
            return [int(c) for c in color]
        else:
            return None

    out = []
    start_time = time.time()

    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        elapsed = time.time() - start_time
        # Safety margin: if we've used 140s, we can't solve the next one in time.
        # But we must return 4 lists. If we fail, we return a dummy valid structure?
        # No, the contract says "An instance whose colouring breaks a list or an edge scores 0".
        # It does not say returning a dummy scores 0 for the whole result, but "wrong length or type makes the whole result invalid".
        # So we must return a list of 4 lists of ints.
        # If solve returns None, we need to provide a fallback. 
        # Since the problem is guaranteed to be solvable by Alon-Tarsi, 
        # and our solver is robust, this is unlikely. 
        # If it happens, we return a list of 0s? No, that might break edge constraints.
        # But returning 0s is at least a valid type. It will score 0 for that instance.
        # It's better to score 0 for one instance than make the whole result invalid.
        
        n, edges, lists = instance(splits, seed)
        sol = solve(n, edges, lists)
        
        if sol is None:
            # Fallback: try a greedy assignment just to have a valid type, 
            # though it will likely be invalid coloring.
            # Or just return zeros.
            sol = [0] * n
            
        out.append([int(c) for c in sol])

    return out

if __name__ == "__main__":
    # Simple self-test
    res = construct()
    print(len(res))
    for r in res:
        print(len(r))
