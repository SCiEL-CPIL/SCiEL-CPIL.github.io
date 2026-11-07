"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin for 150s limit
    
    from random import Random
    
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

    def solve_csp(n, adj, initial_domains):
        """
        Solves the list coloring CSP using backtracking with forward checking.
        Returns a list of colors or None if no solution found.
        """
        # Copy domains to avoid modifying original
        domains = [set(d) for d in initial_domains]
        color = [None] * n
        
        # MRV heuristic: pick variable with fewest remaining values
        # To optimize, we could maintain a heap, but for n~400, linear scan is fast enough
        # and simpler to implement correctly with backtracking.
        
        def find_mrv():
            min_size = float('inf')
            min_var = -1
            for v in range(n):
                if color[v] is None:
                    size = len(domains[v])
                    if size < min_size:
                        min_size = size
                        min_var = v
                        if size == 1:
                            break
            return min_var

        def backtrack():
            # Check time limit
            if time.time() > DEADLINE:
                return False
            
            v = find_mrv()
            if v == -1:
                return True # All assigned
            
            # Try each color in the domain
            # Order colors by least constraining value? For now, just iterate.
            for c in domains[v]:
                # Check if color is valid (it is, because it's in the domain)
                # Assign
                color[v] = c
                domains[v] = {c}
                
                # Forward checking: remove c from neighbors' domains
                neighbors = adj[v]
                saved_domains = []
                consistent = True
                for u in neighbors:
                    if color[u] is None:
                        if c in domains[u]:
                            saved_domains.append((u, domains[u].copy()))
                            domains[u].remove(c)
                            if not domains[u]:
                                consistent = False
                                break
                
                if consistent:
                    if backtrack():
                        return True
                
                # Undo assignment and forward checking
                for u, saved_dom in saved_domains:
                    domains[u] = saved_dom
                domains[v] = initial_domains[v] # Restore full domain for v
                color[v] = None
                
            return False

        if backtrack():
            return color
        return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        
        # Build adjacency list
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Convert lists to sets for easier domain manipulation
        initial_domains = [set(lst) for lst in lists]
        
        # Solve
        coloring = solve_csp(n, adj, initial_domains)
        
        if coloring is None:
            # Fallback: should not happen for these instances, but just in case
            # Try a simple greedy? Or just return empty/invalid?
            # The problem states all are colourable. If our solver fails, it's a bug or time out.
            # If time out, we return whatever partial result? No, must be valid or invalid.
            # Let's try a randomized restart if deterministic fails quickly?
            # Given the size, deterministic should work.
            # If it fails, we return a dummy list to avoid crash, but it will score 0 for this instance.
            coloring = [0] * n # Invalid, will score 0
            
        # Ensure types are correct (Python int)
        # The solver returns ints.
        
        out.append(coloring)
        
        if time.time() > DEADLINE:
            # If we run out of time, we must return something.
            # We have completed some instances. The rest are empty lists?
            # The contract says "wrong length ... makes the whole result invalid".
            # So we must ensure we return 4 lists.
            # If we didn't finish, we can't really "return best result" for the unfinished ones.
            # But the solver is so fast this shouldn't happen.
            pass

    return out

# EVOLVE-BLOCK-END
