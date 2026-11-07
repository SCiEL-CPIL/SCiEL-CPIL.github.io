import time
from random import Random

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill

    def instance(splits, seed, palette=5):
        rng = Random(seed)
        # Cube faces and edges
        # Vertices 0-7
        # Faces:
        # 0-1-2-3 (top? no, let's use standard cube)
        # Let's stick to the seed's generation exactly to ensure same instances.
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        edges = set()
        for f in faces:
            for i in range(4):
                a, b = f[i], f[(i + 1) % 4]
                edges.add((min(a, b), max(a, b)))
        n = 8
        for _ in range(splits):
            face = faces.pop(rng.randrange(len(faces)))
            inner = list(range(n, n + 4))
            n += 4
            for i in range(4):
                a, b = face[i], face[(i + 1) % 4]
                # Connect inner[i] to a
                edges.add((min(a, inner[i]), max(a, inner[i])))
                # Connect inner[i] to inner[i+1]
                edges.add((min(inner[i], inner[(i + 1) % 4]), max(inner[i], inner[(i + 1) % 4])))
                # New face: a, b, inner[i+1], inner[i]
                faces.append([a, b, inner[(i + 1) % 4], inner[i]])
            faces.append(inner)
        
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        return n, sorted(edges), lists

    def solve(n, edges, lists):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Initial domains
        domains = [list(l) for l in lists]
        colour = [-1] * n
        
        # Dynamic MRV backtracking with Forward Checking
        uncolored = set(range(n))
        
        def solve_csp():
            if time.time() > deadline:
                return False
                
            if not uncolored:
                return True
                
            # Select variable with Minimum Remaining Values (MRV)
            min_size = float('inf')
            best_v = None
            
            for v in uncolored:
                sz = len(domains[v])
                if sz == 0:
                    return False # Dead end
                if sz < min_size:
                    min_size = sz
                    best_v = v
                    if min_size == 1:
                        break # Can't get better than 1
            
            if best_v is None:
                return True # All colored? Shouldn't happen if uncolored not empty

            v = best_v
            uncolored.remove(v)
            
            # Identify uncolored neighbors for Forward Checking
            neighbors = [u for u in adj[v] if u in uncolored]
            # Save current domains of neighbors to allow undo
            saved_domains = {u: list(domains[u]) for u in neighbors}
            
            # Try each colour in current domain
            for c in domains[v]:
                colour[v] = c
                
                # Forward Check: prune neighbors
                valid = True
                for u in neighbors:
                    if c in domains[u]:
                        domains[u].remove(c)
                        if not domains[u]:
                            valid = False
                            break
                
                if valid:
                    if solve_csp():
                        return True
                
                # Undo Forward Check
                for u in neighbors:
                    domains[u] = saved_domains[u]
                
                colour[v] = -1
                
            uncolored.add(v)
            return False
            
        if solve_csp():
            return colour
        else:
            # Fallback: return current state (likely invalid, but better than crash)
            return colour

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        
        # Solve
        colour = solve(n, edges, lists)
        out.append(colour)
        
        if time.time() > deadline:
            # If we are running out of time, we should break and return what we have.
            # But we need 4 results.
            # If we can't finish, we might return invalid lists for the rest.
            # The contract says "wrong length ... makes the whole result invalid".
            # So we MUST return 4 lists.
            # If we timed out, it's a failure.
            # To be safe, if we break early, we should try to fill the rest with a trivial 
            # or just reuse the last one? No, that's wrong.
            # The best strategy is to ensure we don't time out.
            # With N=400 and MRV backtracking, it should be milliseconds.
            break

    # Ensure we have 4 results
    while len(out) < 4:
        # If we timed out, we need to return something.
        # But the contract says "wrong length ... makes the whole result invalid".
        # So we MUST return 4 lists.
        # If we didn't solve them, we can't just leave them out.
        # We should have solved them.
        # If we timed out, it's a failure.
        # To be safe, if we break early, we should try to fill the rest with a trivial (likely invalid) 
        # or just reuse the last one? No, that's wrong.
        # The best strategy is to ensure we don't time out.
        # With N=400 and MRV backtracking, it should be milliseconds.
        # Just append an empty list? No, must be correct length.
        # This path should not be reached.
        out.append([])

    # Convert to list of lists of int
    # Ensure types are correct
    result = []
    for c in out:
        result.append([int(x) for x in c])
        
    return result

if __name__ == "__main__":
    # Test
    res = construct()
    print(len(res))
