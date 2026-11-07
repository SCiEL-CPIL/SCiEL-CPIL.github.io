"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
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

    def solve_instance(n, edges, lists, deadline):
        # Build adjacency list
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Precompute degrees for tie-breaking
        degrees = [len(adj[i]) for i in range(n)]
        
        # Initial available colors
        avail = [set(l) for l in lists]
        
        for i in range(n):
            if not avail[i]:
                return None

        colour = [None] * n 
        
        # Track uncolored vertices for faster MRV
        uncolored = set(range(n))
        
        def backtrack():
            if time.time() > deadline:
                return False
            
            # Find uncolored vertex with fewest available colors (MRV)
            # Tie-break by highest degree
            min_count = float('inf')
            best_v = -1
            best_deg = -1
            
            for v in uncolored:
                count = len(avail[v])
                if count == 0:
                    return False # Dead end
                if count < min_count or (count == min_count and degrees[v] > best_deg):
                    min_count = count
                    best_v = v
                    best_deg = degrees[v]
                    if count == 1: # Best possible, break early
                        break
            
            if best_v == -1:
                return True # All colored
            
            # Get colors to try
            # LCV: Try colors that constrain neighbors the least
            colors = list(avail[best_v])
            
            # Precompute uncolored neighbors for this vertex
            uncolored_neighbors = [u for u in adj[best_v] if u in uncolored]
            
            def count_constraints(c):
                count = 0
                for u in uncolored_neighbors:
                    if c in avail[u]:
                        count += 1
                return count
            
            colors.sort(key=count_constraints)
            
            for c in colors:
                # Assign c to best_v
                colour[best_v] = c
                uncolored.discard(best_v)
                
                # Check neighbors
                valid = True
                removed = []
                
                for u in adj[best_v]:
                    if colour[u] is not None:
                        if colour[u] == c:
                            valid = False
                            break
                    else:
                        if c in avail[u]:
                            avail[u].discard(c)
                            removed.append(u)
                            if not avail[u]:
                                valid = False
                                break
                
                if valid:
                    if backtrack():
                        return True
                else:
                    # Restore state
                    for u in removed:
                        avail[u].add(c)
                    
                # Restore state
                colour[best_v] = None
                uncolored.add(best_v)
                for u in removed:
                    avail[u].add(c)
                
            return False
            
        if backtrack():
            # Ensure types are int
            return [int(c) for c in colour]
        else:
            return None

    out = []
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin (5s buffer from 120s limit)
    
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            # Fallback: return whatever we have, though this shouldn't happen
            # Fill remaining with dummy values to maintain correct length
            for remaining_splits, remaining_seed in ((20, 701), (40, 702), (60, 703), (100, 704))[len(out):]:
                n, _, _ = instance(remaining_splits, remaining_seed)
                out.append([0] * n)
            break
        
        n, edges, lists = instance(splits, seed)
        colouring = solve_instance(n, edges, lists, deadline)
        if colouring is None:
            # This should not happen as instances are guaranteed colorable
            # If it does, we might try a different strategy or just fail
            # But the prompt says "All are colourable"
            # If backtracking fails, it's a bug in my solver or the instance is hard.
            # Given the size, it should be fine.
            # Let's add a simple greedy fallback just in case, though it's likely unnecessary
            # Actually, if backtracking fails, returning invalid is bad.
            # But MRV backtracking is exact. It only fails if no solution exists.
            # So this case should be empty.
            colouring = [0] * n # Dummy, will be invalid
        out.append(colouring)
        
    return out
# EVOLVE-BLOCK-END
