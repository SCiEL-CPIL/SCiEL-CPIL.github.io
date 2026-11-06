import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    best_result = {'n': 39, 'edges': []}
    best_n = 39

    def can_add_edge(adj, n, i, j):
        """Check if adding edge (i,j) keeps the graph C4-free."""
        # A new C4 would use edge (i,j) and a path of length 3 from i to j not using (i,j).
        # i - x - y - j where x is neighbor of i, y is neighbor of j, and (x,y) is an edge.
        # We need to check: for each x in N(i) and y in N(j) with x != y, x != j, y != i,
        # is (x,y) an edge?
        # More efficiently: for each x in N(i), check if x is adjacent to any y in N(j) \ {i, x}.
        # Actually, even simpler: for each x in N(i) \ {j}, if x is adjacent to any y in N(j) \ {i, x},
        # then we have a C4. But we also need x != y.
        
        # Let's think again. Adding edge (i,j).
        # New C4s must contain edge (i,j).
        # The other 3 edges form a path of length 3 from i to j: i-x-y-j.
        # So x is a neighbor of i (x != j), y is a neighbor of j (y != i), and (x,y) is an edge.
        # Also x != y (distinct vertices in a cycle).
        # And x != j, y != i (already ensured).
        
        # So: does there exist x in N(i) \ {j} and y in N(j) \ {i} such that x != y and adj[x][y] == 1?
        
        # We can iterate over x in N(i) \ {j}:
        #   For each such x, check if x is adjacent to any y in N(j) \ {i, x}.
        #   If yes, adding (i,j) creates a C4.
        
        ni = np.where(adj[i] == 1)[0]
        nj = np.where(adj[j] == 1)[0]
        
        # Exclude j from ni and i from nj
        ni_mask = np.ones(n, dtype=bool)
        ni_mask[j] = False
        ni_filtered = ni[ni_mask[ni]]
        
        nj_mask = np.ones(n, dtype=bool)
        nj_mask[i] = False
        nj_filtered = nj[nj_mask[nj]]
        
        for x in ni_filtered:
            # Check if x is adjacent to any y in nj_filtered, y != x
            adj_x = adj[x]
            # We want any y in nj_filtered with y != x and adj_x[y] == 1
            candidates = nj_filtered[nj_filtered != x]
            if len(candidates) > 0 and np.any(adj_x[candidates]):
                return False
        
        return True

    def is_valid(n, adj):
        # Check min degree
        for i in range(n):
            if adj[i].sum() < n - 39:
                return False
        # Check C4-free: for all u < v, common neighbors <= 1
        for u in range(n):
            for v in range(u+1, n):
                common = int((adj[u] & adj[v]).sum())
                if common > 1:
                    return False
        return True

    def try_construct(n, seed, max_steps=500):
        random.seed(seed)
        np.random.seed(seed)
        
        # Start with a random graph with appropriate density
        p = (n - 39) / (n - 1) * 1.3
        
        adj = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    adj[i][j] = 1
                    adj[j][i] = 1
        
        # Repair loop
        for step in range(max_steps):
            if time.time() > deadline:
                return None
            
            # Check if valid
            valid = True
            
            # 1. Fix min degree
            for i in range(n):
                deg = int(adj[i].sum())
                if deg < n - 39:
                    missing = n - 39 - deg
                    candidates = [j for j in range(n) if j != i and not adj[i][j]]
                    random.shuffle(candidates)
                    added = 0
                    for j in candidates:
                        if added >= missing:
                            break
                        if can_add_edge(adj, n, i, j):
                            adj[i][j] = 1
                            adj[j][i] = 1
                            added += 1
                    valid = False
                    break
            if not valid:
                continue
            
            # 2. Fix C4s
            found_c4 = False
            for u in range(n):
                if found_c4:
                    break
                for v in range(u+1, n):
                    c = int((adj[u] & adj[v]).sum())
                    if c >= 2:
                        # Remove a random edge incident to u or v that is part of a C4
                        # Get common neighbors
                        commons = np.where(adj[u] & adj[v])[0]
                        commons = [w for w in commons if w != u and w != v]
                        if len(commons) >= 2:
                            # Pick two common neighbors x, y
                            # The C4 is u-x-v-y-u or u-y-v-x-u
                            # Remove edge (u, x) or (v, y) etc.
                            # Let's remove a random edge from the cycle
                            x, y = random.sample(commons, 2)
                            # Options: remove (u,x), (x,v), (v,y), (y,u)
                            # Randomly pick one
                            choice = random.choice([(u,x), (x,v), (v,y), (y,u)])
                            a, b = choice
                            adj[a][b] = 0
                            adj[b][a] = 0
                            found_c4 = True
                            valid = False
                            break
            if not valid:
                continue
            
            # 3. If we get here, check if truly valid
            min_deg = min(int(adj[i].sum()) for i in range(n))
            if min_deg >= n - 39:
                # Double-check C4-free
                c4_free = True
                for u in range(n):
                    if not c4_free:
                        break
                    for v in range(u+1, n):
                        if int((adj[u] & adj[v]).sum()) > 1:
                            c4_free = False
                            break
                if c4_free:
                    edges = []
                    for i in range(n):
                        for j in range(i+1, n):
                            if adj[i][j]:
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
        
        return None

    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        found = False
        # Try multiple seeds
        for seed in range(200):
            if time.time() > deadline:
                break
            res = try_construct(n, seed)
            if res is not None:
                if n > best_n:
                    best_n = n
                    best_result = res
                found = True
                break
        
        if found and n == 47:
            return best_result
    
    return best_result
