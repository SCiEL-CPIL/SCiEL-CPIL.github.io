import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    # 1. Define the fallback: Paley 37
    def paley_37():
        q = 37
        residues = {(x * x) % q for x in range(1, q)}
        return q, [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    
    best_n = 37
    best_edges = paley_37()[1]
    
    # 2. Local Search for larger n
    # We try to find a (5,5)-free graph for n = 42, 41, 40
    for target_n in [42, 41, 40]:
        if time.time() > deadline - 30: # Need time to return
            break
        
        found = search_for_graph(target_n, deadline)
        if found is not None:
            best_n = target_n
            best_edges = found
            # We can stop here if we found the max possible known (42)
            if target_n == 42:
                break
    
    return best_n, best_edges

def search_for_graph(n, deadline):
    # Use bitmasks for adjacency
    # adj[i] is an integer where bit j is 1 if (i,j) is an edge
    
    # Helper: Check if a set of 5 vertices is a K5 or Independent Set
    def is_k5(verts):
        # verts is a tuple of 5 vertices
        v0, v1, v2, v3, v4 = verts
        # Check all 10 pairs
        if (adj[v0] >> v1) & 1 == 0: return False
        if (adj[v0] >> v2) & 1 == 0: return False
        if (adj[v0] >> v3) & 1 == 0: return False
        if (adj[v0] >> v4) & 1 == 0: return False
        if (adj[v1] >> v2) & 1 == 0: return False
        if (adj[v1] >> v3) & 1 == 0: return False
        if (adj[v1] >> v4) & 1 == 0: return False
        if (adj[v2] >> v3) & 1 == 0: return False
        if (adj[v2] >> v4) & 1 == 0: return False
        if (adj[v3] >> v4) & 1 == 0: return False
        return True

    def is_ind5(verts):
        v0, v1, v2, v3, v4 = verts
        if (adj[v0] >> v1) & 1 == 1: return False
        if (adj[v0] >> v2) & 1 == 1: return False
        if (adj[v0] >> v3) & 1 == 1: return False
        if (adj[v0] >> v4) & 1 == 1: return False
        if (adj[v1] >> v2) & 1 == 1: return False
        if (adj[v1] >> v3) & 1 == 1: return False
        if (adj[v1] >> v4) & 1 == 1: return False
        if (adj[v2] >> v3) & 1 == 1: return False
        if (adj[v2] >> v4) & 1 == 1: return False
        if (adj[v3] >> v4) & 1 == 1: return False
        return True

    def find_violation():
        # Iterate through all 5-combinations
        # Optimization: If we find one, return it immediately
        for verts in combinations(range(n), 5):
            if is_k5(verts):
                return verts, True
            if is_ind5(verts):
                return verts, False
        return None, None

    def flip_edge(u, v):
        nonlocal adj
        if u > v: u, v = v, u
        bit = 1 << v
        if (adj[u] & bit):
            adj[u] &= ~bit
            adj[v] &= ~(1 << u)
        else:
            adj[u] |= bit
            adj[v] |= (1 << u)

    # Multiple restarts
    max_restarts = 20
    for r in range(max_restarts):
        if time.time() > deadline:
            return None
        
        # Initialize random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Local Search
        steps = 0
        max_steps = 5000
        while steps < max_steps and time.time() < deadline:
            verts, is_clique = find_violation()
            if verts is None:
                # Success! Convert to edge list
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if (adj[i] >> j) & 1:
                            edges.append((i, j))
                return edges
            
            # Flip a random edge in the violating set
            # If it's a clique, we want to remove an edge.
            # If it's an independent set, we want to add an edge.
            # Picking any edge in the set and flipping it works.
            # For clique: flip removes edge (since it exists).
            # For ind set: flip adds edge (since it doesn't exist).
            u, v = random.sample(list(verts), 2)
            flip_edge(u, v)
            steps += 1
            
            # Occasionally perturb if stuck?
            # If we cycle, random restart is better.
            # Let's just let it run.
            
    return None
