import numpy as np
from itertools import combinations
import random

def construct():
    random.seed(777)
    
    def has_k5_or_indep5(adj, n, new_vertex=-1):
        """Check if graph has K5 or independent 5-set. If new_vertex >= 0, only check tuples involving it."""
        if new_vertex >= 0:
            # Only check 5-tuples containing new_vertex
            others = [v for v in range(n) if v != new_vertex]
            for combo in combinations(others, 4):
                    full = list(combo) + [new_vertex]
                    is_clique = True
                    is_indep = True
                    for idx in range(5):
                        for jdx in range(idx+1, 5):
                            u, v = full[idx], full[jdx]
                            if adj[u, v]:
                                is_indep = False
                            else:
                                is_clique = False
                            if not is_clique and not is_indep:
                                break
                        if not is_clique and not is_indep:
                            break
                    if is_clique or is_indep:
                        return True
            return False
        else:
            for combo in combinations(range(n), 5):
                is_clique = True
                is_indep = True
                for idx in range(5):
                    for jdx in range(idx+1, 5):
                        u, v = combo[idx], combo[jdx]
                        if adj[u, v]:
                            is_indep = False
                        else:
                            is_clique = False
                        if not is_clique and not is_indep:
                            break
                    if not is_clique and not is_indep:
                        break
                if is_clique or is_indep:
                    return True
            return False
    
    # Start with Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj = np.zeros((q, q), dtype=np.int8)
    for i in range(q):
        for j in range(i+1, q):
            if (j - i) % q in residues:
                adj[i, j] = 1
                adj[j, i] = 1
    
    n = q
    
    # Try to add vertices
    for target_n in range(q+1, 47):
        added = False
        for attempt in range(200):
            # Try a random adjacency for the new vertex
            new_adj = np.random.randint(0, 2, size=n)
            # Check if adding this vertex creates a violation
            # Only need to check 5-tuples containing the new vertex
            valid = True
            others = list(range(n))
            for combo in combinations(others, 4):
                full = list(combo) + [n]
                is_clique = True
                is_indep = True
                for idx in range(5):
                    for jdx in range(idx+1, 5):
                        u, v = full[idx], full[jdx]
                        if u == n or v == n:
                            # Edge to new vertex
                            other = u if v == n else v
                            if new_adj[other]:
                                is_indep = False
                            else:
                                is_clique = False
                        else:
                            if adj[u, v]:
                                is_indep = False
                            else:
                                is_clique = False
                        if not is_clique and not is_indep:
                            break
                    if not is_clique and not is_indep:
                        break
                if is_clique or is_indep:
                    valid = False
                    break
            if valid:
                # Add the vertex
                new_adj_full = np.zeros(n+1, dtype=np.int8)
                new_adj_full[:n] = new_adj
                adj = np.vstack([np.hstack([adj, new_adj.reshape(-1,1)]), np.hstack([new_adj, np.zeros(1)])])
                n += 1
                added = True
                break
        if not added:
            break
    
    # Extract edges
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j]:
                edges.append((i, j))
    
    return n, edges
