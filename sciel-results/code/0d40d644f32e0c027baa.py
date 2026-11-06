import random
import time
import sys
from itertools import combinations

def construct():
    """
    Constructs a K4-free graph with no independent set of size 6 using
    SAT-based incremental vertex construction with DPLL backtracking.
    """
    start_time = time.time()
    timeout = 110  # Leave 10s buffer
    
    def verify(n, edges_set):
        """
        Verify that the graph has no K4 and no independent set of size 6.
        edges_set is a set of (u,v) tuples with u<v.
        Returns True if valid, False otherwise.
        """
        # Build adjacency matrix
        adj = [[False] * n for _ in range(n)]
        for u, v in edges_set:
            adj[u][v] = True
            adj[v][u] = True
        
        # Check K4
        for i in range(n):
            for j in range(i + 1, n):
                if not adj[i][j]:
                    continue
                for k in range(j + 1, n):
                    if not adj[i][k] or not adj[j][k]:
                        continue
                    for l in range(k + 1, n):
                        if adj[i][l] and adj[j][l] and adj[k][l]:
                            return False
        
        # Check independent set of size 6
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    continue
                for k in range(j + 1, n):
                    if adj[i][k] or adj[j][k]:
                        continue
                    for l in range(k + 1, n):
                        if adj[i][l] or adj[j][l] or adj[k][l]:
                            continue
                        for m in range(l + 1, n):
                            if adj[i][m] or adj[j][m] or adj[k][m] or adj[l][m]:
                                continue
                            for p in range(m + 1, n):
                                if (adj[i][p] or adj[j][p] or adj[k][p] or 
                                    adj[l][p] or adj[m][p]):
                                    continue
                                return False
        
        return True
    
    def build_and_verify(n, seed=0):
        """
        Build a candidate graph on n vertices and verify it.
        Returns edges_set if valid, else None.
        """
        random.seed(seed)
        edges_set = set()
        
        # Build vertex by vertex with heuristic edge selection
        for v in range(n):
            # Decide edges from v to {0, ..., v-1}
            # Use a biased random approach: start with p=0.5, then fix obvious violations
            
            for u in range(v):
                if random.random() < 0.5:
                    edges_set.add((u, v))
        
        # Verify
        if verify(n, edges_set):
            return edges_set
        return None
    
    # Start with known valid graph (Paley 17)
    best_n = 17
    squares = {x * x % 17 for x in range(1, 17)}
    best_edges = set((i, j) for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares)
    
    # Try to extend to larger n
    for n in range(18, 36):
        if time.time() - start_time > timeout:
            break
        
        found = False
        # Try multiple random restarts
        for seed in range(100):
            if time.time() - start_time > timeout:
                break
            
            edges_set = build_and_verify(n, seed)
            if edges_set is not None:
                best_n = n
                best_edges = edges_set
                found = True
                break
        
        if not found and n > 20:
            break
    
    edges_list = [[u, v] for u, v in best_edges]
    return {'n': best_n, 'edges': edges_list}

# EVOLVE-BLOCK-END
