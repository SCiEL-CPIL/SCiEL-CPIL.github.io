"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 21
    pn = p * n  # 21
    
    # Target: for each (i,j,k), U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] sums to 1
    # For all other (u,v,w) combos, sum should be 0
    target_triples = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                target_triples.add((u, v, w))
    
    def check_decomposition(U, V, W, tol=1e-4):
        """Check if decomposition is valid. Returns True if valid."""
        rank = len(U[0])
        # Check target triples
        for (u, v, w) in target_triples:
            s = 0.0
            for c in range(rank):
                s += U[u][c] * V[v][c] * W[w][c]
            if abs(s - 1.0) > tol:
                return False
        # Check non-target triples (sample for speed)
        all_u = range(nm)
        all_v = range(mp)
        all_w = range(pn)
        for u in all_u:
            for v in all_v:
                for w in all_w:
                    if (u, v, w) not in target_triples:
                        s = 0.0
                        for c in range(rank):
                            s += U[u][c] * V[v][c] * W[w][c]
                        if abs(s) > tol:
                            return False
        return True
    
    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    U[u][col] = 1
                    V[v][col] = 1
                    W[w][col] = 1
                    col += 1
        return U, V, W
    
    best_U, best_V, best_W = schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank using local search
    # This is a heuristic approach
    # For this specific case, let's try ranks from a lower bound up
    
    # Lower bound: max(n*m, m*p, p*n) = 21, but this is very weak
    # Practical lower bound: try from 50 up to 63
    
    # Given time constraints, just return schoolbook for now
    # In a real implementation, we'd do a search
    
    if time.time() < deadline:
        # Try a simple improvement: see if we can merge any terms
        # For now, just return schoolbook
        pass
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
