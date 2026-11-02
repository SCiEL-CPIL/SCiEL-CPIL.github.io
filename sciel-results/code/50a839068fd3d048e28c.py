import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    
    # Try to find a lower rank decomposition using border rank technique
    # Start with rank 54 and try to solve
    for target_rank in range(54, 60):
        result = _try_border_rank(n, m, p, target_rank)
        if result is not None:
            return result
    
    # Fallback to trivial decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

def _try_border_rank(n, m, p, rank):
    """
    Try to find a rank decomposition using border rank technique.
    We parameterize with epsilon and solve for coefficients.
    """
    # The tensor equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid i,j,k
    # We'll use a structured approach: assume most entries are 0 or 1, with some entries involving a parameter.
    
    # Strategy: Start with the trivial rank-n*m*p decomposition and try to merge terms.
    # For border rank, we introduce a parameter t and allow entries to be polynomials in t.
    
    # Let's try a specific structured decomposition.
    # For 2x3x10, we can try to save terms by exploiting the structure.
    
    # Approach: Use a rank-58 decomposition by combining some terms.
    # This is a heuristic approach - we'll try specific patterns.
    
    if rank >= 58:
        # Try a simple merging strategy
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Start with base decomposition for first 58 terms
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col >= rank:
                        break
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
                    if col >= rank:
                        break
                if col >= rank:
                    break
            if col >= rank:
                break
        
        # For the remaining terms, we need to handle the leftover (i,j,k) triples
        # by using a combination of existing terms with coefficients
        # This is complex; let's verify if this works
        if col < rank:
            # We didn't fill all rank terms with base, which means we have extra capacity
            # But we also have unfilled (i,j,k) triples
            # Let's check which triples are covered
            covered = set()
            for r in range(rank):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if U[i*m+j][r] != 0 and V[j*p+k][r] != 0 and W[k*n+i][r] != 0:
                                covered.add((i,j,k))
            
            # Check if all triples are covered
            all_triples = [(i,j,k) for i in range(n) for j in range(m) for k in range(p)]
            if all(t in covered for t in all_triples):
                return {"U": U, "V": V, "W": W}
        
        # If not all covered, try a different approach
        # Use linear algebra to find coefficients for the remaining terms
    
    # For now, return None to trigger fallback
    return None
