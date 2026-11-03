"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W form a valid tensor decomposition for n×m×p matrix mult."""
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U):
        return False
    if any(len(row) != rank for row in V):
        return False
    if any(len(row) != rank for row in W):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _try_strassen_embedded(n, m, p, deadline):
    """
    Try to construct a lower-rank decomposition using Strassen's algorithm ideas.
    For 3x4x5, partition the middle dimension into two halves of size 2.
    C = A1*B1 + A2*B2 where A1,A2 are 3x2 and B1,B2 are 2x5.
    
    We can use the fact that 2x2 matrix multiplication has rank 7 (Strassen).
    But here we have 3x2x5, not 2x2x2.
    
    Alternative: partition into 2x2 blocks in a different way.
    Let's try: think of the 4x4 middle as two 2x2 blocks.
    """
    start_time = time.time()
    
    # Strategy: Use the fact that we can share some computations
    # For 3x4x5, try a structured decomposition with rank < 60
    
    # Approach: Use linear combinations of the schoolbook terms
    # Start with schoolbook and try to find dependencies
    
    n, m, p = 3, 4, 5
    rank_school = n * m * p  # 60
    
    # Build the tensor as a 3D array T[i,j,k] = 1 for all i,j,k
    # We want to find rank-1 decompositions: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
    
    # Try a structured approach: use the fact that some entries can be shared
    # For example, if U[i,j,c] = U[i',j',c] and V[j,k,c] = V[j',k',c],
    # then W[k,i,c] and W[k',i',c] can be adjusted.
    
    # Let's try a specific construction:
    # Group the 60 terms into pairs and see if we can merge them.
    
    # Actually, let's try a more direct approach:
    # Use a random search with a better initialization strategy.
    
    best_rank = rank_school
    best_result = None
    
    # Try ranks from 55 down to 50
    for target_rank in range(55, 49, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            # Initialize with a structured pattern
            # Start from schoolbook and zero out some entries
            U = [[0] * target_rank for _ in range(n * m)]
            V = [[0] * target_rank for _ in range(m * p)]
            W = [[0] * target_rank for _ in range(p * n)]
            
            # Fill with a structured pattern
            for c in range(target_rank):
                # Each rank-1 term should contribute to some (i,j,k) triples
                # Use a hash-like function to determine which entries get non-zero values
                for i in range(n):
                    for j in range(m):
                        # Each (i,j) pair should have some non-zero entries
                        if (i * m + j + c) % 3 == 0:
                            U[i * m + j][c] = 1 if (i + j + c) % 2 == 0 else -1
            
            for j in range(m):
                for k in range(p):
                    if (j * p + k + c) % 3 == 0:
                        V[j * p + k][c] = 1 if (j + k + c) % 2 == 0 else -1
            
            for k in range(p):
                for i in range(n):
                    if (k * n + i + c) % 3 == 0:
                        W[k * n + i][c] = 1 if (k + i + c) % 2 == 0 else -1
            
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    return None

def _try_greedy_elimination(n, m, p, deadline):
    """
    Start with schoolbook and try to eliminate terms by finding linear dependencies.
    """
    start_time = time.time()
    schoolbook = _build_schoolbook(n, m, p)
    U = [row[:] for row in schoolbook["U"]]
    V = [row[:] for row in schoolbook["V"]]
    W = [row[:] for row in schoolbook["W"]]
    rank = n * m * p
    
    # Try to eliminate terms by checking if removing one and adjusting others still works
    # This is complex, so let's try a simpler approach:
    # Check if any two terms are identical (same U,V,W patterns)
    
    eliminated = 0
    for c in range(rank - 1, -1, -1):
        if time.time() > deadline:
            break
        
        # Check if term c is redundant
        # A term is redundant if for all (i,j,k), the contribution from term c
        # can be compensated by other terms.
        
        # Simplified check: if U[:,c], V[:,c], W[:,c] are all zero, remove it
        u_col = [U[i][c] for i in range(n * m)]
        v_col = [V[j][c] for j in range(m * p)]
        w_col = [W[k][c] for k in range(p * n)]
        
        if all(x == 0 for x in u_col) or all(x == 0 for x in v_col) or all(x == 0 for x in w_col):
            # Remove this term
            U = [row[:c] + row[c+1:] for row in U]
            V = [row[:c] + row[c+1:] for row in V]
            W = [row[:c] + row[c+1:] for row in W]
            rank -= 1
            eliminated += 1
            c += 1  # Re-check same index since we shifted
    
    if eliminated > 0:
        return {"U": U, "V": V, "W": W}
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # Leave 15s margin
    
    # Start with schoolbook
    schoolbook = _build_schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = n * m * p
    
    # Try Strassen-embedded approach
    if time.time() < deadline - 10:
        result = _try_strassen_embedded(n, m, p, deadline)
        if result is not None:
            rank = len(result["U"][0])
            if rank < best_rank:
                best_result = result
                best_rank = rank
    
    # Try greedy elimination
    if time.time() < deadline - 5:
        result = _try_greedy_elimination(n, m, p, deadline)
        if result is not None:
            rank = len(result["U"][0])
            if rank < best_rank:
                best_result = result
                best_rank = rank
    
    return best_result
# EVOLVE-BLOCK-END
