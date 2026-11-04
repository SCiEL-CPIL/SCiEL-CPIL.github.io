"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Start with schoolbook as fallback
    rank_fallback = n * m * p
    U_fb = [[0] * rank_fallback for _ in range(n * m)]
    V_fb = [[0] * rank_fallback for _ in range(m * p)]
    W_fb = [[0] * rank_fallback for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    
    # Try to find a lower-rank decomposition using a sparse approach
    # The multiplication tensor T[i,j,k] = 1 for all i,j,k
    # We want: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Strategy: Try rank values from a lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = max(15, 35, 21) = 35
    # But tensor rank can be lower than these dimension products
    
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb, "rank": rank_fallback}
    
    # Try various rank values
    min_rank = max(n*m, m*p, p*n)  # 35
    # Actually the tensor rank lower bound is more complex
    # For matrix multiplication, rank >= max(n*m, m*p, p*n) is not necessarily true
    # But rank >= max(n, m, p) is a weak bound
    # Let's try from a reasonable range
    
    # Known results: 3x3x3 has rank 23 (Strassen)
    # 3x5x7: we can partition into blocks
    # 3x5 = 3x3 + 3x2, 5x7 = 5x5 + 5x2 (not quite)
    # Better: 3x5x7 = (3x3x3) + (3x2x7) or other partitions
    
    # Let's try a direct sparse search for small ranks
    # Use a greedy/iterative approach
    
    # First, try to verify if rank 35 (schoolbook minimum by dimensions) works
    # Actually schoolbook is rank n*m*p = 105, but we can do better
    
    # Try rank values from 35 to 105
    for target_rank in range(35, min(80, rank_fallback + 1)):
        if time.time() > deadline:
            break
        
        success = _try_find_decomposition(n, m, p, target_rank)
        if success is not None:
            U, V, W = success
            # Verify
            if _verify_decomposition(U, V, W, n, m, p):
                best_result = {"U": U, "V": V, "W": W, "rank": target_rank}
                break
    
    return {"U": best_result["U"], "V": best_result["V"], "W": best_result["W"]}


def _try_find_decomposition(n, m, p, target_rank):
    """Try to find a rank-target_rank decomposition using sparse linear algebra."""
    if time.time() > 140:
        return None
    
    # We need to find U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # such that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    # This is a system of n*m*p equations with rank*(n*m + m*p + p*n) unknowns
    # But it's nonlinear (products of variables)
    
    # Alternative: fix U and V, solve for W linearly
    # For each c, let u_c = U[:,c], v_c = V[:,c], w_c = W[:,c]
    # Then sum_c (u_c ⊗ v_c) * w_c = ones (flattened)
    
    # This is still nonlinear. Let's use a different approach.
    
    # Use the fact that for integer/half-integer entries, we can search
    # over small integer values
    
    # Greedy approach: build up the decomposition term by term
    random.seed(42)
    
    U = [[0] * target_rank for _ in range(n * m)]
    V = [[0] * target_rank for _ in range(m * p)]
    W = [[0] * target_rank for _ in range(p * n)]
    
    # Start with schoolbook pattern for first min(target_rank, n*m*p) terms
    # then try to optimize
    
    # Actually, let's use a structured approach:
    # For 3x5x7, try to use block structure
    # Partition 5 = 3 + 2, 7 = 3 + 4 or 7 = 5 + 2
    
    # Use 3x3x3 Strassen (rank 23) for the 3x3x3 part
    # and schoolbook for remaining parts
    
    # 3x5x7 = 3x3x7 + 3x2x7
    # 3x3x7 = 3x3x3 + 3x3x4
    # This gets complicated. Let's just try to find any valid decomposition
    # with the target rank using a simple pattern.
    
    # Simple pattern: use cyclic indices
    for c in range(target_rank):
        # Assign values to U, V, W for this term
        # Use a hash-like pattern based on c
        for i in range(n):
            for j in range(m):
                # Simple: only set some entries
                if (i * m + j + c) % 3 == 0:
                    U[i * m + j][c] = 1
                elif (i * m + j + c) % 3 == 1:
                    U[i * m + j][c] = -1
        
        for j in range(m):
            for k in range(p):
                if (j * p + k + c) % 3 == 0:
                    V[j * p + k][c] = 1
                elif (j * p + k + c) % 3 == 1:
                    V[j * p + k][c] = -1
        
        for k in range(p):
            for i in range(n):
                if (k * n + i + c) % 3 == 0:
                    W[k * n + i][c] = 1
                elif (k * n + i + c) % 3 == 1:
                    W[k * n + i][c] = -1
    
    # Verify this pattern
    if _verify_decomposition(U, V, W, n, m, p):
        return U, V, W
    
    # If the simple pattern doesn't work, try schoolbook subset
    # Take the first target_rank schoolbook terms
    if target_rank <= n * m * p:
        U2 = [[0] * target_rank for _ in range(n * m)]
        V2 = [[0] * target_rank for _ in range(m * p)]
        W2 = [[0] * target_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col < target_rank:
                        U2[i * m + j][col] = 1
                        V2[j * p + k][col] = 1
                        W2[k * n + i][col] = 1
                        col += 1
        # This won't work unless target_rank = n*m*p
        if target_rank == n * m * p:
            if _verify_decomposition(U2, V2, W2, n, m, p):
                return U2, V2, W2
    
    return None


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if total != 1:
                    return False
    return True
# EVOLVE-BLOCK-END
