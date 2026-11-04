"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from fractions import Fraction
from itertools import product

def _schoolbook():
    """Return the standard rank n*m*p decomposition."""
    n, m, p = DIMENSIONS
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

def _verify_exact(U, V, W):
    """
    Verify exact decomposition using Fraction arithmetic.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all (i,j,k)
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for r in range(rank):
                    total += Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r]) * Fraction(W[k*n+i][r])
                if total != 1:
                    return False
    return True

def _try_structured_strassen_like():
    """
    Try a structured decomposition based on Strassen's algorithm for 3x3x3,
    extended to 3x3x8.
    
    For 3x3x3, Strassen gives rank 7. The idea is to find a decomposition
    where the middle dimension (j) is handled with fewer terms.
    
    Actually, let's try a different structured approach: exploit the fact that
    for each j, we need to compute A[:,j] * B[j,:] which is an outer product.
    We can try to find linear combinations that reduce the total rank.
    """
    n, m, p = DIMENSIONS
    
    # Try to find a decomposition with rank < 72 using random search
    # with structured initialization
    
    best_result = None
    best_rank = n * m * p
    
    # Strategy: For each rank r from lower bound to schoolbook, try random searches
    # Lower bound: max(n*m, m*p, p*n) = max(9, 24, 24) = 24
    
    # But for this specific problem, let's try a smarter approach.
    # The tensor can be viewed as a sum of m rank-(n*p) tensors, one for each j.
    # Each such tensor is the outer product of the i-index and k-index for fixed j.
    
    # Key insight: We can potentially combine the j-dimension using linear algebra.
    # If we find matrices U_j, V_j, W_j for j=0,1,2 such that their sum
    # (in a tensor sense) gives the full multiplication, we might reduce rank.
    
    # Let's try random search with small integer entries
    deadline = time.time() + 100
    
    for target_rank in range(24, 72):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Initialize with random values from {-1, 0, 1}
            U = [[0] * target_rank for _ in range(n * m)]
            V = [[0] * target_rank for _ in range(m * p)]
            W = [[0] * target_rank for _ in range(p * n)]
            
            for r in range(target_rank):
                for idx in range(n * m):
                    U[idx][r] = random.choice([-1, 0, 1])
                for idx in range(m * p):
                    V[idx][r] = random.choice([-1, 0, 1])
                for idx in range(p * n):
                    W[idx][r] = random.choice([-1, 0, 1])
            
            # Quick check with floating point first
            U_np = np.array(U, dtype=float)
            V_np = np.array(V, dtype=float)
            W_np = np.array(W, dtype=float)
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(np.dot(U_np[i*m+j], V_np[j*p+k]), W_np[k*n+i])
                        if abs(val - 1.0) > 1e-8:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Verify exactly
                if _verify_exact(U, V, W):
                    return {"U": U, "V": V, "W": W}
    
    return None

def _try_block_decomposition():
    """
    Try a block-based decomposition. Split B into blocks and use
    structured linear combinations.
    
    B is 3x8. We can view it as [B1 B2] where B1 is 3x4 and B2 is 3x4.
    Then C = A*B = A*B1 + A*B2.
    
    This doesn't directly help, but let's try a different block structure.
    
    Actually, let's try to use the fact that 3x3 matrix multiplication
    can be done in rank 7 (Strassen), and see if we can extend this.
    """
    n, m, p = DIMENSIONS
    
    # For now, try a specific structured pattern
    # Based on the idea that we can share computation across k indices
    
    # Let's try rank 24 (the lower bound) with a structured approach
    # For each j, we need n*p = 24 terms, but we can potentially share
    
    # This is complex; let's fall back to random search with more attempts
    deadline = time.time() + 80
    
    for target_rank in range(24, 72):
        if time.time() > deadline:
            break
        
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            # Use a biased random initialization: more zeros, some ±1
            U = [[0] * target_rank for _ in range(n * m)]
            V = [[0] * target_rank for _ in range(m * p)]
            W = [[0] * target_rank for _ in range(p * n)]
            
            for r in range(target_rank):
                for idx in range(n * m):
                    U[idx][r] = random.choice([-1, 0, 0, 1])  # Bias toward 0
                for idx in range(m * p):
                    V[idx][r] = random.choice([-1, 0, 0, 1])
                for idx in range(p * n):
                    W[idx][r] = random.choice([-1, 0, 0, 1])
            
            # Quick float check
            U_np = np.array(U, dtype=float)
            V_np = np.array(V, dtype=float)
            W_np = np.array(W, dtype=float)
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(np.dot(U_np[i*m+j], V_np[j*p+k]), W_np[k*n+i])
                        if abs(val - 1.0) > 1e-8:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid and _verify_exact(U, V, W):
                return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    """
    Find a low-rank exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Returns dict with U, V, W matrices.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Schoolbook as fallback
    schoolbook = _schoolbook()
    best_U, best_V, best_W = schoolbook["U"], schoolbook["V"], schoolbook["W"]
    best_rank = n * m * p  # 72
    
    # Try structured Strassen-like approach
    result = _try_structured_strassen_like()
    if result is not None:
        best_U, best_V, best_W = result["U"], result["V"], result["W"]
        best_rank = len(best_U[0])
    
    # Try block decomposition
    if best_rank == 72:
        result = _try_block_decomposition()
        if result is not None:
            best_U, best_V, best_W = result["U"], result["V"], result["W"]
            best_rank = len(best_U[0])
    
    # If we still have schoolbook, try one more intensive random search
    if best_rank == 72 and time.time() < deadline - 5:
        # Try ranks from 24 to 71 with more attempts
        for target_rank in range(24, 72):
            if time.time() > deadline - 5:
                break
            
            for attempt in range(200):
                if time.time() > deadline - 5:
                    break
                
                U = [[0] * target_rank for _ in range(n * m)]
                V = [[0] * target_rank for _ in range(m * p)]
                W = [[0] * target_rank for _ in range(p * n)]
                
                for r in range(target_rank):
                    for idx in range(n * m):
                        U[idx][r] = random.choice([-1, 0, 1])
                    for idx in range(m * p):
                        V[idx][r] = random.choice([-1, 0, 1])
                    for idx in range(p * n):
                        W[idx][r] = random.choice([-1, 0, 1])
                
                U_np = np.array(U, dtype=float)
                V_np = np.array(V, dtype=float)
                W_np = np.array(W, dtype=float)
                
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(np.dot(U_np[i*m+j], V_np[j*p+k]), W_np[k*n+i])
                            if abs(val - 1.0) > 1e-8:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid and _verify_exact(U, V, W):
                    best_U, best_V, best_W = U, V, W
                    best_rank = target_rank
                    break
            
            if best_rank != 72:
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
