"""Find a rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 5]

def _verify(U, V, W, tol=1e-10):
    """Check if U, V, W exactly decompose the matmul tensor."""
    n, m, p = DIMENSIONS
    r = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(r):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    """
    Find a low-rank decomposition for 2x4 * 4x5 matrix multiplication.
    
    The matmul tensor T[i,j,k] = 1 for all i in [0,2), j in [0,4), k in [0,5).
    We seek U (8 x r), V (20 x r), W (10 x r) such that
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s limit
    
    # Schoolbook solution as fallback: rank = n*m*p = 40
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Strategy: Use structured decompositions
    # Key insight: 2x4 * 4x5 can be decomposed by exploiting structure
    
    # Approach 1: Winograd-style / Strassen-style block decomposition
    # 2x4 = [2x2 | 2x2] horizontally
    # 4x5 = [2x5; 2x5] vertically
    # Product = A1*B1 + A2*B2 where A1,A2 are 2x2, B1,B2 are 2x5
    
    # For 2x2 * 2x5: rank is at most 2*2*5=20, but can be lower
    # Actually, let's think about this differently
    
    # Approach 2: Direct structured construction
    # T[i,j,k] = 1 for all i,j,k
    # We need sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    
    # Let's try to construct a decomposition of rank r where r < 40
    
    # Key observation: If we can find vectors such that the sum works,
    # we can use random constructions or structured ones
    
    # Let's try a systematic approach:
    # For rank r, we need to solve a system of bilinear equations
    
    # Try ranks from low to high with random initialization + verification
    # Focus on promising ranks based on theoretical bounds
    
    # Theoretical: For n x m * m x p, rank is at least max(n*m, m*p, n*p)
    # Here: max(8, 20, 10) = 20
    # So rank >= 20
    
    # Let's try to find rank 20, 21, 22, ... up to 39
    
    # Method: Random initialization with structure
    # For each target rank, try many random initializations
    
    for target_rank in range(20, best_rank):
        if time.time() > deadline:
            break
        
        found = False
        max_attempts = 2000
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            rng = np.random.RandomState(attempt * 7919 + target_rank * 131)
            
            # Initialize with random values
            # Scale to help convergence
            scale = 1.0
            U = rng.randn(n*m, target_rank) * scale
            V = rng.randn(m*p, target_rank) * scale
            W = rng.randn(p*n, target_rank) * scale
            
            # Verify
            if _verify(U.tolist(), V.tolist(), W.tolist()):
                best_U, best_V, best_W = U.tolist(), V.tolist(), W.tolist()
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    # If we haven't found anything better than schoolbook, try structured approaches
    
    # Structured approach for rank 20:
    # Use the fact that T[i,j,k] = 1
    # Try: U[i,j,c] = a_i * b_j * c_c
    #       V[j,k,c] = d_j * e_k * f_c  
    #       W[k,i,c] = g_k * h_i * l_c
    # Then sum_c a_i * b_j * c_c * d_j * e_k * f_c * g_k * h_i * l_c = 1
    # = (a_i * h_i) * (b_j * d_j) * (e_k * g_k) * sum_c (c_c * f_c * l_c)
    # This gives a product of three terms times a scalar, which can't equal 1 for all i,j,k
    # unless the scalar is 1 and each product is 1, which is too restrictive
    
    # Better structured approach:
    # For rank 20, use the structure of the problem
    # 2x4 * 4x5: split into 4 independent 2x1 * 1x5 problems? No, that's rank 4*2*5=40
    
    # Alternative: Use bilinear complexity results
    # For small matrices, exact decompositions exist
    
    # Let's try a different random strategy with more structure
    if best_rank == rank_school and time.time() < deadline:
        # Try rank 20 with a specific structured random approach
        for target_rank in [20, 21, 22, 23, 24]:
            if time.time() > deadline:
                break
            
            max_attempts = 5000
            for attempt in range(max_attempts):
                if time.time() > deadline:
                    break
                
                rng = np.random.RandomState(attempt * 31 + target_rank * 1000)
                
                # Use a more structured initialization
                # Initialize U, V, W with values that might work
                U = np.zeros((n*m, target_rank))
                V = np.zeros((m*p, target_rank))
                W = np.zeros((p*n, target_rank))
                
                # Fill with random values
                for c in range(target_rank):
                    for i in range(n):
                        for j in range(m):
                            U[i*m+j, c] = rng.uniform(-1, 1)
                    for j in range(m):
                        for k in range(p):
                            V[j*p+k, c] = rng.uniform(-1, 1)
                    for k in range(p):
                        for i in range(n):
                            W[k*n+i, c] = rng.uniform(-1, 1)
                
                # Verify
                if _verify(U.tolist(), V.tolist(), W.tolist()):
                    best_U, best_V, best_W = U.tolist(), V.tolist(), W.tolist()
                    best_rank = target_rank
                    break
            
            if best_rank == target_rank:
                break
    
    # Final: try to find any rank below 40 using a more exhaustive search
    if best_rank == rank_school and time.time() < deadline:
        # Try all ranks from 20 to 39 with more attempts
        for target_rank in range(20, 40):
            if time.time() > deadline:
                break
            
            max_attempts = 1000
            for attempt in range(max_attempts):
                if time.time() > deadline:
                    break
                
                rng = np.random.RandomState(attempt * 7 + target_rank * 13)
                
                # Different initialization strategies
                strategy = attempt % 3
                
                if strategy == 0:
                    # Pure random
                    U = rng.randn(n*m, target_rank)
                    V = rng.randn(m*p, target_rank)
                    W = rng.randn(p*n, target_rank)
                elif strategy == 1:
                    # Random with some zeros
                    U = rng.randn(n*m, target_rank)
                    V = rng.randn(m*p, target_rank)
                    W = rng.randn(p*n, target_rank)
                    # Zero out some entries
                    for _ in range(target_rank // 2):
                        idx = rng.randint(n*m)
                        col = rng.randint(target_rank)
                        U[idx, col] = 0
                else:
                    # Sparse random
                    U = np.zeros((n*m, target_rank))
                    V = np.zeros((m*p, target_rank))
                    W = np.zeros((p*n, target_rank))
                    for c in range(target_rank):
                        for _ in range(n*m // 2):
                            idx = rng.randint(n*m)
                            U[idx, c] = rng.uniform(-2, 2)
                        for _ in range(m*p // 2):
                            idx = rng.randint(m*p)
                            V[idx, c] = rng.uniform(-2, 2)
                        for _ in range(p*n // 2):
                            idx = rng.randint(p*n)
                            W[idx, c] = rng.uniform(-2, 2)
                
                if _verify(U.tolist(), V.tolist(), W.tolist()):
                    best_U, best_V, best_W = U.tolist(), V.tolist(), W.tolist()
                    best_rank = target_rank
                    break
            
            if best_rank == target_rank:
                break
    
    # Return the best result found
    U_out = [[round(val, 10) for val in row] for row in best_U]
    V_out = [[round(val, 10) for val in row] for row in best_V]
    W_out = [[round(val, 10) for val in row] for row in best_W]
    
    return {"U": U_out, "V": V_out, "W": W_out}
