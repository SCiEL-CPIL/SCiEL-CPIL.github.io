"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    n, m, p = DIMENSIONS
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Try to find a lower-rank exact decomposition
    # Use a structured approach: try ranks from 50 upward
    for target_rank in range(50, 126):
        if time.time() > deadline - 5:
            break
        result = _try_construct(n, m, p, target_rank, deadline)
        if result is not None:
            return result
    
    # Return schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Check that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,k} for all i,j,k
    # Wait, that's not right. Let me re-read the convention.
    # W[k*n+i, r] - so W has shape (p*n, rank)
    # The tensor is T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For matrix multiplication: (AB)[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] should equal delta_{i,k} ... no wait.
    # Actually the multiplication tensor T satisfies: (A*B)[i,k] = sum_{j} A[i,j] * B[j,k]
    # In tensor form: T_{i,j,k} such that T(A,B)_{i,k} = sum_{j} T_{i,j,k} A_{i,j} B_{j,k}
    # So T_{i,j,k} = delta_{i,k}? No, that's not right either.
    
    # Let me think again. The bilinear map is: (A, B) -> A*B
    # (A*B)[i,k] = sum_j A[i,j] B[j,k]
    # In the tensor decomposition: (A*B)[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
    # Wait, I'm confusing myself. Let me use the standard formulation.
    
    # The matrix multiplication tensor T is a 3-tensor T[i,j,k] where:
    # (A*B)[i,k] = sum_{j} A[i,j] B[j,k] = sum_{j} T[i,j,k] A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all i,j,k? No, that gives sum_j A[i,j]B[j,k] which is correct!
    # Wait, T[i,j,k] = 1 for all i,j,k means the tensor is all ones, which gives:
    # sum_{j} 1 * A[i,j] * B[j,k] = (A*B)[i,k]. Yes!
    
    # So the tensor T[i,j,k] = 1 for all i,j,k in the range [0,n), [0,m), [0,p)
    # And the decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Verify: for all i in [0,n), j in [0,m), k in [0,p):
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1) > 1e-12:
                    return False
    return True

def _try_construct(n, m, p, target_rank, deadline):
    """Try to construct a rank target_rank decomposition."""
    # Use random initialization with structured patterns
    for attempt in range(20):
        if time.time() > deadline - 2:
            break
        
        # Initialize with random small values
        U = [[0.0] * target_rank for _ in range(n * m)]
        V = [[0.0] * target_rank for _ in range(m * p)]
        W = [[0.0] * target_rank for _ in range(p * n)]
        
        # Use a structured initialization based on Kronecker product ideas
        for r in range(target_rank):
            # Assign random indices
            i = (r * 7) % n
            j = (r * 11) % m
            k = (r * 13) % p
            U[i*m+j][r] = 1.0
            V[j*p+k][r] = 1.0
            W[k*n+i][r] = 1.0
        
        # Verify
        if _verify_decomposition(U, V, W, n, m, p):
            # Convert to integer/half-integer
            U_int = [[int(round(2*x))/2 for x in row] for row in U]
            V_int = [[int(round(2*x))/2 for x in row] for row in V]
            W_int = [[int(round(2*x))/2 for x in row] for row in W]
            if _verify_decomposition(U_int, V_int, W_int, n, m, p):
                return {"U": U_int, "V": V_int, "W": W_int}
    
    return None
# EVOLVE-BLOCK-END
