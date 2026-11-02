"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

_start_time = time.time()
_DEADLINE = _start_time + 140.0  # Safety margin before 150s kill

def _construct_strassen():
    """Fallback: Strassen-based construction with rank 53."""
    n, m, p = DIMENSIONS
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(base_r, k0, k1):
        r1, r2, r3, r4, r5, r6, r7 = base_r, base_r+1, base_r+2, base_r+3, base_r+4, base_r+5, base_r+6
        
        # P1: (A00+A11)(B0k0+B1k1)
        U[0][r1] = 1; U[3][r1] = 1
        V[0*p + k0][r1] = 1; V[1*p + k1][r1] = 1
        
        # P2: (A10+A11)B0k0
        U[2][r2] = 1; U[3][r2] = 1
        V[0*p + k0][r2] = 1
        
        # P3: A00(B0k1-B1k1)
        U[0][r3] = 1
        V[0*p + k1][r3] = 1; V[1*p + k1][r3] = -1
        
        # P4: A11(B1k0-B0k0)
        U[3][r4] = 1
        V[1*p + k0][r4] = 1; V[0*p + k0][r4] = -1
        
        # P5: (A00+A01)B1k1
        U[0][r5] = 1; U[1][r5] = 1
        V[1*p + k1][r5] = 1
        
        # P6: (A10-A00)(B0k0+B0k1)
        U[2][r6] = 1; U[0][r6] = -1
        V[0*p + k0][r6] = 1; V[0*p + k1][r6] = 1
        
        # P7: (A01-A11)(B1k0+B1k1)
        U[1][r7] = 1; U[3][r7] = -1
        V[1*p + k0][r7] = 1; V[1*p + k1][r7] = 1
        
        # W reconstruction
        W[2*k0][r1] = 1; W[2*k0][r4] = 1; W[2*k0][r5] = -1; W[2*k0][r7] = 1
        W[2*k0+1][r2] = 1; W[2*k0+1][r4] = 1
        W[2*k1][r3] = 1; W[2*k1][r5] = 1
        W[2*k1+1][r1] = 1; W[2*k1+1][r2] = -1; W[2*k1+1][r3] = 1; W[2*k1+1][r6] = 1
    
    for i in range(7):
        add_strassen_block(7 * i, 2*i, 2*i+1)
    
    # Trivial for column 14
    base_r = 49
    U[0][base_r] = 1; V[14][base_r] = 1; W[28][base_r] = 1
    U[1][base_r+1] = 1; V[29][base_r+1] = 1; W[28][base_r+1] = 1
    U[2][base_r+2] = 1; V[14][base_r+2] = 1; W[29][base_r+2] = 1
    U[3][base_r+3] = 1; V[29][base_r+3] = 1; W[29][base_r+3] = 1
    
    return {"U": U, "V": V, "W": W}

def _build_constraint_matrix(n, m, p):
    """Build the constraint matrix M where M @ vec(U,V,W) = 0 for the tensor identity.
    
    The tensor identity is: for all i,j,k:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta-like structure
    
    Actually, the correct formulation: C[i,k] = sum_j A[i,j]*B[j,k]
    In tensor form: sum_{j,r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal the product.
    
    For exact decomposition, we need:
    For each (i,k): sum_j A[i,j] B[j,k] = sum_r (sum_j U[i*m+j,r] * V[j*p+k,r]) * W[k*n+i,r]
    
    This is nonlinear. Instead, we use the standard approach:
    The multiplication tensor has entries T[i,j,k] = 1 if the output C[i,k] depends on A[i,j]*B[j,k].
    
    A rank-r decomposition means: T[i,j,k] = sum_{r=1}^r U[i,j,r] * V[j,k,r] * W[i,k,r]
    where U has shape (n*m, r), V has shape (m*p, r), W has shape (n*p, r).
    
    Wait, let me re-read the convention. The task says:
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    
    And the output factor follows W[k*n+i, r].
    
    So the tensor is: T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    T[i,j,k] = 1 if C[i,k] uses A[i,j]*B[j,k], which is always true for standard matmul.
    
    Actually for 2x2 * 2x15: C[i,k] = sum_j A[i,j]*B[j,k] for j=0,1.
    So T[i,j,k] = 1 for all valid i,j,k.
    
    A rank-r decomposition: T[i,j,k] = sum_{r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    We can linearize by fixing two of the three matrices. But that's complex.
    
    Alternative: Use the fact that if we fix U and V, then W is determined (or constrained).
    Or use a different approach: find a set of rank-1 tensors that sum to T.
    
    Let me use a simpler approach: randomly generate candidate decompositions and check validity.
    """
    pass

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for k in range(p):
            # Expected: C[i,k] = sum_j A[i,j] * B[j,k]
            # In tensor form: T[i,j,k] = 1 for all i,j,k (since every A[i,j] contributes to C[i,k] via B[j,k])
            # Actually no. The tensor T[i,j,k] represents the coefficient of A[i,j]*B[j,k] in C[i,k].
            # For standard matmul, C[i,k] = sum_j A[i,j]*B[j,k], so T[i,j,k] = 1 for all valid i,j,k.
            # 
            # The decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
            #
            # Wait, that's not right either. Let me think again.
            #
            # The matrix multiplication can be viewed as a tensor T in R^{n x m x p} where T[i,j,k] = 1
            # if the (i,k)-th entry of C depends on A[i,j] and B[j,k]. For standard matmul, every
            # A[i,j] contributes to C[i,k] through B[j,k], so T[i,j,k] = 1 for all i,j,k.
            #
            # A rank-r decomposition of this tensor is: T[i,j,k] = sum_{r=1}^r U[i,j,r] * V[j,k,r] * W[i,k,r]
            # where U in R^{n x m x r}, V in R^{m x p x r}, W in R^{n x p x r}.
            #
            # In the flattened convention: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r].
            #
            # So we need: for all i in [0,n), j in [0,m), k in [0,p):
            #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            #
            # This is a system of n*m*p equations.
            
            for j in range(m):
                lhs = 0.0
                for r in range(rank):
                    lhs += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(lhs - 1.0) > 1e-9:
                    return False
    return True

def _try_random_search(n, m, p, target_rank, num_trials=1000, time_limit=10.0):
    """Try to find a rank-target_rank decomposition using random search."""
    start = time.time()
    
    # The tensor T[i,j,k] = 1 for all i,j,k (n*m*p = 2*2*15 = 60 entries)
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Strategy: Use the Strassen decomposition as a starting point and try to find linear dependencies.
    # Or: randomly initialize and use gradient-based optimization (but that's complex).
    # 
    # Simpler: Try to find a basis for the space of valid decompositions and then find a sparse one.
    
    # For now, just return None to indicate no improvement found.
    # A more sophisticated approach would be needed.
    
    return None

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15
    
    # First, try the global search (with a short time limit)
    if time.time() - _start_time < _DEADLINE - 5.0:
        result = _try_random_search(n, m, p, 50, num_trials=100, time_limit=5.0)
        if result is not None:
            return result
    
    # Fall back to the Strassen construction
    return _construct_strassen()

# EVOLVE-BLOCK-END
