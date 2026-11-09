"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _build_schoolbook():
    """Build the schoolbook decomposition as numpy arrays."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly decomposes matrix multiplication."""
    # The tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # should equal delta (i.e., T[i,j,k] = 1 for all i,j,k in schoolbook sense)
    # Actually, the condition is: for all i,j,k, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # Wait, no. The matrix multiplication tensor is T[i,j,k] = 1 if we're computing (AB)[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor has T[i,j,k] = 1 for all i,j,k (it's the identity tensor for multiplication)
    # The decomposition requires: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For schoolbook, each term is a single (i,j,k) triple with value 1.
    
    # Let's verify by checking that the product reconstructs the multiplication
    # (AB)[i,k] = sum_j A[i,j] B[j,k]
    # With decomposition: (AB)[i,k] = sum_r (sum_j U[i*m+j,r]*A[i,j]) * (sum_j V[j*p+k,r]*B[j,k]) ... no
    
    # Actually the standard form: C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor form: C[i,k] = sum_{j,r} A[i,j] B[j,k] * T[i,j,k,r] where T is the decomposition
    # With U, V, W: C[i,k] = sum_r (sum_j U[i*m+j,r] A[i,j]) * (sum_j V[j*p+k,r] B[j,k]) * W[k*n+i,r]
    
    # Let me just verify numerically with random matrices
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_ref = A @ B
    
    # Compute using decomposition
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    # This is not quite right. Let me think again.
    
    # Standard tensor decomposition for matrix mult:
    # T_{ijk} = sum_r U_{ir} V_{jr} W_{kr}  ... no, that's for rank-1 tensors
    # For matrix multiplication, the tensor is T_{ijk} where i indexes rows of A, j columns of A/rows of B, k columns of B
    # T_{ijk} = 1 for all i,j,k (the multiplication tensor)
    # Decomposition: T_{ijk} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    
    # So: 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for all i,j,k
    
    # Verify this identity
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _build_schoolbook()
    current_rank = n * m * p
    
    # Try to reduce rank by finding linear dependencies
    # Strategy: try to express one column as a linear combination of others
    # and remove it
    
    # Convert to float for numerical work
    U_f = U.astype(float)
    V_f = V.astype(float)
    W_f = W.astype(float)
    
    # Try greedy rank reduction: for each column, check if it can be removed
    # by adjusting others (this is complex)
    
    # Alternative: use known Strassen constructions
    # For 5x5, we can use a block decomposition
    
    # Let's try a direct approach: build a Strassen-like decomposition
    # Partition 5x5 into blocks: use 2x2 Strassen (rank 7) for the main blocks
    
    # Actually, let's try a simpler improvement: 
    # Use the fact that 5 = 2 + 2 + 1
    # We can decompose the multiplication using:
    # - 4 Strassen 2x2 multiplications (each rank 7) = 28
    # - Plus cross terms
    
    # This is getting complex. Let me try a numerical rank reduction search.
    
    # Better approach: use SVD or null space to find dependencies
    # The constraint is: for all (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a nonlinear constraint. Let me try a different strategy.
    
    # Strategy: Start from schoolbook, try to set some entries to 0 or combine columns
    # to reduce the effective rank.
    
    # For now, let's try a simple heuristic: look for columns that are linearly dependent
    # in the U matrix (or V or W) and try to eliminate them.
    
    # Actually, the most promising approach for 5x5 is to use a known construction.
    # Let me implement a Strassen-based construction.
    
    # Strassen 2x2: 
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 5x5, partition as:
    # A = [A11 A12 A13; A21 A22 A23; A31 A32 A33] where A11 is 2x2, etc.
    # This gives a 3x3 block structure with blocks of sizes 2,2,1
    
    # This is complex. Let me just try to do a numerical search for rank reduction.
    
    # Time check
    if time.time() > deadline - 30:
        # Return schoolbook if we're running out of time
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try a simple rank reduction: look for pairs of columns that can be combined
    # This is a heuristic search
    
    # For now, let's return the schoolbook decomposition as a baseline
    # and add a simple improvement attempt
    
    # Attempt: try to find if any column can be zeroed out by adjusting others
    # This requires solving a system, which is complex.
    
    # Let's just return schoolbook for now, but structured for easy evolution
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
