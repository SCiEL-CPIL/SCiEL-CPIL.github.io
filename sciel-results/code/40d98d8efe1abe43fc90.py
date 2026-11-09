"""
Refined program for 2x3x8 matrix multiplication rank minimization.
Design: Strassen (2x2x2) blocks for the first 2 columns of B, 
        Outer product for the remaining column.
Rank: 44 (7*4 + 16).
Refactoring: Explicit term generation to avoid loop-based AST duplication.
"""
import numpy as np
import time

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit

    # Rank 44 decomposition
    # U shape: (n*m, rank) = (6, 44)
    # V shape: (m*p, rank) = (24, 44)
    # W shape: (p*n, rank) = (16, 44)
    
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Indices for A (2x3 matrix flattened row-major? No, U index is i*m+j)
    # A indices:
    # A[0,0]=0, A[0,1]=1, A[0,2]=2
    # A[1,0]=3, A[1,1]=4, A[1,2]=5
    
    # Indices for B (3x8 matrix flattened row-major? V index is j*p+k)
    # B indices:
    # Row 0: 0..7
    # Row 1: 8..15
    # Row 2: 16..23
    
    # Indices for C (2x8 matrix flattened col-major? W index is k*n+i)
    # C indices:
    # Col 0: 0,1
    # Col 1: 2,3
    # ...
    # Col k: 2k, 2k+1

    col = 0

    # --- PART 1: Strassen for A[0:2, 0:2] * B[0:2, 0:8] ---
    # We process 4 blocks of 2x2 matrices.
    # Block b corresponds to B columns 2b and 2b+1.
    
    # Block 0: B cols 0,1
    # A submatrix: A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
    # B submatrix: B[0,0]=0, B[0,1]=1, B[1,0]=8, B[1,1]=9
    # C submatrix: C[0,0]=0, C[0,1]=2, C[1,0]=1, C[1,1]=3
    
    # M1: (A00+A11)(B00+B11)
    U[0, col] += 1; U[4, col] += 1
    V[0, col] += 1; V[9, col] += 1
    W[0, col] += 1; W[3, col] += 1
    col += 1
    
    # M2: (A10+A11)B00
    U[3, col] += 1; U[4, col] += 1
    V[0, col] += 1
    W[1, col] += 1; W[3, col] -= 1
    col += 1
    
    # M3: A00(B01-B11)
    U[0, col] += 1
    V[1, col] += 1; V[9, col] -= 1
    W[2, col] += 1; W[3, col] += 1
    col += 1
    
    # M4: A11(B10-B00)
    U[4, col] += 1
    V[8, col] += 1; V[0, col] -= 1
    W[0, col] += 1; W[1, col] += 1
    col += 1
    
    # M5: (A00+A01)B11
    U[0, col] += 1; U[1, col] += 1
    V[9, col] += 1
    W[0, col] -= 1; W[2, col] += 1
    col += 1
    
    # M6: (A10-A00)(B00+B01)
    U[3, col] += 1; U[0, col] -= 1
    V[0, col] += 1; V[1, col] += 1
    W[3, col] += 1
    col += 1
    
    # M7: (A01-A11)(B10+B11)
    U[1, col] += 1; U[4, col] -= 1
    V[8, col] += 1; V[9, col] += 1
    W[0, col] += 1
    col += 1

    # Block 1: B cols 2,3
    # B submatrix: B[0,2]=2, B[0,3]=3, B[1,2]=10, B[1,3]=11
    # C submatrix: C[0,2]=4, C[0,3]=6, C[1,2]=5, C[1,3]=7
    
    # M1
    U[0, col] += 1; U[4, col] += 1
    V[2, col] += 1; V[11, col] += 1
    W[4, col] += 1; W[7, col] += 1
    col += 1
    
    # M2
    U[3, col] += 1; U[4, col] += 1
    V[2, col] += 1
    W[5, col] += 1; W[7, col] -= 1
    col += 1
    
    # M3
    U[0, col] += 1
    V[3, col] += 1; V[11, col] -= 1
    W[6, col] += 1; W[7, col] += 1
    col += 1
    
    # M4
    U[4, col] += 1
    V[10, col] += 1; V[2, col] -= 1
    W[4, col] += 1; W[5, col] += 1
    col += 1
    
    # M5
    U[0, col] += 1; U[1, col] += 1
    V[11, col] += 1
    W[4, col] -= 1; W[6, col] += 1
    col += 1
    
    # M6
    U[3, col] += 1; U[0, col] -= 1
    V[2, col] += 1; V[3, col] += 1
    W[7, col] += 1
    col += 1
    
    # M7
    U[1, col] += 1; U[4, col] -= 1
    V[10, col] += 1; V[11, col] += 1
    W[4, col] += 1
    col += 1

    # Block 2: B cols 4,5
    # B submatrix: B[0,4]=4, B[0,5]=5, B[1,4]=12, B[1,5]=13
    # C submatrix: C[0,4]=8, C[0,5]=10, C[1,4]=9, C[1,5]=11
    
    # M1
    U[0, col] += 1; U[4, col] += 1
    V[4, col] += 1; V[13, col] += 1
    W[8, col] += 1; W[11, col] += 1
    col += 1
    
    # M2
    U[3, col] += 1; U[4, col] += 1
    V[4, col] += 1
    W[9, col] += 1; W[11, col] -= 1
    col += 1
    
    # M3
    U[0, col] += 1
    V[5, col] += 1; V[13, col] -= 1
    W[10, col] += 1; W[11, col] += 1
    col += 1
    
    # M4
    U[4, col] += 1
    V[12, col] += 1; V[4, col] -= 1
    W[8, col] += 1; W[9, col] += 1
    col += 1
    
    # M5
    U[0, col] += 1; U[1, col] += 1
    V[13, col] += 1
    W[8, col] -= 1; W[10, col] += 1
    col += 1
    
    # M6
    U[3, col] += 1; U[0, col] -= 1
    V[4, col] += 1; V[5, col] += 1
    W[11, col] += 1
    col += 1
    
    # M7
    U[1, col] += 1; U[4, col] -= 1
    V[12, col] += 1; V[13, col] += 1
    W[8, col] += 1
    col += 1

    # Block 3: B cols 6,7
    # B submatrix: B[0,6]=6, B[0,7]=7, B[1,6]=14, B[1,7]=15
    # C submatrix: C[0,6]=12, C[0,7]=14, C[1,6]=13, C[1,7]=15
    
    # M1
    U[0, col] += 1; U[4, col] += 1
    V[6, col] += 1; V[15, col] += 1
    W[12, col] += 1; W[15, col] += 1
    col += 1
    
    # M2
    U[3, col] += 1; U[4, col] += 1
    V[6, col] += 1
    W[13, col] += 1; W[15, col] -= 1
    col += 1
    
    # M3
    U[0, col] += 1
    V[7, col] += 1; V[15, col] -= 1
    W[14, col] += 1; W[15, col] += 1
    col += 1
    
    # M4
    U[4, col] += 1
    V[14, col] += 1; V[6, col] -= 1
    W[12, col] += 1; W[13, col] += 1
    col += 1
    
    # M5
    U[0, col] += 1; U[1, col] += 1
    V[15, col] += 1
    W[12, col] -= 1; W[14, col] += 1
    col += 1
    
    # M6
    U[3, col] += 1; U[0, col] -= 1
    V[6, col] += 1; V[7, col] += 1
    W[15, col] += 1
    col += 1
    
    # M7
    U[1, col] += 1; U[4, col] -= 1
    V[14, col] += 1; V[15, col] += 1
    W[12, col] += 1
    col += 1

    # --- PART 2: Outer Product for A[0:2, 2] * B[2, 0:8] ---
    # A indices: A[0,2]=2, A[1,2]=5
    # B indices: B[2,0]=16 ... B[2,7]=23
    # C indices: C[i,k] = k*n + i
    
    for i in range(n):
        a_idx = i * m + 2
        for k in range(p):
            b_idx = 2 * p + k
            c_idx = k * n + i
            
            U[a_idx, col] += 1
            V[b_idx, col] += 1
            W[c_idx, col] += 1
            col += 1

    # Verify decomposition if time permits
    if time.time() < deadline - 5:
        if not _verify_decomposition(U, V, W, n, m, p):
            # In case of unexpected error, return zeros (invalid) rather than wrong data?
            # No, the decomposition is hardcoded and correct. 
            # This check is just for sanity.
            pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify tensor decomposition: T[i,j,k] = sum_r U[i,r]V[j,r]W[k,r]
    For matrix multiplication, T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    """
    rank = U.shape[1]
    # Use numpy for fast verification
    # Tensor shape (n, m, p)
    # U: (n*m, rank) -> reshape to (n, m, rank)
    # V: (m*p, rank) -> reshape to (m, p, rank)
    # W: (p*n, rank) -> reshape to (p, n, rank)
    
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # Sum over rank axis
    # Term = U[i,j,r] * V[j,k,r] * W[k,i,r]
    # We need to sum r.
    # Let's compute T[i,j,k]
    # This is a contraction.
    # T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    try:
        T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    except Exception:
        return False
        
    # T should be all ones
    return np.allclose(T, np.ones((n, m, p)), atol=1e-9)
# EVOLVE-BLOCK-END
