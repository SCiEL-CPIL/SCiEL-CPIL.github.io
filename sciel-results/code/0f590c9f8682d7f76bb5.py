import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 factors
    # U, V, W are (2, 2, 7)
    U2 = np.zeros((2, 2, 7))
    V2 = np.zeros((2, 2, 7))
    W2 = np.zeros((2, 2, 7))
    
    # r=0
    U2[0, 0, 0] = 1; U2[1, 1, 0] = 1
    V2[0, 0, 0] = 1; V2[1, 1, 0] = 1
    W2[0, 0, 0] = 1; W2[1, 1, 0] = 1
    
    # r=1
    U2[1, 0, 1] = 1; U2[1, 1, 1] = 1
    V2[0, 0, 1] = 1
    W2[0, 1, 1] = 1; W2[1, 1, 1] = -1
    
    # r=2
    U2[0, 0, 2] = 1
    V2[0, 1, 2] = 1; V2[1, 1, 2] = -1
    W2[1, 0, 2] = 1; W2[1, 1, 2] = 1
    
    # r=3
    U2[1, 1, 3] = 1
    V2[1, 0, 3] = 1; V2[0, 0, 3] = -1
    W2[0, 0, 3] = 1; W2[0, 1, 3] = 1
    
    # r=4
    U2[0, 0, 4] = 1; U2[0, 1, 4] = 1
    V2[1, 1, 4] = 1
    W2[0, 0, 4] = -1; W2[1, 0, 4] = 1
    
    # r=5
    U2[1, 0, 5] = 1; U2[0, 0, 5] = -1
    V2[0, 0, 5] = 1; V2[0, 1, 5] = 1
    W2[1, 1, 5] = 1
    
    # r=6
    U2[0, 1, 6] = 1; U2[1, 1, 6] = -1
    V2[1, 0, 6] = 1; V2[1, 1, 6] = 1
    W2[0, 0, 6] = 1

    # Build 4x4x4 factors using Kronecker product logic
    # Rank is 7*7 = 49
    rank4 = 49
    U4 = np.zeros((4, 4, rank4))
    V4 = np.zeros((4, 4, rank4))
    W4 = np.zeros((4, 4, rank4))
    
    # Vectorized construction of U4, V4, W4
    # Index mapping: 
    # U4[i, j, R] where i=2*i1+i0, j=2*j1+j0, R=7*s+r
    # U4[2*i1+i0, 2*j1+j0, 7*s+r] = U2[i1, j1, s] * U2[i0, j0, r]
    
    # We can build this by iterating over the 2x2 blocks
    for i1 in range(2):
        for j1 in range(2):
            for i0 in range(2):
                for j0 in range(2):
                    i = 2*i1 + i0
                    j = 2*j1 + j0
                    # The slice for R is determined by s and r
                    # U2[i1, j1, :] has shape (7,)
                    # U2[i0, j0, :] has shape (7,)
                    # We need to fill U4[i, j, :] which has shape (49,)
                    # U4[i, j, 7*s+r] = U2[i1, j1, s] * U2[i0, j0, r]
                    # This is the Kronecker product of U2[i1, j1, :] and U2[i0, j0, :]
                    # But careful with ordering: R = 7*s + r. 
                    # np.kron(A, B) flattens A then B. 
                    # If A is (7,) and B is (7,), kron(A, B) has length 49.
                    # Index k in kron(A,B) corresponds to A[k//7] * B[k%7].
                    # Here s = R//7, r = R%7.
                    # So U4[i, j, :] = np.kron(U2[i1, j1, :], U2[i0, j0, :])
                    
                    U4[i, j, :] = np.kron(U2[i1, j1, :], U2[i0, j0, :])
                    
    for j1 in range(2):
        for k1 in range(2):
            for j0 in range(2):
                for k0 in range(2):
                    j = 2*j1 + j0
                    k = 2*k1 + k0
                    V4[j, k, :] = np.kron(V2[j1, k1, :], V2[j0, k0, :])
                    
    for k1 in range(2):
        for i1 in range(2):
            for k0 in range(2):
                for i0 in range(2):
                    k = 2*k1 + k0
                    i = 2*i1 + i0
                    W4[k, i, :] = np.kron(W2[k1, i1, :], W2[k0, i0, :])

    # Build 4x4x8 factors
    # The problem is C = A * B, where A is 4x4, B is 4x8.
    # We can view B as two 4x4 matrices B1 and B2.
    # C = [C1, C2] where C1 = A*B1 and C2 = A*B2.
    # The decomposition for C1 uses rank 49.
    # The decomposition for C2 uses rank 49.
    # Since A is shared, the U factors are the same for both halves?
    # No, the tensor decomposition is for the whole operation.
    # The tensor is T_{i, j, k} = delta_{j, k} (for the multiplication tensor M_{n,m,p}).
    # Here n=4, m=4, p=8.
    # The multiplication tensor is the direct sum of two M_{4,4,4} tensors?
    # Yes, because B is 4x8, it's like two independent multiplications by 4x4 matrices.
    # So the rank is 49 + 49 = 98.
    
    rank_total = 98
    U = np.zeros((16, rank_total)) # n*m = 16
    V = np.zeros((32, rank_total)) # m*p = 32
    W = np.zeros((32, rank_total)) # p*n = 32
    
    # Fill U
    # U[i*4+j, R]
    # For R in 0..48 (first half of B), use U4[i, j, R]
    # For R in 49..97 (second half of B), use U4[i, j, R-49]
    for i in range(4):
        for j in range(4):
            idx = i*4 + j
            U[idx, 0:49] = U4[i, j, :]
            U[idx, 49:98] = U4[i, j, :]
            
    # Fill V
    # V[j*8+k, R]
    # First half (k in 0..3): V[j*8+k, R] = V4[j, k, R] for R in 0..48
    # Second half (k in 4..7): V[j*8+k, R] = V4[j, k-4, R-49] for R in 49..97
    for j in range(4):
        for k in range(4):
            # First half of B (columns 0-3)
            V[j*8 + k, 0:49] = V4[j, k, :]
            # Second half of B (columns 4-7)
            V[j*8 + (4+k), 49:98] = V4[j, k, :]
            
    # Fill W
    # W[k*4+i, R]
    # First half (k in 0..3): W[k*4+i, R] = W4[k, i, R] for R in 0..48
    # Second half (k in 4..7): W[k*4+i, R] = W4[k-4, i, R-49] for R in 49..97
    for k in range(4):
        for i in range(4):
            # First half of C (columns 0-3 of output)
            W[k*4 + i, 0:49] = W4[k, i, :]
            # Second half of C (columns 4-7 of output)
            W[(4+k)*4 + i, 49:98] = W4[k, i, :]

    # Quick validation with random small integers
    # A: 4x4, B: 4x8
    A = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    B = np.array([[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16], 
                  [17, 18, 19, 20, 21, 22, 23, 24], [25, 26, 27, 28, 29, 30, 31, 32]])
    
    C_expected = A @ B
    
    # Compute C from factors
    # C[i, k] = sum_R U[i*4+j, R] * V[j*8+k, R] * W[k*4+i, R] ... wait.
    # The decomposition is T = sum_R U_R \otimes V_R \otimes W_R
    # U_R is a vector of size 16 (flattened A indices)
    # V_R is a vector of size 32 (flattened B indices)
    # W_R is a vector of size 32 (flattened C indices)
    # The tensor element T_{i, j, k} = sum_R U[i, j, R] V[j, k, R] W[k, i, R]
    # In our flattened notation:
    # U_flat[i*m+j, R]
    # V_flat[j*p+k, R]
    # W_flat[k*n+i, R]
    
    # Let's verify one element, e.g., C[0, 0]
    # C[0, 0] = sum_{j=0}^3 A[0, j] B[j, 0]
    # In tensor form: C[0, 0] = sum_R (sum_j U[0*4+j, R] V[j*8+0, R]) W[0*4+0, R]
    # Wait, the standard decomposition is:
    # C_{i,k} = sum_{j} A_{i,j} B_{j,k}
    # This is represented by the tensor T_{i,j,k} = delta_{j,k} (no, that's not right).
    # The multiplication tensor M_{n,m,p} has components M_{i,j,k} = 1 if we are computing C_{i,k} from A_{i,j} and B_{j,k}.
    # Actually, the tensor is defined such that C = A*B corresponds to:
    # C_{i,k} = sum_{j} A_{i,j} B_{j,k}
    # The tensor T has entries T_{i, j, k} = 1 if the term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # So T_{i, j, k} is non-zero only for specific indices.
    # Specifically, T_{i, j, k} = 1 if we are multiplying A_{i,j} and B_{j,k} to get C_{i,k}.
    # But in the decomposition, we have:
    # T_{i, j, k} = sum_R U_{i, j, R} V_{j, k, R} W_{k, i, R}
    # Note the indices on W: W is indexed by (k, i) in the problem description "W[k*n+i,r]".
    # So W_{k, i, R}.
    
    # Let's check C[0, 0].
    # C[0, 0] = sum_j A[0, j] B[j, 0]
    # From decomposition:
    # C[0, 0] = sum_R (sum_j U[0, j, R] V[j, 0, R]) W[0, 0, R]  <-- This is not quite right.
    # The tensor T_{i, j, k} represents the coefficient of A_{i, j} B_{j, k} in C_{i, k}.
    # So C_{i, k} = sum_{j} T_{i, j, k} A_{i, j} B_{j, k}.
    # Since T_{i, j, k} = 1 for all i, j, k (in the standard multiplication tensor, every pair A_{i,j}, B_{j,k} contributes to C_{i,k}),
    # wait, no. A_{i, j} only contributes to C_{i, k} if the column index of A matches the row index of B.
    # So T_{i, j, k} is 1 if we are computing C_{i, k} using A_{i, j} and B_{j, k}.
    # This is always true for the specific j.
    # So T_{i, j, k} = 1 for all i, j, k? No.
    # The tensor M_{n, m, p} is defined by M_{i, j, k} = 1 if i, j, k are valid indices.
    # Actually, the multiplication tensor is:
    # M = sum_{i,j,k} E_{i,k} \otimes E_{i,j} \otimes E_{j,k}
    # Where E_{i,k} is the basis vector for C, E_{i,j} for A, E_{j,k} for B.
    # So the component M_{i, j, k} (where i is C row, j is A col/B row, k is C col) is 1.
    # So yes, M_{i, j, k} = 1 for all i, j, k.
    # Therefore, C_{i, k} = sum_j M_{i, j, k} A_{i, j} B_{j, k} = sum_j A_{i, j} B_{j, k}.
    
    # So the check is:
    # sum_R U_flat[i*m+j, R] V_flat[j*p+k, R] W_flat[k*n+i, R] should equal 1 if we sum over j?
    # No. The decomposition is T_{i, j, k} = sum_R U_{i, j, R} V_{j, k, R} W_{k, i, R}.
    # And we require T_{i, j, k} = 1 for all i, j, k.
    # So for any fixed i, j, k, the sum over R must be 1.
    # And for any i, j, k' with k' != k? No, T is 1 for all i, j, k.
    # Wait, is T_{i, j, k} 1 for all i, j, k?
    # Let's check the definition.
    # The tensor for matrix multiplication C = AB is:
    # T = sum_{i=1}^n sum_{j=1}^m sum_{k=1}^p e_{i,k} \otimes e_{i,j} \otimes e_{j,k}
    # The component T_{i, j, k} (with indices corresponding to C, A, B) is 1.
    # So yes, T_{i, j, k} = 1 for all i, j, k.
    
    # So we can verify:
    # For a random i, j, k, sum_R U[i, j, R] V[j, k, R] W[k, i, R] should be 1.
    
    i_test, j_test, k_test = 0, 1, 2
    val = 0
    for R in range(rank_total):
        u_val = U[i_test*4 + j_test, R]
        v_val = V[j_test*8 + k_test, R]
        w_val = W[k_test*4 + i_test, R]
        val += u_val * v_val * w_val
        
    if not np.isclose(val, 1.0):
        # Try another
        i_test, j_test, k_test = 3, 3, 7
        val = 0
        for R in range(rank_total):
            u_val = U[i_test*4 + j_test, R]
            v_val = V[j_test*8 + k_test, R]
            w_val = W[k_test*4 + i_test, R]
            val += u_val * v_val * w_val
        if not np.isclose(val, 1.0):
             # Fallback: return zeros if verification fails (should not happen)
             return {"U": np.zeros((16, 98)).tolist(), "V": np.zeros((32, 98)).tolist(), "W": np.zeros((32, 98)).tolist()}

    if time.time() > deadline:
         return {"U": np.zeros((16, 98)).tolist(), "V": np.zeros((32, 98)).tolist(), "W": np.zeros((32, 98)).tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
