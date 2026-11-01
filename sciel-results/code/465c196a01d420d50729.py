import time
import numpy as np

DIMENSIONS = [2, 2, 7]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 120.0

def _time_remaining():
    return _DEADLINE - time.time()

def construct():
    """
    Construct a rank-25 exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    
    Strategy:
    - Use Strassen's rank-7 algorithm for columns 0,1
    - Use Strassen's rank-7 algorithm for columns 2,3
    - Use Strassen's rank-7 algorithm for columns 4,5
    - Use schoolbook (rank 4) for column 6
    Total rank: 7 + 7 + 7 + 4 = 25
    """
    if _time_remaining() < 5.0:
        # Safety: return a trivially valid schoolbook rank-28 decomposition
        return _schoolbook_decomposition()
    
    n, m, p = 2, 2, 7
    rank = 25
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def add_strassen(col_start, k_start):
        """Add Strassen rank-7 terms for 2x2 block at columns k_start, k_start+1."""
        
        # M1 = (A00 + A11) * (B0,ks + B1,ks+1)
        U[0, col_start+0] += 1
        U[3, col_start+0] += 1
        V[7*0 + k_start, col_start+0] += 1
        V[7*1 + (k_start+1), col_start+0] += 1
        W[2*k_start + 0, col_start+0] += 1
        W[2*(k_start+1) + 1, col_start+0] += 1

        # M2 = (A10 + A11) * B0,ks
        U[2, col_start+1] += 1
        U[3, col_start+1] += 1
        V[7*0 + k_start, col_start+1] += 1
        W[2*k_start + 1, col_start+1] += 1
        W[2*(k_start+1) + 1, col_start+1] -= 1

        # M3 = A00 * (B0,ks+1 - B1,ks+1)
        U[0, col_start+2] += 1
        V[7*0 + (k_start+1), col_start+2] += 1
        V[7*1 + (k_start+1), col_start+2] -= 1
        W[2*(k_start+1) + 0, col_start+2] += 1
        W[2*(k_start+1) + 1, col_start+2] += 1

        # M4 = A11 * (B1,ks - B0,ks)
        U[3, col_start+3] += 1
        V[7*1 + k_start, col_start+3] += 1
        V[7*0 + k_start, col_start+3] -= 1
        W[2*k_start + 0, col_start+3] += 1
        W[2*k_start + 1, col_start+3] += 1

        # M5 = (A00 + A01) * B1,ks+1
        U[0, col_start+4] += 1
        U[1, col_start+4] += 1
        V[7*1 + (k_start+1), col_start+4] += 1
        W[2*k_start + 0, col_start+4] -= 1
        W[2*(k_start+1) + 0, col_start+4] += 1

        # M6 = (A10 - A00) * (B0,ks + B0,ks+1)
        U[2, col_start+5] += 1
        U[0, col_start+5] -= 1
        V[7*0 + k_start, col_start+5] += 1
        V[7*0 + (k_start+1), col_start+5] += 1
        W[2*(k_start+1) + 1, col_start+5] += 1

        # M7 = (A01 - A11) * (B1,ks + B1,ks+1)
        U[1, col_start+6] += 1
        U[3, col_start+6] -= 1
        V[7*1 + k_start, col_start+6] += 1
        V[7*1 + (k_start+1), col_start+6] += 1
        W[2*k_start + 0, col_start+6] += 1
        
    # Add Strassen for columns 0,1
    add_strassen(0, 0)
    
    # Add Strassen for columns 2,3
    add_strassen(7, 2)
    
    # Add Strassen for columns 4,5
    add_strassen(14, 4)
    
    # Add schoolbook for column 6 (4 terms)
    col_start = 21
    # A00 * B0,6 -> C0,6
    U[0, col_start+0] += 1
    V[7*0 + 6, col_start+0] += 1
    W[2*6 + 0, col_start+0] += 1
    
    # A01 * B1,6 -> C0,6
    U[1, col_start+1] += 1
    V[7*1 + 6, col_start+1] += 1
    W[2*6 + 0, col_start+1] += 1
    
    # A10 * B0,6 -> C1,6
    U[2, col_start+2] += 1
    V[7*0 + 6, col_start+2] += 1
    W[2*6 + 1, col_start+2] += 1
    
    # A11 * B1,6 -> C1,6
    U[3, col_start+3] += 1
    V[7*1 + 6, col_start+3] += 1
    W[2*6 + 1, col_start+3] += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook_decomposition():
    """Fallback rank-28 schoolbook decomposition."""
    n, m, p = 2, 2, 7
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, idx] = 1
                V[j*p + k, idx] = 1
                W[k*n + i, idx] = 1
                idx += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    # Simple verification
    import random
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    # Random matrices
    A = np.random.rand(2, 2)
    B = np.random.rand(2, 7)
    C_expected = A @ B
    
    # Reconstruct C from tensor
    # C[i,k] = sum_r U[i, r] * V[k, r] * W[2*k+i, r] ? No.
    # The tensor decomposition is T = sum U_r (v_r) (w_r).
    # For matrix mult, T_{i,j,k} is coeff of A_{i,j} B_{j,k} in C_{i,k}.
    # Here indices are flattened.
    # U is (n*m, rank) -> indices 0..3 correspond to A[0,0], A[0,1], A[1,0], A[1,1]
    # V is (m*p, rank) -> indices 0..13 correspond to B[0,0]..B[1,6]
    # W is (p*n, rank) -> indices 0..13 correspond to C[0,0]..C[1,6] (flattened 2*k+i)
    
    C_reconstructed = np.zeros((2, 7))
    for r in range(U.shape[1]):
        u_r = U[:, r]
        v_r = V[:, r]
        w_r = W[:, r]
        
        # u_r is vector of coeffs for A elements
        # v_r is vector of coeffs for B elements
        # w_r is vector of coeffs for C elements
        
        # Term for C[i,k] from this r is:
        # sum_{j} (A[i,j] coeff in u_r) * (B[j,k] coeff in v_r) * (C[i,k] coeff in w_r)
        # This logic is for checking validity of the tensor.
        # We want to check if the tensor computes A @ B.
        # Value = sum_r (sum_{i,j} U[i,j,r] A[i,j]) * (sum_{j,k} V[j,k,r] B[j,k]) * ...?
        # No, the decomposition is:
        # C[i,k] = sum_r U_{i,j,r} ... wait.
        # The standard form is C_{i,k} = sum_r \alpha_r (A, U_r) \beta_r (B, V_r) ...
        # Actually, the factor matrices are usually defined such that:
        # C = \sum_r (A \cdot U_r) \otimes (V_r \cdot B) ... no.
        
        # Let's use the explicit expansion:
        # The tensor T is a 3-way array T[i, j, k] where i is A index, j is B index, k is C index.
        # The multiplication is C_k = \sum_{i,j} T[i,j,k] A_i B_j.
        # In our case, A has 4 entries, B has 14, C has 14.
        # U is 4 x R. V is 14 x R. W is 14 x R.
        # The term r contributes to C_k:
        # \sum_{i} \sum_{j} U[i, r] A[i] * V[j, r] B[j] * W[k, r] ?
        # No, W is a factor for C.
        # Usually, if we have factors U, V, W, it means:
        # \langle C, W_r \rangle = \langle A, U_r \rangle \langle B, V_r \rangle ?
        # This would imply C is a sum of rank-1 matrices scaled by scalars from A and B.
        # i.e., C = \sum_r (\langle A, U_r \rangle \langle B, V_r \rangle) M_r ?
        # Where M_r is the matrix represented by W_r?
        # If W_r is a vector, it defines a matrix by reshaping.
        
        # Let's test with this interpretation:
        # C = sum_r (U_r^T A_flat) (V_r^T B_flat) W_r_resized ?
        # This doesn't look right dimensionally for arbitrary U, V.
        
        # Standard Tensor Decomposition of Matrix Mult:
        # The tensor is T \in R^{4 \times 14 \times 14}.
        # T_{a, b, c} = 1 if C_c = A_a B_b, else 0.
        # We want T = \sum_r U_r \otimes V_r \otimes W_r.
        # This means T_{a,b,c} = \sum_r U[a,r] V[b,r] W[c,r].
        # Then C_c = \sum_{a,b} T_{a,b,c} A_a B_b = \sum_{a,b} \sum_r U[a,r] V[b,r] W[c,r] A_a B_b
        # = \sum_r W[c,r] (\sum_a U[a,r] A_a) (\sum_b V[b,r] B_b).
        
        # So, for each r, let S_A(r) = sum_a U[a,r] A[a] and S_B(r) = sum_b V[b,r] B[b].
        # Then C[c] = \sum_r W[c,r] S_A(r) S_B(r).
        
        # Let's verify this with the generated matrices.
        
    for r in range(U.shape[1]):
        SA = np.dot(U[:, r], A.flatten())
        SB = np.dot(V[:, r], B.flatten())
        for c in range(14):
            C_reconstructed.flat[c] += W[c, r] * SA * SB
            
    # Check indices
    # A.flatten() is [A00, A01, A10, A11]. Indices 0,1,2,3. Matches U.
    # B.flatten() is [B00..B06, B10..B16]. Indices 0..13. Matches V (7*j+k).
    # C.flatten() in our W indexing is 2*k+i.
    # C_reconstructed is 2x7. flat index is i*7+k (row major).
    # We need to map W index (2*k+i) to C[i,k].
    # Let's create a C_from_W array.
    C_W = np.zeros(14)
    for r in range(U.shape[1]):
        SA = np.dot(U[:, r], A.flatten())
        SB = np.dot(V[:, r], B.flatten())
        for idx in range(14):
            C_W[idx] += W[idx, r] * SA * SB
            
    # Map C_W to C_mat
    C_mat = np.zeros((2,7))
    for k in range(7):
        for i in range(2):
            C_mat[i, k] = C_W[2*k+i]
            
    if np.allclose(C_mat, C_expected):
        print("Verification PASSED")
    else:
        print("Verification FAILED")
        print(C_mat)
        print(C_expected)
