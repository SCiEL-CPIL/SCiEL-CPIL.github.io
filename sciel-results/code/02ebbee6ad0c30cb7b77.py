"""Matrix multiplication 2,5,6 - rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]
START_TIME = time.time()
DEADLINE = START_TIME + 120

def _time_remaining():
    return DEADLINE - time.time()

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS  # n=2, m=5, p=6
    
    # Schoolbook decomposition: rank = n*m*p = 60
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school), dtype=int)
    V_school = np.zeros((m * p, rank_school), dtype=int)
    W_school = np.zeros((p * n, rank_school), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * m * p + j * p + k
                U_school[i*m+j, idx] = 1
                V_school[j*p+k, idx] = 1
                W_school[k*n+i, idx] = 1
    
    def verify(U, V, W):
        """Verify the decomposition is correct."""
        r = U.shape[1]
        U_np = U.astype(float)
        V_np = V.astype(float)
        W_np = W.astype(float)
        
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        U_reshaped = U_np.reshape(n, m, r)
        V_reshaped = V_np.reshape(m, p, r)
        W_reshaped = W_np.reshape(p, n, r)
        
        T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        C_computed = np.einsum('ijk,ij,jk->ik', T, A_test, B_test)
        
        return np.allclose(C_computed, C_expected, atol=1e-10)
    
    # Try Strassen-based decomposition: rank 54
    # Partition j (size 5) into pairs (0,1), (2,3) and single (4)
    # For each j-pair, partition k (size 6) into pairs (0,1), (2,3), (4,5)
    # Each 2x2x2 block uses Strassen (rank 7)
    # Each 2x1x2 block (for j=4) uses schoolbook (rank 4)
    
    rank_54 = 7 * 2 * 3 + 4 * 3  # 42 + 12 = 54
    U = np.zeros((n * m, rank_54), dtype=int)
    V = np.zeros((m * p, rank_54), dtype=int)
    W = np.zeros((p * n, rank_54), dtype=int)
    
    col = 0
    
    def add_strassen(i_base, j_base, k_base):
        """Add Strassen 7-term decomposition for 2x2x2 block starting at (i_base, j_base, k_base)."""
        nonlocal col
        # i indices: i_base, i_base+1 (n=2, so i_base=0 only)
        # j indices: j_base, j_base+1
        # k indices: k_base, k_base+1
        
        # M0: (A00+A11)(B00+B11) -> C00 +=, C11 +=
        U[i_base*m + j_base, col] += 1
        U[(i_base+1)*m + (j_base+1), col] += 1
        V[j_base*p + k_base, col] += 1
        V[(j_base+1)*p + (k_base+1), col] += 1
        W[k_base*n + i_base, col] += 1
        W[(k_base+1)*n + (i_base+1), col] += 1
        col += 1
        
        # M1: (A10+A11)B00 -> C10 +=, C11 -=
        U[(i_base+1)*m + j_base, col] += 1
        U[(i_base+1)*m + (j_base+1), col] += 1
        V[j_base*p + k_base, col] += 1
        W[k_base*n + (i_base+1), col] += 1
        W[(k_base+1)*n + (i_base+1), col] -= 1
        col += 1
        
        # M2: A00(B01-B11) -> C01 +=, C11 +=
        U[i_base*m + j_base, col] += 1
        V[j_base*p + (k_base+1), col] += 1
        V[(j_base+1)*p + (k_base+1), col] -= 1
        W[(k_base+1)*n + i_base, col] += 1
        W[(k_base+1)*n + (i_base+1), col] += 1
        col += 1
        
        # M3: A11(B10-B00) -> C00 +=, C10 +=
        U[(i_base+1)*m + (j_base+1), col] += 1
        V[(j_base+1)*p + k_base, col] += 1
        V[j_base*p + k_base, col] -= 1
        W[k_base*n + i_base, col] += 1
        W[k_base*n + (i_base+1), col] += 1
        col += 1
        
        # M4: (A00+A01)B11 -> C00 -=, C01 +=
        U[i_base*m + j_base, col] += 1
        U[i_base*m + (j_base+1), col] += 1
        V[(j_base+1)*p + (k_base+1), col] += 1
        W[k_base*n + i_base, col] -= 1
        W[(k_base+1)*n + i_base, col] += 1
        col += 1
        
        # M5: (A10-A00)(B00+B01) -> C11 +=
        U[(i_base+1)*m + j_base, col] += 1
        U[i_base*m + j_base, col] -= 1
        V[j_base*p + k_base, col] += 1
        V[j_base*p + (k_base+1), col] += 1
        W[(k_base+1)*n + (i_base+1), col] += 1
        col += 1
        
        # M6: (A01-A11)(B10+B11) -> C00 +=
        U[i_base*m + (j_base+1), col] += 1
        U[(i_base+1)*m + (j_base+1), col] -= 1
        V[(j_base+1)*p + k_base, col] += 1
        V[(j_base+1)*p + (k_base+1), col] += 1
        W[k_base*n + i_base, col] += 1
        col += 1
    
    def add_naive_2x1x2(i_base, j_idx, k_base):
        """Add schoolbook 4-term decomposition for 2x1x2 block: A[i_base:i_base+2, j_idx] * B[j_idx, k_base:k_base+2]."""
        nonlocal col
        # 4 scalar multiplications
        for di in range(2):
            for dk in range(2):
                U[(i_base+di)*m + j_idx, col] += 1
                V[j_idx*p + (k_base+dk), col] += 1
                W[(k_base+dk)*n + (i_base+di), col] += 1
                col += 1
    
    # Apply Strassen blocks for j-pairs (0,1) and (2,3)
    for j_base in [0, 2]:
        for k_base in [0, 2, 4]:
            add_strassen(0, j_base, k_base)
    
    # Apply naive blocks for j=4
    for k_base in [0, 2, 4]:
        add_naive_2x1x2(0, 4, k_base)
    
    # Verify
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

# EVOLVE-BLOCK-END
