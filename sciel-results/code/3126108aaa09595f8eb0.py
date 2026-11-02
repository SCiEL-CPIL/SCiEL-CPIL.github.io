"""Matrix multiplication 2x2x15: rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np

_t0 = time.time()
_DEADLINE = _t0 + 130.0

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15
    
    # Rank 53 decomposition:
    # 7 Strassen blocks (rank 7 each) for 14 columns
    # 1 trivial block (rank 4) for the last column
    # Total rank = 7*7 + 4 = 53
    
    rank = 53
    
    # U: (n*m, rank) = (4, 53)
    # V: (m*p, rank) = (30, 53)
    # W: (p*n, rank) = (30, 53)
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen 2x2x2 factors
    # Indices for U: 0=A00, 1=A01, 2=A10, 3=A11
    # Indices for V: j*p + k
    # Indices for W: k*n + i
    
    # Standard Strassen decomposition for 2x2x2:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    #
    # Reconstruction:
    # C00 = P1 - P3 - P4 - P6 + P7
    # C01 = P5 + P6 - P7
    # C10 = P2 + P3 - P5 + P7
    # C11 = P1 - P2 + P4 + P5 - P7
    
    u_entries = [
        [(0, 1), (3, 1)],   # P1: A00+A11
        [(2, 1), (3, 1)],   # P2: A10+A11
        [(0, 1)],           # P3: A00
        [(3, 1)],           # P4: A11
        [(0, 1), (1, 1)],   # P5: A00+A01
        [(2, 1), (0, -1)],  # P6: A10-A00
        [(1, 1), (3, -1)],  # P7: A01-A11
    ]
    v_entries = [
        [(('B0k0', 1)), (('B1k1', 1))],  # P1: B0k0+B1k1
        [(('B0k0', 1))],                  # P2: B0k0
        [(('B0k1', 1)), (('B1k1', -1))],  # P3: B0k1-B1k1
        [(('B1k0', 1)), (('B0k0', -1))],  # P4: B1k0-B0k0
        [(('B1k1', 1))],                  # P5: B1k1
        [(('B0k0', 1)), (('B0k1', 1))],   # P6: B0k0+B0k1
        [(('B1k0', 1)), (('B1k1', 1))],   # P7: B1k0+B1k1
    ]
    w_entries = [
        [(('C0k0', 1)), (('C1k1', 1))],            # P1: C00+1, C11+1
        [(('C1k0', 1)), (('C1k1', -1))],           # P2: C10+1, C11-1
        [(('C0k0', -1)), (('C1k0', 1))],           # P3: C00-1, C10+1
        [(('C0k0', -1)), (('C1k1', 1))],           # P4: C00-1, C11+1
        [(('C0k1', 1)), (('C1k0', -1)), (('C1k1', 1))],  # P5: C01+1, C10-1, C11+1
        [(('C0k0', -1)), (('C0k1', 1))],           # P6: C00-1, C01+1
        [(('C0k0', 1)), (('C0k1', -1)), (('C1k0', 1))],  # P7: C00+1, C01-1, C10+1
    ]
    
    def get_v_row(code, k0, k1):
        # code is like 'B0k0'
        j = int(code[1])
        k = k0 if code.endswith('k0') else k1
        return j * p + k
        
    def get_w_row(code, k0, k1):
        # code is like 'C0k0'
        i = int(code[1])
        k = k0 if code.endswith('k0') else k1
        return k * n + i

    # Place 7 Strassen blocks for columns (0,1), (2,3), ..., (12,13)
    for blk in range(7):
        k0 = 2 * blk
        k1 = 2 * blk + 1
        base = 7 * blk
        for t in range(7):
            r = base + t
            # U
            for (u_row_i, c) in u_entries[t]:
                U[u_row_i, r] += c
            # V
            for (code, c) in v_entries[t]:
                v_row_idx = get_v_row(code, k0, k1)
                V[v_row_idx, r] += c
            # W
            for (code, c) in w_entries[t]:
                w_row_idx = get_w_row(code, k0, k1)
                W[w_row_idx, r] += c
                
    # Place the residual 2x2x1 block for column 14 (rank 4, trivial)
    # C[0,14] = A[0,0]B[0,14] + A[0,1]B[1,14]
    # C[1,14] = A[1,0]B[0,14] + A[1,1]B[1,14]
    b14 = 49
    # Term 1: A00 * B0,14 -> C0,14
    U[0, b14 + 0] = 1
    V[0 * p + 14, b14 + 0] = 1
    W[14 * n + 0, b14 + 0] = 1
    
    # Term 2: A01 * B1,14 -> C0,14
    U[1, b14 + 1] = 1
    V[1 * p + 14, b14 + 1] = 1
    W[14 * n + 0, b14 + 1] = 1
    
    # Term 3: A10 * B0,14 -> C1,14
    U[2, b14 + 2] = 1
    V[0 * p + 14, b14 + 2] = 1
    W[14 * n + 1, b14 + 2] = 1
    
    # Term 4: A11 * B1,14 -> C1,14
    U[3, b14 + 3] = 1
    V[1 * p + 14, b14 + 3] = 1
    W[14 * n + 1, b14 + 3] = 1

    # Verify the decomposition is correct
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # For each rank component r, we have:
    # C = sum_r U[:,r] * (V[:,r]^T * A) ... actually the tensor decomposition is:
    # T_{i,j,k} = sum_r U_{i,j,r} * V_{j,k,r} * W_{k,i,r}
    # Wait, the standard matrix multiplication tensor is:
    # T_{i,j,k} = sum_l A_{i,l} * B_{l,k}
    # The decomposition is: sum_r U_{i,l,r} * V_{l,k,r} * W_{k,i,r}
    
    # Let's verify by checking that for random A, B:
    # (A*B)_{i,k} = sum_l A_{i,l} * B_{l,k} = sum_r (sum_l U_{i,l,r} * A_{i,l}) * (sum_l V_{l,k,r} * B_{l,k}) * W_{k,i,r}
    
    # Actually, the standard formulation is:
    # C_{i,k} = sum_r M_r * W_{k,i,r}
    # where M_r = sum_{i,l} U_{i,l,r} * A_{i,l} * ... 
    
    # Let me use the direct verification:
    # The tensor M_{i,j,k} for matrix multiplication (A is n x m, B is m x p)
    # M_{i,j,k} = sum_l delta_{i+l, ?} ... 
    
    # Simpler: just check that U^T @ A @ B @ W^T gives the right answer
    # Actually: C = (U^T @ A) * (B @ W) element-wise? No.
    
    # The correct verification:
    # For each r, define:
    # U_r: n x m matrix (from U[:, r] reshaped)
    # V_r: m x p matrix (from V[:, r] reshaped)  
    # W_r: p x n matrix (from W[:, r] reshaped)
    # Then: C = sum_r (U_r * A) * (B * V_r) * W_r? 
    
    # Actually the standard tensor decomposition for matrix multiplication:
    # (A * B)_{i,k} = sum_r (sum_{j} U_{i,j,r} * A_{i,j}) * (sum_{j'} V_{j',k,r} * B_{j',k}) * W_{k,i,r}
    # No wait, that's not right either.
    
    # The correct form: The matrix multiplication tensor T has entries T_{i,j,k}
    # where i indexes rows of A (0..n-1), j indexes columns of A / rows of B (0..m-1),
    # k indexes columns of B (0..p-1).
    # T_{i,j,k} = 1 if j is a valid index, 0 otherwise (it's the "multiply" operation)
    
    # A rank-r decomposition: T_{i,j,k} = sum_{r=0}^{R-1} U_{i,j,r} * V_{j,k,r} * W_{k,i,r}
    
    # Wait, I need to be more careful. Let me just test numerically.
    
    # U has shape (n*m, rank) = (4, rank)
    # V has shape (m*p, rank) = (30, rank)
    # W has shape (p*n, rank) = (30, rank)
    
    # Reshape:
    # U_r[i, l] = U[i*m + l, r] for i in 0..n-1, l in 0..m-1
    # V_r[l, k] = V[l*p + k, r] for l in 0..m-1, k in 0..p-1
    # W_r[k, i] = W[k*n + i, r] for k in 0..p-1, i in 0..n-1
    
    # Then: C_{i,k} = sum_r (sum_l U_r[i,l] * A_{i,l}) * ... 
    # Hmm, this doesn't look right. Let me think again.
    
    # Actually, I think the correct interpretation is:
    # C_{i,k} = sum_r (sum_{l} U_{i,l,r} * A_{i,l}) * (sum_{l'} V_{l',k,r} * B_{l',k}) * W_{k,i,r}
    # But this has an extra product which doesn't make sense for linear operations.
    
    # Let me reconsider. The matrix multiplication is bilinear:
    # C_{i,k} = sum_l A_{i,l} * B_{l,k}
    
    # A rank-r decomposition means:
    # C_{i,k} = sum_r f_r(A) * g_r(B) * W_{k,i,r}
    # where f_r(A) = sum_{i',l'} U_{i',l',r} * A_{i',l'}
    # and g_r(B) = sum_{l'',k''} V_{l'',k'',r} * B_{l'',k''}
    
    # But wait, f_r and g_r are linear forms, so:
    # C_{i,k} = sum_r (sum_{i',l'} U_{i',l',r} * A_{i',l'}) * (sum_{l'',k''} V_{l'',k'',r} * B_{l'',k''}) * W_{k,i,r}
    
    # This should equal sum_l A_{i,l} * B_{l,k} for all i, k.
    
    # Let's verify numerically:
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        # f_r(A) = sum_{i',l'} U_{i',l',r} * A_{i',l'}
        f_r = 0.0
        for i_prime in range(n):
            for l_prime in range(m):
                f_r += U[i_prime * m + l_prime, r] * A_test[i_prime, l_prime]
        
        # g_r(B) = sum_{l'',k''} V_{l'',k'',r} * B_{l'',k''}
        g_r = 0.0
        for l_double in range(m):
            for k_double in range(p):
                g_r += V[l_double * p + k_double, r] * B_test[l_double, k_double]
        
        # C_{i,k} += f_r * g_r * W_{k,i,r}
        for k in range(p):
            for i in range(n):
                C_computed[i, k] += f_r * g_r * W[k * n + i, r]
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Decomposition is wrong, fall back to schoolbook
        rank_schoolbook = n * m * p
        U_s = np.zeros((n * m, rank_schoolbook), dtype=int)
        V_s = np.zeros((m * p, rank_schoolbook), dtype=int)
        W_s = np.zeros((p * n, rank_schoolbook), dtype=int)
        
        idx = 0
        for i in range(n):
            for l in range(m):
                for k in range(p):
                    U_s[i * m + l, idx] = 1
                    V_s[l * p + k, idx] = 1
                    W_s[k * n + i, idx] = 1
                    idx += 1
        
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    # Verify the decomposition is correct
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        f_r = 0.0
        for i_prime in range(n):
            for l_prime in range(m):
                f_r += U[i_prime * m + l_prime, r] * A_test[i_prime, l_prime]
        
        g_r = 0.0
        for l_double in range(m):
            for k_double in range(p):
                g_r += V[l_double * p + k_double, r] * B_test[l_double, k_double]
        
        for k in range(p):
            for i in range(n):
                C_computed[i, k] += f_r * g_r * W[k * n + i, r]
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fall back to schoolbook
        rank_schoolbook = n * m * p
        U_s = np.zeros((n * m, rank_schoolbook), dtype=int)
        V_s = np.zeros((m * p, rank_schoolbook), dtype=int)
        W_s = np.zeros((p * n, rank_schoolbook), dtype=int)
        
        idx = 0
        for i in range(n):
            for l in range(m):
                for k in range(p):
                    U_s[i * m + l, idx] = 1
                    V_s[l * p + k, idx] = 1
                    W_s[k * n + i, idx] = 1
                    idx += 1
        
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
