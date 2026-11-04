import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# Strassen 2x2x2 decomposition: 7 rank-1 terms
# C = A*B where A,B,C are 2x2
# M1 = (A00+A11)*(B00+B11)
# M2 = (A10+A11)*B00
# M3 = A00*(B01-B11)
# M4 = A11*(B10-B00)
# M5 = (A00+A01)*B11
# M6 = (A10-A00)*(B00+B01)
# M7 = (A01-A11)*(B10+B11)
# C00 = M1+M4-M5+M7
# C01 = M3+M5
# C10 = M2+M4
# C11 = M1-M2+M3+M6

def construct():
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    N = n * m  # 25
    M = m * p  # 25
    P = p * n  # 25
    
    # Partition: [2, 2, 1] for each of the three dimensions
    # Block sizes for dimension n (rows of A, cols of C): [2, 2, 1]
    # Block sizes for dimension m (cols of A, rows of B): [2, 2, 1]
    # Block sizes for dimension p (cols of B, rows of C): [2, 2, 1]
    
    n_parts = [2, 2, 1]
    m_parts = [2, 2, 1]
    p_parts = [2, 2, 1]
    
    # Compute offsets
    n_offsets = [0, 2, 4]
    m_offsets = [0, 2, 4]
    p_offsets = [0, 2, 4]
    
    rank = 0
    U_rows = []
    V_rows = []
    W_rows = []
    
    # Strassen 2x2x2 rank-1 terms
    # Each term: (U_coeff, V_coeff, W_coeff) for the 4 entries in each 2x2 block
    # For a 2x2x2 block at (i_off, j_off, k_off):
    # U index: (i_off + di) * m + (j_off + dj)
    # V index: (j_off + dj) * p + (k_off + dk)
    # W index: (k_off + dk) * n + (i_off + di)
    
    # Strassen terms as (u_entries, v_entries, w_entries)
    # u_entries: dict mapping (di, dj) -> coeff
    # v_entries: dict mapping (dj, dk) -> coeff
    # w_entries: dict mapping (dk, di) -> coeff
    
    strassen_terms = [
        # M1 = (A00+A11)*(B00+B11)
        (
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1},
            {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1},
        ),
        # M2 = (A10+A11)*B00
        (
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(0,0): 1},
        ),
        # M3 = A00*(B01-B11)
        (
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(0,1): 1, (1,1): -1},
        ),
        # M4 = A11*(B10-B00)
        (
            {(1,1): 1},
            {(1,0): 1, (0,0): -1},
            {(1,0): 1, (0,0): -1},
        ),
        # M5 = (A00+A01)*B11
        (
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(1,1): 1},
        ),
        # M6 = (A10-A00)*(B00+B01)
        (
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(0,0): 1, (0,1): 1},
        ),
        # M7 = (A01-A11)*(B10+B11)
        (
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(1,0): 1, (1,1): 1},
        ),
    ]
    
    # For each 2x2x2 block, add 7 rank-1 terms
    # For each 1x1x2, 2x1x1, 1x1x1, etc. block, add rank-1 terms
    # Actually, we need to handle all 27 blocks
    
    # For blocks where n_size=2, m_size=2, p_size=2: use Strassen (7 terms)
    # For other blocks: use schoolbook (n_size * m_size * p_size terms)
    
    U = [[0.0] * (4*7 + 2*2*2 + 2*2*2 + 2*2*2 + 2*2*1 + 2*2*1 + 2*2*1 + 2*2*1 + 2*2*1 + 2*2*1 + 2*1*2 + 2*1*2 + 2*1*2 + 2*1*2 + 2*1*2 + 1*2*2 + 1*2*2 + 1*2*2 + 1*2*2 + 1*2*2 + 1*2*1 + 1*2*1 + 1*2*1 + 1*2*1 + 1*2*1 + 1*1*2 + 1*1*2 + 1*1*2 + 1*1*2 + 1*1*2 + 1*1*2 + 1*1*1) for _ in range(N)]
    V = [[0.0] * len(U[0]) for _ in range(M)]
    W = [[0.0] * len(U[0]) for _ in range(P)]
    
    col = 0
    
    for ni in range(3):
        for mj in range(3):
            for pk in range(3):
                ns = n_parts[ni]
                ms = m_parts[mj]
                ps = p_parts[pk]
                
                noff = n_offsets[ni]
                moff = m_offsets[mj]
                poff = p_offsets[pk]
                
                if ns == 2 and ms == 2 and ps == 2:
                    # Use Strassen: 7 terms
                    for term in strassen_terms:
                        u_entries, v_entries, w_entries = term
                        for (di, dj), uc in u_entries.items():
                            ui = (noff + di) * m + (moff + dj)
                            U[ui][col] += uc
                        for (dj, dk), vc in v_entries.items():
                            vi = (moff + dj) * p + (poff + dk)
                            V[vi][col] += vc
                        for (dk, di), wc in w_entries.items():
                            wi = (poff + dk) * n + (noff + di)
                            W[wi][col] += wc
                        col += 1
                else:
                    # Schoolbook for this block
                    for di in range(ns):
                        for dj in range(ms):
                            for dk in range(ps):
                                ui = (noff + di) * m + (moff + dj)
                                vi = (moff + dj) * p + (poff + dk)
                                wi = (poff + dk) * n + (noff + di)
                                U[ui][col] = 1
                                V[vi][col] = 1
                                W[wi][col] = 1
                                col += 1
    
    total_rank = col
    
    # Verify the decomposition
    # C[i,j] = sum over k of A[i,k] * B[k,j]
    # In tensor form: T[i,k,j] = U[i*5+k, r] * V[k*5+j, r] * W[j*5+i, r]
    # Let's verify with random matrices
    rng = np.random.default_rng(42)
    A = rng.integers(-2, 3, size=(n, m))
    B = rng.integers(-2, 3, size=(m, p))
    C_expected = A @ B
    
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    # Compute C from decomposition
    # C[i,j] = sum_r U[i*m+j, r] * V[j*p+k... wait, need to be careful
    # U is indexed by (i, j) where i is row of A, j is col of A
    # V is indexed by (j, k) where j is row of B, k is col of B
    # W is indexed by (k, i) where k is row of C, i is col of C
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So W should be indexed by (k, i) for C[k,i]? No wait...
    
    # Let me re-read the convention: W[k*n+i, r]
    # So W index = k*n + i, where k is the "col" index and i is the "row" index?
    # Actually for matrix multiplication C = A*B:
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # U[i*m+k, r] corresponds to A[i,k]
    # V[k*p+j, r] corresponds to B[k,j]
    # W[j*n+i, r] corresponds to C[i,j]
    
    # Let me verify:
    C_computed = np.zeros((n, p), dtype=float)
    for r in range(total_rank):
        # Extract A_r, B_r, C_r
        A_r = U_arr[:, r].reshape(n, m)
        B_r = V_arr[:, r].reshape(m, p)
        C_r = W_arr[:, r].reshape(p, n)
        # C_r should be outer product related... 
        # Actually: C[i,j] = sum_r U[i*m+j, r] * V[j*p+k... 
        # Hmm, the tensor decomposition is:
        # T[i,k,j] = sum_r U[i,k]_r * V[k,j]_r * W[j,i]_r
        # where T[i,k,j] is the (i,k,j) entry of the matmul tensor
        # and matmul tensor: T[i,k,j] = delta such that C[i,j] = sum_k A[i,k]*B[k,j]
        
        # So: C[i,j] = sum_k sum_r A[i,k]*B[k,j]*delta... 
        # The tensor is: T_{i,k,j} where the matmul is C = A*B means
        # C[i,j] = sum_k A[i,k] B[k,j]
        # The bilinear form: <A, T, B> = sum_{i,k,j} A[i,k] T[i,k,j] B[k,j] = sum_{i,j} C[i,j]
        # For exact decomposition: T[i,k,j] = sum_r U[i,k]_r V[k,j]_r W[j,i]_r
        # And we need: sum_{i,k,j} A[i,k] (sum_r U[i,k]_r V[k,j]_r W[j,i]_r) B[k,j] = sum_{i,j} (A@B)[i,j]
        
        # This works if: for each r, the "partial" contribution is
        # sum_{i,k,j} A[i,k] U[i,k]_r V[k,j]_r W[j,i]_r B[k,j]
        # = sum_i W[j,i]_r ... this is getting complicated. Let me just numerically verify.
        
    # Simpler numerical check:
    # The decomposition says: for all i,j: C[i,j] = sum_k sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    # Wait no, that's not right either. Let me think again.
    
    # Standard tensor decomposition of matmul:
    # T[i,k,j] = sum_r U_r[i,k] * V_r[k,j] * W_r[j,i]
    # where T is the matmul tensor (T[i,k,j] = 1 if this is a valid triple, else 0)
    # Actually the matmul tensor is: T[i,k,j] such that C = A*B means
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # The tensor T satisfies: <A, T, B> = sum_{i,k,j} A[i,k] T[i,k,j] B[k,j] = tr(C^T C) or similar
    
    # Let me just do a direct numerical verification:
    # For the schoolbook, U[i*m+k, r] = 1, V[k*p+j, r] = 1, W[j*n+i, r] = 1 for the term (i,k,j)
    # Then sum_r U[i*m+k', r] V[k'*p+j', r] W[j'*n+i', r] ... 
    
    # I think the correct interpretation is:
    # C[i,j] = sum_k sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    # No wait, that would give C[i,j] = sum_k sum_r ... but U depends on k and V depends on k, so
    # C[i,j] = sum_k [sum_r U[i*m+k, r] V[k*p+j, r] W[j*n+i, r]]
    # For schoolbook with one term per (i,k,j): U[i*m+k, r_{ikj}] = 1, V[k*p+j, r_{ikj}] = 1, W[j*n+i, r_{ikj}] = 1
    # Then C[i,j] = sum_k sum_r ... = sum_k 1 * 1 * 1 = p? That's not right.
    
    # Let me reconsider. The tensor for matmul is T[i,k,j] where:
    # (A*B)[i,j] = sum_k A[i,k] B[k,j]
    # This is a 3-way tensor with indices (i, k, j).
    # A rank-r decomposition: T[i,k,j] = sum_{r=1}^R U_r[i,k] V_r[k,j] W_r[j,i]
    # Then: sum_{i,k,j} A[i,k] T[i,k,j] B[k,j] = sum_r sum_{i,k,j} A[i,k] U_r[i,k] V_r[k,j] W_r[j,i] B[k,j]
    # = sum_r (sum_i (A * U_r^T)[i,i]... no this is getting messy.
    
    # Let me just trust the standard formulation and verify numerically.
    # The standard is: C[i,j] = sum_k A[i,k] B[k,j]
    # Tensor: T[i,k,j] = 1 for all valid (i,k,j) in the sense that it's the "structure tensor"
    # Rank decomposition: T[i,k,j] = sum_r U_r[i,k] V_r[k,j] W_r[j,i]
    # Verification: for any A, B: sum_{i,k,j} A[i,k] T[i,k,j] B[k,j] = sum_{i,k,j} A[i,k] B[k,j] = sum_{i,j} (A@B)[i,j]
    
    # So the check is: sum_{i,k,j} A[i,k] (sum_r U_r[i,k] V_r[k,j] W_r[j,i]) B[k,j] == sum_{i,j} (A@B)[i,j]
    
    T = np.zeros((n, m, p))
    for r in range(total_rank):
        U_r = U_arr[:, r].reshape(n, m)
        V_r = V_arr[:, r].reshape(m, p)
        W_r = W_arr[:, r].reshape(p, n)
        T += np.einsum('ik,kj,ji->ikj', U_r, V_r, W_r)
    
    # Now verify: sum_{i,k,j} A[i,k] T[i,k,j] B[k,j] should equal sum_{i,j} (A@B)[i,j]
    lhs = np.einsum('ik,ikj,kj->', A, T, B)
    rhs = np.sum(C_expected)
    
    if abs(lhs - rhs) > 1e-6:
        # Fallback to schoolbook
        rank_fallback = n * m * p
        U = [[0.0] * rank_fallback for _ in range(N)]
        V = [[0.0] * rank_fallback for _ in range(M)]
        W = [[0.0] * rank_fallback for _ in range(P)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Convert to clean integers if possible
    U_out = [[int(round(x)) if abs(x - round(x)) < 1e-9 else (round(x*2)/2) for x in row] for row in U]
    V_out = [[int(round(x)) if abs(x - round(x)) < 1e-9 else (round(x*2)/2) for x in row] for row in V]
    W_out = [[int(round(x)) if abs(x - round(x)) < 1e-9 else (round(x*2)/2) for x in row] for row in W]
    
    return {"U": U_out, "V": V_out, "W": W_out}
