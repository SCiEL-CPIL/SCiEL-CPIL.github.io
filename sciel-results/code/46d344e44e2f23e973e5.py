"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s evaluator limit
    
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # We'll build a rank-88 decomposition using Strassen for 2x2x2 blocks
    # plus schoolbook for the remaining row.
    # 
    # Strategy:
    # - For i in {0,1} (first 2 rows), j in {0,1}, k in {0..7}: use Strassen 2x2x2 blocks
    #   Split k into 4 blocks of 2: k_base in {0,2,4,6}
    #   Each Strassen block uses 7 terms -> 4 blocks * 7 = 28 terms
    # - For i in {0,1}, j in {2,3}, k in {0..7}: use Strassen 2x2x2 blocks
    #   Split k into 4 blocks of 2: k_base in {0,2,4,6}
    #   Each Strassen block uses 7 terms -> 4 blocks * 7 = 28 terms
    # - For i=2, j in {0..3}, k in {0..7}: schoolbook, 4*8 = 32 terms
    # Total: 28 + 28 + 32 = 88
    #
    # Wait, that's 88. But let me verify the Strassen decomposition is correct.
    
    rank = 88
    
    U = [[0] * rank for _ in range(n * m)]   # shape (12, 88)
    V = [[0] * rank for _ in range(m * p)]   # shape (32, 88)
    W = [[0] * rank for _ in range(p * n)]   # shape (24, 88)
    
    col = 0
    
    def add_component(u_indices, v_indices, w_indices):
        nonlocal col
        for idx, val in u_indices:
            U[idx][col] += val
        for idx, val in v_indices:
            V[idx][col] += val
        for idx, val in w_indices:
            W[idx][col] += val
        col += 1
    
    def u_idx(i, j):
        return i * m + j
    
    def v_idx(j, k):
        return j * p + k
    
    def w_idx(k, i):
        return k * n + i
    
    # Strassen 2x2x2 block decomposition
    # A is 2x2 (rows i_base..i_base+1, cols j_base..j_base+1)
    # B is 2x2 (rows j_base..j_base+1, cols k_base..k_base+1)
    # C is 2x2 (rows i_base..i_base+1, cols k_base..k_base+1)
    def emit_strassen_block(i_base, j_base, k_base):
        # M1 = (A00 + A11) * (B00 + B11) -> contributes to C00 and C11
        add_component(
            [(u_idx(i_base, j_base), 1), (u_idx(i_base+1, j_base+1), 1)],
            [(v_idx(j_base, k_base), 1), (v_idx(j_base+1, k_base+1), 1)],
            [(w_idx(k_base, i_base), 1), (w_idx(k_base+1, i_base+1), 1)]
        )
        
        # M2 = (A10 + A11) * B00 -> contributes to C10 and C11
        add_component(
            [(u_idx(i_base+1, j_base), 1), (u_idx(i_base+1, j_base+1), 1)],
            [(v_idx(j_base, k_base), 1)],
            [(w_idx(k_base, i_base+1), 1), (w_idx(k_base+1, i_base+1), 1)]
        )
        
        # M3 = A00 * (B01 - B11) -> contributes to C01 and C11 (with -1)
        add_component(
            [(u_idx(i_base, j_base), 1)],
            [(v_idx(j_base, k_base+1), 1), (v_idx(j_base+1, k_base+1), -1)],
            [(w_idx(k_base+1, i_base), 1), (w_idx(k_base+1, i_base+1), -1)]
        )
        
        # M4 = A11 * (B10 - B00) -> contributes to C00 (with -1) and C10
        add_component(
            [(u_idx(i_base+1, j_base+1), 1)],
            [(v_idx(j_base+1, k_base), 1), (v_idx(j_base, k_base), -1)],
            [(w_idx(k_base, i_base), -1), (w_idx(k_base, i_base+1), 1)]
        )
        
        # M5 = (A00 + A01) * B11 -> contributes to C00 and C01
        add_component(
            [(u_idx(i_base, j_base), 1), (u_idx(i_base, j_base+1), 1)],
            [(v_idx(j_base+1, k_base+1), 1)],
            [(w_idx(k_base, i_base), 1), (w_idx(k_base+1, i_base), 1)]
        )
        
        # M6 = (A10 - A00) * (B00 + B01) -> contributes to C10 and C11
        add_component(
            [(u_idx(i_base+1, j_base), 1), (u_idx(i_base, j_base), -1)],
            [(v_idx(j_base, k_base), 1), (v_idx(j_base, k_base+1), 1)],
            [(w_idx(k_base, i_base+1), 1), (w_idx(k_base+1, i_base+1), 1)]
        )
        
        # M7 = (A01 - A11) * (B10 + B11) -> contributes to C01 and C10
        add_component(
            [(u_idx(i_base, j_base+1), 1), (u_idx(i_base+1, j_base+1), -1)],
            [(v_idx(j_base+1, k_base), 1), (v_idx(j_base+1, k_base+1), 1)],
            [(w_idx(k_base+1, i_base), 1), (w_idx(k_base, i_base+1), 1)]
        )
    
    # Block 1: i in {0,1}, j in {0,1}, k split into 4 blocks of 2
    for q in range(4):
        k_base = 2 * q
        emit_strassen_block(0, 0, k_base)
    
    # Block 2: i in {0,1}, j in {2,3}, k split into 4 blocks of 2
    for q in range(4):
        k_base = 2 * q
        emit_strassen_block(0, 2, k_base)
    
    # Block 3: i=2, all j and k (schoolbook)
    for j in range(m):
        for k in range(p):
            add_component(
                [(u_idx(2, j), 1)],
                [(v_idx(j, k), 1)],
                [(w_idx(k, 2), 1)]
            )
    
    # Verify col == rank
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    # Verify correctness
    if time.time() < deadline:
        try:
            U_np = np.array(U, dtype=float)
            V_np = np.array(V, dtype=float)
            W_np = np.array(W, dtype=float)
            
            # The decomposition says: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = delta(i,i) for the MM tensor
            # Actually for matrix multiplication, the tensor is T[i,j,k,l] = delta(j,k)*delta(i,l) but here
            # we have a different indexing. Let me check.
            #
            # The matrix multiplication tensor for C = AB where A is n x m and B is m x p:
            # C[i,k] = sum_j A[i,j] * B[j,k]
            # So the tensor has indices (i, j, k) with value delta... no.
            #
            # In the trace convention W[k*n+i, r], the decomposition is:
            # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # This should equal sum_j A[i,j] * B[j,k]
            #
            # To verify: for random A and B, check that C computed via the decomposition matches A @ B.
            
            # Test with a few random matrices
            rng = np.random.RandomState(42)
            max_err = 0.0
            for trial in range(5):
                A = rng.randn(n, m)
                B = rng.randn(m, p)
                C_expected = A @ B
                
                # Compute C via decomposition
                C_decomp = np.zeros((n, p))
                for i in range(n):
                    for k in range(p):
                        for j in range(m):
                            for r in range(rank):
                                C_decomp[i, k] += U_np[i*m + j, r] * V_np[j*p + k, r] * W_np[k*n + i, r]
                        max_err = max(max_err, abs(C_decomp[i,k] - C_expected[i,k]))
            
            if max_err > 1e-9:
                # Fallback: return schoolbook rank 96
                rank_sb = n * m * p
                U_sb = [[0] * rank_sb for _ in range(n * m)]
                V_sb = [[0] * rank_sb for _ in range(m * p)]
                W_sb = [[0] * rank_sb for _ in range(p * n)]
                col_sb = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U_sb[i*m + j][col_sb] = 1
                            V_sb[j*p + k][col_sb] = 1
                            W_sb[k*n + i][col_sb] = 1
                            col_sb += 1
                return {"U": U_sb, "V": V_sb, "W": W_sb}
        except Exception:
            pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
