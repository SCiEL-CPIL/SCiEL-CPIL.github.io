"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = 2, 5, 6
    
    # Strategy: Split the 5 columns of A (and rows of B) into 2+2+1.
    # - First 2x2x6 block: 3 Strassen 2x2x2 = 21 terms
    # - Second 2x2x6 block: 3 Strassen 2x2x2 = 21 terms  
    # - 2x1x6 block: schoolbook = 12 terms
    # Total rank = 54
    
    # Strassen 2x2x2 tensor decomposition.
    # For a 2x2x2 multiplication: C = A*B
    # A[i,j] indexed by U[i*2+j], B[j,k] indexed by V[j*2+k], C[k,i] indexed by W[k*2+i]
    # T[i,j,k] = sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] = 1 for all i,j,k in {0,1}
    
    # The 7 Strassen terms as (U_entries, V_entries, W_entries):
    # Each entry is a dict mapping index -> coefficient
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11) -> contributes to C00, C11
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
        # M2 = (A10+A11)B00 -> contributes to C10
        ({2:1, 3:1}, {0:1}, {2:1}),
        # M3 = A00(B10-B11) -> contributes to C01
        ({0:1}, {2:1, 3:-1}, {1:1}),
        # M4 = A11(B01-B00) -> contributes to C10, C11
        ({3:1}, {1:1, 0:-1}, {2:1, 3:1}),
        # M5 = (A00+A01)B11 -> contributes to C00, C01
        ({0:1, 1:1}, {3:1}, {0:1, 1:1}),
        # M6 = (A10-A00)(B00+B10) -> contributes to C11
        ({2:1, 0:-1}, {0:1, 2:1}, {3:1}),
        # M7 = (A01-A11)(B01+B11) -> contributes to C00
        ({1:1, 3:-1}, {1:1, 3:1}, {0:1}),
    ]
    
    # Verify Strassen: check T[i,j,k] = 1 for all i,j,k
    def verify_strassen():
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    total = 0
                    for u_ent, v_ent, w_ent in strassen_terms:
                        total += u_ent.get(i*2+j, 0) * v_ent.get(j*2+k, 0) * w_ent.get(k*2+i, 0)
                    if total != 1:
                        return False
        return True
    
    if not verify_strassen():
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Build the full 54-rank decomposition
    # Block 1: j in {0,1}, k in {0,1,2} -> 2x2x3, split into two 2x2x2 Strassen blocks
    # Block 2: j in {2,3}, k in {3,4,5} -> 2x2x3, split into two 2x2x2 Strassen blocks
    # Block 3: j in {4}, k in {0..5} -> 2x1x6 schoolbook
    
    # Wait, I need to think about this more carefully.
    # The 5 dimension is the middle dimension (j). We split j into {0,1}, {2,3}, {4}.
    # For j in {0,1}: we have a 2x2x6 sub-problem. Split k into {0,1}, {2,3}, {4,5} -> three 2x2x2 Strassen.
    # For j in {2,3}: same, another 2x2x6 -> three 2x2x2 Strassen.
    # For j in {4}: 2x1x6 -> schoolbook, 12 terms.
    
    # Total: 21 + 21 + 12 = 54
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]  # (10, 54)
    V = [[0] * rank for _ in range(m * p)]  # (30, 54)
    W = [[0] * rank for _ in range(p * n)]  # (12, 54)
    
    col = 0
    
    # Helper: add a Strassen 2x2x2 block for j-offset j0 and k-offset k0
    def add_strassen_block(j0, k0):
        nonlocal col
        for u_ent, v_ent, w_ent in strassen_terms:
            # U: index i*m + j, where i in {0,1}, j = j0 + local_j
            # local_j from u_ent key: key = local_i * 2 + local_j
            for u_idx, u_val in u_ent.items():
                local_i = u_idx // 2
                local_j = u_idx % 2
                global_u = local_i * m + (j0 + local_j)
                U[global_u][col] += u_val
            
            # V: index j*p + k, where j = j0 + local_j, k = k0 + local_k
            for v_idx, v_val in v_ent.items():
                local_j = v_idx // 2
                local_k = v_idx % 2
                global_v = (j0 + local_j) * p + (k0 + local_k)
                V[global_v][col] += v_val
            
            # W: index k*n + i, where k = k0 + local_k, i = local_i
            for w_idx, w_val in w_ent.items():
                local_k = w_idx // 2
                local_i = w_idx % 2
                global_w = (k0 + local_k) * n + local_i
                W[global_w][col] += w_val
            
            col += 1
    
    # Block 1: j in {0,1}, three Strassen blocks for k in {0,1}, {2,3}, {4,5}
    add_strassen_block(0, 0)  # j0=0, k0=0
    add_strassen_block(0, 2)  # j0=0, k0=2
    add_strassen_block(0, 4)  # j0=0, k0=4
    
    # Block 2: j in {2,3}, three Strassen blocks for k in {0,1}, {2,3}, {4,5}
    add_strassen_block(2, 0)  # j0=2, k0=0
    add_strassen_block(2, 2)  # j0=2, k0=2
    add_strassen_block(2, 4)  # j0=2, k0=4
    
    # Block 3: j=4, schoolbook for all k in {0..5}
    for k in range(p):
        for i in range(n):
            U[i * m + 4][col] = 1
            V[4 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    assert col == rank, f"Column count {col} != rank {rank}"
    
    # Verify the full decomposition
    def verify_full():
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if total != 1:
                        return (i, j, k, total)
        return None
    
    result = verify_full()
    if result is not None:
        # Fallback to schoolbook
        rank2 = n * m * p
        U2 = [[0] * rank2 for _ in range(n * m)]
        V2 = [[0] * rank2 for _ in range(m * p)]
        W2 = [[0] * rank2 for _ in range(p * n)]
        column2 = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U2[i * m + j][column2] = 1
                    V2[j * p + k][column2] = 1
                    W2[k * n + i][column2] = 1
                    column2 += 1
        return {"U": U2, "V": V2, "W": W2}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
