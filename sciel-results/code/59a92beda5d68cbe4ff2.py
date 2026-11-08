import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS  # 2, 3, 4

    # Index mapping:
    # U[i] where i = row*m + col_a, row in [0,2), col_a in [0,3)  -> 6 entries
    # V[j] where j = col_a*p + col_b, col_a in [0,3), col_b in [0,4)  -> 12 entries
    # W[k] where k = col_b*n + row, col_b in [0,4), row in [0,2)  -> 8 entries (W[k*n+i] convention)

    def u_idx(r, c): return r * m + c
    def v_idx(c, d): return c * p + d
    def w_idx(r, d): return d * n + r

    # Build rank-22 decomposition using Strassen-like blocks
    rank = 22
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    def add_term(u_entries, v_entries, w_entries, col):
        for idx, val in u_entries:
            U[idx, col] += val
        for idx, val in v_entries:
            V[idx, col] += val
        for idx, val in w_entries:
            W[idx, col] += val

    # Block 1: Handle intermediate columns 0 and 1 (Strassen-like for 2x2 * 2x4 structure)
    # A rows 0,1; A cols 0,1; B rows 0,1; B cols 0,1,2,3
    # Using 7 Strassen multiplications for the 2x2 * 2x2 core, extended to 4 output cols

    # Actually, let me use a more systematic approach.
    # For each pair of intermediate columns, we can use a rank-7 Strassen decomposition
    # for the 2x2 * 2x2 subproblem, but we have 4 output columns per intermediate column.
    
    # Let me use the known rank-22 construction:
    # Split B into B1 (rows 0,1) and B2 (row 2)
    # A1 = A[:,0:2] (2x2), A2 = A[:,2] (2x1)
    # B1 = B[0:2,:] (2x4), B2 = B[2,:] (1x4)
    
    # A*B = A1*B1 + A2*B2
    # A1*B1: 2x2 * 2x4. We can decompose this using 7 Strassen terms for the 2x2*2x2 part
    # and then extend to 4 columns. Actually, 2x2 * 2x4 can be done in 7*2 = 14? No.
    # 2x2 * 2x4: think of B1 as [B1a B1b] where B1a is 2x2, B1b is 2x2.
    # Then A1*B1 = A1*B1a + A1*B1b. Each 2x2*2x2 takes 7 Strassen terms. Total 14.
    # A2*B2: 2x1 * 1x4 = 4 rank-1 terms. Total 14+4 = 18? That's too optimistic.
    
    # Actually, the standard rank-22 construction for M_{2,3,4} is:
    # Use 7 terms for 2x2*2x2 (first two intermediate cols, first two output cols)
    # Use 7 terms for 2x2*2x2 (first two intermediate cols, last two output cols)
    # Use 4 terms for 2x1*1x4 (third intermediate col, all output cols)
    # But 7+7+4 = 18, which is below the known lower bound of 18... 
    # Wait, the lower bound for M_{2,3,4} is 18, and the best known upper bound is 22.
    # The 18 construction doesn't work because the blocks don't align properly.
    
    # Let me just use a verified rank-22 construction.
    
    # Terms 0-6: Strassen for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # M1 = (A00+A11)(B00+B11)
    add_term([(u_idx(0,0),1), (u_idx(1,1),1)], 
             [(v_idx(0,0),1), (v_idx(1,1),1)], 
             [(w_idx(0,0),1), (w_idx(1,1),1)], 0)
    # M2 = (A10+A11)B00
    add_term([(u_idx(1,0),1), (u_idx(1,1),1)], 
             [(v_idx(0,0),1)], 
             [(w_idx(0,0),1), (w_idx(1,0),-1)], 1)
    # M3 = A00(B10-B00)
    add_term([(u_idx(0,0),1)], 
             [(v_idx(1,0),1), (v_idx(0,0),-1)], 
             [(w_idx(0,1),1), (w_idx(1,1),1)], 2)
    # M4 = (A00+A10)B11
    add_term([(u_idx(0,0),1), (u_idx(1,0),1)], 
             [(v_idx(1,1),1)], 
             [(w_idx(0,1),1), (w_idx(1,1),-1)], 3)
    # M5 = (A11-A00)(B00+B10)
    add_term([(u_idx(1,1),1), (u_idx(0,0),-1)], 
             [(v_idx(0,0),1), (v_idx(1,0),1)], 
             [(w_idx(1,0),1), (w_idx(1,1),1)], 4)
    # M6 = (A01-A11)(B10+B11)
    add_term([(u_idx(0,1),1), (u_idx(1,1),-1)], 
             [(v_idx(1,0),1), (v_idx(1,1),1)], 
             [(w_idx(0,0),1), (w_idx(0,1),1)], 5)
    # M7 = (A00+A11)(B11-B01)
    add_term([(u_idx(0,0),1), (u_idx(1,1),1)], 
             [(v_idx(1,1),1), (v_idx(1,0),-1)], 
             [(w_idx(0,0),1), (w_idx(1,1),-1)], 6)
    
    # Terms 7-13: Strassen for A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    # M1' = (A00+A11)(B02+B13)
    add_term([(u_idx(0,0),1), (u_idx(1,1),1)], 
             [(v_idx(0,2),1), (v_idx(1,3),1)], 
             [(w_idx(0,2),1), (w_idx(1,3),1)], 7)
    # M2' = (A10+A11)B02
    add_term([(u_idx(1,0),1), (u_idx(1,1),1)], 
             [(v_idx(0,2),1)], 
             [(w_idx(0,2),1), (w_idx(1,2),-1)], 8)
    # M3' = A00(B12-B02)
    add_term([(u_idx(0,0),1)], 
             [(v_idx(1,2),1), (v_idx(0,2),-1)], 
             [(w_idx(0,3),1), (w_idx(1,3),1)], 9)
    # M4' = (A00+A10)B13
    add_term([(u_idx(0,0),1), (u_idx(1,0),1)], 
             [(v_idx(1,3),1)], 
             [(w_idx(0,3),1), (w_idx(1,3),-1)], 10)
    # M5' = (A11-A00)(B02+B12)
    add_term([(u_idx(1,1),1), (u_idx(0,0),-1)], 
             [(v_idx(0,2),1), (v_idx(1,2),1)], 
             [(w_idx(1,2),1), (w_idx(1,3),1)], 11)
    # M6' = (A01-A11)(B12+B13)
    add_term([(u_idx(0,1),1), (u_idx(1,1),-1)], 
             [(v_idx(1,2),1), (v_idx(1,3),1)], 
             [(w_idx(0,2),1), (w_idx(0,3),1)], 12)
    # M7' = (A00+A11)(B13-B03)
    add_term([(u_idx(0,0),1), (u_idx(1,1),1)], 
             [(v_idx(1,3),1), (v_idx(0,3),-1)], 
             [(w_idx(0,2),1), (w_idx(1,3),-1)], 13)
    
    # Terms 14-21: Handle A[0:2,2] * B[2,0:4] -> C[0:2,0:4]
    # A2 is 2x1, B2 is 1x4. This is 4 rank-1 terms.
    # But we need 8 terms (14-21). Let me check.
    # Actually, A[:,2] * B[2,:] = (A02*A_row0 + A12*A_row1) * B_row2
    # = A02*B[2,0:4] + A12*B[2,0:4]
    # That's 2*4 = 8 rank-1 terms if we don't share.
    # But we can share: for each output col d, we have A02*B[2,d] for row 0 and A12*B[2,d] for row 1.
    # These are separate rank-1 terms.
    
    # Term 14: A[0,2] * B[2,0] -> C[0,0]
    add_term([(u_idx(0,2),1)], [(v_idx(2,0),1)], [(w_idx(0,0),1)], 14)
    # Term 15: A[0,2] * B[2,1] -> C[0,1]
    add_term([(u_idx(0,2),1)], [(v_idx(2,1),1)], [(w_idx(0,1),1)], 15)
    # Term 16: A[0,2] * B[2,2] -> C[0,2]
    add_term([(u_idx(0,2),1)], [(v_idx(2,2),1)], [(w_idx(0,2),1)], 16)
    # Term 17: A[0,2] * B[2,3] -> C[0,3]
    add_term([(u_idx(0,2),1)], [(v_idx(2,3),1)], [(w_idx(0,3),1)], 17)
    # Term 18: A[1,2] * B[2,0] -> C[1,0]
    add_term([(u_idx(1,2),1)], [(v_idx(2,0),1)], [(w_idx(1,0),1)], 18)
    # Term 19: A[1,2] * B[2,1] -> C[1,1]
    add_term([(u_idx(1,2),1)], [(v_idx(2,1),1)], [(w_idx(1,1),1)], 19)
    # Term 20: A[1,2] * B[2,2] -> C[1,2]
    add_term([(u_idx(1,2),1)], [(v_idx(2,2),1)], [(w_idx(1,2),1)], 20)
    # Term 21: A[1,2] * B[2,3] -> C[1,3]
    add_term([(u_idx(1,2),1)], [(v_idx(2,3),1)], [(w_idx(1,3),1)], 21)

    # Verify
    if _verify(U, V, W, n, m, p):
        return _to_int_lists(U, V, W, rank)

    # Fallback: rank 24 (trivial)
    if time.time() < deadline - 10:
        rank = 24
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        idx = 0
        for r in range(n):
            for c in range(m):
                for d in range(p):
                    U[u_idx(r,c), idx] = 1
                    V[v_idx(c,d), idx] = 1
                    W[w_idx(r,d), idx] = 1
                    idx += 1
        
        return _to_int_lists(U, V, W, rank)
    
    # If we get here, something went wrong. Return the trivial decomposition.
    rank = 24
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    idx = 0
    for r in range(n):
        for c in range(m):
            for d in range(p):
                U[u_idx(r,c), idx] = 1
                V[v_idx(c,d), idx] = 1
                W[w_idx(r,d), idx] = 1
                idx += 1
    
    return _to_int_lists(U, V, W, rank)


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = U.shape[1]
    for i in range(n * m):
        for j in range(m * p):
            for k in range(p * n):
                # Determine if this should be 1
                # i = r_i * m + c_i
                r_i, c_i = divmod(i, m)
                # j = c_j * p + d_j
                c_j, d_j = divmod(j, p)
                # k = d_k * n + r_k  (based on W[k*n+i] convention)
                d_k, r_k = divmod(k, n)
                expected = 1.0 if (c_i == c_j and r_i == r_k and d_j == d_k) else 0.0
                val = sum(U[i, r] * V[j, r] * W[k, r] for r in range(rank))
                if abs(val - expected) > 1e-9:
                    return False
    return True


def _to_int_lists(U, V, W, rank):
    """Convert numpy arrays to lists with integer/half-integer values."""
    U_list = [[0] * rank for _ in range(U.shape[0])]
    V_list = [[0] * rank for _ in range(V.shape[0])]
    W_list = [[0] * rank for _ in range(W.shape[0])]
    
    for i in range(U.shape[0]):
        for r in range(rank):
            val = U[i, r]
            val_rounded = round(val * 2) / 2
            U_list[i][r] = val_rounded
    
    for i in range(V.shape[0]):
        for r in range(rank):
            val = V[i, r]
            val_rounded = round(val * 2) / 2
            V_list[i][r] = val_rounded
    
    for i in range(W.shape[0]):
        for r in range(rank):
            val = W[i, r]
            val_rounded = round(val * 2) / 2
            W_list[i][r] = val_rounded
    
    # Verify half-integer constraint
    for mat in [U_list, V_list, W_list]:
        for row in mat:
            for val in row:
                if not (val == int(val) or abs(val * 2 - int(val * 2)) < 1e-9):
                    raise ValueError("Non-half-integer entry")
    
    return {"U": U_list, "V": V_list, "W": W_list}
