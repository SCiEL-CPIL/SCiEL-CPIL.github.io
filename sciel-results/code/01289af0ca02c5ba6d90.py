import time
import numpy as np

def construct():
    n, m, p = 2, 4, 4
    nm, mp, pn = n * m, m * p, p * n  # 8, 16, 8

    start_time = time.time()
    deadline = start_time + 140.0

    # Build the multiplication tensor
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for l in range(m):
            for k in range(p):
                a = i * m + l
                b = l * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify_decomp(U_mat, V_mat, W_mat, rank_val):
        S = np.zeros((nm, mp, pn), dtype=float)
        for r in range(rank_val):
            S += np.einsum('a,b,c->abc', U_mat[:, r], V_mat[:, r], W_mat[:, r])
        return np.max(np.abs(S - T))

    # Greedy peeling with multiple restarts
    def greedy_peel(T_tensor, max_rank=20, n_restarts=5):
        best_result = None
        best_rank = max_rank + 1
        best_err = float('inf')
        
        for restart in range(n_restarts):
            if time.time() > deadline - 15:
                break
            R = T_tensor.copy()
            U_list, V_list, W_list = [], [], []
            
            # Random initialization
            np.random.seed(restart * 12345 + 67890)
            
            for step in range(max_rank):
                if time.time() > deadline - 15:
                    break
                if np.max(np.abs(R)) < 1e-12:
                    break
                
                # Find best rank-1 approximation
                # Initialize
                if step == 0:
                    u = R.sum(axis=(1,2))
                    v = R.sum(axis=(0,2))
                    w = R.sum(axis=(0,1))
                else:
                    u = np.random.randn(nm)
                    v = np.random.randn(mp)
                    w = np.random.randn(pn)
                
                u = u / (np.linalg.norm(u) + 1e-15)
                v = v / (np.linalg.norm(v) + 1e-15)
                w = w / (np.linalg.norm(w) + 1e-15)
                
                # Power iteration
                for it in range(200):
                    vw = np.outer(v, w)
                    u_new = np.einsum('abc,bc->a', R, vw)
                    u_norm = np.linalg.norm(u_new)
                    if u_norm < 1e-15:
                        break
                    u = u_new / u_norm
                    
                    uw = np.outer(u, w)
                    v_new = np.einsum('abc,ac->b', R, uw)
                    v_norm = np.linalg.norm(v_new)
                    if v_norm < 1e-15:
                        break
                    v = v_new / v_norm
                    
                    uv = np.outer(u, v)
                    w_new = np.einsum('abc,ab->c', R, uv)
                    w_norm = np.linalg.norm(w_new)
                    if w_norm < 1e-15:
                        break
                    w = w_new / w_norm
                
                uvw_val = np.einsum('a,b,c,abc->', u, v, w, R)
                if abs(uvw_val) < 1e-12:
                    break
                
                # Scale: we want u⊗v⊗w to have the right magnitude
                # The contribution is uvw_val * u⊗v⊗w (since u,v,w are unit)
                # Actually, the rank-1 tensor is (uvw_val) * u⊗v⊗w
                # But we want to subtract it from R
                U_list.append(u * uvw_val)
                V_list.append(v)
                W_list.append(w)
                R -= np.einsum('a,b,c->abc', U_list[-1], V_list[-1], W_list[-1])
            
            if len(U_list) == 0:
                continue
            
            U_arr = np.array(U_list).T
            V_arr = np.array(V_list).T
            W_arr = np.array(W_list).T
            rank = len(U_list)
            err = verify_decomp(U_arr, V_arr, W_arr, rank)
            
            if err < best_err:
                best_err = err
                best_rank = rank
                best_result = (U_arr, V_arr, W_arr)
            
            if err < 1e-10:
                break
        
        return best_result, best_rank, best_err

    # Try greedy peeling
    result, rank_g, err_g = greedy_peel(T, max_rank=20, n_restarts=3)
    
    if result is not None and err_g < 1e-8:
        U_g, V_g, W_g = result
        # Try to round to half-integers
        U_r = np.round(U_g * 2) / 2.0
        V_r = np.round(V_g * 2) / 2.0
        W_r = np.round(W_g * 2) / 2.0
        
        if np.max(np.abs(U_r - U_g)) < 1e-6 and np.max(np.abs(V_r - V_g)) < 1e-6 and np.max(np.abs(W_r - W_g)) < 1e-6:
            err_r = verify_decomp(U_r, V_r, W_r, rank_g)
            if err_r < 1e-10:
                best_U, best_V, best_W, best_rank = U_r, V_r, W_r, rank_g
            else:
                best_U, best_V, best_W, best_rank = U_g, V_g, W_g, rank_g
        else:
            best_U, best_V, best_W, best_rank = U_g, V_g, W_g, rank_g
    else:
        best_U, best_V, best_W, best_rank = None, None, None, 999

    # Fallback: improved Strassen-based construction
    # For 2x4 * 4x4, we can exploit that both 2x2 blocks in A share rows with C
    # A = [A_top; A_bot] where A_top, A_bot are 2x4
    # C = [C_top; C_bot] where C_top, C_bot are 2x4
    # C_top = A_top * B_top + A_bot * B_bot  (B_top, B_bot are 4x2 blocks of B)
    # C_bot = A_top * B_bot + A_bot * B_top
    
    # We use 7 Strassen products for (A_top, B_top) -> C_top contribution
    # and 7 for (A_bot, B_bot) -> C_top contribution  
    # and 7 for (A_top, B_bot) -> C_bot contribution
    # and 7 for (A_bot, B_top) -> C_bot contribution
    # Total: 28, but we can potentially share some intermediates
    
    # Let me try a direct rank-28 construction with explicit verification
    if best_rank > 28 or best_U is None:
        # Try to find a rank-28 or lower decomposition
        # Using the structure: C[i,k] = sum_l A[i,l] * B[l,k]
        # where i in {0,1}, l in {0,1,2,3}, k in {0,1,2,3}
        
        # Rank-28 via 4 independent Strassen 2x2x2 blocks
        rank = 28
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        # Strassen for 2x2 * 2x2 -> 2x2
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B10-B11)
        # M4 = A11*(B01-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B10)
        # M7 = (A01-A11)*(B01+B11)
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        
        def strassen_block(u_base, v_base, w_base, col_off, 
                          a_rows, b_cols, c_rows):
            """Add 7 Strassen terms for a 2x2*2x2 block"""
            # a_rows: [a00, a01, a10, a11] indices in U
            # b_cols: [b00, b01, b10, b11] indices in V  
            # c_rows: [c00, c01, c10, c11] indices in W
            
            a00, a01, a10, a11 = a_rows
            b00, b01, b10, b11 = b_cols
            c00, c01, c10, c11 = c_rows
            
            # M1 = (A00+A11)*(B00+B11)
            U[a00, col_off] += 1
            U[a11, col_off] += 1
            V[b00, col_off] += 1
            V[b11, col_off] += 1
            W[c00, col_off] += 1
            W[c11, col_off] += 1
            
            # M2 = (A10+A11)*B00
            U[a10, col_off+1] += 1
            U[a11, col_off+1] += 1
            V[b00, col_off+1] += 1
            W[c10, col_off+1] += 1
            W[c11, col_off+1] -= 1
            
            # M3 = A00*(B10-B11)
            U[a00, col_off+2] += 1
            V[b10, col_off+2] += 1
            V[b11, col_off+2] -= 1
            W[c01, col_off+2] += 1
            W[c11, col_off+2] += 1
            
            # M4 = A11*(B01-B00)
            U[a11, col_off+3] += 1
            V[b01, col_off+3] += 1
            V[b00, col_off+3] -= 1
            W[c00, col_off+3] += 1
            W[c10, col_off+3] += 1
            
            # M5 = (A00+A01)*B11
            U[a00, col_off+4] += 1
            U[a01, col_off+4] += 1
            V[b11, col_off+4] += 1
            W[c00, col_off+4] -= 1
            W[c01, col_off+4] += 1
            
            # M6 = (A10-A00)*(B00+B10)
            U[a10, col_off+5] += 1
            U[a00, col_off+5] -= 1
            V[b00, col_off+5] += 1
            V[b10, col_off+5] += 1
            W[c11, col_off+5] += 1
            
            # M7 = (A01-A11)*(B01+B11)
            U[a01, col_off+6] += 1
            U[a11, col_off+6] -= 1
            V[b01, col_off+6] += 1
            V[b11, col_off+6] += 1
            W[c00, col_off+6] += 1
        
        # Block 1: A[0:2,0:2] * B[0:2,0:2] contributes to C[0:2,0:2]
        # A rows 0,1; B cols 0,1; C rows 0,1
        strassen_block(0, 0, 0, 0, 
                      [0, 1, 2, 3], [0, 1, 4, 5], [0, 1, 2, 3])
        
        # Block 2: A[2:4,0:2] * B[0:2,0:2] contributes to C[0:2,0:2]
        # A rows 4,5; B cols 0,1; C rows 0,1
        strassen_block(0, 0, 0, 7,
                      [4, 5, 6, 7], [0, 1, 4, 5], [0, 1, 2, 3])
        
        # Block 3: A[0:2,2:4] * B[2:4,0:2] contributes to C[0:2,0:2]
        # A rows 0,1; B cols 2,3; C rows 0,1
        strassen_block(0, 0, 0, 14,
                      [0, 1, 2, 3], [2, 3, 6, 7], [0, 1, 2, 3])
        
        # Block 4: A[2:4,2:4] * B[2:4,0:2] contributes to C[0:2,0:2]
        # A rows 4,5; B cols 2,3; C rows 0,1
        strassen_block(0, 0, 0, 21,
                      [4, 5, 6, 7], [2, 3, 6, 7], [0, 1, 2, 3])
        
        # Wait, this gives C[0:2, 0:2] but we need C[0:2, 0:4] and C[2:4, 0:4]
        # Let me reconsider the indexing
        
        # Actually for 2x4 * 4x4:
        # A is 2x4: rows 0,1; cols 0,1,2,3
        # B is 4x4: rows 0,1,2,3; cols 0,1,2,3
        # C is 2x4: rows 0,1; cols 0,1,2,3
        # C[i,k] = sum_{l=0}^{3} A[i,l] * B[l,k]
        
        # So C has only 2 rows! Not 4.
        # The output factor W has shape (p*n, rank) = (8, rank)
        # W[k*n+i, r] so W[0] = C[0,0], W[1] = C[1,0], W[2] = C[0,1], W[3] = C[1,1], etc.
        
        # Let me redo this properly
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        # C[i,k] for i in {0,1}, k in {0,1,2,3}
        # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
        
        # Group by pairs of k: (k=0,1) and (k=2,3)
        # For each pair, we have a 2x2 * 4x2 -> 2x2 problem
        # Which can be seen as two 2x2*2x2 blocks
        
        # Pair 1: k=0,1
        # C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0] + A[i,2]*B[2,0] + A[i,3]*B[3,0]
        # C[i,1] = A[i,0]*B[0,1] + A[i,1]*B[1,1] + A[i,2]*B[2,1] + A[i,3]*B[3,1]
        
        # Split A into top 2 cols and bottom 2 cols:
        # A_top = A[:, 0:2] (2x2), A_bot = A[:, 2:4] (2x2)
        # B_top = B[0:2, 0:2] (2x2), B_bot = B[2:4, 0:2] (2x2)
        
        # C[:, 0:2] = A_top * B_top + A_bot * B_bot
        
        # Use Strassen for each product, but share the A factors
        # This is where we can potentially save rank
        
        # For now, use 4 independent Strassen blocks (rank 28)
        # Block 1: A_top * B_top
        # A rows: 0,1 (cols 0,1 of A)
        # B rows: 0,1 (cols 0,1 of B) 
        # C: rows 0,1 (cols 0,1)
        
        # U indexing: A[i,l] is at U[i*4+l]
        # So A[0,0]=U[0], A[0,1]=U[1], A[0,2]=U[2], A[0,3]=U[3]
        #    A[1,0]=U[4], A[1,1]=U[5], A[1,2]=U[6], A[1,3]=U[7]
        
        # V indexing: B[l,k] is at V[l*4+k]
        # So B[0,0]=V[0], B[0,1]=V[1], B[0,2]=V[2], B[0,3]=V[3]
        #    B[1,0]=V[4], B[1,1]=V[5], B[1,2]=V[6], B[1,3]=V[7]
        #    B[2,0]=V[8], B[2,1]=V[9], B[2,2]=V[10], B[2,3]=V[11]
        #    B[3,0]=V[12], B[3,1]=V[13], B[3,2]=V[14], B[3,3]=V[15]
        
        # W indexing: C[i,k] is at W[k*2+i]
        # So C[0,0]=W[0], C[1,0]=W[1], C[0,1]=W[2], C[1,1]=W[3]
        #    C[0,2]=W[4], C[1,2]=W[5], C[0,3]=W[6], C[1,3]=W[7]
        
        # Block 1: A[:,0:2] * B[0:2,0:2] -> C[:,0:2]
        # A indices: [0,1,4,5] for [A00,A01,A10,A11]
        # B indices: [0,1,4,5] for [B00,B01,B10,B11]
        # C indices: [0,2,1,3] for [C00,C01,C10,C11]
        strassen_block(0, 0, 0, 0,
                      [0, 1, 4, 5], [0, 1, 4, 5], [0, 2, 1, 3])
        
        # Block 2: A[:,2:4] * B[2:4,0:2] -> C[:,0:2]
        # A indices: [2,3,6,7]
        # B indices: [8,9,12,13]
        # C indices: [0,2,1,3]
        strassen_block(0, 0, 0, 7,
                      [2, 3, 6, 7], [8, 9, 12, 13], [0, 2, 1, 3])
        
        # Block 3: A[:,0:2] * B[0:2,2:4] -> C[:,2:4]
        # A indices: [0,1,4,5]
        # B indices: [2,3,6,7]
        # C indices: [4,6,5,7]
        strassen_block(0, 0, 0, 14,
                      [0, 1, 4, 5], [2, 3, 6, 7], [4, 6, 5, 7])
        
        # Block 4: A[:,2:4] * B[2:4,2:4] -> C[:,2:4]
        # A indices: [2,3,6,7]
        # B indices: [10,11,14,15]
        # C indices: [4,6,5,7]
        strassen_block(0, 0, 0, 21,
                      [2, 3, 6, 7], [10, 11, 14, 15], [4, 6, 5, 7])
        
        # Verify
        err = verify_decomp(U, V, W, rank)
        if err < 1e-10:
            best_U, best_V, best_W, best_rank = U, V, W, rank
        else:
            # Fallback to schoolbook rank 32
            rank = 32
            U = np.zeros((nm, rank))
            V = np.zeros((mp, rank))
            W = np.zeros((pn, rank))
            for i in range(n):
                for l in range(m):
                    for k in range(p):
                        a = i * m + l
                        b = l * p + k
                        c = k * n + i
                        r = i * (m*p) + l * p + k
                        U[a, r] = 1
                        V[b, r] = 1
                        W[c, r] = 1
            best_U, best_V, best_W, best_rank = U, V, W, rank

    def to_list(mat):
        result = []
        for row in mat:
            row_list = []
            for val in row:
                rounded = round(val * 2) / 2.0
                if abs(rounded - val) < 1e-10:
                    row_list.append(rounded)
                else:
                    row_list.append(float(val))
            result.append(row_list)
        return result

    # Final safety check
    if best_U is None or best_rank > 32:
        # Absolute fallback: schoolbook
        rank = 32
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        for i in range(n):
            for l in range(m):
                for k in range(p):
                    a = i * m + l
                    b = l * p + k
                    c = k * n + i
                    r = i * (m*p) + l * p + k
                    U[a, r] = 1
                    V[b, r] = 1
                    W[c, r] = 1
        best_U, best_V, best_W, best_rank = U, V, W, rank

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
