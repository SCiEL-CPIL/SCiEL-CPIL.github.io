import time
import sys

DIMENSIONS = [4, 4, 5]
_START_TIME = time.time()
TIME_LIMIT = 120.0  # seconds

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Deadline guard
    if _time_remaining() < 1.0:
        # Return a fallback schoolbook decomposition (rank 80)
        rank = 80
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # We will use a decomposition of rank 65.
    rank = 65
    U = [[0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * rank for _ in range(p * n)]  # 20 x 65
    
    def add_term(col, u_list, v_list, w_list):
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i][col] += c

    col = 0
    
    # --- Part 1: 4x4 * 4x4 using Strassen ---
    # Super terms: 7
    # Inner terms: 7
    # Total: 49
    
    # Define Strassen parameters for 2x2 blocks
    # P = A00 + A11, Q = B00 + B11, etc.
    
    # Helper to generate indices for a specific super/inner term
    def get_indices(super_idx, inner_idx):
        # p_blocks: coefficients for A00, A01, A10, A11
        # q_blocks: coefficients for B00, B01, B10, B11
        # blocks: (block_row, block_col, sign) for C contribution
        
        if super_idx == 0:
            p_blocks = [1, 0, 0, 1]
            q_blocks = [1, 0, 0, 1]
            blocks = [(0, 0, 1), (1, 1, 1)]
        elif super_idx == 1:
            p_blocks = [0, 0, 1, 1]
            q_blocks = [1, 0, 0, 0]
            blocks = [(1, 0, 1), (1, 1, -1)]
        elif super_idx == 2:
            p_blocks = [1, 0, 0, 0]
            q_blocks = [0, 1, 0, -1]
            blocks = [(0, 1, 1), (1, 1, 1)]
        elif super_idx == 3:
            p_blocks = [0, 0, 0, 1]
            q_blocks = [-1, 0, 1, 0]
            blocks = [(0, 0, 1), (1, 0, 1)]
        elif super_idx == 4:
            p_blocks = [1, 1, 0, 0]
            q_blocks = [0, 0, 0, 1]
            blocks = [(0, 0, -1), (0, 1, 1)]
        elif super_idx == 5:
            p_blocks = [-1, 0, 1, 0]
            q_blocks = [1, 1, 0, 0]
            blocks = [(1, 1, 1)]
        elif super_idx == 6:
            p_blocks = [0, 1, 0, -1]
            q_blocks = [0, 0, 1, 1]
            blocks = [(0, 0, 1)]
        else:
            return [], [], []
            
        # Inner Strassen terms for 2x2
        # 0: (P00+P11)(Q00+Q11) -> C00+=1, C11+=1
        # 1: (P10+P11)Q00     -> C10+=1, C11-=1
        # 2: P00(Q01-Q11)      -> C01+=1, C11+=1
        # 3: P11(Q10-Q00)      -> C00+=1, C10+=1
        # 4: (P00+P01)Q11      -> C00-=1, C01+=1
        # 5: (P10-P00)(Q00+Q01)-> C11+=1
        # 6: (P01-P11)(Q10+Q11)-> C00+=1
        
        # p_elems: list of (elem_idx, coeff) where elem_idx is 0:00, 1:01, 2:10, 3:11
        # q_elems: list of (elem_idx, coeff)
        # c_contrib: list of (local_row, local_col, coeff)
        
        if inner_idx == 0:
            p_elems = [(0, 1), (3, 1)]
            q_elems = [(0, 1), (3, 1)]
            c_contrib = [(0, 0, 1), (1, 1, 1)]
        elif inner_idx == 1:
            p_elems = [(2, 1), (3, 1)]
            q_elems = [(0, 1)]
            c_contrib = [(1, 0, 1), (1, 1, -1)]
        elif inner_idx == 2:
            p_elems = [(0, 1)]
            q_elems = [(1, 1), (3, -1)]
            c_contrib = [(0, 1, 1), (1, 1, 1)]
        elif inner_idx == 3:
            p_elems = [(3, 1)]
            q_elems = [(2, 1), (0, -1)]
            c_contrib = [(0, 0, 1), (1, 0, 1)]
        elif inner_idx == 4:
            p_elems = [(0, 1), (1, 1)]
            q_elems = [(3, 1)]
            c_contrib = [(0, 0, -1), (0, 1, 1)]
        elif inner_idx == 5:
            p_elems = [(2, 1), (0, -1)]
            q_elems = [(0, 1), (1, 1)]
            c_contrib = [(1, 1, 1)]
        elif inner_idx == 6:
            p_elems = [(1, 1), (3, -1)]
            q_elems = [(2, 1), (3, 1)]
            c_contrib = [(0, 0, 1)]
        else:
            return [], [], []
            
        u_list = []
        v_list = []
        
        # Expand P elements
        for p_elem, coeff in p_elems:
            r = p_elem // 2
            c = p_elem % 2
            for b_idx, b_coeff in enumerate(p_blocks):
                if b_coeff == 0: continue
                if b_idx == 0: gi, gj = r, c
                elif b_idx == 1: gi, gj = r, 2+c
                elif b_idx == 2: gi, gj = 2+r, c
                elif b_idx == 3: gi, gj = 2+r, 2+c
                u_list.append((gi, gj, coeff * b_coeff))
                
        # Expand Q elements
        for q_elem, coeff in q_elems:
            r = q_elem // 2
            c = q_elem % 2
            for b_idx, b_coeff in enumerate(q_blocks):
                if b_coeff == 0: continue
                if b_idx == 0: gj, gk = r, c
                elif b_idx == 1: gj, gk = r, 2+c
                elif b_idx == 2: gj, gk = 2+r, c
                elif b_idx == 3: gj, gk = 2+r, 2+c
                v_list.append((gj, gk, coeff * b_coeff))
                
        # Expand W elements
        w_list = []
        for br, bc, sign in blocks:
            for lr, lc, local_coeff in c_contrib:
                gi = br * 2 + lr
                gk = bc * 2 + lc
                w_list.append((gk, gi, sign * local_coeff))
                
        return u_list, v_list, w_list

    for s in range(7):
        for i in range(7):
            u_l, v_l, w_l = get_indices(s, i)
            if u_l and v_l and w_l:
                add_term(col, u_l, v_l, w_l)
                col += 1

    # Part 2: 4x4 * 4x1 (rank 16)
    for i in range(4):
        for j in range(4):
            U[i*4 + j][col] = 1
            V[j*5 + 4][col] = 1
            W[4*4 + i][col] = 1
            col += 1

    # Optional: Verification
    # If time permits, verify a few random cases
    if _time_remaining() > 5.0:
        try:
            import numpy as np
            # Convert to numpy for speed
            U_np = np.array(U)
            V_np = np.array(V)
            W_np = np.array(W)
            
            # Generate random integer matrices
            A_test = np.random.randint(-10, 10, (n, m))
            B_test = np.random.randint(-10, 10, (m, p))
            C_expected = A_test @ B_test
            
            C_actual = np.zeros((n, p))
            for r in range(rank):
                # M_r = sum(U * A) * sum(V * B)
                # U shape (n*m, rank). Flatten A to (n*m,)
                u_flat = U_np[:, r]
                v_flat = V_np[:, r]
                w_flat = W_np[:, r]
                
                # Calculate scalar product of U_r and A
                # U_r is indexed by i*m+j. A is i,j.
                # Reshape U_r to (n, m)
                u_mat = u_flat.reshape((n, m))
                v_mat = v_flat.reshape((m, p))
                
                val_u = np.sum(u_mat * A_test)
                val_v = np.sum(v_mat * B_test)
                
                # Add to C
                # W_r is indexed by k*n+i.
                # Reshape W_r to (p, n)
                w_mat = w_flat.reshape((p, n))
                C_actual += val_u * val_v * w_mat.T # w_mat is (p,n), we need (n,p) for C?
                # C[i,k] gets contribution. W[k*n+i] is W_r[k,i].
                # So C += (val_u * val_v) * W_r.T
                # W_r.T is (n, p). C is (n, p). Correct.
            
            if not np.allclose(C_actual, C_expected):
                # Fallback to schoolbook if verification fails
                rank_fb = n * m * p
                U_fb = [[0] * rank_fb for _ in range(n * m)]
                V_fb = [[0] * rank_fb for _ in range(m * p)]
                W_fb = [[0] * rank_fb for _ in range(p * n)]
                col_fb = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U_fb[i * m + j][col_fb] = 1
                            V_fb[j * p + k][col_fb] = 1
                            W_fb[k * n + i][col_fb] = 1
                            col_fb += 1
                return {"U": U_fb, "V": V_fb, "W": W_fb}
        except Exception:
            pass

    return {"U": U, "V": V, "W": W}
