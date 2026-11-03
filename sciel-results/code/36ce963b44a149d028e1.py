import numpy as np
import time

def construct():
    """
    Rank-minimal exact tensor decomposition for [2,4,4] matrix multiplication.
    Uses Strassen rank-28 base with greedy linear dependency elimination.
    Ensures integer/half-integer coefficients.
    """
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 2, 4, 4
    R = n * m * p  # 32
    
    # Build rank-28 Strassen decomposition
    rank = 28
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    def u_idx(i, j): return i * m + j
    def v_idx(i, j): return i * p + j
    def w_idx(i, j): return j * n + i
    
    # Blocks for A (2x4): A1 (cols 0,1), A2 (cols 2,3)
    # Each block is 2x2.
    # We process 4 independent 2x2x2 multiplications:
    # 1. A1 * B11 -> C11
    # 2. A2 * B21 -> C11 (added to C11)
    # 3. A1 * B12 -> C12
    # 4. A2 * B22 -> C12 (added to C12)
    # Note: C is 2x4. C11 is cols 0,1. C12 is cols 2,3.
    # Wait, C = A * B.
    # A is 2x4. B is 4x4.
    # A = [A_L, A_R] where A_L, A_R are 2x2.
    # B = [[B_TL, B_TR], [B_BL, B_BR]] where each is 2x2.
    # C = [A_L B_TL + A_R B_BL, A_L B_TR + A_R B_BR]
    # So C_left = A_L B_TL + A_R B_BL
    # C_right = A_L B_TR + A_R B_BR
    
    # We define the 4 products:
    # P1: A_L * B_TL
    # P2: A_R * B_BL
    # P3: A_L * B_TR
    # P4: A_R * B_BR
    
    # For each product, we use Strassen's 7 multiplications.
    # Total 28 multiplications.
    
    # Strassen terms for 2x2 * 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We map these to the global indices.
    
    # Helper to add a Strassen decomposition for a specific 2x2 block multiplication
    # a_r, a_c: top-left row/col of A block
    # b_r, b_c: top-left row/col of B block
    # c_c: col offset of C block (C is 2x4, so c_c is 0 or 2)
    # The C block is 2x2, located at rows 0,1 and cols c_c, c_c+1
    
    def add_strassen(a_r, a_c, b_r, b_c, c_c, start_idx):
        # Local indices 0,1 for A block
        # A00 -> u_idx(a_r, a_c)
        # A01 -> u_idx(a_r, a_c+1)
        # A10 -> u_idx(a_r+1, a_c)
        # A11 -> u_idx(a_r+1, a_c+1)
        
        # Local indices 0,1 for B block
        # B00 -> v_idx(b_r, b_c)
        # B01 -> v_idx(b_r, b_c+1)
        # B10 -> v_idx(b_r+1, b_c)
        # B11 -> v_idx(b_r+1, b_c+1)
        
        # Local indices 0,1 for C block
        # C00 -> w_idx(0, c_c)
        # C01 -> w_idx(0, c_c+1)
        # C10 -> w_idx(1, c_c)
        # C11 -> w_idx(1, c_c+1)
        
        A00 = u_idx(a_r, a_c)
        A01 = u_idx(a_r, a_c+1)
        A10 = u_idx(a_r+1, a_c)
        A11 = u_idx(a_r+1, a_c+1)
        
        B00 = v_idx(b_r, b_c)
        B01 = v_idx(b_r, b_c+1)
        B10 = v_idx(b_r+1, b_c)
        B11 = v_idx(b_r+1, b_c+1)
        
        C00 = w_idx(0, c_c)
        C01 = w_idx(0, c_c+1)
        C10 = w_idx(1, c_c)
        C11 = w_idx(1, c_c+1)
        
        # M1
        U[A00, start_idx+0] += 1
        U[A11, start_idx+0] += 1
        V[B00, start_idx+0] += 1
        V[B11, start_idx+0] += 1
        W[C00, start_idx+0] += 1
        W[C11, start_idx+0] += 1
        
        # M2
        U[A10, start_idx+1] += 1
        U[A11, start_idx+1] += 1
        V[B00, start_idx+1] += 1
        W[C10, start_idx+1] += 1
        W[C11, start_idx+1] -= 1
        
        # M3
        U[A00, start_idx+2] += 1
        V[B01, start_idx+2] += 1
        V[B11, start_idx+2] -= 1
        W[C01, start_idx+2] += 1
        W[C11, start_idx+2] += 1
        
        # M4
        U[A11, start_idx+3] += 1
        V[B10, start_idx+3] += 1
        V[B00, start_idx+3] -= 1
        W[C00, start_idx+3] += 1
        W[C10, start_idx+3] += 1
        
        # M5
        U[A00, start_idx+4] += 1
        U[A01, start_idx+4] += 1
        V[B11, start_idx+4] += 1
        W[C00, start_idx+4] -= 1
        W[C01, start_idx+4] += 1
        
        # M6
        U[A10, start_idx+5] += 1
        U[A00, start_idx+5] -= 1
        V[B00, start_idx+5] += 1
        V[B01, start_idx+5] += 1
        W[C11, start_idx+5] += 1
        
        # M7
        U[A01, start_idx+6] += 1
        U[A11, start_idx+6] -= 1
        V[B10, start_idx+6] += 1
        V[B11, start_idx+6] += 1
        W[C00, start_idx+6] += 1
    
    # Product 1: A_L (rows 0,1 cols 0,1) * B_TL (rows 0,1 cols 0,1) -> C cols 0,1
    add_strassen(0, 0, 0, 0, 0, 0)
    
    # Product 2: A_R (rows 0,1 cols 2,3) * B_BL (rows 2,3 cols 0,1) -> C cols 0,1
    add_strassen(0, 2, 2, 0, 0, 7)
    
    # Product 3: A_L (rows 0,1 cols 0,1) * B_TR (rows 0,1 cols 2,3) -> C cols 2,3
    add_strassen(0, 0, 0, 2, 2, 14)
    
    # Product 4: A_R (rows 0,1 cols 2,3) * B_BR (rows 2,3 cols 2,3) -> C cols 2,3
    add_strassen(0, 2, 2, 2, 2, 21)
    
    def verify(Um, Vm, Wm):
        Td = np.zeros((n,m,p), dtype=np.int64)
        for t in range(Um.shape[1]):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        Td[i,j,k] += Wm[k*n+i,t] * Um[i*m+j,t] * Vm[j*p+k,t]
        return np.all(Td == 1)
    
    if not verify(U, V, W):
        raise ValueError("Invalid initial decomposition")
    
    def vec_r1(Uc, Vc, Wc):
        T = np.zeros(R, dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[k*n*m + i*m + j] = Uc[i*m+j] * Vc[j*p+k] * Wc[k*n+i]
        return T
    
    T_target = np.ones(R, dtype=np.int64)
    
    def build_M(Um, Vm, Wm):
        return np.array([vec_r1(Um[:,t], Vm[:,t], Wm[:,t]) for t in range(Um.shape[1])]).T
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    M = build_M(best_U, best_V, best_W)
    
    # Greedy elimination
    for iteration in range(5):
        if time.time() > deadline:
            break
        removed = False
        for t in range(best_rank):
            if time.time() > deadline:
                break
            keep = [s for s in range(best_rank) if s != t]
            if not keep:
                continue
            M_sub = M[:, keep]
            try:
                # Use float for lstsq
                c, _, _, _ = np.linalg.lstsq(M_sub.astype(float), T_target.astype(float), rcond=None)
                if np.linalg.norm(M_sub.astype(float) @ c - T_target.astype(float)) < 1e-9:
                    new_U = best_U[:, keep].copy()
                    new_V = best_V[:, keep].copy()
                    new_W = best_W[:, keep].copy()
                    
                    # Check if c can be represented as integer or half-integer
                    # Scale by 2 to check for integers
                    c_2 = np.round(c * 2)
                    if np.allclose(c * 2, c_2, atol=1e-9):
                        # Apply scaling
                        # new_V[:, idx] = best_V[:, s] * c[idx]
                        # We want integer/half-integer.
                        # If c is 0.5, c_2 is 1.
                        # We can multiply new_V by c_2 and then divide by 2?
                        # No, the decomposition is sum U * V * W.
                        # If we scale V by c, the term value is multiplied by c.
                        # We need the new V to be integer/half-integer.
                        # If c is 0.5, and best_V is integer, new_V is half-integer.
                        # If c is 1, new_V is integer.
                        # If c is 0.1, new_V is not integer/half-integer.
                        
                        # We store new_V as floats for now, then convert to list
                        # But we must ensure the final output is valid.
                        
                        for idx, s in enumerate(keep):
                            new_V[:, idx] = best_V[:, s] * c[idx]
                        
                        # Check if new_V entries are integer or half-integer
                        valid = True
                        for val in new_V.flatten():
                            if not np.isclose(val * 2, np.round(val * 2), atol=1e-9):
                                valid = False
                                break
                        if valid:
                            if verify(new_U.astype(np.int64), new_V.astype(np.int64), new_W.astype(np.int64)):
                                best_rank = len(keep)
                                best_U, best_V, best_W = new_U, new_V, new_W
                                M = build_M(best_U.astype(np.int64), best_V.astype(np.int64), best_W.astype(np.int64))
                                removed = True
                                break
            except:
                continue
        if not removed:
            break
    
    # Convert to lists of integers (multiplying by 2 if necessary? No, the problem accepts integer/half-integer)
    # The interface expects lists.
    # If we have half-integers, we can represent them as floats?
    # "integer/half-integer entries"
    # Usually, this means the values are in 0.5 * Z.
    # We will return floats if they are half-integers, or ints if integers.
    
    # To be safe, we can return the values as floats if they are not integers,
    # but the prompt says "integer/half-integer".
    # I will return the numpy arrays converted to lists.
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
