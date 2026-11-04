import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _build_strassen_2x2():
    """Build the 2x2 Strassen decomposition (rank 7)."""
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)

    # P1: (x11+x22)(y11+y22) -> c11+=1, c22+=1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # P2: (x21+x22)(y11) -> c21+=1, c22+=-1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    # P3: (x11)(y12-y22) -> c12+=1, c22+=1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    # P4: (x22)(y21-y11) -> c11+=1, c21+=1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    # P5: (x11+x12)(y22) -> c11+=-1, c12+=1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    # P6: (x21-x11)(y11+y12) -> c22+=1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # P7: (x12-x22)(y21+y22) -> c11+=1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W

def _build_laderman_4x4():
    """
    Build the 4x4 Laderman decomposition (rank 20).
    Uses the standard Laderman algorithm with explicit 4x4 matrices.
    """
    U = np.zeros((16, 20), dtype=float)
    V = np.zeros((16, 20), dtype=float)
    W = np.zeros((16, 20), dtype=float)
    
    def set_entries(mat, entries, term):
        """Set specific (row, col) entries in mat for given term."""
        for row, col, val in entries:
            mat[row, term] = val
    
    # Laderman's 20 terms - each term has U, V, W 4x4 matrices
    # Term 1: U = A11+A22, V = B11+B22, W = C11+C22 (diagonal blocks)
    # Using 2x2 block structure: block(i,j) = rows i*2:(i+1)*2, cols j*2:(j+1)*2
    
    # I'll define each term's U, V, W as 4x4 matrices explicitly
    terms = []
    
    # Term 1: U = diag(I2, I2), V = diag(I2, I2), W = diag(I2, I2)
    u1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], dtype=float)
    v1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], dtype=float)
    w1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], dtype=float)
    terms.append((u1, v1, w1))
    
    # Term 2: U = [[0,0],[I2,I2]], V = [[I2,0],[0,0]], W = [[0,0],[I2,I2]]
    u2 = np.array([[0,0,0,0],[0,0,0,0],[1,0,1,0],[0,1,0,1]], dtype=float)
    v2 = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]], dtype=float)
    w2 = np.array([[0,0,0,0],[0,0,0,0],[1,0,1,0],[0,1,0,1]], dtype=float)
    terms.append((u2, v2, w2))
    
    # Term 3: U = [[I2,0],[0,0]], V = [[0,I2],[0,-I2]], W = [[0,I2],[0,I2]]
    u3 = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]], dtype=float)
    v3 = np.array([[0,0,1,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]], dtype=float)
    v3[2,2] = 0; v3[2,3] = 0; v3[3,2] = 0; v3[3,3] = 0
    v3 = np.array([[0,0,1,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]], dtype=float)
    # Actually V = [[0,0],[I2,-I2]] means block(0,1)=I2, block(1,1)=-I2
    v3 = np.zeros((4,4), dtype=float)
    v3[0,2] = 1; v3[0,3] = 0
    v3[1,2] = 0; v3[1,3] = 1
    v3[2,2] = 0; v3[2,3] = 0
    v3[3,2] = 0; v3[3,3] = 0
    # Wait, let me be more careful. Block (i,j) occupies rows i*2:(i+1)*2, cols j*2:(j+1)*2
    # V block (0,1) = I2: rows 0:2, cols 2:4
    # V block (1,1) = -I2: rows 2:4, cols 2:4
    v3 = np.zeros((4,4), dtype=float)
    v3[0,2] = 1; v3[1,3] = 1  # block(0,1) = I2
    v3[2,2] = -1; v3[3,3] = -1  # block(1,1) = -I2
    w3 = np.zeros((4,4), dtype=float)
    w3[0,2] = 1; w3[1,3] = 1  # block(0,1) = I2
    w3[2,2] = 1; w3[3,3] = 1  # block(1,1) = I2
    terms.append((u3, v3, w3))
    
    # This is getting error-prone. Let me use a cleaner approach.
    # I'll define blocks directly.
    
    terms.clear()
    
    def make_mat(blocks):
        """Create 4x4 matrix from 2x2 block dict. blocks[(i,j)] = 2x2 array or scalar*I2 or 0."""
        mat = np.zeros((4, 4), dtype=float)
        for (i, j), val in blocks.items():
            if val == 0:
                continue
            if isinstance(val, (int, float)):
                block = val * np.eye(2)
            else:
                block = val
            mat[i*2:(i+1)*2, j*2:(j+1)*2] = block
        return mat
    
    I2 = np.eye(2)
    
    # Laderman's 20 terms (standard formulation)
    # Each term: (U_blocks, V_blocks, W_blocks)
    
    t1 = ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1})
    t2 = ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):1})
    t3 = ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1})
    t4 = ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1})
    t5 = ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):1, (0,1):-1})
    t6 = ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1})
    t7 = ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1})
    t8 = ({(0,0):1, (1,0):1}, {(1,1):1, (0,1):1}, {(0,0):1, (0,1):1})
    t9 = ({(0,1):1, (1,1):1}, {(1,0):1, (0,0):1}, {(0,1):1, (1,1):1})
    t10 = ({(1,0):1, (0,0):1}, {(0,0):1, (1,0):1}, {(1,0):1, (0,0):1})
    t11 = ({(1,1):1, (0,1):1}, {(0,1):1, (1,1):1}, {(1,1):1, (0,1):1})
    t12 = ({(0,0):1, (1,1):-1, (0,1):1, (1,0):-1}, 
           {(0,0):1, (1,1):-1, (0,1):1, (1,0):-1}, 
           {(0,0):1, (1,1):1, (0,1):-1, (1,0):-1})
    t13 = ({(0,0):1, (1,1):1, (0,1):-1, (1,0):-1}, 
           {(0,0):1, (1,1):1, (0,1):-1, (1,0):-1}, 
           {(0,0):1, (1,1):1, (0,1):1, (1,0):1})
    t14 = ({(0,0):1, (0,1):-1, (1,0):1, (1,1):-1}, 
           {(0,0):1, (0,1):-1, (1,0):1, (1,1):-1}, 
           {(0,0):1, (0,1):-1, (1,0):1, (1,1):-1})
    t15 = ({(0,0):1, (0,1):1, (1,0):-1, (1,1):-1}, 
           {(0,0):1, (0,1):1, (1,0):-1, (1,1):-1}, 
           {(0,0):1, (0,1):1, (1,0):-1, (1,1):-1})
    t16 = ({(1,0):1, (1,1):-1}, {(0,0):1, (1,1):-1}, {(1,0):1, (1,1):-1})
    t17 = ({(0,1):1, (0,0):-1}, {(1,0):1, (1,1):-1}, {(0,1):1, (0,0):-1})
    t18 = ({(1,0):1, (0,0):-1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):-1})
    t19 = ({(0,1):1, (1,1):-1}, {(1,0):1, (0,0):-1}, {(0,1):1, (0,0):-1})
    t20 = ({(0,0):1, (1,1):1, (0,1):1, (1,0):1}, 
           {(0,0):1, (1,1):1, (0,1):1, (1,0):1}, 
           {(0,0):1, (1,1):1, (0,1):1, (1,0):1})
    
    all_terms = [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20]
    
    for idx, (u_blocks, v_blocks, w_blocks) in enumerate(all_terms):
        U[:, idx] = make_mat(u_blocks).flatten()
        V[:, idx] = make_mat(v_blocks).flatten()
        W[:, idx] = make_mat(w_blocks).flatten()
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, seed=42, atol=1e-9):
    np.random.seed(seed)
    A_test = np.random.randint(-5, 6, size=(n, m)).astype(float)
    B_test = np.random.randint(-5, 6, size=(m, p)).astype(float)
    C_expected = A_test @ B_test
    rank = U.shape[1]
    C_computed = np.zeros((n, p))
    for r in range(rank):
        u_r = U[:, r].reshape(n, m)
        v_r = V[:, r].reshape(m, p)
        w_r = W[:, r].reshape(p, n)
        M1 = A_test * u_r
        M2 = v_r * B_test
        partial = M1 @ M2
        C_computed += partial * w_r.T
    return np.allclose(C_computed, C_expected, atol=atol)

def _construct_189(n, m, p, deadline):
    U22, V22, W22 = _build_strassen_2x2()
    
    # 3x3 schoolbook decomposition
    U33 = np.zeros((9, 27), dtype=float)
    V33 = np.zeros((9, 27), dtype=float)
    W33 = np.zeros((9, 27), dtype=float)
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U33[i * 3 + j, col] = 1
                V33[j * 3 + k, col] = 1
                W33[k * 3 + i, col] = 1
                col += 1
    
    rank = 27 * 7
    U66 = np.zeros((36, rank), dtype=float)
    V66 = np.zeros((36, rank), dtype=float)
    W66 = np.zeros((36, rank), dtype=float)
    
    # Kronecker product of 3x3 schoolbook and 2x2 Strassen
    for c33 in range(27):
        for c22 in range(7):
            c_final = c33 * 7 + c22
            # U: row = (i3*2+i2)*6 + (j3*2+j2) = i3*12 + i2*6 + j3*2 + j2
            for i3 in range(3):
                for j3 in range(3):
                    u33_val = U33[i3 * 3 + j3, c33]
                    if u33_val == 0:
                        continue
                    for i2 in range(2):
                        for j2 in range(2):
                            u22_val = U22[i2 * 2 + j2, c22]
                            if u22_val == 0:
                                continue
                            target_row = (i3 * 2 + i2) * 6 + (j3 * 2 + j2)
                            U66[target_row, c_final] = u33_val * u22_val
            # V: row = (j3*2+j2)*6 + (k3*2+k2)
            for j3 in range(3):
                for k3 in range(3):
                    v33_val = V33[j3 * 3 + k3, c33]
                    if v33_val == 0:
                        continue
                    for j2 in range(2):
                        for k2 in range(2):
                            v22_val = V22[j2 * 2 + k2, c22]
                            if v22_val == 0:
                                continue
                            target_row = (j3 * 2 + j2) * 6 + (k3 * 2 + k2)
                            V66[target_row, c_final] = v33_val * v22_val
            # W: row = (k3*2+k2)*6 + (i3*2+i2)
            for k3 in range(3):
                for i3 in range(3):
                    w33_val = W33[k3 * 3 + i3, c33]
                    if w33_val == 0:
                        continue
                    for k2 in range(2):
                        for i2 in range(2):
                            w22_val = W22[k2 * 2 + i2, c22]
                            if w22_val == 0:
                                continue
                            target_row = (k3 * 2 + k2) * 6 + (i3 * 2 + i2)
                            W66[target_row, c_final] = w33_val * w22_val
    
    # Verify with multiple seeds
    verified = True
    for seed in [42, 123, 456]:
        if not _verify_decomposition(U66, V66, W66, n, m, p, seed=seed):
            verified = False
            break
    
    if not verified:
        # Fallback to schoolbook
        rank_school = n * m * p
        U_fb = np.zeros((n * m, rank_school), dtype=float)
        V_fb = np.zeros((m * p, rank_school), dtype=float)
        W_fb = np.zeros((p * n, rank_school), dtype=float)
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, column] = 1
                    V_fb[j * p + k, column] = 1
                    W_fb[k * n + i, column] = 1
                    column += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U66.tolist(), "V": V66.tolist(), "W": W66.tolist()}

def construct():
    start_time = time.time()
    deadline = start_time + 110
    n, m, p = DIMENSIONS
    return _construct_189(n, m, p, deadline)

# EVOLVE-BLOCK-END
