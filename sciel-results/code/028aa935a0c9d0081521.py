"""
Tensor decomposition for 4x5 * 5x5 matrix multiplication.
Strategy: decompose using 4x4 Strassen block + cross terms, then aggressively prune.
"""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Return U, V, W (each 4x7) for 2x2 * 2x2 matrix multiplication using Strassen.
    Layout: A[i*2+j], B[j*2+k], C[k*2+i]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1; W[0, 4] = -1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W


def _build_4x5x5():
    """
    Build decomposition for 4x5 * 5x5 using block decomposition with 4x4 Strassen.
    Split: A = [A_4 | A_1], B = [B_4; B_1] where B_4 is 4x5, B_1 is 1x5.
    C = A_4 * B_4 + A_1 * B_1
    
    A_4 * B_4: split B_4 = [B_44 | B_41] (4x4 and 4x1)
      A_4 * B_44: 4x4*4x4 Strassen, rank 49
      A_4 * B_41: 4x4*4x1 schoolbook, rank 16
    A_1 * B_1: 4x1*1x5, rank 1
    
    Total initial rank: 66
    """
    n, m, p = 4, 5, 5
    nm = n * m  # 20
    mp = m * p  # 25
    pn = p * n  # 20
    
    U2, V2, W2 = _strassen_2x2()
    rank2 = 7
    rank4 = rank2 * rank2  # 49
    
    U4 = np.zeros((16, rank4))
    V4 = np.zeros((16, rank4))
    W4 = np.zeros((16, rank4))
    
    def block_offset_4x4(br, bc):
        idxs = []
        for lr in range(2):
            for lc in range(2):
                idxs.append((br * 2 + lr) * 4 + (bc * 2 + lc))
        return idxs
    
    strassen_products = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    for s, (A_coeffs, B_coeffs, C_coeffs) in enumerate(strassen_products):
        for r2 in range(rank2):
            col4 = s * rank2 + r2
            for (abr, abc), c_a in A_coeffs.items():
                a_idxs = block_offset_4x4(abr, abc)
                for li, a_idx in enumerate(a_idxs):
                    U4[a_idx, col4] += c_a * U2[li, r2]
            for (bbr, bbc), c_b in B_coeffs.items():
                b_idxs = block_offset_4x4(bbr, bbc)
                for li, b_idx in enumerate(b_idxs):
                    V4[b_idx, col4] += c_b * V2[li, r2]
            for (cbr, cbc), c_c in C_coeffs.items():
                c_idxs = block_offset_4x4(cbr, cbc)
                for li, c_idx in enumerate(c_idxs):
                    W4[c_idx, col4] += c_c * W2[li, r2]
    
    U_part1 = np.zeros((nm, 49))
    V_part1 = np.zeros((mp, 49))
    W_part1 = np.zeros((pn, 49))
    
    for i in range(4):
        for j in range(4):
            U_part1[i * 5 + j, :] = U4[i * 4 + j, :]
    for j in range(4):
        for k in range(4):
            V_part1[j * 5 + k, :] = V4[j * 4 + k, :]
    for k in range(4):
        for i in range(4):
            W_part1[k * 4 + i, :] = W4[k * 4 + i, :]
    
    U_part2 = np.zeros((nm, 16))
    V_part2 = np.zeros((mp, 16))
    W_part2 = np.zeros((pn, 16))
    
    col2 = 0
    for i in range(4):
        for j in range(4):
            U_part2[i * 5 + j, col2] = 1
            V_part2[j * 5 + 4, col2] = 1
            W_part2[4 * 4 + i, col2] = 1
            col2 += 1
    
    U_part3 = np.zeros((nm, 1))
    V_part3 = np.zeros((mp, 1))
    W_part3 = np.zeros((pn, 1))
    
    for i in range(4):
        U_part3[i * 5 + 4, 0] = 1
    for k in range(5):
        V_part3[4 * 5 + k, 0] = 1
    for k in range(5):
        for i in range(4):
            W_part3[k * 4 + i, 0] = 1
    
    U = np.hstack([U_part1, U_part2, U_part3])
    V = np.hstack([V_part1, V_part2, V_part3])
    W = np.hstack([W_part1, W_part2, W_part3])
    
    return U, V, W


def _compute_tensor(U, V, W):
    """Compute the full tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]"""
    return np.einsum('ar,br,cr->abc', U, V, W)


def _expected_tensor(n, m, p):
    """Compute the expected matrix multiplication tensor."""
    nm, mp, pn = n * m, m * p, p * n
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T


def _try_remove_columns(U, V, W, n, m, p, max_time=60):
    """
    Aggressively remove redundant columns using multiple strategies:
    1. Greedy forward removal with incremental tensor updates
    2. Random restarts with different removal orders
    3. Pairwise removal attempts
    """
    start_time = time.time()
    nm, mp, pn = n * m, m * p, p * n
    rank = U.shape[1]
    expected = _expected_tensor(n, m, p)
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    def try_remove_with_order(U_cur, V_cur, W_cur, order, time_limit):
        """Try removing columns in the given order, greedily."""
        active = list(range(U_cur.shape[1]))
        T_cur = _compute_tensor(U_cur, V_cur, W_cur)
        
        for idx in order:
            if time.time() > time_limit:
                break
            if idx not in active:
                continue
            pos = active.index(idx)
            test_active = active[:pos] + active[pos+1:]
            if not test_active:
                break
            
            # Compute tensor for test_active
            T_test = _compute_tensor(U_cur[:, test_active], V_cur[:, test_active], W_cur[:, test_active])
            if np.max(np.abs(T_test - expected)) < 1e-9:
                active = test_active
                T_cur = T_test
        
        if len(active) < U_cur.shape[1]:
            return U_cur[:, active], V_cur[:, active], W_cur[:, active], len(active)
        return None, None, None, U_cur.shape[1]
    
    # Strategy 1: Forward greedy
    U1, V1, W1, r1 = try_remove_with_order(U, V, W, list(range(rank)), start_time + max_time * 0.4)
    if U1 is not None and r1 < best_rank:
        best_U, best_V, best_W, best_rank = U1, V1, W1, r1
    
    # Strategy 2: Reverse order
    U2, V2, W2, r2 = try_remove_with_order(U, V, W, list(range(rank-1, -1, -1)), start_time + max_time * 0.6)
    if U2 is not None and r2 < best_rank:
        best_U, best_V, best_W, best_rank = U2, V2, W2, r2
    
    # Strategy 3: Random shuffles from best so far
    if time.time() < start_time + max_time * 0.9:
        rng = np.random.RandomState(42)
        for trial in range(5):
            if time.time() > start_time + max_time * 0.9:
                break
            order = list(range(best_U.shape[1]))
            rng.shuffle(order)
            U3, V3, W3, r3 = try_remove_with_order(best_U, best_V, best_W, order, start_time + max_time * 0.95)
            if U3 is not None and r3 < best_rank:
                best_U, best_V, best_W, best_rank = U3, V3, W3, r3
    
    return best_U, best_V, best_W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U,V,W decompose the matrix multiplication tensor."""
    nm, mp, pn = n * m, m * p, p * n
    expected = _expected_tensor(n, m, p)
    T = _compute_tensor(U, V, W)
    max_err = np.max(np.abs(T - expected))
    return max_err < tol, max_err


def _schoolbook(n, m, p):
    """Schoolbook decomposition as fallback."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def construct():
    start_time = time.time()
    deadline = 140.0
    
    n, m, p = DIMENSIONS
    
    U, V, W = _build_4x5x5()
    
    valid, err = _verify(U, V, W, n, m, p)
    if not valid:
        U, V, W = _schoolbook(n, m, p)
    
    if time.time() < deadline - 15:
        remaining = min(90, deadline - time.time() - 15)
        if remaining > 10:
            U, V, W = _try_remove_columns(U, V, W, n, m, p, max_time=remaining)
    
    # Final verification
    valid, err = _verify(U, V, W, n, m, p)
    if not valid:
        U, V, W = _schoolbook(n, m, p)
    
    U_list = [[float(x) for x in row] for row in U.tolist()]
    V_list = [[float(x) for x in row] for row in V.tolist()]
    W_list = [[float(x) for x in row] for row in W.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
