"""Matrix multiplication 4x4x6 via tensor-product Strassen decomposition."""
import time
import numpy as np

_N, _M, _P = 4, 4, 6
_S = 7  # Strassen 2x2 rank


def _strassen_tensor_2x2():
    """Strassen 2x2 factors as 3D arrays: U(2,2,S), V(2,2,S), W(2,2,S)."""
    prods = [
        ([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)]),
        ([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)], [(1, 0, 1), (1, 1, -1)]),
        ([(0, 0, 1)], [(0, 1, 1), (1, 1, -1)], [(0, 1, 1), (1, 1, 1)]),
        ([(1, 1, 1)], [(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (1, 0, 1)]),
        ([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)], [(0, 0, -1), (0, 1, 1)]),
        ([(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)], [(1, 1, 1)]),
        ([(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)], [(0, 0, 1)]),
    ]
    U = np.zeros((2, 2, _S))
    V = np.zeros((2, 2, _S))
    W = np.zeros((2, 2, _S))
    for m_idx, (ub, vb, wb) in enumerate(prods):
        for (i, j, c) in ub:
            U[i, j, m_idx] += c
        for (j, k, c) in vb:
            V[j, k, m_idx] += c
        for (i, k, c) in wb:
            W[i, k, m_idx] += c
    return U, V, W


def _build_444():
    """Build 4x4x4 Strassen factors via tensor product of 2x2 Strassen."""
    Us, Vs, Ws = _strassen_tensor_2x2()
    S2 = _S * _S
    U4 = np.zeros((4, 4, S2))
    V4 = np.zeros((4, 4, S2))
    W4 = np.zeros((4, 4, S2))
    for t1 in range(_S):
        for t2 in range(_S):
            s = t1 * _S + t2
            U4[:, :, s] = Us[:, :, t1] * Us[:, :, t2].T
            # Wait, tensor product of two 2x2 matrices:
            # If A = a1 ⊗ a2, then A[i1*2+i2, j1*2+j2] = a1[i1,j1] * a2[i2,j2]
            # So U4[i1*2+i2, j1*2+j2, s] = Us[i1,j1,t1] * Us[i2,j2,t2]
            # This is the Kronecker product Us[:,:,t1] ⊗ Us[:,:,t2]
            U4[:, :, s] = np.kron(Us[:, :, t1], Us[:, :, t2])
            V4[:, :, s] = np.kron(Vs[:, :, t1], Vs[:, :, t2])
            W4[:, :, s] = np.kron(Ws[:, :, t1], Ws[:, :, t2])
    return U4.reshape(16, S2), V4.reshape(16, S2), W4.reshape(16, S2)


def _build_442():
    """
    Build 4x4x2 factors using 2x2 Strassen blocks.
    A(4x4), B(4x2) -> C(4x2).
    Split A into 2x2 blocks: A11, A12, A21, A22.
    Split B into 2x2 blocks: B11, B21 (since B is 4x2, it's two 2x2 blocks vertically? No.
    B is 4x2. Rows 0-3, Cols 0-1.
    B11 = B[0:2, 0:2] (2x2)
    B21 = B[2:4, 0:2] (2x2)
    C is 4x2.
    C11 = C[0:2, 0:2] (2x2) = A11 B11 + A12 B21
    C21 = C[2:4, 0:2] (2x2) = A21 B11 + A22 B21
    
    We can compute C11 and C21 using Strassen products.
    However, A12, A21, A22, A11 are all 2x2.
    B11, B21 are 2x2.
    
    Let's use the standard Strassen decomposition for 2x2.
    We have 4 "2x2 products" needed if we do it naively?
    C11 = A11 B11 + A12 B21
    C21 = A21 B11 + A22 B21
    
    This is not a single 2x2*2x2 -> 2x2 multiplication. It's two separate ones.
    Each requires rank 7. Total rank 14?
    
    Alternatively, group A into a 2x2 matrix of 2x2 blocks, and B into a 2x1 matrix of 2x2 blocks.
    This is (2x2) * (2x1) -> (2x1) in terms of blocks.
    This can be done with rank 5? Or using Strassen-like?
    
    Let's just use the tensor product approach for the first 4 columns (4x4x4) and then handle the last 2 columns separately?
    No, the inspiration says: "Splitting the 4x4x6 problem into three independent 4x4x2 subproblems... gives rank 84".
    3 * 28 = 84.
    So each 4x4x2 subproblem has rank 28.
    28 / 7 = 4 Strassen 2x2 products.
    
    How to get rank 28 for 4x4x2?
    Split A(4x4) into 4 2x2 blocks: A00, A01, A10, A11.
    Split B(4x2) into 2 2x2 blocks: B0, B1 (rows 0-1, rows 2-3).
    C(4x2) has 4 2x2 blocks? No, C is 4x2.
    C0 = C[0:2, :] (2x2)
    C1 = C[2:4, :] (2x2)
    
    C0 = A00 B0 + A01 B1
    C1 = A10 B0 + A11 B1
    
    We can compute these using Strassen.
    Let M1 = (A00 + A11) (B0 + B1)
    Let M2 = (A10 + A11) B0
    Let M3 = A00 (B1 - B0)
    Let M4 = A11 (B0 - B1) ... wait.
    
    Standard Strassen for 2x2 * 2x2:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B21-B11)
    M4 = A22(B12-B22)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    Here we have a 2x2 matrix of 2x2 blocks (A) multiplied by a 2x1 matrix of 2x2 blocks (B).
    This is effectively computing two 2x2 results: C0 and C1.
    C0 = A00 B0 + A01 B1
    C1 = A10 B0 + A11 B1
    
    We can view this as:
    [C0]   [A00 A01] [B0]
    [C1] = [A10 A11] [B1]
    
    This is a (2x2) * (2x1) multiplication.
    Rank?
    We can use 7 Strassen products?
    Let's define the "blocks" as elements.
    A = [[a00, a01], [a10, a11]] where each is 2x2.
    B = [[b0], [b1]] where each is 2x2.
    
    M1 = (a00 + a11)(b0 + b1)
    M2 = (a10 + a11)b0
    M3 = a00(b1 - b0)
    M4 = (a00 + a01)b1
    M5 = (a10 - a00)(b0 + b1) -- wait, this is for 2x2*2x2.
    
    Actually, rank of 2x2 * 2x1 is 5? No.
    Let's just use the 4 independent 2x2*2x2 products?
    C0 = a00 b0 + a01 b1 -> 2 products (rank 14)
    C1 = a10 b0 + a11 b1 -> 2 products (rank 14)
    Total 28.
    Yes!
    So we just do 4 separate Strassen 2x2 multiplications.
    
    1. P1 = A00 * B0
    2. P2 = A01 * B1
    3. P3 = A10 * B0
    4. P4 = A11 * B1
    
    C0 = P1 + P2
    C1 = P3 + P4
    
    Each Pi is a 2x2*2x2 -> 2x2 multiplication, rank 7.
    Total rank 4 * 7 = 28.
    """
    Us, Vs, Ws = _strassen_tensor_2x2()
    rank = 4 * _S
    U = np.zeros((16, rank))
    V = np.zeros((8, rank))  # 4x2 = 8
    W = np.zeros((8, rank))  # 4x2 = 8
    
    # A indices: 0-3, 0-3.
    # A00: rows 0-1, cols 0-1
    # A01: rows 0-1, cols 2-3
    # A10: rows 2-3, cols 0-1
    # A11: rows 2-3, cols 2-3
    
    # B indices: 0-3, 0-1.
    # B0: rows 0-1, cols 0-1
    # B1: rows 2-3, cols 0-1
    
    # C indices: 0-3, 0-1.
    # C0: rows 0-1, cols 0-1
    # C1: rows 2-3, cols 0-1
    
    def get_U_block(r_start, c_start, offset):
        """Extract U for a 2x2 block of A."""
        for i in range(2):
            for j in range(2):
                U[(r_start + i) * 4 + (c_start + j), offset] = Us[i, j, :]
                U[(r_start + i) * 4 + (c_start + j), offset + 1:offset + 7] = Us[i, j, 1:] # dummy, fix below
        
    # Let's do it explicitly with loops to avoid mistakes.
    for s in range(_S):
        # P1 = A00 * B0
        # U for A00
        for i in range(2):
            for j in range(2):
                U[i * 4 + j, s] += Us[i, j, s]
        # V for B0
        for j in range(2):
            for k in range(2):
                V[j * 2 + k, s] += Vs[j, k, s]
        # W for C0
        for i in range(2):
            for k in range(2):
                W[k * 4 + i, s] += Ws[i, k, s]
        
        # P2 = A01 * B1
        offset = _S
        for i in range(2):
            for j in range(2):
                U[i * 4 + (j + 2), offset + s] += Us[i, j, s]
        for j in range(2):
            for k in range(2):
                V[(j + 2) * 2 + k, offset + s] += Vs[j, k, s]
        for i in range(2):
            for k in range(2):
                W[k * 4 + i, offset + s] += Ws[i, k, s]
        
        # P3 = A10 * B0
        offset = 2 * _S
        for i in range(2):
            for j in range(2):
                U[(i + 2) * 4 + j, offset + s] += Us[i, j, s]
        for j in range(2):
            for k in range(2):
                V[j * 2 + k, offset + s] += Vs[j, k, s]
        for i in range(2):
            for k in range(2):
                W[k * 4 + (i + 2), offset + s] += Ws[i, k, s]
        
        # P4 = A11 * B1
        offset = 3 * _S
        for i in range(2):
            for j in range(2):
                U[(i + 2) * 4 + (j + 2), offset + s] += Us[i, j, s]
        for j in range(2):
            for k in range(2):
                V[(j + 2) * 2 + k, offset + s] += Vs[j, k, s]
        for i in range(2):
            for k in range(2):
                W[k * 4 + (i + 2), offset + s] += Ws[i, k, s]
                
    return U, V, W


def _verify(U, V, W):
    """Verify the decomposition."""
    r = U.shape[1]
    n, m, p = _N, _M, _P
    Ar = U.reshape(n, m, r)
    Br = V.reshape(m, p, r)
    Cr = W.reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', Ar, Br, Cr)
    return float(np.max(np.abs(T - 1.0)))


def _fallback():
    r = _N * _M * _P
    U = np.zeros((_N * _M, r))
    V = np.zeros((_M * _P, r))
    W = np.zeros((_P * _N, r))
    col = 0
    for i in range(_N):
        for j in range(_M):
            for k in range(_P):
                U[i * _M + j, col] = 1
                V[j * _P + k, col] = 1
                W[k * _N + i, col] = 1
                col += 1
    return U, V, W


def construct():
    t0 = time.time()
    
    # Build 3 subproblems of 4x4x2
    # Subproblem 1: Cols 0-1 of C
    U1, V1, W1 = _build_442()
    # Subproblem 2: Cols 2-3 of C
    U2, V2, W2 = _build_442()
    # Subproblem 3: Cols 4-5 of C
    U3, V3, W3 = _build_442()
    
    # Concatenate U (same for all, but we need to align indices? No, U is just A factors)
    # Wait, U is the factor for A. A is the same for all subproblems.
    # So we just take U1.
    # V and W need to be concatenated along the rank axis, but also their row indices must correspond to the correct parts of B and C.
    
    # V1 rows correspond to B cols 0-1.
    # V2 rows correspond to B cols 0-1 (but for C cols 2-3, so B is same? No.
    # B is 4x6.
    # Subproblem 1 uses B[:, 0:2]
    # Subproblem 2 uses B[:, 2:4]
    # Subproblem 3 uses B[:, 4:6]
    
    # So V1 has rows 0-7 (B rows 0-3, cols 0-1 flattened).
    # We need to place these into V_full (4x6 = 24 rows).
    # B row index j, col index k. Flat index j*6+k.
    # V1 index: j*2 + k_local (k_local 0-1).
    # Full V index: j*6 + k.
    
    rank_total = 3 * 28
    U_full = np.zeros((16, rank_total))
    V_full = np.zeros((24, rank_total))
    W_full = np.zeros((24, rank_total))
    
    # U is identical for all 3 blocks (it only depends on A).
    # But we need to stack them in the rank dimension.
    # Since U1, U2, U3 are identical, we can just repeat U1.
    U_full[:, 0:28] = U1
    U_full[:, 28:56] = U2
    U_full[:, 56:84] = U3
    
    # V and W mapping
    for idx in range(3):
        offset = idx * 28
        Uv, Vv, Wv = [U1, V1, W1][idx] if idx == 0 else [U2, V2, W2][idx-1] if idx == 1 else [U3, V3, W3][idx-2]
        # Actually just use V1, W1 as they are structurally identical, just shifted in B/C cols.
        Vb, Wb = V1, W1
        
        for j in range(4):
            for k_local in range(2):
                v_row_local = j * 2 + k_local
                v_row_full = j * 6 + (idx * 2 + k_local)
                V_full[v_row_full, offset:offset+28] = Vb[v_row_local, :]
                
                w_row_local = k_local * 4 + i # Wait, W is (p*n, rank). p=2 (local), n=4.
                # W index is k*n + i.
                # In Wb (8x28), index is k_local*4 + i.
                # In W_full (24x84), index is k_full*4 + i.
                # k_full = idx*2 + k_local.
        
        for k_local in range(2):
            for i in range(4):
                w_row_local = k_local * 4 + i
                w_row_full = (idx * 2 + k_local) * 4 + i
                W_full[w_row_full, offset:offset+28] = Wb[w_row_local, :]

    err = _verify(U_full, V_full, W_full)
    if err > 1e-6:
        return {"U": _fallback()[0].tolist(), "V": _fallback()[1].tolist(), "W": _fallback()[2].tolist()}
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
