"""Program for matrix multiplication 3,5,6 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W, rank


def _strassen_222():
    """Return the 7-rank Strassen decomposition for 2x2x2."""
    # The 7 bilinear forms of Strassen:
    # M1 = (A11-A22)(B21+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B11-B22)
    # M4 = A22(B22-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # U: A-index (2x2) -> 7 coefficients
    # V: B-index (2x2) -> 7 coefficients
    # W: C-index (2x2) -> 7 coefficients
    #
    # A indices: 00, 01, 10, 11  (row-major: A11=0, A12=1, A21=2, A22=3)
    # B indices: 00, 01, 10, 11  (B11=0, B12=1, B21=2, B22=3)
    # C indices: 00, 01, 10, 11  (C11=0, C12=1, C21=2, C22=3)
    
    U = np.array([
        [1,  0,  1,  0,  1, -1,  0],  # A11
        [0,  0,  0,  0,  1,  0, -1],  # A12
        [0,  1,  0,  0,  0,  1,  0],  # A21
        [-1, 1,  0,  1,  0,  0, -1],  # A22
    ], dtype=float)
    
    V = np.array([
        [0,  1,  1, -1,  0,  1,  0],  # B11
        [0,  0,  0,  0,  0,  1,  0],  # B12
        [1,  0, -1,  0,  0,  0,  1],  # B21
        [1,  0,  0,  1,  1,  0,  1],  # B22
    ], dtype=float)
    
    W = np.array([
        [1,  0,  0,  1, -1,  0,  1],  # C11
        [0,  0,  1,  0,  1,  0,  0],  # C12
        [0,  1,  0,  1,  0,  0,  0],  # C21
        [1, -1, 1,  0,  0,  1,  0],  # C22
    ], dtype=float)
    
    return U, V, W


def _embed_block(U_sub, V_sub, W_sub, n, m, p, i0, j0, k0, ni, mj, pk):
    """Embed a sub-problem decomposition into the full problem.
    
    U_sub: (ni*mj, r)
    V_sub: (mj*pk, r)
    W_sub: (pk*ni, r)
    Maps to positions in the full (n*m), (m*p), (p*n) spaces.
    """
    r = U_sub.shape[1]
    U_full = np.zeros((n * m, r))
    V_full = np.zeros((m * p, r))
    W_full = np.zeros((p * n, r))
    
    for a in range(ni):
        for b in range(mj):
            src_idx = a * mj + b
            dst_idx = (i0 + a) * m + (j0 + b)
            U_full[dst_idx, :] = U_sub[src_idx, :]
    
    for b in range(mj):
        for c in range(pk):
            src_idx = b * pk + c
            dst_idx = (j0 + b) * p + (k0 + c)
            V_full[dst_idx, :] = V_sub[src_idx, :]
    
    for c in range(pk):
        for a in range(ni):
            src_idx = c * ni + a
            dst_idx = (k0 + c) * n + (i0 + a)
            W_full[dst_idx, :] = W_sub[src_idx, :]
    
    return U_full, V_full, W_full


def _verify(U, V, W, n, m, p, atol=1e-10):
    """Verify that the decomposition is exact."""
    rank = U.shape[1]
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    S = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
    return np.allclose(S, 1.0, atol=atol)


def _construct_recursive(n, m, p, memo={}):
    """Construct a low-rank decomposition using recursive block structure.
    
    Returns (U, V, W, rank) where U has shape (n*m, rank), etc.
    """
    key = (n, m, p)
    if key in memo:
        return memo[key]
    
    # Base cases
    if n == 1 and m == 1 and p == 1:
        U = np.ones((1, 1))
        V = np.ones((1, 1))
        W = np.ones((1, 1))
        result = (U, V, W, 1)
        memo[key] = result
        return result
    
    if n == 1:
        result = _schoolbook(1, m, p)
        memo[key] = result
        return result
    
    if m == 1:
        result = _schoolbook(n, 1, p)
        memo[key] = result
        return result
    
    if p == 1:
        result = _schoolbook(n, m, 1)
        memo[key] = result
        return result
    
    if n == 2 and m == 2 and p == 2:
        U, V, W = _strassen_222()
        result = (U, V, W, 7)
        memo[key] = result
        return result
    
    # Try all possible splits
    best_rank = n * m * p
    best_result = None
    
    # Try splitting n
    for n1 in range(1, n):
        n2 = n - n1
        for m1 in range(1, m):
            m2 = m - m1
            for p1 in range(1, p):
                p2 = p - p1
                # Split into 4 blocks
                # Block (1,1): n1 x m1 x p1
                # Block (1,2): n1 x m2 x p1
                # Block (2,1): n2 x m1 x p2
                # Block (2,2): n2 x m2 x p2
                # Wait, this isn't quite right. The standard approach is:
                # C = A*B where A is n x m, B is m x p
                # Split A into rows (n1, n2) and B into cols (p1, p2)
                # Then C = A_top * B_left + A_top * B_right + A_bot * B_left + A_bot * B_right
                # But we can also split columns of A and rows of B.
                
                # Let's use the standard Kronecker/block approach:
                # A = [A11 A12; A21 A22] (n1+m2) x (m1+m2)
                # B = [B11 B21; B12 B22]... 
                # Actually, let's think of it as:
                # A: n x m, split rows into n1, n2 and cols into m1, m2
                # B: m x p, split rows into m1, m2 and cols into p1, p2
                # C: n x p
                # C[i,k] = sum_j A[i,j] B[j,k]
                # C_top_left = A_top * B_left (where B_left is first p1 cols)
                # Actually the cleanest is:
                # C = [A1 A2] * B where A1 is n x m1, A2 is n x m2
                # Or C = A * [B1 B2]
                # Or C = [A1; A2] * [B1; B2]
                
                # Let's do: split A's columns into m1, m2 and B's rows into m1, m2
                # C = A[:, :m1] @ B[:m1, :] + A[:, m1:] @ B[m1:, :]
                # First term: n x p from n x m1 times m1 x p -> rank n*m1*p
                # Second term: n x p from n x m2 times m2 x p -> rank n*m2*p
                # Total: n*p*(m1+m2) = n*m*p. That's just schoolbook.
                
                # Better: use the 4-block approach
                # C = A11*B11 + A12*B21 + A21*B12 + A22*B22... no that's not right either.
                
                # Correct 4-block:
                # A = [[A11, A12], [A21, A22]] where A11 is n1 x m1, A12 is n1 x m2, etc.
                # B = [[B11, B12], [B21, B22]] where B11 is m1 x p1, B12 is m1 x p2, etc.
                # C = [[C11, C12], [C21, C22]]
                # C11 = A11*B11 + A12*B21
                # C12 = A11*B12 + A12*B22
                # C21 = A21*B11 + A22*B21
                # C22 = A21*B12 + A22*B22
                #
                # So we need:
                # A11*B11: n1 x m1 x p1
                # A12*B21: n1 x m2 x p1
                # A11*B12: n1 x m1 x p2
                # A12*B22: n1 x m2 x p2
                # A21*B11: n2 x m1 x p1
                # A22*B21: n2 x m2 x p1
                # A21*B12: n2 x m1 x p2
                # A22*B22: n2 x m2 x p2
                #
                # Total rank = sum of ranks of all 8 sub-problems
                
                sub_problems = [
                    (n1, m1, p1, 0, 0, 0),
                    (n1, m2, p1, 0, m1, 0),
                    (n1, m1, p2, 0, 0, p1),
                    (n1, m2, p2, 0, m1, p1),
                    (n2, m1, p1, n1, 0, 0),
                    (n2, m2, p1, n1, m1, 0),
                    (n2, m1, p2, n1, 0, p1),
                    (n2, m2, p2, n1, m1, p1),
                ]
                
                total_rank = 0
                sub_results = []
                for (sn, sm, sp, si, sj, sk) in sub_problems:
                    res = _construct_recursive(sn, sm, sp, memo)
                    total_rank += res[3]
                    sub_results.append((res, sn, sm, sp, si, sj, sk))
                
                if total_rank < best_rank:
                    best_rank = total_rank
                    U_full = np.zeros((n * m, total_rank))
                    V_full = np.zeros((m * p, total_rank))
                    W_full = np.zeros((p * n, total_rank))
                    offset = 0
                    for (res, sn, sm, sp, si, sj, sk) in sub_results:
                        U_sub, V_sub, W_sub, r_sub = res
                        U_e, V_e, W_e = _embed_block(U_sub, V_sub, W_sub, n, m, p, si, sj, sk, sn, sm, sp)
                        U_full[:, offset:offset + r_sub] = U_e
                        V_full[:, offset:offset + r_sub] = V_e
                        W_full[:, offset:offset + r_sub] = W_e
                        offset += r_sub
                    best_result = (U_full, V_full, W_full, total_rank)
    
    if best_result is None:
        best_result = _schoolbook(n, m, p)
    
    memo[key] = best_result
    return best_result


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Try recursive construction
    U, V, W, rank = _construct_recursive(n, m, p)
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fall back to schoolbook
        U, V, W, rank = _schoolbook(n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
