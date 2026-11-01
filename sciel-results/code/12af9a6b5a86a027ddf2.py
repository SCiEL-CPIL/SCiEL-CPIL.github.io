"""Rewritten program for matrix multiplication 2,2,15 tensor rank minimization."""
import time
import itertools
import numpy as np
from numpy import random as npr

DIMENSIONS = [2, 2, 15]
START_TIME = time.time()
TIME_LIMIT = 130  # keep a safety margin under the 150 s wall-clock


def _is_valid(U, V, W, tol=1e-9):
    """Check exactness of the decomposition on a set of test matrices."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]

    # Build the multiplication tensor T with indices (i, j, k)
    # T[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
    # For matrix multiplication, T[i, j, k] should be 1 if (i==0 and j==k) or (i==1 and j==k)
    # i.e., A[i, j] * B[j, k] = C[i, k]
    # So the tensor is: T[i, j, k] = 1 if i==j and k is free? No...
    # Actually: C[i, k] = sum_j A[i, j] B[j, k]
    # So the tensor T has T[i, j, k] = 1 for all (i, j, k) where the product A[i,j]*B[j,k] contributes to C[i,k]
    # That's all triples (i, j, k) - each contributes 1.
    # So T[i, j, k] = 1 for all i in [0,2), j in [0,2), k in [0,15)

    # Check on random matrices
    for trial in range(5):
        A = npr.rand(n, m)
        B = npr.rand(m, p)
        C_expected = A @ B

        # Compute C via decomposition
        # C[i, k] = sum_j A[i, j] * B[j, k] = sum_r (sum_i U[i,r]*A[i,?]) ... 
        # Actually: C[i,k] = sum_r U[i,r] * V[:,r] dot B ... 
        # Let's do it properly:
        # a_r = sum_i A[i,j] * U[i, r]  ... wait, U has indices (i, r) where i is row of A
        # But A is n x m = 2 x 2, so A[i, j] with i in {0,1}, j in {0,1}
        # U has shape (n*m, rank) = (4, rank). Index 0:A00, 1:A01, 2:A10, 3:A11
        # So U[i*m + j, r] corresponds to A[i,j]
        # V has shape (m*p, rank) = (30, rank). Index j*p + k corresponds to B[j,k]
        # W has shape (p*n, rank) = (30, rank). Index k*n + i corresponds to C[i,k]

        # a_r = sum_{i,j} A[i,j] * U[i*m+j, r]
        # b_r = sum_{j,k} B[j,k] * V[j*p+k, r]
        # c_r = a_r * b_r
        # C[i,k] = sum_r W[k*n+i, r] * c_r

        A_flat = A.flatten()  # [A00, A01, A10, A11]
        B_flat = B.flatten()  # [B00, B01, ..., B0_14, B10, B11, ..., B1_14]

        a = U.T @ A_flat  # (rank,)
        b = V.T @ B_flat  # (rank,)
        c = a * b         # (rank,)
        C_flat = W.T @ c  # (p*n,) = (30,)
        C_decomp = C_flat.reshape(p, n).T  # reshape to (n, p) = (2, 15)

        if np.max(np.abs(C_decomp - C_expected)) > tol:
            return False
    return True


def _rank_of_decomposition(U, V, W):
    """Return the rank (number of columns) of the decomposition."""
    return len(U[0]) if isinstance(U, list) else U.shape[1]


def _build_strassen_2x2x2():
    """Build Strassen decomposition for 2x2 * 2x2, returning U(4x7), V(4x7), W(4x7).
    Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))

    # A indices: 0:A00, 1:A01, 2:A10, 3:A11
    # B indices: 0:B00, 1:B01, 2:B10, 3:B11
    # C indices: 0:C00, 1:C01, 2:C10, 3:C11

    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C00 += M1, C11 += M1

    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1  # C10 += M2, C11 -= M2

    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1  # C01 += M3, C11 += M3

    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1  # C00 += M4, C10 += M4

    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1  # C00 -= M5, C01 += M5

    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C11 += M6

    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # C00 += M7

    return U, V, W


def _build_schoolbook_2x2x2():
    """Build schoolbook decomposition for 2x2 * 2x2, returning U(4x4), V(4x4), W(4x4)."""
    rank = 4
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))

    for r in range(4):
        i, j = r // 2, r % 2  # A[i,j]
        V[j, r] = 1
        # C[i,k] for each k: C[i,k] += A[i,j]*B[j,k]
        for k in range(2):
            W[k * 2 + i, r] = 1

    for i in range(2):
        for j in range(2):
            r = i * 2 + j
            U[i * 2 + j, r] = 1

    return U, V, W


def _build_2x2x1():
    """Build decomposition for 2x2 * 2x1 (single column), returning U(4x4), V(2x4), W(2x4)."""
    rank = 4
    U = np.zeros((4, rank))
    V = np.zeros((2, rank))
    W = np.zeros((2, rank))

    # r=0: A00 * B00 -> C00
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    # r=1: A01 * B10 -> C00
    U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
    # r=2: A10 * B00 -> C10
    U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
    # r=3: A11 * B10 -> C10
    U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1

    return U, V, W


def _block_decomposition(block_ranks, p=15, m=2, n=2):
    """
    Build a block decomposition for 2x2 * 2x15.
    block_ranks: list of ranks for each block of columns.
    Total rank = sum(block_ranks)
    
    For a block covering columns [k_start, k_end) of B and C:
    - U: 4 x rank_block
    - V: (m * block_size) x rank_block, but we need to map to global indices
    - W: (block_size * n) x rank_block, mapped to global indices
    
    Returns (U, V, W) as lists of lists.
    """
    total_rank = sum(block_ranks)
    U = [[0.0] * total_rank for _ in range(n * m)]  # 4 x total_rank
    V = [[0.0] * total_rank for _ in range(m * p)]  # 30 x total_rank
    W = [[0.0] * total_rank for _ in range(p * n)]  # 30 x total_rank

    # We need to figure out block boundaries
    # 15 columns: we can split into groups
    # Let's support flexible splits
    # block_ranks has len = number of blocks
    # We need to assign column ranges to blocks

    # Default: pairs of 2 + one single
    if len(block_ranks) == 8:
        # 7 blocks of 2 columns + 1 block of 1 column
        col_ranges = [(2*i, 2*i+2) for i in range(7)] + [(14, 15)]
    elif len(block_ranks) == 1:
        col_ranges = [(0, 15)]
    else:
        # Try to make even blocks
        num_blocks = len(block_ranks)
        base_size = p // num_blocks
        remainder = p % num_blocks
        col_ranges = []
        start = 0
        for b in range(num_blocks):
            size = base_size + (1 if b < remainder else 0)
            col_ranges.append((start, start + size))
            start += size

    offset = 0
    for bi, (rank_b, (k0, k1)) in enumerate(zip(block_ranks, col_ranges)):
        block_size = k1 - k0
        if block_size == 2:
            U_b, V_b, W_b = _build_strassen_2x2x2()
        elif block_size == 1:
            U_b, V_b, W_b = _build_2x2x1()
        else:
            # Fallback: schoolbook for this block
            # For a 2x2 * 2xb block, schoolbook uses 4*b rank
            U_b = np.zeros((4, 4 * block_size))
            V_b = np.zeros((2 * block_size, 4 * block_size))
            W_b = np.zeros((block_size * 2, 4 * block_size))
            for kk in range(block_size):
                for r in range(4):
                    i, j = r // 2, r % 2
                    U_b[i * 2 + j, kk * 4 + r] = 1
                    V_b[j * block_size + kk, kk * 4 + r] = 1
                    W_b[kk * 2 + i, kk * 4 + r] = 1
            rank_b = 4 * block_size

        rb = U_b.shape[1]
        if rb != rank_b:
            # Mismatch; pad or truncate
            if rb < rank_b:
                U_b = np.pad(U_b, ((0, 0), (0, rank_b - rb)))
                V_b = np.pad(V_b, ((0, 0), (0, rank_b - rb)))
                W_b = np.pad(W_b, ((0, 0), (0, rank_b - rb)))
            else:
                U_b = U_b[:, :rank_b]
                V_b = V_b[:, :rank_b]
                W_b = W_b[:, :rank_b]

        # Merge U
        for i in range(4):
            for r in range(rank_b):
                U[i][offset + r] += U_b[i, r]

        # Merge V: local index j*block_size + kk -> global j*p + (k0+kk)
        for jj in range(2):
            for kk in range(block_size):
                local_idx = jj * block_size + kk
                global_idx = jj * p + (k0 + kk)
                for r in range(rank_b):
                    V[global_idx][offset + r] += V_b[local_idx, r]

        # Merge W: local index kk*2 + i -> global (k0+kk)*2 + i
        for kk in range(block_size):
            for ii in range(2):
                local_idx = kk * 2 + ii
                global_idx = (k0 + kk) * 2 + ii
                for r in range(rank_b):
                    W[global_idx][offset + r] += W_b[local_idx, r]

        offset += rank_b

    return U, V, W


def _try_find_223_rank10():
    """
    Try to find a rank-10 decomposition for 2x2 * 2x3.
    If found, we could use 5 blocks of rank 10 = rank 50 total.
    Uses randomized search with gradient descent on the factor matrices.
    """
    n, m, p = 2, 2, 3
    target_rank = 10

    # Target tensor T[i, j, k] = 1 for all i, j, k
    # T has shape (2, 2, 3) = 12 entries, all 1.

    # We want U (4xR), V (6xR), W (6xR) such that
    # sum_r U[i,r] V[j,r] W[k,r] = 1 for all (i,j,k)

    # Initialize with Strassen (rank 7) for first 2 cols + schoolbook (rank 4) for 3rd col = rank 11
    # Then try to optimize to rank 10

    # Actually, let's try a direct optimization approach
    # Start with a rank-11 valid decomposition and try to find a rank-10 one

    # Initialize factors
    np.random.seed(42)
    U = np.random.randn(4, target_rank)
    V = np.random.randn(6, target_rank)
    W = np.random.randn(6, target_rank)

    # Target: T[i,j,k] = 1 for all
    # Loss = sum_{i,j,k} (sum_r U[i,r]V[j,r]W[k,r] - 1)^2

    best_loss = float('inf')
    best_params = (U.copy(), V.copy(), W.copy())

    lr = 0.01
    for step in range(5000):
        # Compute loss and gradients
        # T_decomp[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
        T_decomp = np.zeros((2, 2, 3))
        for r in range(target_rank):
            T_decomp += np.outer(np.outer(U[:, r], V[:, r]), W[:, r]).reshape(2, 2, 3)

        loss = np.sum((T_decomp - 1.0) ** 2)

        if loss < best_loss:
            best_loss = loss
            best_params = (U.copy(), V.copy(), W.copy())
            if loss < 1e-10:
                return best_params

        # Gradients
        # dL/dU[i,r] = sum_{j,k} 2*(T_decomp[i,j,k]-1) * V[j,r] * W[k,r]
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)

        err = T_decomp - 1.0  # (2,2,3)

        for r in range(target_rank):
            # dU[i,r] = sum_{j,k} 2*err[i,j,k] * V[j,r] * W[k,r]
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        dU[i, r] += 2.0 * err[i, j, k] * V[j, r] * W[k, r]
                        dV[j, r] += 2.0 * err[i, j, k] * U[i, r] * W[k, r]
                        dW[k, r] += 2.0 * err[i, j, k] * U[i, r] * V[j, r]

        U -= lr * dU
        V -= lr * dV
        W -= lr * dW

        if time.time() - START_TIME > TIME_LIMIT - 10:
            break

    if best_loss < 1e-9:
        return best_params
    return None


def _try_find_225_rank16():
    """
    Try to find a rank-16 decomposition for 2x2 * 2x5.
    If found, 3 blocks of rank 16 = rank 48 total.
    Uses a more structured initialization.
    """
    n, m, p = 2, 2, 5
    target_rank = 16

    np.random.seed(123)

    # Initialize with a valid higher-rank decomposition and perturb
    # 2x2 * 2x5: schoolbook would be 4*5=20 rank
    # Strassen on first 2 (rank 7) + schoolbook on last 3 (rank 12) = rank 19
    # Let's start from that and optimize down to 16

    # Build initial valid decomposition
    U_init = np.zeros((4, 19))
    V_init = np.zeros((10, 19))
    W_init = np.zeros((10, 19))

    # Strassen for cols 0,1
    Us, Vs, Ws = _build_strassen_2x2x2()
    U_init[:, :7] = Us
    for j in range(2):
        for k in range(2):
            V_init[j * 5 + k, k] = Vs[j * 2 + k, k]
    for k in range(2):
        for i in range(2):
            W_init[k * 2 + i, i] = Ws[k * 2 + i, i]

    # Schoolbook for cols 2,3,4
    for kk in range(3):
        for r in range(4):
            i, jj = r // 2, r % 2
            U_init[i * 2 + jj, 7 + kk * 4 + r] = 1
            V_init[jj * 5 + (2 + kk), 7 + kk * 4 + r] = 1
            W_init[(2 + kk) * 2 + i, 7 + kk * 4 + r] = 1

    # Now try to find a rank-16 solution
    U = U_init[:, :16] + np.random.randn(4, 16) * 0.1
    V = V_init[:, :16] + np.random.randn(10, 16) * 0.1
    W = W_init[:, :16] + np.random.randn(10, 16) * 0.1

    best_loss = float('inf')
    best_params = (U.copy(), V.copy(), W.copy())

    lr = 0.005
    for step in range(3000):
        T_decomp = np.zeros((2, 2, 5))
        for r in range(target_rank):
            T_decomp += np.outer(np.outer(U[:, r], V[:, r]), W[:, r]).reshape(2, 2, 5)

        loss = np.sum((T_decomp - 1.0) ** 2)

        if loss < best_loss:
            best_loss = loss
            best_params = (U.copy(), V.copy(), W.copy())
            if loss < 1e-10:
                return best_params

        err = T_decomp - 1.0
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)

        for r in range(target_rank):
            for i in range(2):
                for j in range(2):
                    for k in range(5):
                        dU[i, r] += 2.0 * err[i, j, k] * V[j, r] * W[k, r]
                        dV[j, r] += 2.0 * err[i, j, k] * U[i, r] * W[k, r]
                        dW[k, r] += 2.0 * err[i, j, k] * U[i, r] * V[j, r]

        U -= lr * dU
        V -= lr * dV
        W -= lr * dW

        if time.time() - START_TIME > TIME_LIMIT - 10:
            break

    if best_loss < 1e-9:
        return best_params
    return None


def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15

    # Best known: rank 53 (7 Strassen blocks of rank 7 + 1 schoolbook block of rank 4)
    U, V, W = _block_decomposition([7] * 7 + [4], p=p, m=m, n=n)
    best_rank = 53

    # Try to find better decompositions
    # Attempt 1: rank 50 via 5 blocks of 2x2*2x3 rank 10
    if time.time() - START_TIME < TIME_LIMIT - 20:
        result_223 = _try_find_223_rank10()
        if result_223 is not None:
            U223, V223, W223 = result_223
            # Build full decomposition using 5 blocks of rank 10
            U_full = [[0.0] * 50 for _ in range(4)]
            V_full = [[0.0] * 50 for _ in range(30)]
            W_full = [[0.0] * 50 for _ in range(30)]

            for bi in range(5):
                k0 = 3 * bi
                offset = 10 * bi
                for i in range(4):
                    for r in range(10):
                        U_full[i][offset + r] += U223[i, r]
                for j in range(2):
                    for k in range(3):
                        local_v = j * 3 + k
                        global_v = j * 15 + (k0 + k)
                        for r in range(10):
                            V_full[global_v][offset + r] += V223[local_v, r]
                for k in range(3):
                    for i in range(2):
                        local_w = k * 2 + i
                        global_w = (k0 + k) * 2 + i
                        for r in range(10):
                            W_full[global_w][offset + r] += W223[local_w, r]

            if _is_valid(U_full, V_full, W_full):
                return {"U": U_full, "V": V_full, "W": W_full}

    # Attempt 2: rank 48 via 3 blocks of 2x2*2x5 rank 16
    if time.time() - START_TIME < TIME_LIMIT - 15:
        result_225 = _try_find_225_rank16()
        if result_225 is not None:
            U225, V225, W225 = result_225
            U_full = [[0.0] * 48 for _ in range(4)]
            V_full = [[0.0] * 48 for _ in range(30)]
            W_full = [[0.0] * 48 for _ in range(30)]

            for bi in range(3):
                k0 = 5 * bi
                offset = 16 * bi
                for i in range(4):
                    for r in range(16):
                        U_full[i][offset + r] += U225[i, r]
                for j in range(2):
                    for k in range(5):
                        local_v = j * 5 + k
                        global_v = j * 15 + (k0 + k)
                        for r in range(16):
                            V_full[global_v][offset + r] += V225[local_v, r]
                for k in range(5):
                    for i in range(2):
                        local_w = k * 2 + i
                        global_w = (k0 + k) * 2 + i
                        for r in range(16):
                            W_full[global_w][offset + r] += W225[local_w, r]

            if _is_valid(U_full, V_full, W_full):
                return {"U": U_full, "V": V_full, "W": W_full}

    # Attempt 3: Try mixed block sizes
    # 2x2*2x4 (rank 14?) + 2x2*2x4 (rank 14?) + 2x2*2x4 (rank 14?) + 2x2*2x3 (rank 10?)
    # = 14+14+14+10 = 52
    # Or 2x2*2x6 + 2x2*2x6 + 2x2*2x3
    # Just try a few combinations

    if time.time() - START_TIME < TIME_LIMIT - 10:
        # Try rank 52: [7,7,7,7,7,7,7,4] is 53. Try [14,14,14,10] = 52
        for ranks in [[14, 14, 14, 10], [7, 7, 7, 7, 7, 7, 12], [10, 10, 10, 10, 13]]:
            try:
                U_t, V_t, W_t = _block_decomposition(ranks, p=p, m=m, n=n)
                if _is_valid(U_t, V_t, W_t):
                    tr = _rank_of_decomposition(U_t, V_t, W_t)
                    if tr < best_rank:
                        return {"U": U_t, "V": V_t, "W": W_t}
            except Exception:
                pass

    # Return the best known valid decomposition
    if _is_valid(U, V, W):
        return {"U": U, "V": V, "W": W}

    # Fallback: build a simple schoolbook
    U_fb = [[0.0] * 60 for _ in range(4)]
    V_fb = [[0.0] * 60 for _ in range(30)]
    W_fb = [[0.0] * 60 for _ in range(30)]
    r = 0
    for i in range(2):
        for j in range(2):
            for k in range(15):
                U_fb[i * 2 + j][r] = 1.0
                V_fb[j * 15 + k][r] = 1.0
                W_fb[k * 2 + i][r] = 1.0
                r += 1

    return {"U": U_fb, "V": V_fb, "W": W_fb}
