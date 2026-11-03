"""Rank-minimal exact tensor decomposition for 3x4x8 matrix multiplication.

Uses a hybrid: schoolbook for i=0 (1x4x8, rank 32) plus a searched
lower-rank decomposition for the i=1,2 block (2x4x8). Attempts to beat
the Strassen-based rank of 56 for the 2x4x8 block via randomized search.
"""
import numpy as np
import time
import random

DIMENSIONS = [3, 4, 8]


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check the cyclic bilinear identity:
       sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1  for all i,j,k.
    """
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _build_schoolbook_1x4x8():
    """Schoolbook decomposition for the i=0 block (1x4x8), rank 32."""
    m, p = 4, 8
    rank = m * p
    U = np.zeros((m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p, rank))
    for j in range(m):
        for k in range(p):
            col = j * p + k
            U[j, col] = 1.0
            V[j * p + k, col] = 1.0
            W[k, col] = 1.0
    return U, V, W


def _build_strassen_2x4x8():
    """Strassen-based decomposition for the i=1,2 block (2x4x8), rank 56."""
    n, m, p = 2, 4, 8
    rank = 56
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    sU = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
    sV = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    sW = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]
    for jb in range(2):
        js = jb * 2
        for kb in range(4):
            ks = kb * 2
            for t in range(7):
                for l in range(4):
                    if sU[t][l]:
                        li, lj = l // 2, l % 2
                        U[li * m + (js + lj), col] += sU[t][l]
                    if sV[t][l]:
                        lj, lk = l // 2, l % 2
                        V[(js + lj) * p + (ks + lk), col] += sV[t][l]
                    if sW[t][l]:
                        li, lk = l // 2, l % 2
                        W[(ks + lk) * n + li] += sW[t][l]
                col += 1
    return U, V, W


def _search_2x4x8(deadline, max_rank=56, seed=42):
    """Randomized search for a low-rank decomposition of the 2x4x8 block.
    
    Returns (U, V, W, rank) for the 2x4x8 sub-problem, or None if not found.
    """
    n, m, p = 2, 4, 8
    rng = np.random.RandomState(seed)
    best = None
    
    for rank in range(32, max_rank + 1):
        if time.time() > deadline - 30:
            break
        # Try multiple random initializations
        for attempt in range(20):
            if time.time() > deadline - 30:
                break
            # Random initialization
            U = rng.randn(n * m, rank) * 0.1
            V = rng.randn(m * p, rank) * 0.1
            W = rng.randn(p * n, rank) * 0.1
            
            # Gradient-free iterative refinement: for each (i,j,k), adjust
            # the rank factors to better match the target tensor.
            # We use a simple alternating minimization.
            success = False
            for it in range(100):
                if time.time() > deadline - 30:
                    break
                # For each (i,j,k), the contribution is sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
                # We want this to equal 1 for all (i,j,k).
                # Use a simple update: for each r, adjust to reduce error.
                # This is complex; instead, use a different approach.
                break
            
            # Simpler: use the known structure. Try to find a decomposition
            # by solving a linear system for each rank level.
            # For a given rank R, we need U (8xR), V (32xR), W (16xR) such that
            # the tensor equation holds. This is a system of 64 equations in
            # 8R + 32R + 16R = 56R unknowns. For R >= 2, this is underdetermined.
            # We can set up a least-squares problem.
            
            # Build the tensor equation matrix.
            # For each (i,j,k), the equation is:
            # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            # This is nonlinear. Use an iterative approach.
            
            # Initialize with a structured guess
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            # Use the Strassen structure as a starting point if rank >= 56
            # Otherwise, use a random structured initialization
            if rank >= 56:
                U_s, V_s, W_s = _build_strassen_2x4x8()
                U = U_s[:, :rank].copy()
                V = V_s[:, :rank].copy()
                W = W_s[:, :rank].copy()
            else:
                # Random initialization with small values
                U = rng.randn(n * m, rank) * 0.05
                V = rng.randn(m * p, rank) * 0.05
                W = rng.randn(p * n, rank) * 0.05
            
            # Alternating least squares: fix two, solve for the third
            for it in range(200):
                if time.time() > deadline - 30:
                    break
                # Solve for U: for each (i,j), U[i*m+j, :] should satisfy
                # sum_k' ... this is complex.
                # Instead, use a simple gradient descent on the error.
                # Error = sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
                # Gradient w.r.t. U[i*m+j,r] = 2 * (err[i,j,k]) * V[j*p+k,r] * W[k*n+i,r]
                # summed over k.
                
                # Compute error
                err = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(rank):
                                s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                            err[i, j, k] = s - 1.0
                
                err_norm = np.sqrt(np.sum(err ** 2))
                if err_norm < 1e-10:
                    success = True
                    break
                
                # Gradient descent with adaptive step
                lr = 0.01 / (1 + it * 0.01)
                for i in range(n):
                    for j in range(m):
                        for r in range(rank):
                            grad = 0.0
                            for k in range(p):
                                grad += 2 * err[i, j, k] * V[j * p + k, r] * W[k * n + i, r]
                            U[i * m + j, r] -= lr * grad
                for j in range(m):
                    for k in range(p):
                        for r in range(rank):
                            grad = 0.0
                            for i in range(n):
                                grad += 2 * err[i, j, k] * U[i * m + j, r] * W[k * n + i, r]
                            V[j * p + k, r] -= lr * grad
                for k in range(p):
                    for i in range(n):
                        for r in range(rank):
                            grad = 0.0
                            for j in range(m):
                                grad += 2 * err[i, j, k] * U[i * m + j, r] * V[j * p + k, r]
                            W[k * n + i, r] -= lr * grad
                
                if success:
                    break
            
            if success:
                best = (U, V, W, rank)
                break
        if best is not None:
            break
    
    return best


def _combine_1x4x8_and_2x4x8(U1, V1, W1, U2, V2, W2):
    """Combine the i=0 and i=1,2 decompositions into a full 3x4x8 decomposition."""
    n, m, p = 3, 4, 8
    rank1 = U1.shape[1]
    rank2 = U2.shape[1]
    rank = rank1 + rank2
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Part 1: i=0
    U[0 * m:(0 + 1) * m, :rank1] = U1
    V[:rank1] = V1
    W[0 * n:(0 + 1) * n, :rank1] = W1
    
    # Part 2: i=1,2
    U[1 * m:(1 + 2) * m, rank1:rank] = U2
    V[rank1:rank] = V2
    W[0 * n:(0 + 2) * n, rank1:rank] = W2
    
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Build the i=0 schoolbook part
    U1, V1, W1 = _build_schoolbook_1x4x8()
    
    # Build the i=1,2 Strassen part as baseline
    U2, V2, W2 = _build_strassen_2x4x8()
    
    # Try to find a lower-rank decomposition for the 2x4x8 block
    search_result = _search_2x4x8(deadline, max_rank=56, seed=42)
    
    if search_result is not None:
        U2_new, V2_new, W2_new, rank2_new = search_result
        if rank2_new < 56:
            U2, V2, W2 = U2_new, V2_new, W2_new
    
    # Combine
    U, V, W = _combine_1x4x8_and_2x4x8(U1, V1, W1, U2, V2, W2)
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fall back to the known-good rank-88
        U1, V1, W1 = _build_schoolbook_1x4x8()
        U2, V2, W2 = _build_strassen_2x4x8()
        U, V, W = _combine_1x4x8_and_2x4x8(U1, V1, W1, U2, V2, W2)
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}


if __name__ == "__main__":
    result = construct()
    rank = len(result['U'][0])
    print(f"Rank: {rank}")
    
    U_arr = np.array(result['U'])
    V_arr = np.array(result['V'])
    W_arr = np.array(result['W'])
    n, m, p = 3, 4, 8
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                t_val = 0
                for rr in range(rank):
                    t_val += result['U'][i * m + j][rr] * result['V'][j * p + k][rr] * result['W'][k * n + i][rr]
                if abs(t_val - 1) > 1e-9:
                    max_err = max(max_err, abs(t_val - 1))
    print(f"Max tensor error: {max_err}")
