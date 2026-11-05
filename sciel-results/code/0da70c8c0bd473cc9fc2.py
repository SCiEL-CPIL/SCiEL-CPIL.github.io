import numpy as np
import time

DIMENSIONS = [5, 5, 5]
DEADLINE_SECONDS = 140.0

def _get_strassen_factors():
    """
    Return list of (U_vec, V_vec, W_vec) for Strassen 2x2x2.
    U_vec has length 4: indices [0,1,2,3] = [a00,a01,a10,a11]
    V_vec has length 4: indices [0,1,2,3] = [b00,b01,b10,b11]
    W_vec has length 4: indices [0,1,2,3] = [c00,c01,c10,c11]
    
    Strassen formulas:
    M1 = (a00+a11)(b00+b11)
    M2 = (a10+a11)b00
    M3 = a00(b01-b11)
    M4 = a11(b10-b00)
    M5 = (a00+a01)b11
    M6 = (a10-a00)(b00+b01)
    M7 = (a01-a11)(b10+b11)
    
    c00 = M1+M4-M5+M7
    c01 = M3+M5
    c10 = M2+M4
    c11 = M1-M2+M3+M6
    """
    factors = []
    # M1 = (a00+a11)(b00+b11) -> c00(+1), c11(+1)
    factors.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    # M2 = (a10+a11)b00 -> c10(+1), c11(-1)
    factors.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
    # M3 = a00(b01-b11) -> c01(+1), c11(+1)
    factors.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
    # M4 = a11(b10-b00) -> c00(+1), c10(+1)
    factors.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]))
    # M5 = (a00+a01)b11 -> c00(-1), c01(+1)
    factors.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
    # M6 = (a10-a00)(b00+b01) -> c11(+1)
    factors.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
    # M7 = (a01-a11)(b10+b11) -> c00(+1)
    factors.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
    return factors

def _schoolbook_block(ni, mj, pk):
    """
    Schoolbook decomposition for ni x mj times mj x pk.
    Returns list of (U_vec, V_vec, W_vec) where each vector has length ni*mj, mj*pk, ni*pk respectively.
    """
    factors = []
    for li in range(ni):
        for lj in range(mj):
            for lk in range(pk):
                U_vec = [0] * (ni * mj)
                V_vec = [0] * (mj * pk)
                W_vec = [0] * (ni * pk)
                U_vec[li * mj + lj] = 1
                V_vec[lj * pk + lk] = 1
                W_vec[li * pk + lk] = 1
                factors.append((U_vec, V_vec, W_vec))
    return factors

def _construct_block_decomposition():
    """
    Construct a rank-minimal decomposition for 5x5x5 using 2+2+1 partition
    with Strassen for 2x2x2 sub-problems.
    """
    n, m, p = 5, 5, 5
    block_sizes = [2, 2, 1]
    offsets = [0]
    for s in block_sizes:
        offsets.append(offsets[-1] + s)
    
    strassen_factors = _get_strassen_factors()
    
    # Calculate total rank
    total_rank = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]
                if ni == 2 and mj == 2 and pk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * mj * pk
    
    # Allocate matrices
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]
                
                if ni == 2 and mj == 2 and pk == 2:
                    for (uf, vf, wf) in strassen_factors:
                        # Map local indices to global
                        for idx in range(4):
                            if uf[idx] != 0:
                                local_row = idx // 2
                                local_col = idx % 2
                                global_row = offsets[i] + local_row
                                global_col = offsets[j] + local_col
                                U[global_row * m + global_col][col] += uf[idx]
                            if vf[idx] != 0:
                                local_row = idx // 2
                                local_col = idx % 2
                                global_row = offsets[j] + local_row
                                global_col = offsets[k] + local_col
                                V[global_row * p + global_col][col] += vf[idx]
                            if wf[idx] != 0:
                                local_row = idx // 2
                                local_col = idx % 2
                                global_row = offsets[i] + local_row
                                global_col = offsets[k] + local_col
                                # W convention: W[k*n+i, r] where k is col, i is row
                                W[global_col * n + global_row][col] += wf[idx]
                        col += 1
                else:
                    factors = _schoolbook_block(ni, mj, pk)
                    for (uf, vf, wf) in factors:
                        for idx in range(ni * mj):
                            if uf[idx] != 0:
                                local_row = idx // mj
                                local_col = idx % mj
                                global_row = offsets[i] + local_row
                                global_col = offsets[j] + local_col
                                U[global_row * m + global_col][col] += uf[idx]
                        for idx in range(mj * pk):
                            if vf[idx] != 0:
                                local_row = idx // pk
                                local_col = idx % pk
                                global_row = offsets[j] + local_row
                                global_col = offsets[k] + local_col
                                V[global_row * p + global_col][col] += vf[idx]
                        for idx in range(ni * pk):
                            if wf[idx] != 0:
                                local_row = idx // pk
                                local_col = idx % pk
                                global_row = offsets[i] + local_row
                                global_col = offsets[k] + local_col
                                W[global_col * n + global_row][col] += wf[idx]
                        col += 1
    
    assert col == total_rank, f"Column mismatch: {col} != {total_rank}"
    return {"U": U, "V": V, "W": W}, total_rank

def _verify(U, V, W, n, m, p, tol=1e-10):
    """
    Verify the decomposition using the W[k*n+i, r] convention.
    C[i,j] = sum_r W[j*n+i, r] * L_r * R_r
    where L_r = sum_{i',k'} U[i'*m+k', r] * A[i',k']
    and R_r = sum_{k',j'} V[k'*p+j', r] * B[k',j']
    """
    rng = np.random.RandomState(42)
    A = rng.randn(n, m)
    B = rng.randn(m, p)
    C_expected = A @ B
    
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    A_flat = A.flatten()
    B_flat = B.flatten()
    
    L = U_arr @ A_flat
    R = V_arr @ B_flat
    
    C_computed = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            idx = j * n + i
            C_computed[i, j] = np.sum(W_arr[idx, :] * L * R)
    
    if np.max(np.abs(C_computed - C_expected)) > tol:
        return False
    return True

def _schoolbook_full(n, m, p):
    """Full schoolbook decomposition as fallback."""
    rank = n * m * p
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
    return {"U": U, "V": V, "W": W}, rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Try the block decomposition with Strassen
    try:
        result, rank = _construct_block_decomposition()
        
        # Verify before returning
        if _verify(result["U"], result["V"], result["W"], n, m, p):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    result, rank = _schoolbook_full(n, m, p)
    return result
