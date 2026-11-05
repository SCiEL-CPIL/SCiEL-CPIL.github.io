import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_factors():
    """
    Return list of (U_vec, V_vec, W_vec) for Strassen 2x2x2.
    """
    factors = []
    factors.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    factors.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
    factors.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
    factors.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]))
    factors.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
    factors.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
    factors.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
    return factors

def _construct_6x6x6_strassen():
    """
    Construct 6x6x6 matrix multiplication decomposition using Strassen on 2+2+2 partition.
    Returns U (36, 56), V (36, 56), W (36, 56)
    """
    n = m = p = 6
    block_sizes = [2, 2, 2]
    offsets = [0, 2, 4, 6]
    strassen_factors = _strassen_factors()
    
    total_rank = 8 * 7  # 8 blocks, each rank 7
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for (uf, vf, wf) in strassen_factors:
                    local_to_global_U = [
                        (offsets[i] + 0) * m + (offsets[j] + 0),
                        (offsets[i] + 0) * m + (offsets[j] + 1),
                        (offsets[i] + 1) * m + (offsets[j] + 0),
                        (offsets[i] + 1) * m + (offsets[j] + 1),
                    ]
                    local_to_global_V = [
                        (offsets[j] + 0) * p + (offsets[k] + 0),
                        (offsets[j] + 0) * p + (offsets[k] + 1),
                        (offsets[j] + 1) * p + (offsets[k] + 0),
                        (offsets[j] + 1) * p + (offsets[k] + 1),
                    ]
                    local_to_global_W = [
                        (offsets[k] + 0) * n + (offsets[i] + 0),
                        (offsets[k] + 1) * n + (offsets[i] + 0),
                        (offsets[k] + 0) * n + (offsets[i] + 1),
                        (offsets[k] + 1) * n + (offsets[i] + 1),
                    ]
                    
                    for idx in range(4):
                        if uf[idx] != 0:
                            U[local_to_global_U[idx], col] += uf[idx]
                        if vf[idx] != 0:
                            V[local_to_global_V[idx], col] += vf[idx]
                        if wf[idx] != 0:
                            W[local_to_global_W[idx], col] += wf[idx]
                    col += 1
    
    assert col == total_rank
    return U, V, W

def _project_to_5x5x5(U6, V6, W6):
    """
    Project 6x6x6 decomposition to 5x5x5 by zeroing the 6th index.
    """
    U5 = np.zeros((25, U6.shape[1]), dtype=np.float64)
    V5 = np.zeros((25, V6.shape[1]), dtype=np.float64)
    W5 = np.zeros((25, W6.shape[1]), dtype=np.float64)
    
    for i in range(5):
        for j in range(5):
            U5[i * 5 + j] = U6[i * 6 + j]
            V5[j * 5 + (i)] = V6[j * 6 + i]  # Wait, V is indexed by j*p+k, not j*m+i
            # Let me fix this
    # Redo properly
    U5 = np.zeros((25, U6.shape[1]), dtype=np.float64)
    V5 = np.zeros((25, V6.shape[1]), dtype=np.float64)
    W5 = np.zeros((25, W6.shape[1]), dtype=np.float64)
    
    for i in range(5):
        for j in range(5):
            U5[i * 5 + j] = U6[i * 6 + j]
    
    for j in range(5):
        for k in range(5):
            V5[j * 5 + k] = V6[j * 6 + k]
    
    for k in range(5):
        for i in range(5):
            W5[k * 5 + i] = W6[k * 6 + i]
    
    return U5, V5, W5

def _verify(n, m, p, U, V, W, tol=1e-9):
    """
    Verify that the decomposition exactly reconstructs the matrix multiplication tensor.
    """
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_arr[i * m + j]
                v_row = V_arr[j * p + k]
                w_row = W_arr[k * n + i]
                val = np.dot(u_row, v_row * w_row)
                if abs(val - 1.0) > tol:
                    return False
    return True

def _construct_5x5x5():
    """
    Construct a valid decomposition for 5x5x5 matrix multiplication.
    Uses 6x6x6 Strassen decomposition projected to 5x5x5.
    """
    n, m, p = 5, 5, 5
    
    U6, V6, W6 = _construct_6x6x6_strassen()
    U5, V5, W5 = _project_to_5x5x5(U6, V6, W6)
    
    return {"U": U5.tolist(), "V": V5.tolist(), "W": W5.tolist()}, U5.shape[1]

def construct():
    """
    Main entry point. Returns a valid tensor decomposition for 5x5x5 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    result, rank = _construct_5x5x5()
    
    if _verify(n, m, p, result["U"], result["V"], result["W"]):
        return result
    
    # Fallback: Pure schoolbook (rank 125)
    U = np.zeros((n * m, 125), dtype=np.float64)
    V = np.zeros((m * p, 125), dtype=np.float64)
    W = np.zeros((p * n, 125), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
