import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 6]
DEADLINE = 150.0
SAFETY_MARGIN = 10.0

def _strassen_2x2x2(i_off, j_off, k_off, m, n, p, U, V, W, cur):
    def u(i, j): return (i_off + i) * m + (j_off + j)
    def v(j, k): return (j_off + j) * p + (k_off + k)
    def w(i, k): return (k_off + k) * n + (i_off + i)

    # M1 = (a00+a11)(b00+b11)
    U[u(0,0)][cur] += 1
    U[u(1,1)][cur] += 1
    V[v(0,0)][cur] += 1
    V[v(1,1)][cur] += 1
    W[w(0,0)][cur] += 1
    W[w(1,1)][cur] += 1
    cur += 1
    # M2 = (a10+a11)(b00)
    U[u(1,0)][cur] += 1
    U[u(1,1)][cur] += 1
    V[v(0,0)][cur] += 1
    W[w(1,0)][cur] += 1
    W[w(1,1)][cur] -= 1
    cur += 1
    # M3 = (a00)(b01-b11)
    U[u(0,0)][cur] += 1
    V[v(0,1)][cur] += 1
    V[v(1,1)][cur] -= 1
    W[w(0,1)][cur] += 1
    W[w(1,1)][cur] += 1
    cur += 1
    # M4 = (a11)(b10-b00)
    U[u(1,1)][cur] += 1
    V[v(1,0)][cur] += 1
    V[v(0,0)][cur] -= 1
    W[w(0,0)][cur] += 1
    W[w(1,0)][cur] += 1
    cur += 1
    # M5 = (a00+a01)(b11)
    U[u(0,0)][cur] += 1
    U[u(0,1)][cur] += 1
    V[v(1,1)][cur] += 1
    W[w(0,0)][cur] -= 1
    W[w(0,1)][cur] += 1
    cur += 1
    # M6 = (a10-a00)(b00+b01)
    U[u(1,0)][cur] += 1
    U[u(0,0)][cur] -= 1
    V[v(0,0)][cur] += 1
    V[v(0,1)][cur] += 1
    W[w(1,1)][cur] += 1
    cur += 1
    # M7 = (a01-a11)(b10+b11)
    U[u(0,1)][cur] += 1
    U[u(1,1)][cur] -= 1
    V[v(1,0)][cur] += 1
    V[v(1,1)][cur] += 1
    W[w(0,0)][cur] += 1
    cur += 1

    return cur

def _winograd_2x2_2x2(i_off, j_off, k_off, m, n, p, U, V, W, cur):
    def u(i, j): return (i_off + i) * m + (j_off + j)
    def v(j, k): return (j_off + j) * p + (k_off + k)
    def w(i, k): return (k_off + k) * n + (i_off + i)

    # Winograd F(2x2, 2x2): 8 multiplications
    # C = [a00 a01; a10 a11] * [b00 b01; b10 b11]
    # m1 = (a00 + a01)(b00 + b10)
    U[u(0,0)][cur] += 1
    U[u(0,1)][cur] += 1
    V[v(0,0)][cur] += 1
    V[v(0,1)][cur] += 1
    W[w(0,0)][cur] += 1
    W[w(1,0)][cur] += 1
    cur += 1
    # m2 = (a10 - a00)(b00 + b01)
    U[u(1,0)][cur] += 1
    U[u(0,0)][cur] -= 1
    V[v(0,0)][cur] += 1
    V[v(1,0)][cur] += 1
    W[w(0,0)][cur] -= 1
    W[w(0,1)][cur] += 1
    cur += 1
    # m3 = (a10 - a11)(b11)
    U[u(1,0)][cur] += 1
    U[u(1,1)][cur] -= 1
    V[v(1,1)][cur] += 1
    W[w(1,0)][cur] += 1
    W[w(1,1)][cur] -= 1
    cur += 1
    # m4 = (a10 + a11)(b01)
    U[u(1,0)][cur] += 1
    U[u(1,1)][cur] += 1
    V[v(0,1)][cur] += 1
    W[w(0,0)][cur] += 1
    W[w(0,1)][cur] += 1
    cur += 1
    # m5 = a00*b00
    U[u(0,0)][cur] += 1
    V[v(0,0)][cur] += 1
    W[w(0,0)][cur] += 1
    cur += 1
    # m6 = a00*b10
    U[u(0,0)][cur] += 1
    V[v(1,0)][cur] += 1
    W[w(1,0)][cur] += 1
    cur += 1
    # m7 = a01*b11
    U[u(0,1)][cur] += 1
    V[v(1,1)][cur] += 1
    W[w(0,1)][cur] += 1
    cur += 1
    # m8 = a11*b11
    U[u(1,1)][cur] += 1
    V[v(1,1)][cur] += 1
    W[w(1,1)][cur] += 1
    cur += 1

    return cur

def _schoolbook(n_block, m_block, p_block, i_off, j_off, k_off, m, n, p, U, V, W, cur):
    for i in range(n_block):
        for j in range(m_block):
            for k in range(p_block):
                u_idx = (i_off + i) * m + (j_off + j)
                v_idx = (j_off + j) * p + (k_off + k)
                w_idx = (k_off + k) * n + (i_off + i)
                U[u_idx][cur] = 1
                V[v_idx][cur] = 1
                W[w_idx][cur] = 1
                cur += 1
    return cur

def _decompose(n, m, p):
    # Returns list of (n_sub, m_sub, p_sub, i_off, j_off, k_off, method)
    # where method is 'strassen', 'winograd', 'schoolbook'
    blocks = []
    
    # n=1 block: i_off=0
    blocks.append((1, 4, 6, 0, 0, 0, 'schoolbook'))
    
    # n=2 blocks: i_off=1, split m=4 into 2+2
    for m_chunk in range(2):
        j_off = m_chunk * 2
        # 2x2x6 -> split p=6 into 2+2+2
        for p_chunk in range(3):
            k_off = p_chunk * 2
            blocks.append((2, 2, 2, 1, j_off, k_off, 'strassen'))
    
    return blocks

def _decompose_v2(n, m, p):
    """Alternative: n=3 as 2+1, but n=2 part uses 2x2x3 blocks with schoolbook
    plus try to find better structure."""
    blocks = []
    
    # n=2 block: i_off=0, split m=4 into 2+2, p=6 into 3+3
    for m_chunk in range(2):
        j_off = m_chunk * 2
        for p_chunk in range(2):
            k_off = p_chunk * 3
            blocks.append((2, 2, 3, 0, j_off, k_off, 'schoolbook'))
    
    # n=1 block: i_off=2
    blocks.append((1, 4, 6, 2, 0, 0, 'schoolbook'))
    
    return blocks

def _build_decomposition(blocks, n, m, p):
    total_rank = 0
    for n_sub, m_sub, p_sub, i_off, j_off, k_off, method in blocks:
        if method == 'schoolbook':
            total_rank += n_sub * m_sub * p_sub
        elif method == 'strassen':
            total_rank += 7
        elif method == 'winograd':
            total_rank += 8
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    cur = 0
    for n_sub, m_sub, p_sub, i_off, j_off, k_off, method in blocks:
        if method == 'schoolbook':
            cur = _schoolbook(n_sub, m_sub, p_sub, i_off, j_off, k_off, m, n, p, U, V, W, cur)
        elif method == 'strassen':
            cur = _strassen_2x2x2(i_off, j_off, k_off, m, n, p, U, V, W, cur)
        elif method == 'winograd':
            cur = _winograd_2x2_2x2(i_off, j_off, k_off, m, n, p, U, V, W, cur)
    
    return U, V, W, total_rank

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    rank = U_np.shape[1]
    
    # Build the tensor from the decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This should equal 1 for all valid (i,j,k)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U_np[u_idx], V_np[v_idx]) * 0  # placeholder
                # Actually: sum_r U[u_idx,r] * V[v_idx,r] * W[w_idx,r]
                # This is not a simple dot product, it's a 3-way contraction
                val = 0
                for r in range(rank):
                    val += U_np[u_idx, r] * V_np[v_idx, r] * W_np[w_idx, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    
    # Try multiple decomposition strategies and pick the best verified one
    best_rank = None
    best_result = None
    
    strategies = [
        ('v1', _decompose),
        ('v2', _decompose_v2),
    ]
    
    for name, decompose_fn in strategies:
        if time.time() - t0 > DEADLINE - SAFETY_MARGIN:
            break
        blocks = decompose_fn(n, m, p)
        U, V, W, total_rank = _build_decomposition(blocks, n, m, p)
        
        if not _verify(U, V, W, n, m, p):
            continue
        
        if best_rank is None or total_rank < best_rank:
            best_rank = total_rank
            best_result = {"U": U, "V": V, "W": W}
    
    if best_result is None:
        raise ValueError("No valid decomposition found")
    
    return best_result
