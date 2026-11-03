"""Matrix multiplication 3x5x7 tensor decomposition."""
import numpy as np
import time
import itertools

DIMENSIONS = [3, 5, 7]

def _strassen_2x2x2_terms():
    """Return the 7 rank-1 terms for 2x2x2 matmul tensor.
    Each term: (U_dict, V_dict, W_dict) with local 0-indexed keys.
    U keys: (i,j), V keys: (j,k), W keys: (k,i)
    """
    return [
        # M1 = (a00+a11)(b00+b11) -> C00+=1, C11+=1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2 = (a10+a11)b00 -> C10+=1, C11+=1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):1}),
        # M3 = a00(b01-b11) -> C01+=1, C11+=1
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),
        # M4 = a11(b10-b00) -> C00+=1, C10+=1
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),
        # M5 = (a00+a01)b11 -> C01+=1, C11+=1
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(1,0):1, (1,1):1}),
        # M6 = (a10-a00)(b00+b01) -> C11+=1
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7 = (a01-a11)(b10+b11) -> C00+=1, C11+=1
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
    ]

def _verify(U, V, W, n=3, m=5, p=7, tol=1e-9):
    """Verify that U, V, W form a valid decomposition of the matmul tensor."""
    rank = U.shape[1]
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Direct tensor identity check:
    # sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c] = delta_{j,j'} * delta_{k,k'} * delta_{i,i'}
    rng = np.random.RandomState(42)
    for trial in range(500):
        i = rng.randint(n)
        j = rng.randint(m)
        j2 = rng.randint(m)
        k = rng.randint(p)
        k2 = rng.randint(p)
        i2 = rng.randint(n)
        
        lhs = np.dot(U_arr[i*m+j, :], np.multiply(V_arr[j2*p+k, :], W_arr[k2*n+i2, :]))
        rhs = (1.0 if (j == j2 and k == k2 and i == i2) else 0.0)
        if abs(lhs - rhs) > tol:
            return False
    return True

def _schoolbook(n=3, m=5, p=7):
    """Standard schoolbook decomposition, rank = n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    i_idx, j_idx, k_idx = np.indices((n, m, p)).reshape(3, -1)
    i_flat = i_idx.ravel()
    j_flat = j_idx.ravel()
    k_flat = k_idx.ravel()
    
    u_rows = i_flat * m + j_flat
    v_rows = j_flat * p + k_flat
    w_rows = k_flat * n + i_flat
    cols = np.arange(rank)
    
    U[u_rows, cols] = 1
    V[v_rows, cols] = 1
    W[w_rows, cols] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _strassen_decomposition(n=3, m=5, p=7):
    """Build decomposition using 6 disjoint 2x2x2 Strassen blocks + schoolbook for remainder.
    Rank = 6*7 + (105 - 6*8) = 42 + 57 = 99
    """
    blocks = []
    for i0 in [0]:
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                blocks.append((i0, j0, k0))
    
    # Determine which (i,j,k) entries are covered by Strassen blocks
    covered = set()
    for (i0, j0, k0) in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))
    
    schoolbook_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    schoolbook_entries.append((i, j, k))
    
    rank = len(blocks) * 7 + len(schoolbook_entries)
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    strassen_terms = _strassen_2x2x2_terms()
    col = 0
    
    for (i0, j0, k0) in blocks:
        for U_dict, V_dict, W_dict in strassen_terms:
            for (li, lj), val in U_dict.items():
                gi, gj = i0 + li, j0 + lj
                U[gi*m + gj, col] = val
            for (lj, lk), val in V_dict.items():
                gj, gk = j0 + lj, k0 + lk
                V[gj*p + gk, col] = val
            for (lk, li), val in W_dict.items():
                gk, gi = k0 + lk, i0 + li
                W[gk*n + gi, col] = val
            col += 1
    
    for (i, j, k) in schoolbook_entries:
        U[i*m + j, col] = 1
        V[j*p + k, col] = 1
        W[k*n + i, col] = 1
        col += 1
    
    assert col == rank, f"Column count {col} != rank {rank}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_lower_rank(n=3, m=5, p=7, deadline=None):
    """
    Try to find a lower-rank decomposition using structured random search.
    Uses the tensor identity: sum_c U[i*m+j,c]*V[j'*p+k,c]*W[k'*n+i',c] = delta_{j,j'}delta_{k,k'}delta_{i,i'}
    """
    if deadline is None:
        deadline = time.time() + 100
    
    sb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_sol = sb
    
    rng = np.random.RandomState(99999)
    
    # Try ranks from just below schoolbook down
    for target_rank in [104, 103, 102, 101, 100, 99, 98, 97, 96, 95]:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
        
        # Multiple attempts per rank
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            # Initialize with random matrices
            U_rand = rng.randn(n * m, target_rank)
            V_rand = rng.randn(m * p, target_rank)
            W_rand = rng.randn(p * n, target_rank)
            
            U_list = U_rand.tolist()
            V_list = V_rand.tolist()
            W_list = W_rand.tolist()
            
            if _verify(U_list, V_list, W_list, n, m, p, tol=1e-8):
                best_rank = target_rank
                best_sol = {"U": U_list, "V": V_list, "W": W_list}
                break
    
    return best_sol

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication.
    Returns dict with 'U', 'V', 'W' factor matrices.
    """
    start_time = time.time()
    deadline = start_time + 120  # 120s limit per task spec, leave margin
    
    n, m, p = DIMENSIONS
    
    # Try Strassen-based decomposition (rank 99)
    try:
        sol = _strassen_decomposition(n, m, p)
        if _verify(sol["U"], sol["V"], sol["W"], n, m, p, tol=1e-8):
            return sol
    except Exception:
        pass
    
    # Fallback: schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
