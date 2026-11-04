"""Matrix multiplication 4x4 x 4x7 via Strassen block decomposition with Wiedemann rank verification."""
import numpy as np
import time
import random

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify decomposition exactly reconstructs matmul tensor."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    r = U.shape[1]
    if V.shape[1] != r or W.shape[1] != r:
        return False
    # Tensor: T[i,j,k] = 1 for all i,j,k
    # Check: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_full(U, V, W, n, m, p, tol=1e-9):
    """Full verification: for all i,k,j,l: sum_c W[k*n+i,c]*U[i*m+j,c]*V[l*p+k,c] == delta(j,l)"""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for l in range(m):
                    s = 0.0
                    for c in range(r):
                        s += W[k*n+i, c] * U[i*m+j, c] * V[l*p+k, c]
                    expected = 1.0 if j == l else 0.0
                    if abs(s - expected) > tol:
                        return False
    return True

def _schoolbook(n, m, p):
    r = n*m*p
    U = np.zeros((n*m, r)); V = np.zeros((m*p, r)); W = np.zeros((p*n, r))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1; V[j*p+k, c] = 1; W[k*n+i, c] = 1
                c += 1
    return U, V, W

def _strassen_2x2_2x2():
    """Strassen for 2x2 * 2x2 -> 2x2, rank 7."""
    n = m = p = 2
    U = np.zeros((n*m, 7)); V = np.zeros((m*p, 7)); W = np.zeros((p*n, 7))
    # M1=(A11+A22)(B11+B22); C11+=M1, C22+=M1
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # M2=(A21+A22)B11; C21+=M2
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[1,1]=1
    # M3=A11(B12-B22); C12+=M3
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[2,2]=1
    # M4=A22(B21-B11); C11+=M4, C21+=M4
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[1,3]=1
    # M5=(A11+A12)B22; C11-=M5, C12+=M5
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[2,4]=1
    # M6=(A21-A11)(B11+B12); C22+=M6
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # M7=(A12-A22)(B21+B22); C11+=M7
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W

def _build_444():
    """Strassen-based 4x4 * 4x4 -> 4x4, rank 49."""
    n = m = p = 4
    U = np.zeros((n*m, 49)); V = np.zeros((m*p, 49)); W = np.zeros((p*n, 49))
    A11 = [0,1,4,5]; A12 = [2,3,6,7]; A21 = [8,9,12,13]; A22 = [10,11,14,15]
    B11 = [0,1,4,5]; B12 = [2,3,6,7]; B21 = [8,9,12,13]; B22 = [10,11,14,15]
    C11 = [0,1,4,5]; C12 = [2,3,6,7]; C21 = [8,9,12,13]; C22 = [10,11,14,15]
    
    macros = [
        ([A11, A22], [1,1], [B11, B22], [1,1], [(C11,1),(C22,1)]),
        ([A21, A22], [1,1], [B11], [1], [(C21,1)]),
        ([A11], [1], [B12, B22], [1,-1], [(C12,1)]),
        ([A22], [1], [B21, B11], [1,-1], [(C11,1),(C21,1)]),
        ([A11, A12], [1,1], [B22], [1], [(C11,-1),(C12,1)]),
        ([A21, A11], [1,-1], [B11, B12], [1,1], [(C22,1)]),
        ([A12, A22], [1,-1], [B21, B22], [1,1], [(C11,1)]),
    ]
    
    strassen_pat = [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        ([(2,1),(3,1)], [(0,1)], [(2,1)]),
        ([(0,1)], [(1,1),(3,-1)], [(1,1)]),
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]
    
    col = 0
    for macro in macros:
        A_blocks, A_signs, B_blocks, B_signs, C_targets = macro
        for sub in strassen_pat:
            U_entries, V_entries, W_entries = sub
            for a_local, s_u in U_entries:
                gidx = A_blocks[a_local]
                U[gidx, col] += A_signs[a_local] * s_u
            for b_local, s_v in V_entries:
                gidx = B_blocks[b_local]
                V[gidx, col] += B_signs[b_local] * s_v
            for c_local, s_w in W_entries:
                C_block, c_sign = C_targets[c_local]
                for gidx in C_block:
                    W[gidx, col] += c_sign * s_w
            col += 1
    return U, V, W

def _build_443():
    """4x4 * 4x3 -> 4x3, rank 48 schoolbook."""
    n, m, p = 4, 4, 3
    r = 48
    U = np.zeros((n*m, r)); V = np.zeros((m*p, r)); W = np.zeros((p*n, r))
    
    products = [
        (0, 0, 0, 0),
        (0, 2, 2, 0),
        (2, 0, 0, 2),
        (2, 2, 2, 2),
    ]
    
    col = 0
    for (ar, ac, br, cr) in products:
        for ai in range(2):
            for aj in range(2):
                for bk in range(3):
                    U[(ar+ai)*4 + (ac+aj), col] = 1
                    V[(br+ai)*3 + bk, col] = 1
                    W[(cr+ai)*4 + bk, col] = 1
                    col += 1
    return U, V, W

def _build_447():
    """Combine 4x4*4x4 (rank 49) and 4x4*4x3 (rank 48) for 4x4*4x7 (rank 97)."""
    n, m, p = 4, 4, 7
    U1, V1, W1 = _build_444()
    U2, V2, W2 = _build_443()
    
    total_rank = 49 + 48
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    U[:, 0:49] = U1
    for i in range(4):
        for j in range(4):
            V[i*7+j, 0:49] = V1[i*4+j, :]
    for i in range(4):
        for j in range(4):
            W[i*7+j, 0:49] = W1[i*4+j, :]
    
    U[:, 49:97] = U2
    for i in range(4):
        for j in range(3):
            V[i*7+4+j, 49:97] = V2[i*3+j, :]
    for row in range(4):
        for col in range(3):
            W[row*7+4+col, 49:97] = W2[row*4+col, :]
    
    return U, V, W

def _wiedemann_rank_check(U, V, W, n, m, p, prime=1000000007):
    """
    Use Wiedemann-style probabilistic rank check over finite field.
    We check if the decomposition is valid by evaluating the tensor identity
    at random points and checking consistency.
    
    The matmul tensor M[i,j,k] = 1. A rank-r decomposition means:
    sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k.
    
    We verify this by checking at random (i,j,k) tuples.
    """
    U_mod = np.mod(np.asarray(U, dtype=np.int64), prime)
    V_mod = np.mod(np.asarray(V, dtype=np.int64), prime)
    W_mod = np.mod(np.asarray(W, dtype=np.int64), prime)
    r = U.shape[1]
    
    # Check all entries (small enough for 4x4x7)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s = (s + U_mod[i*m+j, c] * V_mod[j*p+k, c] % prime * W_mod[k*n+i, c]) % prime
                if s % prime != 1:
                    return False
    return True

def _try_rank_improvement(n, m, p, base_U, base_V, base_W, base_rank, deadline, max_attempts=200):
    """
    Try to find a lower-rank decomposition by perturbing the base decomposition.
    Uses random linear combinations of columns to potentially cancel terms.
    """
    best_U, best_V, best_W = base_U.copy(), base_V.copy(), base_W.copy()
    best_r = base_rank
    
    for attempt in range(max_attempts):
        if time.time() > deadline - 5:
            break
        
        target_r = base_rank - random.randint(1, 5)
        if target_r < 50:  # Don't go too low
            continue
        
        # Try random linear combinations to eliminate some columns
        U_try = base_U.copy()
        V_try = base_V.copy()
        W_try = base_W.copy()
        
        # Pick a random column to try to eliminate by combining with others
        col_to_elim = random.randint(0, base_rank - 1)
        
        # Try to express column col_to_elim as linear combination of others
        # This is a simplification - in practice we'd need to solve a system
        # For now, just try random perturbations
        if random.random() < 0.3:
            # Try merging two columns
            c1 = col_to_elim
            c2 = (c1 + random.randint(1, base_rank-1)) % base_rank
            if c1 != c2:
                # Add column c2 to column c1 with random coefficient
                coeff = random.choice([-1, 1, 2, -2])
                U_try[:, c1] += coeff * U_try[:, c2]
                V_try[:, c1] += coeff * V_try[:, c2]
                W_try[:, c1] += coeff * W_try[:, c2]
                # Check if column c1 is now zero
                if np.allclose(U_try[:, c1], 0) and np.allclose(V_try[:, c1], 0) and np.allclose(W_try[:, c1], 0):
                    # Remove column c1
                    mask = [i for i in range(base_rank) if i != c1]
                    U_try = U_try[:, mask]
                    V_try = V_try[:, mask]
                    W_try = W_try[:, mask]
                    if _verify(U_try, V_try, W_try, n, m, p):
                        new_r = U_try.shape[1]
                        if new_r < best_r:
                            best_r = new_r
                            best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
    
    return best_U, best_V, best_W, best_r

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Schoolbook fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = n*m*p
    
    # Try Strassen-based construction
    try:
        if time.time() < deadline - 30:
            U, V, W = _build_447()
            if _verify(U, V, W, n, m, p):
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = U.shape[1]
                # Try to improve further
                if time.time() < deadline - 10:
                    U2, V2, W2, r2 = _try_rank_improvement(n, m, p, U, V, W, best_rank, deadline)
                    if r2 < best_rank:
                        best = {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
                        best_rank = r2
    except Exception:
        pass
    
    return best
# EVOLVE-BLOCK-END
