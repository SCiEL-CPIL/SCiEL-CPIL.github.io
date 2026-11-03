"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _strassen_222():
    """Return U, V, W (each 4x7) for 2x2x2 matmul using Strassen.
    Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # r=0: (A00+A11)(B00+B11)
    U[0, 0] = 1
    U[3, 0] = 1
    V[0, 0] = 1
    V[1, 0] = 1
    V[2, 0] = 1
    V[3, 0] = 1
    W[0, 0] = 1  # k=0,i=0
    W[1, 0] = 1  # k=1,i=0
    W[2, 0] = 1  # k=0,i=1
    W[3, 0] = 1  # k=1,i=1
    
    # r=1: (A10+A11)B00
    U[2, 1] = 1
    U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1  # k=0,i=1
    
    # r=2: A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1
    V[3, 2] = -1
    W[1, 2] = 1  # k=1,i=0
    
    # r=3: A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1
    V[0, 3] = -1
    W[2, 3] = 1  # k=0,i=1
    
    # r=4: (A00+A01)B11
    U[0, 4] = 1
    U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1  # k=1,i=0
    
    # r=5: (A10-A00)(B00+B10)
    U[2, 5] = 1
    U[0, 5] = -1
    V[0, 5] = 1
    V[2, 5] = 1
    W[0, 5] = 1  # k=0,i=0
    
    # r=6: (A01-A11)(B01+B11)
    U[1, 6] = 1
    U[3, 6] = -1
    V[1, 6] = 1
    V[3, 6] = 1
    W[0, 6] = -1  # k=0,i=0
    
    return U, V, W

def _verify_222():
    """Verify the Strassen 2x2x2 decomposition."""
    U, V, W = _strassen_222()
    n = m = p = 2
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(7):
                    total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(total - 1.0) > 1e-10:
                    return False
    return True

def _build_2x5x6():
    """
    Build 2x5x6 decomposition using block Strassen.
    Split 5 = 2+3, 6 = 2+2+2.
    
    A = [A1 (2x2) | A2 (2x3)]
    B = [B1 (2x6); B2 (3x6)]
    C = A1*B1 + A2*B2
    
    A1*B1: 2x2x6, split B1 into three 2x2 blocks. Each uses Strassen (rank 7). Total: 21.
    A2*B2: 2x3x6, split A2 = [A2a (2x1) | A2b (2x2)], B2 = [B2a (1x6); B2b (2x6)].
      A2a*B2a: 2x1x6 (rank 12 schoolbook)
      A2b*B2b: 2x2x6 (rank 21 via Strassen)
      Total for A2*B2: 33
    Grand total: 21 + 33 = 54
    """
    n, m, p = 2, 5, 6
    rank = 54
    
    U = np.zeros((n*m, rank))  # 10 x 54
    V = np.zeros((m*p, rank))  # 30 x 54
    W = np.zeros((p*n, rank))  # 12 x 54
    
    sU, sV, sW = _strassen_222()  # 4x7 each
    
    col = 0
    
    # Part 1: A1 * B1 (2x2 x 6), three Strassen blocks
    for block in range(3):
        k_offset = block * 2
        for r in range(7):
            # U: A1 indices, i in {0,1}, j in {0,1} -> U[i*5+j, col]
            U[0*5 + 0, col] = sU[0, r]  # i=0,j=0
            U[0*5 + 1, col] = sU[1, r]  # i=0,j=1
            U[1*5 + 0, col] = sU[2, r]  # i=1,j=0
            U[1*5 + 1, col] = sU[3, r]  # i=1,j=1
            
            # V: B1_block indices, j in {0,1} globally, k in {k_offset, k_offset+1}
            V[0*6 + k_offset + 0, col] = sV[0, r]  # j=0,k=k_offset
            V[0*6 + k_offset + 1, col] = sV[1, r]  # j=0,k=k_offset+1
            V[1*6 + k_offset + 0, col] = sV[2, r]  # j=1,k=k_offset
            V[1*6 + k_offset + 1, col] = sV[3, r]  # j=1,k=k_offset+1
            
            # W: C indices, k in {k_offset, k_offset+1}, i in {0,1}
            W[(k_offset+0)*2 + 0, col] = sW[0, r]  # k=k_offset, i=0
            W[(k_offset+0)*2 + 1, col] = sW[1, r]  # k=k_offset, i=1
            W[(k_offset+1)*2 + 0, col] = sW[2, r]  # k=k_offset+1, i=0
            W[(k_offset+1)*2 + 1, col] = sW[3, r]  # k=k_offset+1, i=1
            
            col += 1
    # col should be 21 here
    
    # Part 2a: A2a * B2a (2x1 x 6), schoolbook rank 12
    for i in range(2):
        for k in range(6):
            U[i*5 + 2, col] = 1   # A2a: i in {0,1}, j=2
            V[2*6 + k, col] = 1   # B2a: j=2, k in {0..5}
            W[k*2 + i, col] = 1   # C: k, i
            col += 1
    # col should be 33 here
    
    # Part 2b: A2b * B2b (2x2 x 6), three Strassen blocks
    for block in range(3):
        k_offset = block * 2
        for r in range(7):
            # U: A2b indices, i in {0,1}, j in {3,4}
            U[0*5 + 3, col] = sU[0, r]  # i=0,j=0 (local) -> global j=3
            U[0*5 + 4, col] = sU[1, r]  # i=0,j=1 (local) -> global j=4
            U[1*5 + 3, col] = sU[2, r]  # i=1,j=0 (local) -> global j=3
            U[1*5 + 4, col] = sU[3, r]  # i=1,j=1 (local) -> global j=4
            
            # V: B2b_block, j in {3,4} globally, k in {k_offset, k_offset+1}
            V[3*6 + k_offset + 0, col] = sV[0, r]  # j=3, k=k_offset
            V[3*6 + k_offset + 1, col] = sV[1, r]  # j=3, k=k_offset+1
            V[4*6 + k_offset + 0, col] = sV[2, r]  # j=4, k=k_offset
            V[4*6 + k_offset + 1, col] = sV[3, r]  # j=4, k=k_offset+1
            
            # W: same as before
            W[(k_offset+0)*2 + 0, col] = sW[0, r]
            W[(k_offset+0)*2 + 1, col] = sW[1, r]
            W[(k_offset+1)*2 + 0, col] = sW[2, r]
            W[(k_offset+1)*2 + 1, col] = sW[3, r]
            
            col += 1
    # col should be 54 here
    
    assert col == rank, f"Expected col={rank}, got {col}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_full(U, V, W, n=2, m=5, p=6):
    """Verify the full decomposition."""
    rank = len(U[0])
    Ua = np.array(U)
    Va = np.array(V)
    Wa = np.array(W)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def _check_valid_entries(U, V, W):
    """Check that all entries are integer or half-integer."""
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                # Check if val is integer or half-integer
                if abs(val * 2 - round(val * 2)) > 1e-12:
                    return False
    return True

def _schoolbook_decomposition():
    """Fallback schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _verify_fast(Ua, Va, Wa, n=2, m=5, p=6):
    """Vectorized verification: check all (i,j,k) triples sum to 1."""
    rank = Ua.shape[1]
    # Build tensors
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # For each r, contribution to C[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1 for all i,j,k
    
    # Reshape to 3D tensors
    U3 = Ua.reshape(n, m, rank)  # U3[i,j,r]
    V3 = Va.reshape(m, p, rank)  # V3[j,k,r]
    W3 = Wa.reshape(p, n, rank)  # W3[k,i,r]
    
    # Compute the tensor contraction: sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # This is equivalent to: for each (i,j,k), dot product of U3[i,j,:], V3[j,k,:], W3[k,i,:]
    # Use einsum
    result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.all(np.abs(result - 1.0) < 1e-9)

def _try_local_search(U, V, W, time_limit=90):
    """Try to reduce rank via greedy column elimination with vectorized verification."""
    start_time = time.time()
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    n, m, p = 2, 5, 6
    current_rank = Ua.shape[1]
    best_rank = current_rank
    best_U, best_V, best_W = Ua.copy(), Va.copy(), Wa.copy()
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for col in range(Ua.shape[1]):
            if time.time() - start_time > time_limit:
                break
            mask = [i for i in range(Ua.shape[1]) if i != col]
            U_r = Ua[:, mask]
            V_r = Va[:, mask]
            W_r = Wa[:, mask]
            if _verify_fast(U_r, V_r, W_r, n, m, p):
                Ua, Va, Wa = U_r, V_r, W_r
                best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                best_rank = Ua.shape[1]
                improved = True
                break
    
    if best_rank < current_rank:
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    return None

def _try_pairwise_elimination(U, V, W, time_limit=60):
    """Try to reduce rank by removing pairs of columns simultaneously."""
    start_time = time.time()
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    n, m, p = 2, 5, 6
    current_rank = Ua.shape[1]
    best_rank = current_rank
    best_U, best_V, best_W = Ua.copy(), Va.copy(), Wa.copy()
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        rc = Ua.shape[1]
        for c1 in range(rc):
            if time.time() - start_time > time_limit:
                break
            for c2 in range(c1+1, rc):
                if time.time() - start_time > time_limit:
                    break
                mask = [i for i in range(rc) if i != c1 and i != c2]
                U_r = Ua[:, mask]
                V_r = Va[:, mask]
                W_r = Wa[:, mask]
                if _verify_fast(U_r, V_r, W_r, n, m, p):
                    Ua, Va, Wa = U_r, V_r, W_r
                    best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                    best_rank = Ua.shape[1]
                    improved = True
                    break
            if improved:
                break
    
    if best_rank < current_rank:
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    return None

def construct():
    deadline = time.time() + 140  # safety margin
    
    # Try Strassen-based rank-54 decomposition
    try:
        if _verify_222():
            result = _build_2x5x6()
            if _verify_full(result['U'], result['V'], result['W']):
                if _check_valid_entries(result['U'], result['V'], result['W']):
                    # Try to reduce rank via single-column local search
                    remaining_time = deadline - time.time() - 20
                    if remaining_time > 15:
                        reduced = _try_local_search(result['U'], result['V'], result['W'], time_limit=remaining_time)
                        if reduced is not None:
                            ru, rv, rw = reduced
                            if _verify_fast(np.array(ru), np.array(rv), np.array(rw)) and _check_valid_entries(ru, rv, rw):
                                # Now try pairwise elimination on the reduced result
                                remaining_time2 = deadline - time.time() - 10
                                if remaining_time2 > 10:
                                    reduced2 = _try_pairwise_elimination(ru, rv, rw, time_limit=remaining_time2)
                                    if reduced2 is not None:
                                        ru2, rv2, rw2 = reduced2
                                        if _verify_fast(np.array(ru2), np.array(rv2), np.array(rw2)) and _check_valid_entries(ru2, rv2, rw2):
                                            return {'U': ru2, 'V': rv2, 'W': rw2}
                                return {'U': ru, 'V': rv, 'W': rw}
                    # If single-column search didn't help, try pairwise directly
                    remaining_time3 = deadline - time.time() - 10
                    if remaining_time3 > 10:
                        reduced3 = _try_pairwise_elimination(result['U'], result['V'], result['W'], time_limit=remaining_time3)
                        if reduced3 is not None:
                            ru3, rv3, rw3 = reduced3
                            if _verify_fast(np.array(ru3), np.array(rv3), np.array(rw3)) and _check_valid_entries(ru3, rv3, rw3):
                                return {'U': ru3, 'V': rv3, 'W': rw3}
                    return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook_decomposition()
# EVOLVE-BLOCK-END
