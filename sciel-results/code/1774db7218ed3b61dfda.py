# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

_START_TIME = time.time()
_DEADLINE = _START_TIME + 105.0

def _time_remaining():
    return _DEADLINE - time.time()

def _build_rank25():
    """
    Rank-25 decomposition: 3 Strassen blocks (cols 0-1, 2-3, 4-5) + schoolbook (col 6).
    """
    n, m, p = 2, 2, 7
    rank = 25
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))

    def add_strassen(col_start, k_start):
        U[0, col_start+0] += 1; U[3, col_start+0] += 1
        V[7*0+k_start, col_start+0] += 1; V[7*1+(k_start+1), col_start+0] += 1
        W[2*k_start+0, col_start+0] += 1; W[2*(k_start+1)+1, col_start+0] += 1
        U[2, col_start+1] += 1; U[3, col_start+1] += 1
        V[7*0+k_start, col_start+1] += 1
        W[2*k_start+1, col_start+1] += 1; W[2*(k_start+1)+1, col_start+1] -= 1
        U[0, col_start+2] += 1
        V[7*0+(k_start+1), col_start+2] += 1; V[7*1+(k_start+1), col_start+2] -= 1
        W[2*(k_start+1)+0, col_start+2] += 1; W[2*(k_start+1)+1, col_start+2] += 1
        U[3, col_start+3] += 1
        V[7*1+k_start, col_start+3] += 1; V[7*0+k_start, col_start+3] -= 1
        W[2*k_start+0, col_start+3] += 1; W[2*k_start+1, col_start+3] += 1
        U[0, col_start+4] += 1; U[1, col_start+4] += 1
        V[7*1+(k_start+1), col_start+4] += 1
        W[2*k_start+0, col_start+4] -= 1; W[2*(k_start+1)+0, col_start+4] += 1
        U[2, col_start+5] += 1; U[0, col_start+5] -= 1
        V[7*0+k_start, col_start+5] += 1; V[7*0+(k_start+1), col_start+5] += 1
        W[2*(k_start+1)+1, col_start+5] += 1
        U[1, col_start+6] += 1; U[3, col_start+6] -= 1
        V[7*1+k_start, col_start+6] += 1; V[7*1+(k_start+1), col_start+6] += 1
        W[2*k_start+0, col_start+6] += 1

    add_strassen(0, 0)
    add_strassen(7, 2)
    add_strassen(14, 4)

    cs = 21
    U[0, cs+0] += 1; V[7*0+6, cs+0] += 1; W[2*6+0, cs+0] += 1
    U[1, cs+1] += 1; V[7*1+6, cs+1] += 1; W[2*6+0, cs+1] += 1
    U[2, cs+2] += 1; V[7*0+6, cs+2] += 1; W[2*6+1, cs+2] += 1
    U[3, cs+3] += 1; V[7*1+6, cs+3] += 1; W[2*6+1, cs+3] += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _build_rank28_fallback():
    n, m, p = 2, 2, 7
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, idx] = 1
                V[j*p+k, idx] = 1
                W[k*n+i, idx] = 1
                idx += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify(U, V, W, n=2, m=2, p=7, seed=42):
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
    np.random.seed(seed)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_expected = A @ B
    C_computed = np.zeros((n, p))
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for r in range(rank):
        SA = np.dot(Ua[:, r], A.flatten())
        SB = np.dot(Va[:, r], B.flatten())
        for k in range(p):
            for i in range(n):
                C_computed[i, k] += Wa[k*n+i, r] * SA * SB
    return np.allclose(C_computed, C_expected, atol=1e-10)

def _search_column_mixing_rank23():
    """
    Try to find rank-23 decomposition using column mixing.
    
    Strategy: Use 7 linear combinations of B columns in V.
    For each rank term r, V[:,r] = M @ e_r where M is 14x7 mixing matrix.
    Then for each (i,k) output, we need:
      sum_r W[k*n+i, r] * (U[:,r] dot A_vec) * (V[:,r] dot B_vec) = C[i,k]
    
    This is equivalent to: for each (i,k), the bilinear form in (A_vec, B_vec)
    must equal A[i,:] @ B[:,k].
    
    We parameterize by choosing 23 rank-1 terms. Each term has:
      - u_r in Z^4 (A-side coefficient)
      - v_r in Z^14 (B-side coefficient) 
      - w_r in Z^14 (output coefficient)
    
    The constraint is that the sum of outer products equals the tensor.
    
    We use a greedy approach: start from rank-25 and try to merge terms.
    """
    n, m, p = 2, 2, 7
    target_rank = 23
    
    # Start with rank-25 and try to find linear dependencies to reduce rank
    base = _build_rank25()
    U25 = np.array(base['U'], dtype=float)
    V25 = np.array(base['V'], dtype=float)
    W25 = np.array(base['W'], dtype=float)
    
    # Check if any columns are linearly dependent
    # Try all pairs of columns to see if one can be expressed as combo of others
    # This is expensive, so use a simpler heuristic: try removing pairs
    
    best_result = None
    
    # Try removing 2 columns from rank-25 to get rank-23
    # Check if the remaining 23 columns still span the tensor
    for i in range(25):
        if _time_remaining() < 5.0:
            break
        for j in range(i+1, 25):
            if _time_remaining() < 3.0:
                break
            
            # Remove columns i and j
            keep = [k for k in range(25) if k != i and k != j]
            U23 = U25[:, keep]
            V23 = V25[:, keep]
            W23 = W25[:, keep]
            
            # Verify if rank-23 still works
            if _verify(U23, V23, W23, n=n, m=m, p=p, seed=42):
                best_result = {"U": U23.tolist(), "V": V23.tolist(), "W": W23.tolist()}
                return best_result
    
    # If simple removal doesn't work, try column mixing approach
    # Generate random mixing matrices and check if we can find valid U
    np.random.seed(123)
    
    for attempt in range(100):
        if _time_remaining() < 5.0:
            break
        
        # Random mixing matrix M: 14 x 7 (combines 7 B-columns into 7 mixed columns)
        # Actually, we want V to be 14 x 23, where each column is a linear combo of B-cols
        # Let's try: V[:, r] = T @ b_r where T is 14x23 and b_r are basis vectors
        # This doesn't help directly.
        
        # Alternative: try specific structured mixings
        # For now, just return None to use fallback
        pass
    
    return None

def construct():
    if _time_remaining() < 2.0:
        return _build_rank28_fallback()
    
    # Try rank-23 search first
    result23 = _search_column_mixing_rank23()
    if result23 is not None:
        U = np.array(result23['U'], dtype=float)
        V = np.array(result23['V'], dtype=float)
        W = np.array(result23['W'], dtype=float)
        if _verify(U, V, W):
            return result23
    
    # Fall back to rank-25
    result = _build_rank25()
    U = np.array(result['U'], dtype=float)
    V = np.array(result['V'], dtype=float)
    W = np.array(result['W'], dtype=float)
    
    if not _verify(U, V, W):
        return _build_rank28_fallback()
    
    return result

# EVOLVE-BLOCK-END
