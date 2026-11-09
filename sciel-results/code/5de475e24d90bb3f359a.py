"""
Global Linear Dependency Search for rank-minimal 3x4x4 matrix multiplication.

Strategy:
1. Construct a dense rank-48 decomposition (3 blocks of schoolbook 4x4x4).
2. Flatten each rank-1 term into a vector in R^(12*16*12).
3. Check for linear dependencies among these 48 vectors.
4. If a dependency exists, remove the dependent term.
5. Repeat until no more dependencies are found.
6. Additionally, try to find rank-43 by searching for a 1-dim null space
   in the 48-term system that corresponds to a valid removal.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = delta[i,k]*delta[j,i]... 
    T is indexed: A index = i*m+j, B index = j*p+k, C index = k*n+i
    """
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _validate(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T = _build_tensor(n, m, p)
    rank = U.shape[1]
    result = np.zeros((n * m, m * p, p * n))
    for c in range(rank):
        result += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
    return np.allclose(result, T, atol=tol)

def _schoolbook_3x4x4_dense():
    """
    Construct a dense rank-48 decomposition for 3x4x4.
    This is 3 independent schoolbook 4x4x4 multiplications, one for each row of A.
    
    A is 3x4. We can view it as 3 rows, each of length 4.
    Each row i of A (length 4) multiplies B (4x4) to produce row i of C (length 4).
    Wait, C is 3x4. C[i,k] = sum_j A[i,j] B[j,k].
    So for each fixed i, we are doing a 1x4 * 4x4 multiplication.
    This is a 1x4x4 problem, which has rank 16 (schoolbook).
    Since there are 3 such independent rows, the total rank is 3 * 16 = 48.
    
    Let's construct this explicitly.
    """
    n, m, p = 3, 4, 4
    rank = 48
    U = np.zeros((n * m, rank)) # 12 x 48
    V = np.zeros((m * p, rank)) # 16 x 48
    W = np.zeros((p * n, rank)) # 12 x 48
    
    col = 0
    for i in range(n):
        # For this row i, we do a 1x4x4 multiplication
        for j in range(m):
            for k in range(p):
                # A index: i*m + j
                # B index: j*p + k
                # C index: k*n + i
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    return U, V, W

def _find_and_remove_dependency(U, V, W, n, m, p, deadline):
    """
    Find if any term can be expressed as a linear combination of the others.
    If so, remove it.
    Returns the reduced U, V, W or the original if no reduction found.
    """
    current_rank = U.shape[1]
    nm, mp, pn = n * m, m * p, p * n
    dim = nm * mp * pn
    
    # Build the matrix of terms
    # Each column c is the flattened tensor U_c ⊗ V_c ⊗ W_c
    terms = np.zeros((dim, current_rank))
    for c in range(current_rank):
        term = np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        terms[:, c] = term.flatten()
        
    # We want to check if any column is in the span of the others.
    # This is equivalent to checking if the rank of the matrix is < current_rank.
    # But we want to identify *which* column is dependent to remove it.
    
    # Strategy: For each column c, solve for alpha in terms[:, ~c] @ alpha = terms[:, c]
    # If a solution exists with small residual, column c is dependent.
    
    best_removable = -1
    best_residual = np.inf
    
    for c_remove in range(current_rank):
        if time.time() > deadline:
            break
            
        other_cols = [i for i in range(current_rank) if i != c_remove]
        A_mat = terms[:, other_cols]
        b_vec = terms[:, c_remove]
        
        # Solve least squares
        try:
            alpha, residuals, rank_svd, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            residual = np.linalg.norm(A_mat @ alpha - b_vec)
            
            if residual < 1e-8:
                # Found a dependency
                best_removable = c_remove
                best_residual = residual
                break # Found one, we can stop early if we just want one reduction
        except Exception:
            continue
            
    if best_removable >= 0:
        mask = [i for i in range(current_rank) if i != best_removable]
        U_red = U[:, mask]
        V_red = V[:, mask]
        W_red = W[:, mask]
        
        # Validate the reduction
        if _validate(U_red, V_red, W_red, n, m, p):
            return U_red, V_red, W_red
        else:
            # Should not happen if residual was truly 0, but safety check
            pass
            
    return U, V, W

def _try_global_reduction(U, V, W, n, m, p, deadline):
    """
    Iteratively apply dependency removal until no more are found.
    """
    current_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    while best_U.shape[1] > 1:
        if time.time() > deadline:
            break
            
        U_new, V_new, W_new = _find_and_remove_dependency(best_U, best_V, best_W, n, m, p, deadline)
        
        if U_new.shape[1] < best_U.shape[1]:
            best_U, best_V, best_W = U_new, V_new, W_new
            print(f"Reduced rank to {best_U.shape[1]}")
        else:
            break
            
    return best_U, best_V, best_W

def _try_strassen_based_44():
    """
    Construct the rank-44 decomposition from the current program (Strassen 2x4x4 + Schoolbook 1x4x4).
    This serves as a better starting point than rank-48 if we can construct it correctly.
    """
    n, m, p = 3, 4, 4
    # We reuse the logic from the current program to build the rank-44 candidate
    # This is faster than starting from 48 and hoping to reduce 4 steps.
    
    # Build 2x4x4 Strassen (Rank 28)
    # And 1x4x4 Schoolbook (Rank 16)
    # Total 44.
    
    # For simplicity and robustness, let's just return the dense 48 for now,
    # as the dependency search should be able to find the Strassen-like reductions
    # if they are linear dependencies in the 48-term space.
    # However, Strassen reductions are not simple linear dependencies of the schoolbook terms.
    # They require *recombination* of terms, not just removal.
    # So starting from 48 and only removing terms will likely get stuck at 48 or 47.
    
    # Therefore, we MUST start from a better candidate like 44.
    # Let's implement the 44 candidate directly.
    
    U2, V2, W2 = _build_strassen_2x4x4()
    U1, V1, W1 = _build_schoolbook_1x4x4()
    
    total_rank = 44
    U = np.zeros((12, total_rank))
    V = np.zeros((16, total_rank))
    W = np.zeros((12, total_rank))
    
    # 2x4x4 part (cols 0-27)
    for i in range(2):
        for j in range(4):
            U[i*4+j, 0:28] = U2[i*4+j, :]
    V[0:16, 0:28] = V2[:, :]
    for i in range(2):
        for k in range(4):
            W[k*3+i, 0:28] = W2[k*2+i, :]
            
    # 1x4x4 part (cols 28-43)
    for j in range(4):
        U[8+j, 28:44] = U1[j, :]
    V[0:16, 28:44] = V1[:, :]
    for k in range(4):
        W[k*3+2, 28:44] = W1[k, :]
        
    return U, V, W

def _build_strassen_2x4x4():
    """Strassen 2x4x4: rank 28."""
    rank = 28
    U = np.zeros((8, rank))   # A: 2x4
    V = np.zeros((16, rank))  # B: 4x4
    W = np.zeros((8, rank))   # C: 4x2 (k*2+i)
    
    col = 0
    for s in range(7):
        if s == 0:  # M1 = (A11+A22)(B11+B22)
            X = {0: [(0,0,1),(1,2,1)], 1: [(0,1,1),(1,3,1)]}
            Y = {(0,0): [(0,0,1),(2,0,1)], (0,1): [(0,1,1),(2,1,1)],
                 (1,0): [(1,0,1),(3,0,1)], (1,1): [(1,1,1),(3,1,1)]}
            out = {0: [(0,0,1),(0,1,1)], 1: [(1,2,1),(1,3,1)]}
        elif s == 1:  # M2 = (A21+A22)B11
            X = {0: [(1,0,1),(1,2,1)], 1: [(1,1,1),(1,3,1)]}
            Y = {(0,0): [(0,0,1)], (0,1): [(0,1,1)], (1,0): [(1,0,1)], (1,1): [(1,1,1)]}
            out = {0: [(1,0,1),(1,1,1)], 1: [(1,2,-1),(1,3,-1)]}
        elif s == 2:  # M3 = A11(B12-B22)
            X = {0: [(0,0,1)], 1: [(0,1,1)]}
            Y = {(0,0): [(0,2,1),(2,2,-1)], (0,1): [(0,3,1),(2,3,-1)],
                 (1,0): [(1,2,1),(3,2,-1)], (1,1): [(1,3,1),(3,3,-1)]}
            out = {0: [(0,2,1),(0,3,1)], 1: [(1,2,1),(1,3,1)]}
        elif s == 3:  # M4 = A22(B21-B11)
            X = {0: [(1,2,1)], 1: [(1,3,1)]}
            Y = {(0,0): [(2,0,1),(0,0,-1)], (0,1): [(2,1,1),(0,1,-1)],
                 (1,0): [(3,0,1),(1,0,-1)], (1,1): [(3,1,1),(1,1,-1)]}
            out = {0: [(0,0,1),(0,1,1)], 1: [(1,0,1),(1,1,1)]}
        elif s == 4:  # M5 = (A11+A12)B22
            X = {0: [(0,0,1),(0,2,1)], 1: [(0,1,1),(0,3,1)]}
            Y = {(0,0): [(2,2,1)], (0,1): [(2,3,1)], (1,0): [(3,2,1)], (1,1): [(3,3,1)]}
            out = {0: [(0,0,-1),(0,1,-1)], 1: [(0,2,1),(0,3,1)]}
        elif s == 5:  # M6 = (A21-A11)(B11+B12)
            X = {0: [(1,0,1),(0,0,-1)], 1: [(1,1,1),(0,1,-1)]}
            Y = {(0,0): [(0,0,1),(0,2,1)], (0,1): [(0,1,1),(0,3,1)],
                 (1,0): [(1,0,1),(1,2,1)], (1,1): [(1,1,1),(1,3,1)]}
            out = {0: [], 1: [(1,2,1),(1,3,1)]}
        elif s == 6:  # M7 = (A12-A22)(B21+B22)
            X = {0: [(0,2,1),(1,2,-1)], 1: [(0,3,1),(1,3,-1)]}
            Y = {(0,0): [(2,0,1),(2,2,1)], (0,1): [(2,1,1),(2,3,1)],
                 (1,0): [(3,0,1),(3,2,1)], (1,1): [(3,1,1),(3,3,1)]}
            out = {0: [(0,0,1),(0,1,1)], 1: [(1,2,1),(1,3,1)]}
        
        for j in range(2):
            for k in range(2):
                for (i, a, coeff) in X[j]:
                    U[i * 4 + a, col] += coeff
                for (b, c, coeff) in Y[(j,k)]:
                    V[b * 4 + c, col] += coeff
                for (i, k_out, coeff) in out[k]:
                    W[k_out * 2 + i, col] += coeff
                col += 1
    
    return U, V, W

def _build_schoolbook_1x4x4():
    """Schoolbook 1x4x4: rank 16."""
    rank = 16
    U = np.zeros((4, rank))
    V = np.zeros((16, rank))
    W = np.zeros((4, rank))
    for j in range(4):
        for k in range(4):
            idx = j * 4 + k
            U[j, idx] = 1
            V[idx, idx] = 1
            W[k, idx] = 1
    return U, V, W

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Start with the known rank-44 Strassen-based decomposition
    try:
        U, V, W = _try_strassen_based_44()
        if not _validate(U, V, W, n, m, p):
            # Fallback to schoolbook
            U_s, V_s, W_s = _schoolbook(n, m, p)
            return {"U": U_s, "V": V_s, "W": W_s}
        
        best_rank = U.shape[1]
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Strategy: Global Linear Dependency Search
        # Try to remove any term that is a linear combination of the others.
        U_red, V_red, W_red = _try_global_reduction(best_U, best_V, best_W, n, m, p, deadline)
        
        if U_red.shape[1] < best_rank and _validate(U_red, V_red, W_red, n, m, p):
            best_rank = U_red.shape[1]
            best_U, best_V, best_W = U_red, V_red, W_red
            print(f"Global reduction rank to {best_rank}")
            
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    except Exception as e:
        print(f"Error: {e}")
        U_s, V_s, W_s = _schoolbook(n, m, p)
        return {"U": U_s, "V": V_s, "W": W_s}

# EVOLVE-BLOCK-END
