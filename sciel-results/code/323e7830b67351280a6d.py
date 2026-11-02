"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, rank):
    """Verify tensor decomposition exactly (for integer/half-integer entries)."""
    def scale2(mat):
        out = np.array(mat, dtype=np.float64) * 2
        if np.max(np.abs(out - np.round(out))) > 1e-9:
            return None
        return np.round(out).astype(np.int64)
    U2 = scale2(U)
    V2 = scale2(V)
    W2 = scale2(W)
    if U2 is None or V2 is None or W2 is None:
        return False
    # Build target tensor
    target = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                target[a*m + b, b*p + c, c*n + a] = 1
    T_computed = np.einsum('ic,jc,kc->ijk', U2, V2, W2, optimize=True)
    T_target_8 = target * 8
    if np.max(np.abs(T_computed - T_target_8)) > 1e-6:
        return False
    return True

def _build_target(n, m, p):
    """Build the target tensor for matrix multiplication."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                T[a*m + b, b*p + c, c*n + a] = 1
    return T

def _schoolbook(n, m, p):
    """Return schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for a in range(n):
        for b in range(m):
            for c in range(p):
                U[a*m + b][col] = 1
                V[b*p + c][col] = 1
                W[c*n + a][col] = 1
                col += 1
    return U, V, W

def _strassen_2x2():
    """Return Strassen decomposition for 2x2 matrix multiplication.
    Returns U, V, W as lists of lists with 7 columns."""
    # Strassen's 7 products for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Indexing: U index = a*2+b (a=row of A, b=col of A)
    #           V index = b*2+c (b=row of B, c=col of B)
    #           W index = c*2+a (c=row of C, a=col of C)
    
    U = [[0]*7 for _ in range(4)]  # 2*2 = 4 rows
    V = [[0]*7 for _ in range(4)]  # 2*2 = 4 rows
    W = [[0]*7 for _ in range(4)]  # 2*2 = 4 rows
    
    # U: A matrix entries
    # U[a*2+b] for a in {0,1}, b in {0,1}
    # V: B matrix entries  
    # V[b*2+c] for b in {0,1}, c in {0,1}
    # W: C matrix entries
    # W[c*2+a] for c in {0,1}, a in {0,1}
    
    # M1 = (A11+A22)(B11+B22)
    # A11=U[0], A22=U[3], B11=V[0], B22=V[3]
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    # C11 += M1, C22 += M1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A21+A22)B11
    # A21=U[2], A22=U[3], B11=V[0]
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # C21 += M2, C22 -= M2
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = A11(B21-B22)
    # A11=U[0], B21=V[2], B22=V[3]
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    # C12 += M3, C22 += M3
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A22(B12-B11)
    # A22=U[3], B12=V[1], B11=V[0]
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    # C11 += M4, C21 += M4
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A11+A12)B22
    # A11=U[0], A12=U[1], B22=V[3]
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    # C11 -= M5, C12 += M5
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    # A21=U[2], A11=U[0], B11=V[0], B12=V[1]
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    # C22 += M6
    W[3][5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    # A12=U[1], A22=U[3], B21=V[2], B22=V[3]
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    # C11 += M7
    W[0][6] = 1
    
    return U, V, W

def _build_5x5_strassen_based():
    """Build a 5x5 matrix multiplication decomposition using block Strassen.
    Partition 5 = 2 + 3. Use Strassen for 2x2 blocks, schoolbook for others.
    This gives a valid decomposition with rank = 7 + 27 + 12 + 12 + 18 + 18 + 27 = let's compute.
    
    Actually, let's use a simpler approach: decompose 5x5 into blocks and use
    appropriate decompositions for each subproduct.
    
    Block structure:
    A = [[A11(2x2), A12(2x3)],
         [A21(3x2), A22(3x3)]]
    B = [[B11(2x2), B12(2x3)],
         [B21(3x2), B22(3x3)]]
    C = [[C11(2x2), C12(2x3)],
         [C21(3x2), C22(3x3)]]
    
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22  
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    
    For each term, we need a decomposition. Let's use schoolbook for simplicity
    to get a valid starting point, then optimize.
    
    Actually, let's just return the schoolbook for now and use it as initialization.
    """
    return _schoolbook(5, 5, 5)

def _als_step_fix_UV(U, V, T_flat, nm, mp, pn, target_rank):
    """Given U and V, solve for W. Returns W solution."""
    # B_flat[r] = outer(U[:,r], V[:,r]).flatten()
    B_flat = np.zeros((target_rank, nm * mp))
    for r in range(target_rank):
        B_flat[r] = np.outer(U[:, r], V[:, r]).flatten()
    
    try:
        W_sol, residuals, rank_sv, sv = np.linalg.lstsq(B_flat.T, T_flat.T, rcond=1e-12)
    except:
        return None
    
    return W_sol.T  # shape (pn, target_rank)

def _als_step_fix_VW(V, W, T_flat, nm, mp, pn, target_rank):
    """Given V and W, solve for U. 
    T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    For fixed V,W: T_flat[k, i*mp+j] = sum_r W[k,r] * U[i,r] * V[j,r]
    This is: for each k, T_flat[k,:] = W[k,:] @ diag(V_reshaped) @ U.T ... complex.
    
    Alternative: T[i,j,k] = sum_r U[i,r] * (V[j,r] * W[k,r])
    Let M[j,k] = sum_r V[j,r]*W[k,r] (mp x pn matrix)
    Then T[i,j,k] = sum_r U[i,r] * M[j,k] ... no, that's not right either.
    
    Actually: T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    Fix V and W. For each r, define X_r[i,j] = U[i,r] * V[j,r] (nm x mp matrix)
    Then T[i,j,k] = sum_r X_r[i,j] * W[k,r]
    So T_flat[k, i*mp+j] = sum_r W[k,r] * X_r[i*mp+j]
    = sum_r W[k,r] * U[i,r] * V[j,r]
    
    This is a bilinear problem in U. Let's vectorize differently.
    T[i,j,k] = sum_r U[i,r] * (V[j,r] * W[k,r])
    
    Define for each (j,k): S[j,k,r] = V[j,r] * W[k,r]
    Then T[i,j,k] = sum_r U[i,r] * S[j,k,r]
    
    So for fixed i: T[i,j,k] over all (j,k) = U[i,:] @ S_reshaped where S_reshaped has rows indexed by (j,k)
    
    Let S_mat be shape (mp*pn, target_rank) where S_mat[j*pn+k, r] = V[j,r]*W[k,r]
    Then T_reshaped[i, j*pn+k] = sum_r U[i,r] * S_mat[j*pn+k, r]
    i.e., T_reshaped = U @ S_mat.T
    
    So U = T_reshaped @ pinv(S_mat.T)
    """
    # Build S_mat: (mp*pn, target_rank)
    S_mat = np.zeros((mp * pn, target_rank))
    for j in range(mp):
        for k in range(pn):
            S_mat[j*pn+k, :] = V[j, :] * W[k, :]
    
    # T_reshaped: (nm, mp*pn) where T_reshaped[i, j*pn+k] = T[i,j,k]
    # T_flat is (pn, nm*mp) where T_flat[k, i*mp+j] = T[i,j,k]
    # So T_reshaped[i, j*pn+k] = T_flat[k, i*mp+j]
    T_reshaped = np.zeros((nm, mp * pn))
    for i in range(nm):
        for j in range(mp):
            for k in range(pn):
                T_reshaped[i, j*pn+k] = T_flat[k, i*mp+j]
    
    try:
        U_sol, residuals, rank_sv, sv = np.linalg.lstsq(S_mat.T, T_reshaped.T, rcond=1e-12)
    except:
        return None
    
    return U_sol.T  # shape (nm, target_rank)

def _als_step_fix_UW(U, W, T_flat, nm, mp, pn, target_rank):
    """Given U and W, solve for V.
    T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    Fix U and W. For each (i,k): P[i,k,r] = U[i,r] * W[k,r]
    Then T[i,j,k] = sum_r V[j,r] * P[i,k,r]
    
    So for fixed j: T[i,j,k] over all (i,k) = V[j,:] @ P_reshaped
    P_mat: (nm*pn, target_rank) where P_mat[i*pn+k, r] = U[i,r]*W[k,r]
    T_reshaped2[j, i*pn+k] = T[i,j,k]
    T_reshaped2 = V @ P_mat.T
    V = T_reshaped2 @ pinv(P_mat.T)
    """
    # Build P_mat: (nm*pn, target_rank)
    P_mat = np.zeros((nm * pn, target_rank))
    for i in range(nm):
        for k in range(pn):
            P_mat[i*pn+k, :] = U[i, :] * W[k, :]
    
    # T_reshaped2: (mp, nm*pn) where T_reshaped2[j, i*pn+k] = T[i,j,k]
    T_reshaped2 = np.zeros((mp, nm * pn))
    for i in range(nm):
        for j in range(mp):
            for k in range(pn):
                T_reshaped2[j, i*pn+k] = T_flat[k, i*mp+j]
    
    try:
        V_sol, residuals, rank_sv, sv = np.linalg.lstsq(P_mat.T, T_reshaped2.T, rcond=1e-12)
    except:
        return None
    
    return V_sol.T  # shape (mp, target_rank)

def _quantize_half(mat):
    """Quantize matrix to half-integers."""
    return np.round(mat * 2) / 2.0

def _try_rank_als(target_rank, n, m, p, T_flat, deadline, init_U=None, init_V=None, init_W=None):
    """Try to find a valid decomposition of given rank using multi-round ALS with quantization."""
    if time.time() > deadline - 15:
        return None
    
    nm, mp, pn = n*m, m*p, p*n
    max_inits = 20
    
    for init in range(max_inits):
        if time.time() > deadline - 10:
            break
        
        # Initialize
        if init_U is not None and init_V is not None and init_W is not None:
            U = np.array(init_U, dtype=np.float64)
            V = np.array(init_V, dtype=np.float64)
            W = np.array(init_W, dtype=np.float64)
            # Pad or truncate to target_rank
            if U.shape[1] < target_rank:
                U = np.pad(U, ((0,0),(0,target_rank-U.shape[1])))
            else:
                U = U[:, :target_rank]
            if V.shape[1] < target_rank:
                V = np.pad(V, ((0,0),(0,target_rank-V.shape[1])))
            else:
                V = V[:, :target_rank]
            if W.shape[1] < target_rank:
                W = np.pad(W, ((0,0),(0,target_rank-W.shape[1])))
            else:
                W = W[:, :target_rank]
        else:
            # Random initialization with small integers
            U = np.random.randint(-1, 2, size=(nm, target_rank)).astype(np.float64)
            V = np.random.randint(-1, 2, size=(mp, target_rank)).astype(np.float64)
            W = np.random.randint(-1, 2, size=(pn, target_rank)).astype(np.float64)
        
        # Run ALS for several rounds
        for als_round in range(5):
            if time.time() > deadline - 8:
                break
            
            # Solve for W given U, V
            W_new = _als_step_fix_UV(U, V, T_flat, nm, mp, pn, target_rank)
            if W_new is None:
                break
            
            # Quantize W
            W_q = _quantize_half(W_new)
            
            # Solve for U given V, W_q
            U_new = _als_step_fix_VW(V, W_q, T_flat, nm, mp, pn, target_rank)
            if U_new is None:
                break
            U_q = _quantize_half(U_new)
            
            # Solve for V given U_q, W_q
            V_new = _als_step_fix_UW(U_q, W_q, T_flat, nm, mp, pn, target_rank)
            if V_new is None:
                break
            V_q = _quantize_half(V_new)
            
            U, V, W = U_q, V_q, W_q
        
        # Final verification
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        if _verify(U_list, V_list, W_list, n, m, p, target_rank):
            return U_list, V_list, W_list
        
        # Also try integer quantization
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        U_list_i = U_int.tolist()
        V_list_i = V_int.tolist()
        W_list_i = W_int.tolist()
        if _verify(U_list_i, V_list_i, W_list_i, n, m, p, target_rank):
            return U_list_i, V_list_i, W_list_i
    
    return None

def construct():
    deadline = time.time() + 135.0  # 15s safety margin
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    T = _build_target(n, m, p)
    T_flat = T.transpose(2, 0, 1).reshape(p*n, n*m * m*p)
    
    # Try ranks from 125 down to 85
    for target_rank in range(125, 84, -1):
        if time.time() > deadline - 20:
            break
        
        result = _try_rank_als(target_rank, n, m, p, T_flat, deadline)
        if result is not None:
            U_r, V_r, W_r = result
            best_rank = target_rank
            best_result = {"U": U_r, "V": V_r, "W": W_r}
            print(f"Found valid rank {best_rank}")
            break
    
    # If we found a high rank, try to go lower using the found decomposition as init
    if best_rank > 100:
        for target_rank in range(min(best_rank - 1, 100), 84, -1):
            if time.time() > deadline - 10:
                break
            result = _try_rank_als(target_rank, n, m, p, T_flat, deadline,
                                   init_U=best_result['U'], init_V=best_result['V'], init_W=best_result['W'])
            if result is not None:
                U_r, V_r, W_r = result
                best_rank = target_rank
                best_result = {"U": U_r, "V": V_r, "W": W_r}
                print(f"Found valid rank {best_rank}")
                break
    
    return best_result
# EVOLVE-BLOCK-END
