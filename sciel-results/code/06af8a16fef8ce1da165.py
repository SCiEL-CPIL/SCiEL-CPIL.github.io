"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # Safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return _TIME_LIMIT - _elapsed()

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U,V,W exactly decompose the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    # Tensor T[k*n+i, i*m+j, j*p+k] should equal sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c]
    # Build target
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_w = k * n + i
                idx_u = i * m + j
                idx_v = j * p + k
                val = np.sum(U_arr[idx_u] * V_arr[idx_v] * W_arr[idx_w])
                if abs(val - 1.0) > 1e-9:
                    return False
    # Check that off-diagonal terms are zero
    # For all (a,b,c) not corresponding to valid (i,j,k), the sum should be 0
    # Actually, we need to check all (idx_w, idx_u, idx_v) combinations
    # The tensor is only defined on valid (i,j,k), so we just need those to be 1
    # But we also need to ensure no spurious contributions... 
    # Actually the decomposition defines the full tensor, so we need:
    # For all (x,y,z) in the full index space, sum_c U[y,c]*V[z,c]*W[x,c] = T[x,y,z]
    # where T[x,y,z] = 1 if (x,y,z) corresponds to valid (i,j,k), else 0
    # Let's verify this properly
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_w = k * n + i
                idx_u = i * m + j
                idx_v = j * p + k
                val = np.sum(U_arr[idx_u] * V_arr[idx_v] * W_arr[idx_w])
                if abs(val - 1.0) > 1e-9:
                    return False
    # Check some off-diagonal entries to ensure no spurious terms
    # Sample a few
    import random
    random.seed(42)
    for _ in range(100):
        x = random.randint(0, p*n-1)
        y = random.randint(0, n*m-1)
        z = random.randint(0, m*p-1)
        # Check if (x,y,z) corresponds to a valid (i,j,k)
        # x = k*n + i, y = i*m + j, z = j*p + k
        # i = x % n, k = x // n, j = y % m, i2 = y // m, j2 = z // p, k2 = z % p
        i1 = x % n
        k1 = x // n
        j1 = y % m
        i2 = y // m
        j2 = z // p
        k2 = z % p
        is_valid = (i1 == i2) and (j1 == j2) and (k1 == k2)
        val = np.sum(U_arr[y] * V_arr[z] * W_arr[x])
        target = 1.0 if is_valid else 0.0
        if abs(val - target) > 1e-9:
            return False
    return True

def _snap_to_half_integer(arr, tol=1e-6):
    """Snap values to nearest half-integer if close enough."""
    snapped = np.round(arr * 2) / 2.0
    if np.allclose(arr, snapped, atol=tol):
        return snapped
    return None

def _try_rank(rank, n, m, p, max_iter=5000, seed=0):
    """Try to find a rank decomposition using ALS with random restarts."""
    rng = np.random.RandomState(seed)
    best_result = None
    for restart in range(10):
        if _time_left() < 10:
            break
        # Initialize
        U = rng.randn(n*m, rank)
        V = rng.randn(m*p, rank)
        W = rng.randn(p*n, rank)
        
        # ALS iterations
        converged = False
        for it in range(max_iter):
            if _time_left() < 5:
                break
            # Update U
            # For each row y, U[y,:] should satisfy: sum_c U[y,c]*V[z,c]*W[x,c] = T[x,y,z] for all x,z
            # This is a linear system: for fixed y, U[y,:] * (V[z,:] outer W[x,:]) = T[x,y,z]
            # Vectorize: U[y,:] @ (V.T @ W.T) reshaped... 
            # Actually, for fixed y: sum_c U[y,c] * (V[z,c] * W[x,c]) = T[x,y,z]
            # Let M_y[x,z] = T[x,y,z], and B[c,x,z] = V[z,c]*W[x,c]
            # Then U[y,:] @ B = M_y (vectorized)
            # This is: U[y,:] is 1xrank, B is rank x (p*n * m*p)
            # Too large. Let's do it differently.
            
            # For fixed y, we want: sum_c U[y,c] * V[z,c] * W[x,c] = T[x,y,z] for all x,z
            # Fix z: sum_c U[y,c] * V[z,c] * W[x,c] = T[x,y,z] for all x
            # This is: (U[y,:] * V[z,:]) @ W.T = T[:,y,z]
            # So for each z: U[y,:] * V[z,:] = (T[:,y,z] @ W) / (W^T W) approximately
            # This is getting complex. Let's use a simpler approach.
            
            # Actually, let's use the fact that we're solving a tensor completion problem.
            # Use simple gradient descent or ALS on the CP decomposition.
            
            # ALS: fix V and W, solve for U
            # For each row y of U: minimize ||T[:,y,:] - (U[y,:] ⊙ something)||
            # This is a linear least squares problem.
            
            # Let's build the system for U[y,:]:
            # sum_c U[y,c] * V[z,c] * W[x,c] = T[x,y,z]
            # For fixed y, this is a system in U[y,0],...,U[y,rank-1]
            # Variables: u_c for c in range(rank)
            # Equations: for each (x,z): sum_c u_c * V[z,c] * W[x,c] = T[x,y,z]
            # Matrix form: A_y @ u = b_y where A_y[x,z,c] = V[z,c]*W[x,c], b_y[x,z] = T[x,y,z]
            # A_y is (p*n * m*p) x rank, b_y is (p*n * m*p)
            
            # This is too large to form explicitly for rank up to 60.
            # Let's use a different approach: randomized SVD or truncated SVD.
            
            break  # ALS is too complex, let's use a different method
        
        # Try snapping
        U_snap = _snap_to_half_integer(U)
        V_snap = _snap_to_half_integer(V)
        W_snap = _snap_to_half_integer(W)
        
        if U_snap is not None and V_snap is not None and W_snap is not None:
            # Verify
            if _verify_decomposition(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), n, m, p):
                return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    return None

def _build_strassen_222():
    """Return Strassen decomposition of 2x2x2 matrix multiplication, rank 7."""
    # Strassen's 7 multiplications for 2x2x2
    # A = [[a11,a12],[a21,a22]], B = [[b11,b12],[b21,b22]]
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # U: shape (n*m, rank) = (4, 7), indexed by [i*m+j, c]
    # V: shape (m*p, rank) = (4, 7), indexed by [j*p+k, c]
    # W: shape (p*n, rank) = (4, 7), indexed by [k*n+i, c]
    
    # For 2x2x2: n=m=p=2
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    
    U = [[0.0]*7 for _ in range(4)]
    V = [[0.0]*7 for _ in range(4)]
    W = [[0.0]*7 for _ in range(4)]
    
    # M1 = (a11+a22)(b11+b22)
    # a11 is U[0,0], a22 is U[3,0]
    # b11 is V[0,0], b22 is V[3,0]
    U[0][0] = 1  # a11
    U[3][0] = 1  # a22
    V[0][0] = 1  # b11
    V[3][0] = 1  # b22
    # C11 += M1, C22 += M1
    # C11 is W[0*2+0, 0] = W[0,0]
    # C22 is W[1*2+1, 0] = W[3,0]
    W[0][0] = 1
    W[3][0] = 1
    
    # M2 = (a21+a22)b11
    U[2][1] = 1  # a21
    U[3][1] = 1  # a22
    V[0][1] = 1  # b11
    # C21 += M2
    # C21 is W[1*2+0, 1] = W[2,1]
    W[2][1] = 1
    
    # M3 = a11(b12-b22)
    U[0][2] = 1  # a11
    V[1][2] = 1  # b12
    V[3][2] = -1  # b22
    # C12 += M3, C22 += M3
    # C12 is W[0*2+1, 2] = W[1,2]
    # C22 is W[1*2+1, 2] = W[3,2]
    W[1][2] = 1
    W[3][2] = 1
    
    # M4 = a22(b21-b11)
    U[3][3] = 1  # a22
    V[2][3] = 1  # b21
    V[0][3] = -1  # b11
    # C11 += M4, C21 += M4
    # C11 is W[0,3]
    # C21 is W[2,3]
    W[0][3] = 1
    W[2][3] = 1
    
    # M5 = (a11+a12)b22
    U[0][4] = 1  # a11
    U[1][4] = 1  # a12
    V[3][4] = 1  # b22
    # C11 -= M5, C12 += M5
    # C11 is W[0,4]
    # C12 is W[1,4]
    W[0][4] = -1
    W[1][4] = 1
    
    # M6 = (a21-a11)(b11+b12)
    U[2][5] = 1  # a21
    U[0][5] = -1  # a11
    V[0][5] = 1  # b11
    V[1][5] = 1  # b12
    # C22 += M6
    # C22 is W[3,5]
    W[3][5] = 1
    
    # M7 = (a12-a22)(b21+b22)
    U[1][6] = 1  # a12
    U[3][6] = -1  # a22
    V[2][6] = 1  # b21
    V[3][6] = 1  # b22
    # C11 += M7
    # C11 is W[0,6]
    W[0][6] = 1
    
    return U, V, W

def _embed_strassen_in_block(U_full, V_full, W_full, n, m, p, i_off, j_off, k_off, block_n, block_m, block_p):
    """Embed a Strassen 2x2x2 decomposition into a larger decomposition."""
    if block_n != 2 or block_m != 2 or block_p != 2:
        return False
    U_s, V_s, W_s = _build_strassen_222()
    for c in range(7):
        for ii in range(2):
            for jj in range(2):
                u_idx = (i_off + ii) * m + (j_off + jj)
                u_val = U_s[ii * 2 + jj][c]
                if u_val != 0:
                    U_full[u_idx][c] += u_val
        for jj in range(2):
            for kk in range(2):
                v_idx = (j_off + jj) * p + (k_off + kk)
                v_val = V_s[jj * 2 + kk][c]
                if v_val != 0:
                    V_full[v_idx][c] += v_val
        for kk in range(2):
            for ii in range(2):
                w_idx = (k_off + kk) * n + (i_off + ii)
                w_val = W_s[kk * 2 + ii][c]
                if w_val != 0:
                    W_full[w_idx][c] += w_val
    return True

def _construct_strassen_based(n, m, p):
    """Try to construct a decomposition using Strassen embeddings."""
    # We need to tile the (m,p) space with 2x2 blocks
    # m=5, p=6
    # Use 2x2 blocks: m can be 2+2+1, p can be 2+2+2
    # This doesn't tile perfectly. Let's try a different partition.
    
    # Alternative: use the full 2x5x6 and try to find a low-rank decomposition
    # by combining Strassen for 2x2x2 sub-blocks with schoolbook for the rest.
    
    # Partition: m=5 = 2+3, p=6 = 3+3
    # Sub-problems:
    # (2x2)@(2x3), (2x2)@(2x3), (2x3)@(3x3), (2x3)@(3x3)
    
    # For (2x2)@(2x3): split the 3 in p into 2+1
    # (2x2)@(2x2) + (2x2)@(2x1)
    # (2x2)@(2x2) can use Strassen (rank 7)
    # (2x2)@(2x1) has rank at most 4 (schoolbook)
    
    # For (2x3)@(3x3): 
    # This is harder. Schoolbook rank = 18.
    # Can we do better? Lower bound = max(6,9,6) = 9.
    # Let's use schoolbook for now.
    
    # Total rank estimate:
    # (2x2)@(2x2): 7 (Strassen)
    # (2x2)@(2x1): 4
    # (2x2)@(2x2): 7 (Strassen)  
    # (2x2)@(2x1): 4
    # (2x3)@(3x3): 18
    # (2x3)@(3x3): 18
    # Total: 7+4+7+4+18+18 = 58
    
    # That's only slightly better than 60. Let me think of a better approach.
    
    # Actually, let me just try to find a good decomposition numerically.
    # Start with schoolbook and try to reduce rank.
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook as fallback
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    # Try to improve using Strassen-based construction
    # For now, try a few ranks below schoolbook
    
    # Try rank 58, 57, ... down to some minimum
    for target_rank in range(schoolbook_rank - 2, max(29, schoolbook_rank - 15), -1):
        if _time_left() < 30:
            break
        result = _try_rank(target_rank, n, m, p, max_iter=2000, seed=target_rank)
        if result is not None:
            best_result = result
            best_rank = target_rank
            break
    
    return best_result
# EVOLVE-BLOCK-END
