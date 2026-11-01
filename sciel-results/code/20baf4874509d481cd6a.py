"""Matrix multiplication 2x2x15 - rank-minimal tensor decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [2, 2, 15]

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Check if the decomposition exactly reproduces matrix multiplication."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_2x2x2():
    """Return Strassen rank-7 decomposition for 2x2x2 matrix multiplication.
    U shape (4,7), V shape (4,7), W shape (4,7).
    Indices: U[i*2+j], V[j*2+k], W[k*2+i] for i,j,k in {0,1}.
    """
    # Strassen's 7 multiplications for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # U[i*2+j, c] for A[i,j], V[j*2+k, c] for B[j,k], W[k*2+i, c] for C[i,k]
    # U rows: A00=0, A01=1, A10=2, A11=3
    # V rows: B00=0, B01=1, B10=2, B11=3
    # W rows: C00=0, C01=1, C10=2, C11=3 (W[k*2+i]: k=0,i=0->0; k=0,i=1->1; k=1,i=0->2; k=1,i=1->3)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1  # C00 += M1, C11 += M1
    
    # M2 = (A10+A11)B00
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1; W[3,1] = -1  # C10 += M2, C11 -= M2
    
    # M3 = A00(B01-B11)
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[1,2] = 1; W[3,2] = 1  # C01 += M3, C11 += M3
    
    # M4 = A11(B10-B00)
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    W[0,3] = 1; W[2,3] = 1  # C00 += M4, C10 += M4
    
    # M5 = (A00+A01)B11
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = -1; W[1,4] = 1  # C00 -= M5, C01 += M5
    
    # M6 = (A10-A00)(B00+B01)
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1  # C11 += M6
    
    # M7 = (A01-A11)(B10+B11)
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = 1  # C00 += M7
    
    return U, V, W

def _schoolbook_2x2x15():
    """Schoolbook rank-60 decomposition."""
    n, m, p = 2, 2, 15
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _build_strassen_blocks():
    """Build decomposition using Strassen on each 2x2x2 block.
    For 2x2x15: split p into 15 blocks of size 1... wait, p=15 is odd.
    Actually, we can group the p-dimension. For each pair of p-columns, use Strassen.
    But 15 is odd, so we have 7 pairs + 1 single.
    
    Better approach: Use Strassen on the 2x2 (n x m) part for each k.
    For each k in 0..14, we compute C[:,k] = A * B[:,k] which is 2x2 times 2x1.
    This is just 4 multiplications per k, so 60 total (same as schoolbook).
    
    Alternative: Group k's. For k=0,1: compute C[:,0:2] = A * B[:,0:2] (2x2 * 2x2).
    Use Strassen: 7 mults for 2x2x2. For 7 pairs: 7*7=49, plus 1 for the last: 49+4=53.
    """
    n, m, p = 2, 2, 15
    # 7 pairs of k + 1 single
    num_pairs = p // 2  # 7
    remainder = p % 2   # 1
    
    total_rank = num_pairs * 7 + remainder * 4
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    Su, Sv, Sw = _strassen_2x2x2()
    
    col = 0
    for pair in range(num_pairs):
        k0 = pair * 2
        k1 = pair * 2 + 1
        for c in range(7):
            # U: same for all pairs (acts on A)
            U[:, col] = Su[:, c]
            # V: acts on B[:, k0:k1]
            # B rows: B[0,k0]=k0, B[0,k1]=k0+1, B[1,k0]=2+k0, B[1,k1]=2+k0+1
            # V rows in local: B00=0, B01=1, B10=2, B11=3
            # V[j*2+k_local] where j in {0,1}, k_local in {0,1}
            # Global: V[j*p + k]
            V[k0, col] = Sv[0, c]      # B[0,k0]
            V[k0+1, col] = Sv[1, c]    # B[0,k1]
            V[p+k0, col] = Sv[2, c]    # B[1,k0]
            V[p+k0+1, col] = Sv[3, c]  # B[1,k1]
            # W: acts on C[:, k0:k1]
            # W rows: C[i,k] -> global index k*n+i
            # W[k_local*2 + i]
            W[k0*2, col] = Sw[0, c]    # C[0,k0]
            W[k0*2+1, col] = Sw[1, c]  # C[1,k0]
            W[(k0+1)*2, col] = Sw[2, c]  # C[0,k1]
            W[(k0+1)*2+1, col] = Sw[3, c] # C[1,k1]
            col += 1
    
    # Handle remainder (single k)
    if remainder:
        k0 = num_pairs * 2
        for j in range(m):
            for i in range(n):
                # Simple multiplication: A[i,j] * B[j,k0] -> C[i,k0]
                U[i*m+j, col] = 1
                V[j*p+k0, col] = 1
                W[k0*n+i, col] = 1
                col += 1
    
    return U, V, W

def _try_eliminate(U, V, W, n, m, p, max_attempts=200, time_budget=60):
    """Try to reduce rank by eliminating terms via linear combination."""
    current_rank = U.shape[1]
    best_rank = current_rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    start_time = time.time()
    
    for attempt in range(max_attempts):
        if time.time() - start_time > time_budget:
            break
        
        # Pick a random term to try to eliminate
        c_elim = random.randint(0, current_rank - 1)
        
        # Check if term c_elim is a linear combination of others
        # For each (i,j,k): U[i*m+j,c_elim]*V[j*p+k,c_elim]*W[k*n+i,c_elim]
        # should equal sum_{c != c_elim} alpha_c * U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        
        n_eq = n * m * p
        n_unknowns = current_rank - 1
        
        if n_unknowns < n_eq:
            continue  # Can't solve (underdetermined from wrong side)
        
        # Build system
        A_mat = np.zeros((n_eq, n_unknowns))
        b_vec = np.zeros(n_eq)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    b_vec[eq_idx] = U[i*m+j, c_elim] * V[j*p+k, c_elim] * W[k*n+i, c_elim]
                    unknown_idx = 0
                    for c in range(current_rank):
                        if c == c_elim:
                            continue
                        A_mat[eq_idx, unknown_idx] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        unknown_idx += 1
                    eq_idx += 1
        
        # Solve least squares
        try:
            alpha, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            res = np.linalg.norm(A_mat @ alpha - b_vec)
            
            if res < 1e-8:
                # Build new decomposition
                mask = [c for c in range(current_rank) if c != c_elim]
                new_U = np.zeros((U.shape[0], n_unknowns))
                new_V = np.zeros((V.shape[0], n_unknowns))
                new_W = np.zeros((W.shape[0], n_unknowns))
                
                for idx, c in enumerate(mask):
                    scale = 1.0 + alpha[idx]
                    new_U[:, idx] = U[:, c] * scale
                    new_V[:, idx] = V[:, c]
                    new_W[:, idx] = W[:, c]
                
                # Verify
                if _verify(new_U, new_V, new_W, n, m, p):
                    U, V, W = new_U, new_V, new_W
                    current_rank = n_unknowns
                    best_rank = current_rank
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        except Exception:
            pass
    
    return best_U, best_V, best_W, best_rank

def _try_random_perturbation(U, V, W, n, m, p, time_budget=30):
    """Try random perturbations to find lower-rank decompositions."""
    current_rank = U.shape[1]
    best_rank = current_rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    start_time = time.time()
    
    while time.time() - start_time < time_budget and current_rank > 1:
        # Try to reduce by 1
        c_elim = random.randint(0, current_rank - 1)
        n_eq = n * m * p
        n_unknowns = current_rank - 1
        
        if n_unknowns < n_eq:
            break
        
        A_mat = np.zeros((n_eq, n_unknowns))
        b_vec = np.zeros(n_eq)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    b_vec[eq_idx] = U[i*m+j, c_elim] * V[j*p+k, c_elim] * W[k*n+i, c_elim]
                    unknown_idx = 0
                    for c in range(current_rank):
                        if c == c_elim:
                            continue
                        A_mat[eq_idx, unknown_idx] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        unknown_idx += 1
                    eq_idx += 1
        
        try:
            alpha, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            res = np.linalg.norm(A_mat @ alpha - b_vec)
            
            if res < 1e-8:
                mask = [c for c in range(current_rank) if c != c_elim]
                new_U = np.zeros((U.shape[0], n_unknowns))
                new_V = np.zeros((V.shape[0], n_unknowns))
                new_W = np.zeros((W.shape[0], n_unknowns))
                
                for idx, c in enumerate(mask):
                    scale = 1.0 + alpha[idx]
                    new_U[:, idx] = U[:, c] * scale
                    new_V[:, idx] = V[:, c]
                    new_W[:, idx] = W[:, c]
                
                if _verify(new_U, new_V, new_W, n, m, p):
                    U, V, W = new_U, new_V, new_W
                    current_rank = n_unknowns
                    best_rank = current_rank
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                else:
                    break
            else:
                break
        except Exception:
            break
    
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for 150s limit
    
    # Start with Strassen-based decomposition (rank 53)
    U, V, W = _build_strassen_blocks()
    current_rank = U.shape[1]
    
    # Verify initial decomposition
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = _schoolbook_2x2x15()
        current_rank = U.shape[1]
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = current_rank
    
    # Try to reduce rank further
    while time.time() < deadline:
        U_new, V_new, W_new, rank_new = _try_eliminate(
            U, V, W, n, m, p, 
            max_attempts=100, 
            time_budget=min(30, deadline - time.time() - 5)
        )
        
        if rank_new < best_rank:
            best_rank = rank_new
            best_U, best_V, best_W = U_new, V_new, W_new
            U, V, W = U_new, V_new, W_new
            current_rank = rank_new
        else:
            # Try a different approach: random reordering
            if time.time() < deadline - 10:
                # Shuffle columns and try again
                perm = list(range(current_rank))
                random.shuffle(perm)
                U_shuf = U[:, perm]
                V_shuf = V[:, perm]
                W_shuf = W[:, perm]
                
                U_new2, V_new2, W_new2, rank_new2 = _try_eliminate(
                    U_shuf, V_shuf, W_shuf, n, m, p,
                    max_attempts=50,
                    time_budget=min(20, deadline - time.time() - 5)
                )
                
                if rank_new2 < best_rank:
                    best_rank = rank_new2
                    best_U, best_V, best_W = U_new2, V_new2, W_new2
                    U, V, W = U_new2, V_new2, W_new2
                    current_rank = rank_new2
                else:
                    break
    
    # Final verification
    if not _verify(best_U, best_V, best_W, n, m, p):
        # Fallback to schoolbook
        U_sb, V_sb, W_sb = _schoolbook_2x2x15()
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
