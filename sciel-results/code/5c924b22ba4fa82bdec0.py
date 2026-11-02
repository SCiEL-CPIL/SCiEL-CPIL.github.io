<<<<<<< SEARCH
import numpy as np
import time
from scipy.optimize import minimize
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precompute the target tensor T
    # T[u_idx, v_idx, w_idx] = 1 where u_idx = i*m+j, v_idx = j*p+k, w_idx = k*n+i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i*m + j
                v_idx = j*p + k
                w_idx = k*n + i
                T[u_idx, v_idx, w_idx] = 1.0
    
    # Initialize schoolbook as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_U = U_sb.tolist()
    best_V = V_sb.tolist()
    best_W = W_sb.tolist()
    best_rank = rank_sb
    
    # Try ranks from low to high
    # For 3x3x4 matrix multiplication, known results suggest rank could be around 18-24
    target_ranks = [18, 20, 22, 24, 26, 28]
    
    def verify_decomposition(U, V, W, R):
        """Verify that the decomposition exactly reconstructs T"""
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_recon, T, atol=1e-10)
    
    def snap_to_rationals(U, V, W, max_denom=4):
        """Snap values to nearest rational with denominator up to max_denom"""
        U_snap = np.zeros_like(U)
        V_snap = np.zeros_like(V)
        W_snap = np.zeros_like(W)
        
        for arr, snap_arr in [(U, U_snap), (V, V_snap), (W, W_snap)]:
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    val = arr[i, j]
                    # Try to snap to rational with small denominator
                    best_frac = None
                    best_diff = float('inf')
                    for denom in range(1, max_denom+1):
                        num = round(val * denom)
                        frac = num / denom
                        diff = abs(val - frac)
                        if diff < best_diff:
                            best_diff = diff
                            best_frac = frac
                    snap_arr[i, j] = best_frac
        
        return U_snap, V_snap, W_snap
    
    # Strategy: For each target rank, use local optimization from multiple random starts
    # then snap to rationals and verify
    for R in target_ranks:
        if time.time() > deadline - 30:
            break
            
        # Number of restarts based on rank
        max_restarts = 8 if R <= 22 else 5
        
        found = False
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
                
            # Random initialization with different seeds
            seed = 100 + R * 10 + restart
            np.random.seed(seed)
            
            # Initialize with small random values
            U0 = np.random.randn(n*m, R) * 0.5
            V0 = np.random.randn(m*p, R) * 0.5
            W0 = np.random.randn(p*n, R) * 0.5
            
            # Flatten parameters
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            def objective(x):
                U = x[:n*m*R].reshape(n*m, R)
                V = x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                W = x[n*m*R + m*p*R :].reshape(p*n, R)
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                diff = T_recon - T
                return np.sum(diff * diff)
            
            # Use L-BFGS-B optimization
            try:
                res = minimize(objective, x0, method='L-BFGS-B', 
                              options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
                
                if res.fun < 1e-8:
                    U_opt = res.x[:n*m*R].reshape(n*m, R)
                    V_opt = res.x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                    W_opt = res.x[n*m*R + m*p*R :].reshape(p*n, R)
                    
                    # Try different snap strategies
                    for max_denom in [2, 3, 4, 6, 8]:
                        U_snap, V_snap, W_snap = snap_to_rationals(U_opt, V_opt, W_opt, max_denom)
                        
                        if verify_decomposition(U_snap, V_snap, W_snap, R):
                            best_rank = R
                            best_U = U_snap.tolist()
                            best_V = V_snap.tolist()
                            best_W = W_snap.tolist()
                            found = True
                            break
                    
                    if found:
                        break
                        
            except Exception:
                continue
        
        if found:
            # Found a valid decomposition at rank R
            # Since we iterate from low to high, this is the minimal rank found
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
=======
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def _strassen_2x2x2():
    """Return rank-7 Strassen decomposition for 2x2 matrix multiplication.
    Returns U (4,7), V (4,7), W (4,7) such that T = sum_r U[:,r] V[:,r] W[:,r]^T
    where T[i,j,k] = 1 if i+k==j (standard 2x2x2 matmul tensor).
    """
    # Strassen's 7 products for 2x2:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # m1 = (a+d)(e+h), m2 = (c+d)e, m3 = a(f-h), m4 = d(g-e)
    # m5 = (a+b)h, m6 = (c-a)(e+f), m7 = (b-d)(g+h)
    # Result: [[m1+m4-m5+m7, m3+m5], [m2+m4, m1-m2+m3+m6]]
    
    # For tensor T[u,v,w] where u=row of A (0,1), v=row of B (0,1) mapped to (j,k), w=col of result
    # Standard convention: T[i*m+j, j*p+k, k*n+i] for n=m=p=2
    # u = i*2+j (i=row of A, j=col of A = row of B)
    # v = j*2+k (j=row of B, k=col of B)
    # w = k*2+i (k=col of B, i=row of result = col of A)
    
    # Let me use a simpler indexing: for 2x2x2, u in {0,1,2,3}, v in {0,1,2,3}, w in {0,1,2,3}
    # T[u,v,w] = 1 if the corresponding (i,j,k) satisfies the matmul constraint
    
    # Actually, let's just build it directly.
    # A has entries A[i,j] at position u=i*2+j
    # B has entries B[j,k] at position v=j*2+k  
    # C has entries C[i,k] at position w=k*2+i
    
    # T[u,v,w] = 1 iff there exist i,j,k with u=i*2+j, v=j*2+k, w=k*2+i
    # i.e., T[i*2+j, j*2+k, k*2+i] = 1
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen terms for 2x2:
    # m1 = (a+d)(e+h): a=A[0,0], d=A[1,1], e=B[0,0], h=B[1,1]
    #   u=0 (a) and u=3 (d) -> U[0,0]=1, U[3,0]=1
    #   v=0 (e) and v=3 (h) -> V[0,0]=1, V[3,0]=1
    #   w=0 (C[0,0]) and w=3 (C[1,1]) -> W[0,0]=1, W[3,0]=1
    #   But we need C[0,0] = m1+m4-m5+m7 and C[1,1] = m1-m2+m3+m6
    #   So for m1: contributes +1 to C[0,0] and +1 to C[1,1]
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    
    # m2 = (c+d)e: c=A[1,0], d=A[1,1], e=B[0,0]
    #   u=2 (c), u=3 (d) -> U[2,1]=1, U[3,1]=1
    #   v=0 (e) -> V[0,1]=1
    #   C[1,0] = m2+m4, so m2 contributes +1 to C[1,0], w=2
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1
    
    # m3 = a(f-h): a=A[0,0], f=B[0,1], h=B[1,1]
    #   u=0 (a) -> U[0,2]=1
    #   v=1 (f), v=3 (h) -> V[1,2]=1, V[3,2]=-1
    #   C[0,1] = m3+m5, C[1,1] = m1-m2+m3+m6, so m3 contributes +1 to C[0,1] and +1 to C[1,1]
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[1,2] = 1; W[3,2] = 1
    
    # m4 = d(g-e): d=A[1,1], g=B[1,0], e=B[0,0]
    #   u=3 (d) -> U[3,3]=1
    #   v=2 (g), v=0 (e) -> V[2,3]=1, V[0,3]=-1
    #   C[0,0] = m1+m4-m5+m7, C[1,0] = m2+m4, so m4 contributes +1 to C[0,0] and +1 to C[1,0]
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    W[0,3] = 1; W[2,3] = 1
    
    # m5 = (a+b)h: a=A[0,0], b=A[0,1], h=B[1,1]
    #   u=0 (a), u=1 (b) -> U[0,4]=1, U[1,4]=1
    #   v=3 (h) -> V[3,4]=1
    #   C[0,0] = m1+m4-m5+m7 (m5 contributes -1), C[0,1] = m3+m5 (m5 contributes +1)
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = -1; W[1,4] = 1
    
    # m6 = (c-a)(e+f): c=A[1,0], a=A[0,0], e=B[0,0], f=B[0,1]
    #   u=2 (c), u=0 (a) -> U[2,5]=1, U[0,5]=-1
    #   v=0 (e), v=1 (f) -> V[0,6]=1, V[1,6]=1  (wait, let me re-index)
    #   Actually v=0 (e), v=1 (f) -> V[0,5]=1, V[1,5]=1
    #   C[1,1] = m1-m2+m3+m6, so m6 contributes +1 to C[1,1]
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1
    
    # m7 = (b-d)(g+h): b=A[0,1], d=A[1,1], g=B[1,0], h=B[1,1]
    #   u=1 (b), u=3 (d) -> U[1,6]=1, U[3,6]=-1
    #   v=2 (g), v=3 (h) -> V[2,6]=1, V[3,6]=1
    #   C[0,0] = m1+m4-m5+m7 (m7 contributes +1), C[0,1] = m3+m5 (no m7)
    #   Wait, C[0,1] should not have m7. Let me recheck.
    #   C[0,0] = m1+m4-m5+m7, C[0,1] = m3+m5, C[1,0] = m2+m4, C[1,1] = m1-m2+m3+m6
    #   So m7 only appears in C[0,0]
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = 1
    
    return U, V, W

def _build_rank34_strassen():
    """Build rank-34 decomposition for 3x3x4 matrix multiplication using Strassen.
    
    Strategy: Partition 3x3x4 into:
    - Top-left 2x2x3 block: use Strassen for the 2x2 part, extended to 3 columns
    - Remaining rows/cols: schoolbook
    
    Actually, let's use a cleaner partition:
    - Rows of A: {0,1,2}, Cols of A: {0,1,2}, Rows of B: {0,1,2}, Cols of B: {0,1,2,3}
    
    Partition A into [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    Partition B into [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x2, B21 is 1x2, B22 is 1x2
    
    C = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
         [A21*B11 + A22*B21, A21*B12 + A22*B22]]
    
    Using Strassen for the 2x2x2 blocks:
    - A11*B11: rank 7
    - A11*B12: this is 2x2 * 2x2, rank 7
    - A21*B11: 1x2 * 2x2 = 1x2, rank 4 (schoolbook)
    - A22*B21: 1x1 * 1x2 = 1x2, rank 2 (schoolbook)
    - A12*B21: 2x1 * 1x2 = 2x2, rank 4 (schoolbook)
    - A12*B22: 2x1 * 1x2 = 2x2, rank 4 (schoolbook)
    - A21*B12: 1x2 * 2x2 = 1x2, rank 4 (schoolbook)
    - A22*B22: 1x1 * 1x2 = 1x2, rank 2 (schoolbook)
    
    Total: 7+7+4+2+4+4+4+2 = 34
    """
    n, m, p = 3, 3, 4
    R = 34
    U = np.zeros((n*m, R))  # (9, 34)
    V = np.zeros((m*p, R))  # (12, 34)
    W = np.zeros((p*n, R))  # (12, 34)
    
    # Indexing: u = i*3+j (i=row of A, j=col of A), v = j*4+k (j=row of B, k=col of B), w = k*3+i (k=col of B, i=row of C)
    
    # Partition: A rows {0,1} vs {2}, A cols {0,1} vs {2}
    # B rows {0,1} vs {2}, B cols {0,1} vs {2,3}
    
    col_offset = 0
    
    # Block 1: A11*B11 (2x2 * 2x2 -> 2x2), Strassen rank 7
    # A11 entries: A[0,0]=u0, A[0,1]=u1, A[1,0]=u3, A[1,1]=u4
    # B11 entries: B[0,0]=v0, B[0,1]=v1, B[1,0]=v4, B[1,1]=v5
    # C11 entries: C[0,0]=w0, C[0,1]=w3, C[1,0]=w4, C[1,1]=w7
    S_U, S_V, S_W = _strassen_2x2x2()
    # Map 2x2 indices to 3x3/3x4 indices
    # For A: 2x2 index (i,j) -> 3x3 index i*3+j
    # For B: 2x2 index (j,k) -> 3x4 index j*4+k
    # For C: 2x2 index (i,k) -> 3x4 index k*3+i
    a_map = [0, 1, 3, 4]  # (0,0)->0, (0,1)->1, (1,0)->3, (1,1)->4
    b_map = [0, 1, 4, 5]  # (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
    c_map = [0, 3, 4, 7]  # (0,0)->0, (0,1)->3, (1,0)->4, (1,1)->7
    
    for r in range(7):
        for idx_2x2, val in enumerate(S_U[:, r]):
            if val != 0:
                U[a_map[idx_2x2], col_offset + r] = val
        for idx_2x2, val in enumerate(S_V[:, r]):
            if val != 0:
                V[b_map[idx_2x2], col_offset + r] = val
        for idx_2x2, val in enumerate(S_W[:, r]):
            if val != 0:
                W[c_map[idx_2x2], col_offset + r] = val
    col_offset += 7
    
    # Block 2: A11*B12 (2x2 * 2x2 -> 2x2), Strassen rank 7
    # A11 same as above
    # B12 entries: B[0,2]=v2, B[0,3]=v3, B[1,2]=v6, B[1,3]=v7
    # C12 entries: C[0,2]=w2, C[0,3]=w5, C[1,2]=w6, C[1,3]=w9
    b_map2 = [2, 3, 6, 7]
    c_map2 = [2, 5, 6, 9]
    
    for r in range(7):
        for idx_2x2, val in enumerate(S_U[:, r]):
            if val != 0:
                U[a_map[idx_2x2], col_offset + r] = val
        for idx_2x2, val in enumerate(S_V[:, r]):
            if val != 0:
                V[b_map2[idx_2x2], col_offset + r] = val
        for idx_2x2, val in enumerate(S_W[:, r]):
            if val != 0:
                W[c_map2[idx_2x2], col_offset + r] = val
    col_offset += 7
    
    # Block 3: A21*B11 (1x2 * 2x2 -> 1x2), schoolbook rank 4
    # A21: A[2,0]=u6, A[2,1]=u7
    # B11: B[0,0]=v0, B[0,1]=v1, B[1,0]=v4, B[1,1]=v5
    # C21: C[2,0]=w8, C[2,1]=w11
    # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0]
    # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1]
    terms = [(6, 0, 8), (7, 4, 8), (6, 1, 11), (7, 5, 11)]
    for u_idx, v_idx, w_idx in terms:
        U[u_idx, col_offset] = 1
        V[v_idx, col_offset] = 1
        W[w_idx, col_offset] = 1
        col_offset += 1
    
    # Block 4: A22*B21 (1x1 * 1x2 -> 1x2), schoolbook rank 2
    # A22: A[2,2]=u8
    # B21: B[2,0]=v8, B[2,1]=v9
    # C21: C[2,0]=w8, C[2,1]=w11
    terms = [(8, 8, 8), (8, 9, 11)]
    for u_idx, v_idx, w_idx in terms:
        U[u_idx, col_offset] = 1
        V[v_idx, col_offset] = 1
        W[w_idx, col_offset] = 1
        col_offset += 1
    
    # Block 5: A12*B21 (2x1 * 1x2 -> 2x2), schoolbook rank 4
    # A12: A[0,2]=u2, A[1,2]=u5
    # B21: B[2,0]=v8, B[2,1]=v9
    # C12: C[0,2]=w2, C[0,3]=w5, C[1,2]=w6, C[1,3]=w9
    # C[0,2] = A[0,2]*B[2,0]
    # C[0,3] = A[0,2]*B[2,1]
    # C[1,2] = A[1,2]*B[2,0]
    # C[1,3] = A[1,2]*B[2,1]
    terms = [(2, 8, 2), (2, 9, 5), (5, 8, 6), (5, 9, 9)]
    for u_idx, v_idx, w_idx in terms:
        U[u_idx, col_offset] = 1
        V[v_idx, col_offset] = 1
        W[w_idx, col_offset] = 1
        col_offset += 1
    
    # Block 6: A12*B22 (2x1 * 1x2 -> 2x2), schoolbook rank 4
    # A12: A[0,2]=u2, A[1,2]=u5
    # B22: B[2,2]=v10, B[2,3]=v11
    # C12: C[0,2]=w2, C[0,3]=w5, C[1,2]=w6, C[1,3]=w9
    terms = [(2, 10, 2), (2, 11, 5), (5, 10, 6), (5, 11, 9)]
    for u_idx, v_idx, w_idx in terms:
        U[u_idx, col_offset] = 1
        V[v_idx, col_offset] = 1
        W[w_idx, col_offset] = 1
        col_offset += 1
    
    # Block 7: A21*B12 (1x2 * 2x2 -> 1x2), schoolbook rank 4
    # A21: A[2,0]=u6, A[2,1]=u7
    # B12: B[0,2]=v2, B[0,3]=v3, B[1,2]=v6, B[1,3]=v7
    # C22: C[2,2]=w10, C[2,3]=w13
    # C[2,2] = A[2,0]*B[0,2] + A[2,1]*B[1,2]
    # C[2,3] = A[2,0]*B[0,3] + A[2,1]*B[1,3]
    terms = [(6, 2, 10), (7, 6, 10), (6, 3, 13), (7, 7, 13)]
    for u_idx, v_idx, w_idx in terms:
        U[u_idx, col_offset] = 1
        V[v_idx, col_offset] = 1
        W[w_idx, col_offset] = 1
        col_offset += 1
    
    # Block 8: A22*B22 (1x1 * 1x2 -> 1x2), schoolbook rank 2
    # A22: A[2,2]=u8
    # B22: B[2,2]=v10, B[2,3]=v11
    # C22: C[2,2]=w10, C[2,3]=w13
    terms = [(8, 10, 10), (8, 11, 13)]
    for u_idx, v_idx, w_idx in terms:
        U[u_idx, col_offset] = 1
        V[v_idx, col_offset] = 1
        W[w_idx, col_offset] = 1
        col_offset += 1
    
    return U, V, W, R

def _build_target_tensor():
    """Build the target tensor T for 3x3x4 matrix multiplication."""
    n, m, p = 3, 3, 4
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i*m + j
                v_idx = j*p + k
                w_idx = k*n + i
                T[u_idx, v_idx, w_idx] = 1.0
    return T

def _verify_decomposition(U, V, W, T, atol=1e-10):
    """Verify that the decomposition exactly reconstructs T."""
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=atol)

def _snap_to_rationals(U, V, W, max_denom=4):
    """Snap values to nearest rational with denominator up to max_denom."""
    U_snap = np.zeros_like(U)
    V_snap = np.zeros_like(V)
    W_snap = np.zeros_like(W)
    
    for arr, snap_arr in [(U, U_snap), (V, V_snap), (W, W_snap)]:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = arr[i, j]
                best_frac = None
                best_diff = float('inf')
                for denom in range(1, max_denom + 1):
                    num = round(val * denom)
                    frac = num / denom
                    diff = abs(val - frac)
                    if diff < best_diff:
                        best_diff = diff
                        best_frac = frac
                snap_arr[i, j] = best_frac
    
    return U_snap, V_snap, W_snap

def _random_basis_reduction(U, V, W, T, R, max_iterations=200, seed=42):
    """Try to reduce the rank of a decomposition by applying random basis transformations
    and checking for linearly dependent columns via SVD.
    
    The key insight: if we apply the same invertible transformation P to the 'rank' dimension
    of U, V, W, the tensor is unchanged but the effective rank might reveal dependencies.
    Specifically, if some columns become linearly dependent in the matricizations, we can
    eliminate them.
    
    Returns (U_reduced, V_reduced, W_reduced, new_rank) or None if no reduction found.
    """
    np.random.seed(seed)
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = R
    
    for iteration in range(max_iterations):
        # Apply random invertible transformation to the rank dimension
        P = np.random.randn(R, R)
        # Make it more structured to increase chance of revealing dependencies
        # Mix of random and sparse
        if iteration % 3 == 0:
            P = np.eye(R) + np.random.randn(R, R) * 0.1
        elif iteration % 3 == 1:
            P = np.diag(np.random.choice([0.5, 1, 2], R)) + np.random.randn(R, R) * 0.05
        else:
            P = np.random.randn(R, R)
        
        if np.abs(np.linalg.det(P)) < 1e-10:
            continue
        
        U_t = U @ P
        V_t = V @ P
        W_t = W @ P
        
        # Check if the decomposition is still valid
        if not _verify_decomposition(U_t, V_t, W_t, T, atol=1e-8):
            continue
        
        # Try to find linearly dependent columns by looking at matricizations
        # If we can find a subset of columns that spans the same space, we can reduce rank
        
        # Strategy: use SVD on the matricizations to find the effective rank
        # Matricize U: (n*m, R) -> already in that form
        # The effective rank of the tensor is the minimum rank over all matricizations
        
        # Try to eliminate columns by checking if removing column r still gives valid decomposition
        # with adjusted other columns (this is more complex)
        
        # Simpler approach: check if any column is a linear combination of others
        # by looking at the null space of each factor
        
        # For each factor, compute the column space dimension
        # If dim(col(U_t)) < R or dim(col(V_t)) < R or dim(col(W_t)) < R,
        # then there's redundancy
        
        U_rank = np.linalg.matrix_rank(U_t, tol=1e-8)
        V_rank = np.linalg.matrix_rank(V_t, tol=1e-8)
        W_rank = np.linalg.matrix_rank(W_t, tol=1e-8)
        
        min_rank = min(U_rank, V_rank, W_rank)
        
        if min_rank < best_rank:
            # We found a lower effective rank!
            # Now we need to actually construct a decomposition with that rank
            # Use SVD to find the basis
            
            # Use the factor with minimum rank to define the new basis
            if U_rank <= V_rank and U_rank <= W_rank:
                # U has the minimum rank, use its column space
                U_svd, s, Vt_svd = np.linalg.svd(U_t, full_matrices=False)
                # Keep only the top min_rank components
                k = min_rank
                U_new = U_svd[:, :k] * s[:k]  # This gives back U_t approximately
                # Actually, we need U_t = U_svd @ diag(s) @ Vt_svd
                # So U_t[:, :k] is not quite right. Let's think differently.
                
                # Better: find a transformation Q (R x k) such that U_t @ Q ≈ U_t
                # and the new decomposition uses k terms
                # U_new = U_t @ Q, V_new = V_t @ Q, W_new = W_t @ Q
                # But this doesn't work because we need U_new to have k columns
                
                # Actually, if U_t has rank k, then U_t = M @ N where M is (nm x k) and N is (k x R)
                # Then U_t @ P^{-1} @ ... hmm this is getting complicated.
                
                # Let me try a different approach: if the matricization has rank k < R,
                # it means there exists a left null vector or the columns span a k-dim space.
                # We can find a basis for the column space and re-express.
                
                # Use economy SVD: U_t = U_svd @ S @ Vt_svd
                # U_svd: (nm, R), S: (R,), Vt_svd: (R, R)
                # The column space is spanned by U_svd[:, :k]
                # So U_t = U_svd[:, :k] @ diag(s[:k]) @ Vt_svd[:k, :]
                # Let U_new = U_svd[:, :k] @ diag(s[:k])  (nm x k)
                # Let Q = Vt_svd[:k, :]  (k x R)
                # Then U_t = U_new @ Q
                # Similarly for V and W: V_t = V_new @ Q, W_t = W_new @ Q
                # where V_new = V_t @ Q.T @ inv(Q @ Q.T) ... this is getting messy.
                
                # Simpler: just keep the top k singular vectors and project
                # This is approximate, so let's just record that we found rank k
                # and try to construct it properly.
                
                # For now, let's just note the rank and try to build from scratch
                pass
            
            # For simplicity in this iteration, let's just try to directly eliminate
            # columns that are near-zero after transformation
            for r in range(R):
                if np.allclose(U_t[:, r], 0, atol=1e-10) and \
                   np.allclose(V_t[:, r], 0, atol=1e-10) and \
                   np.allclose(W_t[:, r], 0, atol=1e-10):
                    # This column is zero, remove it
                    mask = [i for i in range(R) if i != r]
                    U_new = U_t[:, mask]
                    V_new = V_t[:, mask]
                    W_new = W_t[:, mask]
                    if _verify_decomposition(U_new, V_new, W_new, T, atol=1e-10):
                        best_U, best_V, best_W = U_new.copy(), V_new.copy(), W_new.copy()
                        best_rank = len(mask)
                        R = len(mask)
                        break
    
    if best_rank < R:
        return best_U, best_V, best_W, best_rank
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Build target tensor
    T = _build_target_tensor()
    
    # Start with rank-34 Strassen-based decomposition
    U34, V34, W34, R34 = _build_rank34_strassen()
    
    # Verify the rank-34 decomposition
    if not _verify_decomposition(U34, V34, W34, T, atol=1e-10):
        # Fall back to schoolbook if Strassen construction has errors
        rank_sb = n * m * p
        U_sb = np.zeros((n*m, rank_sb))
        V_sb = np.zeros((m*p, rank_sb))
        W_sb = np.zeros((p*n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k, col] = 1
                    W_sb[k*n+i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    best_U = U34.copy()
    best_V = V34.copy()
    best_W = W34.copy()
    best_rank = R34
    
    # Try to reduce rank using random basis transformations
    if time.time() < deadline - 30:
        # Multiple seeds for robustness
        for seed in range(20):
            if time.time() > deadline - 20:
                break
            
            result = _random_basis_reduction(U34, V34, W34, T, R34, 
                                             max_iterations=50, seed=seed)
            if result is not None:
                U_red, V_red, W_red, new_rank = result
                if new_rank < best_rank:
                    best_U, best_V, best_W = U_red.copy(), V_red.copy(), W_red.copy()
                    best_rank = new_rank
    
    # Try to snap to rationals for cleaner output
    for max_denom in [2, 4, 8]:
        U_snap, V_snap, W_snap = _snap_to_rationals(best_U, best_V, best_W, max_denom)
        if _verify_decomposition(U_snap, V_snap, W_snap, T, atol=1e-10):
            best_U, best_V, best_W = U_snap, V_snap, W_snap
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
>>>>>>> REPLACE
