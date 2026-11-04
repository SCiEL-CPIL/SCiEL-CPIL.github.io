import numpy as np
import time

def construct():
    n, m, p = 3, 3, 5
    nm, mp, pn = n * m, m * p, p * n
    deadline = time.time() + 140.0
    
    # Strategy: Block decomposition of 3x3x5 into 3x3x3 + 3x3x2
    # For 3x3x3: use a rank-23 decomposition (Bini) or a simpler valid one
    # For 3x3x2: schoolbook rank 18, but we can do better with structure
    #
    # Actually, let's use a simpler but provably correct approach:
    # Split B into two parts: B_left (3x3) and B_right (3x2)
    # C = A * B_left + A * B_right
    #
    # For A * B_left (3x3 * 3x3): use Strassen-based 3x3 in rank 19 or use schoolbook 27
    # For A * B_right (3x3 * 3x2): schoolbook rank 18
    #
    # To guarantee correctness with integer/half-integer entries, let's use
    # a verified block construction. The safest sub-45 is:
    #
    # Use the "split" where we compute A*B by:
    #   1. Compute A * B[:, 0:3] using a rank-23 algorithm for 3x3x3
    #   2. Compute A * B[:, 3:5] using schoolbook (rank 6 for 3x1x1 each? No)
    #
    # Let me just use a direct verified construction.
    #
    # SIMPLEST VALID IMPROVEMENT: 
    # The 3x3x5 tensor can be decomposed by noting that for each column k of C,
    # C[:,k] = A * B[:,k]. We can share the A-side linear combinations.
    #
    # Actually, the cleanest sub-45 construction I can verify:
    # Use block diagonal structure. Split the 3x3 A into:
    #   A = [[a11, a12, a13],
    #        [a21, a22, a23],
    #        [a31, a32, a33]]
    #
    # Let me just implement a rank-40 construction by merging some terms.
    #
    # BETTER: Let me use the known fact that matmul(3,3,3) has rank 23.
    # I'll hardcode a verified rank-23 decomposition for 3x3x3, then add
    # rank-6 for the remaining 3x3x2 (using 3 independent 3x1x1 with shared A).
    #
    # For 3x3x2: C[:,3] = A*B[:,3], C[:,4] = A*B[:,4]
    # Schoolbook: 3*3*2 = 18. But we can share: for each (i,j), A[i,j] is used
    # in both C[i,3] and C[i,4]. So we need 3*3*2 = 18 terms still.
    #
    # Total: 23 + 18 = 41. That's better than 45.
    #
    # But I need actual rank-23 factors for 3x3x3. Let me use a simpler approach:
    # Use rank 27 for 3x3x3 (schoolbook) and find savings elsewhere.
    #
    # ALTERNATIVE: Use a 2+1 split on the summation index j:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    # Group j=0,1 together and j=2 separately:
    # C[i,k] = (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + A[i,2]*B[2,k]
    #
    # The first part is a 3x2x5 tensor, the second is 3x1x5.
    # Rank of 3x2x5 schoolbook: 3*2*5 = 30
    # Rank of 3x1x5 schoolbook: 3*1*5 = 15
    # Total: 45. No gain.
    #
    # Let me try a different split. Use the structure of B being 3x5:
    # Think of B as 5 vectors b_k in R^3. C[:,k] = A * b_k.
    # This is 5 matrix-vector multiplications of 3x3 * 3x1.
    # Each 3x3x1 has rank 9 (schoolbook). Total 45.
    #
    # To beat 45, I need to exploit that all 5 use the same A.
    #
    # KEY INSIGHT: Use a rank-r decomposition where the U factors share structure.
    # 
    # Let me just try the numerical approach but with better verification and
    # a fallback to a verified 41-rank construction.
    #
    # Actually, let me implement a concrete 41-rank construction:
    # Use schoolbook for 3x3x3 (rank 27) + schoolbook for 3x3x2 (rank 18) = 45.
    # That's not an improvement.
    #
    # Let me use a different approach: the "Strassen-like" 3x3 decomposition.
    # For 2x2x2, Strassen uses 7 instead of 8.
    # For 3x3x3, there's no simple Strassen, but there are decompositions in rank 23.
    #
    # Given the complexity, let me use a NUMERICAL approach with strong verification:
    # Try to find rank-40, 39, ... decompositions using least_squares,
    # then verify with exact rational arithmetic if possible.
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all (i,j,k)."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    def make_schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][idx] = 1
                    V[j * p + k][idx] = 1
                    W[k * n + i][idx] = 1
                    idx += 1
        return U, V, W, rank
    
    # Try to find a lower rank decomposition
    best_U, best_V, best_W, best_rank = None, None, None, 45
    
    # First, try a simple block construction that we can verify
    # Split into 3x3x3 (rank 27) + 3x3x2 (rank 18) = 45 (no gain)
    # 
    # Try a merged construction:
    # Use the fact that for the last 2 columns, we can share some A-side terms.
    
    # Let me try: construct a rank-40 decomposition by merging 5 pairs of terms.
    # For each (i,j), the terms for k=0..4 all use the same U[i*m+j, c].
    # If we can make some U entries zero and compensate, we save rank.
    
    # Actually, let me just do the numerical search properly with deadline guard.
    
    from scipy.optimize import least_squares
    
    def residuals(params, R):
        nm, mp, pn = 3 * 3, 3 * 5, 5 * 3
        U_flat = params[:nm * R].reshape(nm, R)
        V_flat = params[nm * R:nm * R + mp * R].reshape(mp, R)
        W_flat = params[nm * R + mp * R:].reshape(pn, R)
        
        res = np.zeros(45)
        idx = 0
        for i in range(3):
            for j in range(3):
                for k in range(5):
                    u = U_flat[i * 3 + j]
                    v = V_flat[j * 5 + k]
                    w = W_flat[k * 3 + i]
                    res[idx] = np.dot(u, v * w) - 1.0
                    idx += 1
        return res
    
    # Try ranks from 44 down to 30
    for target_rank in range(44, 29, -1):
        if time.time() > deadline - 20:
            break
        
        found = False
        for restart in range(3):
            if time.time() > deadline - 20:
                break
            
            rng = np.random.default_rng(42 + restart)
            U_init = rng.normal(size=(9, target_rank)) * 0.1
            V_init = rng.normal(size=(15, target_rank)) * 0.1
            W_init = rng.normal(size=(15, target_rank)) * 0.1
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = least_squares(
                    lambda p: residuals(p, target_rank),
                    x0,
                    method='lm',
                    x_scale='jac',
                    max_nfev=10000
                )
                
                if res.cost < 1e-10:
                    U_sol = res.x[:9 * target_rank].reshape(9, target_rank)
                    V_sol = res.x[9 * target_rank:9 * target_rank + 15 * target_rank].reshape(15, target_rank)
                    W_sol = res.x[9 * target_rank + 15 * target_rank:].reshape(15, target_rank)
                    
                    # Round to half-integers
                    U_r = np.round(U_sol * 2) / 2
                    V_r = np.round(V_sol * 2) / 2
                    W_r = np.round(W_sol * 2) / 2
                    
                    # Convert to lists
                    U_list = U_r.tolist()
                    V_list = V_r.tolist()
                    W_list = W_r.tolist()
                    
                    if verify_decomposition(U_list, V_list, W_list, 3, 3, 5):
                        best_U, best_V, best_W, best_rank = U_list, V_list, W_list, target_rank
                        found = True
                        break
                        
            except Exception:
                pass
        
        if found:
            # Keep trying for lower ranks
            continue
    
    if best_U is None:
        # Fallback to schoolbook
        best_U, best_V, best_W, best_rank = make_schoolbook(3, 3, 5)
    
    return {"U": best_U, "V": best_V, "W": best_W}
