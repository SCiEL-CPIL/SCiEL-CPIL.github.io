"""Tensor decomposition for 3x4 * 4x8 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def _verify(U, V, W, n=3, m=4, p=8, tol=1e-9):
    """Verify decomposition: C[i,k] = sum_c W[k*n+i,c] * (sum_j U[i*m+j,c]*A[i,j]) * (sum_j' V[j'*p+k,c]*B[j',k])"""
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U + V + W):
        return False
    rng = np.random.default_rng(777)
    for _ in range(2):
        A = rng.standard_normal((n, m))
        B = rng.standard_normal((m, p))
        C = A @ B
        C_dec = np.zeros((n, p))
        for c in range(rank):
            uA = np.array([sum(U[i*m+j][c] * A[i,j] for j in range(m)) for i in range(n)])
            vB = np.array([sum(V[j*p+k][c] * B[j,k] for j in range(m)) for k in range(p)])
            for i in range(n):
                for k in range(p):
                    C_dec[i,k] += W[k*n+i][c] * uA[i] * vB[k]
        if np.max(np.abs(C - C_dec)) > tol:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook fallback: rank = n*m*p = 96
    rank_sb = nm * mp // m  # n*m*p
    U_sb = [[0]*rank_sb for _ in range(nm)]
    V_sb = [[0]*rank_sb for _ in range(mp)]
    W_sb = [[0]*rank_sb for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    best = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    # Lower bound for rank: at least max(n*p, n*m, m*p) in some sense
    # For matrix mult (n,m,p), rank >= max(n*p, n*m, m*p) is not standard
    # But rank >= n*p = 24 is a reasonable lower bound
    lb = max(n*p, n*m, m*p)  # 24, 12, 32 -> 32
    # Actually for 3x4x8, known best is around 32-40 or so
    
    # Use algebraic approach: fix structure and solve
    # The key insight: if we fix U and V randomly (full rank), W is determined linearly
    # C[i,k] = sum_j A[i,j] B[j,k] must equal sum_c W[k*n+i,c] * (sum_j U[i*m+j,c] A[i,j]) * (sum_j' V[j'*p+k,c] B[j',k])
    
    # Let's define for each c:
    #   a_c(i) = sum_j U[i*m+j, c] A[i,j]  (scalar per i)
    #   b_c(k) = sum_j V[j*p+k, c] B[j,k]  (scalar per k)
    # Then C[i,k] = sum_c W[k*n+i,c] * a_c(i) * b_c(k)
    
    # This means: for fixed A,B, C = sum_c W_col_c * (a_c * b_c^T) where a_c is length n, b_c length p
    # So C is a sum of rank-1 matrices scaled by W entries
    
    # Key: for the equality to hold for ALL A,B, we need the bilinear forms to match.
    # Let's use a different approach: enumerate over a basis of test matrices.
    
    # For A = E_ij (identity at position i,j) and B = E_jk:
    # C should be E_ik
    # a_c(i') = U[i*m+j, c] * delta(i',i) = U[i*m+j,c] if i'=i else 0
    # b_c(k') = V[j*p+k, c] * delta(k',k) = V[j*p+k,c] if k'=k else 0
    # So C[i,k] = sum_c W[k*n+i,c] * U[i*m+j,c] * V[j*p+k,c]
    # And we need this to equal delta(i,k) * ... wait no.
    # C = A @ B = E_ij @ E_jk = E_ik
    # So C[i',k'] = 1 if i'=i and k'=k, else 0.
    # Thus: sum_c W[k'*n+i',c] * U[i'*m+j,c] * V[j*p+k',c] = delta(i',i)*delta(k',k)
    
    # This gives us a system! For each (i,j,k), we need:
    # sum_c W[k*n+i,c] * U[i*m+j,c] * V[j*p+k,c] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # Wait, that's not right. The sum over j is inside.
    
    # Let me redo: C[i,k] = sum_j A[i,j] B[j,k]
    # With A=E_a b, B=E_c d: C = E_ac @ E_cd = delta(b,c) * E_ad
    # So C[i,k] = 1 if i=a, k=d, and b=c; else 0.
    
    # From decomposition: C[i,k] = sum_c' W[k*n+i,c'] * (sum_j U[i*m+j,c'] A[i,j]) * (sum_j' V[j'*p+k,c'] B[j',k])
    # With A[i,j] = delta(i,a)*delta(j,b), B[j',k] = delta(j',c)*delta(k,d):
    # = sum_c' W[k*n+i,c'] * (U[i*m+b,c'] * delta(i,a)) * (V[c*p+k,c'] * delta(k,d))
    # = delta(i,a)*delta(k,d) * sum_c' W[k*n+i,c'] * U[i*m+b,c'] * V[c*p+k,c']
    # We need this to equal delta(i,a)*delta(k,d)*delta(b,c)
    # So for i=a, k=d: sum_c' W[d*n+a,c'] * U[a*m+b,c'] * V[c*p+d,c'] = delta(b,c)
    
    # This must hold for all a,b,c,d. That's n*m*m*p = 3*4*4*8 = 384 equations.
    # With rank r, we have r*n*p + r*m*p + r*n*m = r*(24+32+12) = 68r unknowns.
    # For r=32: 68*32 = 2176 unknowns, 384 equations. Underdetermined.
    
    # Strategy: Fix U and V randomly, solve for W.
    # But W appears in each equation with a different (i,k) index.
    # Equation: for fixed (a,b,c,d) with b!=c: sum_r W[d*n+a,r] * U[a*m+b,r] * V[c*p+d,r] = 0
    # For b=c: sum_r W[d*n+a,r] * U[a*m+b,r] * V[b*p+d,r] = 1
    
    # This is a system where each equation involves only one row of W (indexed by d*n+a).
    # So we can solve for each (a,d) independently!
    
    # For fixed (a,d), the equations are over (b,c):
    # sum_r W[d*n+a,r] * U[a*m+b,r] * V[c*p+d,r] = delta(b,c)
    # This is: let w_ad be the vector W[d*n+a, :] (length r)
    # Then sum_r w_ad[r] * U[a*m+b,r] * V[c*p+d,r] = delta(b,c) for all b,c
    # This is a bilinear constraint on w_ad.
    
    # Define M_{bc} = sum_r U[a*m+b,r] * V[c*p+d,r] * w_ad[r]
    # We need M = I_m (identity matrix of size m)
    
    # This is: M = sum_r w_ad[r] * (U_col_b * V_col_c^T) where U_col_b is the b-th entry of U[a*m: a*m+m, r]
    # Actually for fixed (a,d,r): u_ar is length m (U[a*m+(a*m+j) -> no, U[i*m+j] for i=a, j=0..m-1])
    # v_dr is length p (V[j*p+k] for j=0..m-1, k=d) -> V[0*p+d], V[1*p+d], ..., V[(m-1)*p+d]
    
    # So M_{bc} = sum_r w_ad[r] * U[a*m+b, r] * V[c*p+d, r]
    # We need this to equal delta(b,c).
    
    # Let's form: for each r, define outer product o_r = u_r ⊗ v_r where u_r[j] = U[a*m+j, r], v_r[j] = V[j*p+d, r]
    # Then M = sum_r w_ad[r] * o_r
    # We need M = I, so w_ad must satisfy: sum_r w_ad[r] * o_r = I
    
    # This is a linear system in w_ad! The unknowns are w_ad[0..r-1].
    # The "matrix" of the system: for each (b,c), the coefficient of w_ad[r] is U[a*m+b,r]*V[c*p+d,r].
    # So we have m^2 equations, r unknowns.
    
    # For m=4: 16 equations, r unknowns. Need r >= 16 for a solution to exist generically.
    
    # Let's try: fix U and V randomly, solve for W for each (a,d).
    
    rng = np.random.default_rng(2024)
    
    # Try ranks from lower bound up
    for r in range(24, min(rank_sb + 1, 60)):
        if time.time() > deadline - 10:
            break
        
        found = False
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            U_arr = rng.standard_normal((nm, r))
            V_arr = rng.standard_normal((mp, r))
            
            # Solve for W
            W_arr = np.zeros((pn, r))
            success = True
            for a in range(n):
                for d in range(p):
                    # Build system: 16 equations, r unknowns
                    # A_sys[b*m+c, r_idx] = U[a*m+b, r_idx] * V[c*p+d, r_idx]
                    A_sys = np.zeros((m*m, r))
                    for b in range(m):
                        for c in range(m):
                            for ri in range(r):
                                A_sys[b*m+c, ri] = U_arr[a*m+b, ri] * V_arr[c*p+d, ri]
                    
                    # Target: I_m flattened
                    b_vec = np.zeros(m*m)
                    for b in range(m):
                        b_vec[b*m+b] = 1.0
                    
                    # Solve least squares
                    try:
                        sol, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_vec, rcond=None)
                        if len(residuals) > 0 and residuals[0] > 1e-7:
                            success = False
                            break
                        W_arr[d*n+a, :] = sol
                    except:
                        success = False
                        break
            
            if not success:
                continue
            
            # Verify
            U_list = [[float(U_arr[i,j]) for j in range(r)] for i in range(nm)]
            V_list = [[float(V_arr[i,j]) for j in range(r)] for i in range(mp)]
            W_list = [[float(W_arr[i,j]) for j in range(r)] for i in range(pn)]
            
            if _verify(U_list, V_list, W_list, n, m, p):
                # Clean to rational-looking numbers
                def clean(x):
                    if abs(x - round(x)) < 1e-9:
                        return int(round(x))
                    elif abs(2*x - round(2*x)) < 1e-9:
                        return round(2*x) / 2
                    else:
                        return round(x, 10)
                
                U_list = [[clean(v) for v in row] for row in U_list]
                V_list = [[clean(v) for v in row] for row in V_list]
                W_list = [[clean(v) for v in row] for row in W_list]
                
                best = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = r
                found = True
                break
        
        if found:
            break
    
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
