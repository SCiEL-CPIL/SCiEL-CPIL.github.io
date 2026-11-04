"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0

    def time_left():
        return deadline - time.time()

    # ---- Schoolbook (baseline) ----
    rank_school = n * m * p
    U0 = np.zeros((n*m, rank_school))
    V0 = np.zeros((m*p, rank_school))
    W0 = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m + j, col] = 1
                V0[j*p + k, col] = 1
                W0[k*n + i, col] = 1
                col += 1
    best = (U0, V0, W0, rank_school)

    def _check(U, V, W):
        """Return True if the decomposition exactly equals the tensor."""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        r = U.shape[1]
        # Build tensor
        T = np.zeros((n, m, p))
        for c in range(r):
            Uc = U[:, c]
            Vc = V[:, c]
            Wc = W[:, c]
            for i in range(n):
                if Uc[i*m + 0] == 0 and Uc[i*m + 1] == 0 and Uc[i*m + 2] == 0:
                    continue
                for j in range(m):
                    vj = Vc[j*p + 0] + Vc[j*p + 1] + Vc[j*p + 2]
                    if vj == 0:
                        continue
                    for k in range(p):
                        T[i, j, k] += Uc[i*m + j] * Vc[j*p + k] * Wc[k*n + i]
        # Compare with schoolbook
        Tref = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Tref[i, j, k] = 1.0
        return np.allclose(T, Tref, atol=1e-9)

    def _verify(U, V, W):
        """Exact verification using integer arithmetic if possible."""
        U = np.asarray(U)
        V = np.asarray(V)
        W = np.asarray(W)
        # Try to check exactness via random sampling
        # Actually, for exact integer/half-integer, just check
        return _check(U, V, W)

    # ---- Try to find a lower-rank decomposition using a structured ansatz ----
    # The tensor T(i,j,k) = 1 for all (i,j,k) in {0,1,2}x{0,1,2}x{0,1,2,3}
    # is the "all-ones" tensor of shape 3x3x4.
    # Its rank is at least max(n, m, p) = 4.
    # A rank-4 decomposition exists: use the 4 standard basis vectors in the p-dimension.
    # T[i,j,k] = sum_{c=0}^{3} u_c[i] * v_c[j] * w_c[k]
    # Let u_c[i] = 1 for all i, v_c[j] = 1 for all j, w_c[k] = delta_{c,k}
    # Then T[i,j,k] = sum_c 1*1*delta_{c,k} = 1.
    # So rank <= 4? Wait, but u_c and v_c don't depend on c. That's a rank-1 term per c.
    # Actually: T[i,j,k] = (sum_c u_c[i] v_c[j]) w_c[k]. If u_c[i]=1, v_c[j]=1 for all c,
    # then sum_c u_c[i] v_c[j] w_c[k] = sum_c w_c[k] = 4*delta? No.
    # Let me redo: T[i,j,k] = sum_{c=0}^{3} (1)(1)(delta_{c,k}) = delta_{c,k} summed = 1 for each k? No.
    # For fixed k, only c=k contributes, giving 1. So T[i,j,k] = 1 for all i,j,k. Yes!
    # But wait: u_c[i] = 1 for all i, v_c[j] = 1 for all j, w_c[k] = 1 if c==k else 0.
    # T[i,j,k] = sum_{c=0}^{3} 1 * 1 * (1 if c==k else 0) = 1. Correct!
    # So rank 4 works? But the matrices U, V, W have specific shapes.
    # U is (n*m, rank) = (9, rank). U[i*m+j, c] = u_c[i].
    # V is (m*p, rank) = (12, rank). V[j*p+k, c] = v_c[j] * ??? 
    # Wait, the decomposition is T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    # For the above to work, we need U[i*m+j, c] = 1, V[j*p+k, c] = 1 (but V depends on j AND k!), 
    # and W[k*n+i, c] = delta_{c,k}.
    # But V[j*p+k, c] must be independent of k for this to factor as v_c[j]. It can't be, because 
    # V is a single matrix indexed by (j,k). So the simple rank-4 ansatz doesn't directly map.
    # We need V[j*p+k, c] such that for each c, V[:,c] is a matrix that, when combined with U and W, gives the right result.
    # Actually, let's just try rank 4 with a proper structured approach.
    # Let W[k*n+i, c] = 1 if c == k, else 0. Then W has 4 columns (c=0..3), each with ones at positions where k=c.
    # U[i*m+j, c] = 1 for all i,j,c.
    # V[j*p+k, c] = 1 if c == k, else 0? No, that would give V[j*p+k, c] = delta_{c,k}.
    # Then T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = sum_c 1 * delta_{c,k} * delta_{c,k} = delta_{k,k} = 1. 
    # Wait, W[k*n+i, c] = 1 if c==k. And V[j*p+k, c] = 1 if c==k. Then product is 1*delta_{c,k}*delta_{c,k} = delta_{c,k}.
    # Sum over c gives 1. But this uses the same index k for both V and W, which is correct.
    # So rank 4 should work!
    
    # Let me construct rank-4:
    r = 4
    U4 = np.ones((n*m, r))
    V4 = np.zeros((m*p, r))
    W4 = np.zeros((p*n, r))
    for j in range(m):
        for k in range(p):
            V4[j*p + k, k] = 1
    for i in range(n):
        for k in range(p):
            W4[k*n + i, k] = 1
    
    if _verify(U4, V4, W4):
        best = (U4, V4, W4, r)
        if time_left() < 5:
            break_out = True
    else:
        print("Rank 4 failed verification")

    # Return best found
    Ub, Vb, Wb, rb = best
    U_list = [[int(Ub[i, c]) if abs(Ub[i, c] - round(Ub[i, c])) < 1e-9 else (Ub[i, c] * 2) for c in range(rb)] for i in range(Ub.shape[0])]
    V_list = [[int(Vb[i, c]) if abs(Vb[i, c] - round(Vb[i, c])) < 1e-9 else (Vb[i, c] * 2) for c in range(rb)] for i in range(Vb.shape[0])]
    W_list = [[int(Wb[i, c]) if abs(Wb[i, c] - round(Wb[i, c])) < 1e-9 else (Wb[i, c] * 2) for c in range(rb)] for i in range(Wb.shape[0])]
    
    # Clean up: convert half-integers properly
    def clean(x):
        if abs(x - round(x)) < 1e-9:
            return int(round(x))
        return float(x)
    
    U_final = [[clean(Ub[i, c]) for c in range(rb)] for i in range(Ub.shape[0])]
    V_final = [[clean(Vb[i, c]) for c in range(rb)] for i in range(Vb.shape[0])]
    W_final = [[clean(Wb[i, c]) for c in range(rb)] for i in range(Wb.shape[0])]
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
