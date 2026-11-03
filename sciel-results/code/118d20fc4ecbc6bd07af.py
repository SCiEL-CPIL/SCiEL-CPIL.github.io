"""Rewritten program for the suite task; attempts rank-27 refinement of Strassen-28."""
import numpy as np
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    deadline = time.time() + 140  # Safety margin

    def build_tensor():
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T

    T = build_tensor()

    def verify(U, V, W, tol=1e-8):
        recon = np.einsum('at,bt,ct->abc', U, V, W)
        return np.max(np.abs(recon - T)) < tol

    def to_lists(U, V, W, r):
        U_list = [[float(U[i, t]) for t in range(r)] for i in range(n * m)]
        V_list = [[float(V[i, t]) for t in range(r)] for i in range(m * p)]
        W_list = [[float(W[i, t]) for t in range(r)] for i in range(p * n)]
        return {"U": U_list, "V": V_list, "W": W_list}

    def strassen_2x2_decomp():
        U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
        # C[i,j] -> index j*2 + i
        U[0,0]=1.0; U[3,0]=1.0; V[0,0]=1.0; V[3,0]=1.0; W[0,0]=1.0; W[3,0]=1.0
        U[2,1]=1.0; U[3,1]=1.0; V[0,1]=1.0; W[1,1]=1.0; W[3,1]=-1.0
        U[0,2]=1.0; V[1,2]=1.0; V[3,2]=-1.0; W[2,2]=1.0; W[3,2]=1.0
        U[3,3]=1.0; V[2,3]=1.0; V[0,3]=-1.0; W[0,3]=1.0; W[1,3]=1.0
        U[0,4]=1.0; U[1,4]=1.0; V[3,4]=1.0; W[0,4]=-1.0; W[2,4]=1.0
        U[2,5]=1.0; U[0,5]=-1.0; V[0,5]=1.0; V[1,5]=1.0; W[3,5]=1.0
        U[1,6]=1.0; U[3,6]=-1.0; V[2,6]=1.0; V[3,6]=1.0; W[0,6]=1.0
        return U, V, W

    def build_strassen28():
        U_s, V_s, W_s = strassen_2x2_decomp()
        full_rank = 28
        U_full = np.zeros((n * m, full_rank))
        V_full = np.zeros((m * p, full_rank))
        W_full = np.zeros((p * n, full_rank))
        blocks = [
            (lambda li,lj: li*4+lj,          lambda lj,lk: lj*4+lk,          lambda lk,li: lk*2+li),
            (lambda li,lj: li*4+(lj+2),      lambda lj,lk: (lj+2)*4+lk,      lambda lk,li: lk*2+li),
            (lambda li,lj: li*4+lj,          lambda lj,lk: lj*4+(lk+2),      lambda lk,li: (lk+2)*2+li),
            (lambda li,lj: li*4+(lj+2),      lambda lj,lk: (lj+2)*4+(lk+2),  lambda lk,li: (lk+2)*2+li),
        ]
        for bi,(ga,gb,gc) in enumerate(blocks):
            for t in range(7):
                col = bi*7+t
                for li in range(2):
                    for lj in range(2):
                        U_full[ga(li,lj), col] = U_s[li*2+lj, t]
                for lj in range(2):
                    for lk in range(2):
                        V_full[gb(lj,lk), col] = V_s[lj*2+lk, t]
                for lk in range(2):
                    for li in range(2):
                        W_full[gc(lk,li), col] = W_s[lk*2+li, t]
        return U_full, V_full, W_full

    U28, V28, W28 = build_strassen28()
    
    # Fallback check
    if not verify(U28, V28, W28, tol=1e-8):
        r_school = n*m*p
        U_sch = np.zeros((n*m, r_school)); V_sch = np.zeros((m*p, r_school)); W_sch = np.zeros((p*n, r_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sch[i*m+j, col]=1.0; V_sch[j*p+k, col]=1.0; W_sch[k*n+i, col]=1.0; col+=1
        return to_lists(U_sch, V_sch, W_sch, r_school)

    # Attempt Rank 27 via Linear Dependency of Strassen-28 Tensors
    # Flatten T
    T_flat = T.flatten()
    
    # Build the 28 rank-1 tensors
    basis = np.zeros((28, T_flat.size))
    for t in range(28):
        u = U28[:, t]
        v = V28[:, t]
        w = W28[:, t]
        tensor = np.einsum('a,b,c->abc', u, v, w)
        basis[t] = tensor.flatten()
    
    # Check rank of basis matrix
    try:
        U_svd, s, Vt = np.linalg.svd(basis, full_matrices=False)
        rank_28 = np.sum(s > 1e-8)
    except:
        rank_28 = 28
        
    if rank_28 < 28:
        # Find null space vector z of basis (28 x N)
        # basis @ z = 0
        try:
            A_t = basis.T
            U_A, s_A, Vt_A = np.linalg.svd(A_t, full_matrices=True)
            tol_svd = 1e-8
            num_non_zero = np.sum(s_A > tol_svd)
            
            if num_non_zero < 28:
                z = Vt_A[num_non_zero, :].T
                
                # Solve basis @ coeff = T_flat
                coeff, residuals, rank_ls, sv = np.linalg.lstsq(basis, T_flat, rcond=None)
                
                # Adjust coeff to zero out one component
                k = np.argmax(np.abs(z))
                if abs(z[k]) > 1e-8:
                    alpha = -coeff[k] / z[k]
                    coeff_new = coeff + alpha * z
                    coeff_new[k] = 0.0
                    
                    U_new = np.zeros((n*m, 27))
                    V_new = np.zeros((m*p, 27))
                    W_new = np.zeros((p*n, 27))
                    
                    col_idx = 0
                    for t in range(28):
                        if t == k:
                            continue
                        U_new[:, col_idx] = coeff_new[t] * U28[:, t]
                        V_new[:, col_idx] = V28[:, t]
                        W_new[:, col_idx] = W28[:, t]
                        col_idx += 1
                        
                    if verify(U_new, V_new, W_new, tol=1e-8):
                        return to_lists(U_new, V_new, W_new, 27)
        except Exception:
            pass

    # If dependency check failed, try random search for rank 27
    # This is a long shot but might catch a nearby solution
    def try_rank_27(seed):
        rng = np.random.RandomState(seed)
        r = 27
        best_err = float('inf')
        best_sol = None
        
        for it in range(20):
            if time.time() > deadline:
                break
            
            # Initialize near Strassen-28? Hard to project.
            # Use random init with small variance
            U = rng.randn(n*m, r) * 0.5
            V = rng.randn(m*p, r) * 0.5
            W = rng.randn(p*n, r) * 0.5
            
            for step in range(100):
                if time.time() > deadline:
                    break
                
                recon = np.einsum('at,bt,ct->abc', U, V, W)
                F = recon - T
                err = np.sum(F*F)
                
                if err < best_err:
                    best_err = err
                    best_sol = (U.copy(), V.copy(), W.copy())
                    
                if err < 1e-8:
                    break
                
                gU = np.einsum('abc,bt,ct->at', F, V, W)
                gV = np.einsum('abc,at,ct->bt', F, U, W)
                gW = np.einsum('abc,at,bt->ct', F, U, V)
                
                lr = 0.1 / (1.0 + err**0.5)
                
                U -= lr * gU
                V -= lr * gV
                W -= lr * gW
                
            # Snap to halves
            if best_sol:
                Us = np.round(best_sol[0]*2)/2
                Vs = np.round(best_sol[1]*2)/2
                Ws = np.round(best_sol[2]*2)/2
                if verify(Us, Vs, Ws, tol=1e-8):
                    return Us, Vs, Ws
        return None

    sol = try_rank_27(42)
    if sol:
        return to_lists(sol[0], sol[1], sol[2], 27)

    # Return rank 28
    return to_lists(U28, V28, W28, 28)

# EVOLVE-BLOCK-END
