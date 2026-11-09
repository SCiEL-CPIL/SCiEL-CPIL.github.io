"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Deadline guard
    deadline = time.time() + 110

    # Build target tensor T as (nm, mp, pn)
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def verify(U, V, W, T):
        """Check decomposition: sum_r U[a,r]*V[b,r]*W[c,r] == T[a,b,c] for all a,b,c"""
        err = 0.0
        for a in range(nm):
            implied = (V * U[a, :]) @ W.T
            err += np.sum((implied - T[a]) ** 2)
        return np.sqrt(err)

    def solve_U(U, V, W, T):
        """Solve for U given V, W, T using least squares"""
        U_new = np.zeros_like(U)
        r = U.shape[1]
        Phi_VW = np.einsum('br,cr->bcr', V, W).reshape(mp*pn, r)
        for a in range(nm):
            target = T[a].flatten()
            U_new[a, :] = np.linalg.lstsq(Phi_VW, target, rcond=None)[0]
        return U_new

    def solve_V(U, V, W, T):
        """Solve for V given U, W, T"""
        V_new = np.zeros_like(V)
        r = V.shape[1]
        Phi_UW = np.einsum('ar,cr->acr', U, W).reshape(nm*pn, r)
        for b in range(mp):
            target = T[:, b, :].flatten()
            V_new[b, :] = np.linalg.lstsq(Phi_UW, target, rcond=None)[0]
        return V_new

    def solve_W(U, V, W, T):
        """Solve for W given U, V, T"""
        W_new = np.zeros_like(W)
        r = W.shape[1]
        Phi_UV = np.einsum('ar,br->abr', U, V).reshape(nm*mp, r)
        for c in range(pn):
            target = T[:, :, c].flatten()
            W_new[c, :] = np.linalg.lstsq(Phi_UV, target, rcond=None)[0]
        return W_new

    def solve_all(U, V, W, T, n_iters=100, tol=1e-8):
        """Run ALS iterations to convergence"""
        best_err = verify(U, V, W, T)
        for iteration in range(n_iters):
            U = solve_U(U, V, W, T)
            V = solve_V(U, V, W, T)
            W = solve_W(U, V, W, T)
            
            err = verify(U, V, W, T)
            if err < best_err:
                best_err = err
            if err < tol:
                break
        return U, V, W, best_err

    def try_round(U, V, W, T):
        """Try to round to half-integers or integers and verify"""
        # Try rounding to half-integers
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        err_r = verify(U_r, V_r, W_r, T)
        if err_r < 1e-4:
            return U_r, V_r, W_r, err_r
        
        # Try rounding to integers
        U_r = np.round(U)
        V_r = np.round(V)
        W_r = np.round(W)
        err_r = verify(U_r, V_r, W_r, T)
        if err_r < 1e-4:
            return U_r, V_r, W_r, err_r
            
        return None, None, None, err_r

    best_result = None
    best_rank = n * m * p  # schoolbook = 96

    # Generate schoolbook as safe fallback
    U_fb = np.zeros((nm, best_rank))
    V_fb = np.zeros((mp, best_rank))
    W_fb = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m + j, col] = 1
                V_fb[j*p + k, col] = 1
                W_fb[k*n + i, col] = 1
                col += 1

    # --- New Strategy: Tensor Product / Kronecker Decomposition ---
    # We view the 3x4 x 4x8 multiplication as a combination of smaller problems.
    # Specifically, we can exploit the structure of the dimensions.
    # m=4, p=8.
    # Let's try to construct a rank-56 decomposition based on the prompt's hint:
    # "factor the 4x8 multiplication into a (2x4) x (2x2) structure, utilizing the rank-7 Strassen algorithm for the 2x2 sub-problems and the rank-8 schoolbook for the 2x4 part, yielding a total rank of 7 x 8 = 56."
    
    # This hint is slightly abstract. Let's interpret it as a Kronecker product of decompositions.
    # If we can decompose MM_{n1, m1, p1} with rank r1 and MM_{n2, m2, p2} with rank r2,
    # then MM_{n1*n2, m1*m2, p1*p2} can be done with rank r1*r2.
    # Here n=3, m=4, p=8.
    # We can't factor n=3 easily into 1*3.
    # But we can factor m=4=2*2 and p=8=2*4? No, 2*4=8.
    # So we have (n1=3, m1=2, p1=4) and (n2=1, m2=2, p2=2)?
    # MM_{3,2,4} rank? Schoolbook is 24.
    # MM_{1,2,2} rank? Schoolbook is 4.
    # Total rank 24*4 = 96. Not helpful.
    
    # Let's look at the hint again: "factor the 4x8 multiplication...".
    # Maybe it means we treat the multiplication A(3x4) * B(4x8) as:
    # Split B into two 4x4 blocks? B = [B1, B2].
    # Then C = A*B1 + A*B2.
    # This doesn't reduce rank directly unless we share computations.
    
    # Alternative interpretation:
    # The "4x8" part might refer to the middle dimension m=4 and output p=8.
    # Maybe we can use a decomposition for 2x2 matrices (Strassen, rank 7) and 2x4 matrices (rank 8?).
    # If we view the problem as a Kronecker product of a 3x2x4 problem and a 1x2x2 problem?
    # Rank(3x2x4) * Rank(1x2x2).
    # If we use rank 7 for the 1x2x2 part? No, 1x2x2 is just a vector-matrix mult, rank 4 (or 2 if we allow half-integers? No, exact).
    # Actually, 1x2x2 MM is just:
    # [a b] * [[c d], [e f]] = [ac+be, ad+bf]
    # This is rank 4 (schoolbook). Can it be lower?
    # T[0,0,0]=1, T[0,1,0]=1 (a*c, a*e? No, indices are (i*m+j, j*p+k, k*n+i))
    # n=1, m=2, p=2.
    # a=0: i=0, j=0. b=0: j=0, k=0. c=0: k=0, i=0. T[0,0,0]=1.
    # a=0, b=1 (j=0, k=1), c=1 (k=1, i=0). T[0,1,1]=1.
    # a=1 (i=0, j=1), b=0 (j=1, k=0), c=0 (k=0, i=0). T[1,0,0]=1.
    # a=1, b=1 (j=1, k=1), c=1 (k=1, i=0). T[1,1,1]=1.
    # This is 4 disjoint rank-1 tensors. Rank 4.
    
    # So the Kronecker approach with 1x2x2 gives factor 4.
    # We need Rank(3x2x4) * 4 < 96 => Rank(3x2x4) < 24.
    # Schoolbook for 3x2x4 is 24.
    # Can we do better for 3x2x4?
    # Lower bound for 3x2x4 is max(6, 8, 12) = 12.
    # If we find a rank-12 decomposition for 3x2x4, total rank is 48.
    # If rank-10, total 40.
    
    # Let's try to find a low rank decomposition for 3x2x4 first, then Kronecker with 1x2x2.
    # Or maybe factor differently.
    # n=3, m=4, p=8.
    # Try n=3, m=2, p=8? No.
    
    # Let's stick to the prompt's specific hint: "total rank of 7 x 8 = 56".
    # This implies a factorization into two sub-problems with ranks 7 and 8.
    # Which sub-problems have ranks 7 and 8?
    # 2x2x2 has rank 7 (Strassen).
    # What has rank 8?
    # 2x4x2? Schoolbook 16.
    # 1x4x2? Schoolbook 8.
    # So if we factor 3x4x8 into (3x1x4) x (1x4x2)? No, dimensions don't match for Kronecker.
    # Kronecker MM_{n1,m1,p1} x MM_{n2,m2,p2} gives MM_{n1*n2, m1*m2, p1*p2}.
    # We need n1*n2=3, m1*m2=4, p1*p2=8.
    # Options:
    # 1) n1=3, n2=1. m1=2, m2=2. p1=4, p2=2.
    #    Sub-problem 1: 3x2x4. Rank r1.
    #    Sub-problem 2: 1x2x2. Rank r2=4.
    #    Total rank r1*4.
    #    If r1=14, total 56.
    #    Is there a rank-14 decomposition for 3x2x4?
    #    Schoolbook is 24.
    #    If we can find one, we win with 56.
    
    # 2) n1=1, n2=3. m1=2, m2=2. p1=4, p2=2.
    #    Sub-problem 1: 1x2x4. Rank r1. Schoolbook 8.
    #    Sub-problem 2: 3x2x2. Rank r2. Schoolbook 12.
    #    Total rank 8*12 = 96. No gain.
    #    If r2 < 12? 3x2x2. Lower bound 6.
    #    If r2=7? Total 56.
    #    Is there a rank-7 decomposition for 3x2x2?
    
    # 3) n1=3, n2=1. m1=4, m2=1. p1=2, p2=4.
    #    Sub-problem 1: 3x4x2. Rank r1. Schoolbook 24.
    #    Sub-problem 2: 1x1x4. Rank r2=4.
    #    Total 24*4 = 96.
    
    # 4) n1=1, n2=3. m1=4, m2=1. p1=2, p2=4.
    #    Sub-problem 1: 1x4x2. Rank r1=8.
    #    Sub-problem 2: 3x1x4. Rank r2=12.
    #    Total 96.
    
    # So the most promising paths to 56 are:
    # A) Find rank-14 for 3x2x4, then Kronecker with 1x2x2 (rank 4). Total 56.
    # B) Find rank-7 for 3x2x2, then Kronecker with 1x2x4 (rank 8). Total 56.
    
    # Let's try to construct a rank-7 decomposition for 3x2x2.
    # n=3, m=2, p=2.
    # T[i*2+j, j*2+k, k*3+i] = 1.
    # Indices:
    # a in 0..5, b in 0..3, c in 0..5.
    # Non-zeros:
    # i=0, j=0, k=0: a=0, b=0, c=0.
    # i=0, j=0, k=1: a=0, b=1, c=3.
    # i=0, j=1, k=0: a=1, b=2, c=0.
    # i=0, j=1, k=1: a=1, b=3, c=3.
    # i=1, j=0, k=0: a=2, b=0, c=1.
    # i=1, j=0, k=1: a=2, b=1, c=4.
    # i=1, j=1, k=0: a=3, b=2, c=1.
    # i=1, j=1, k=1: a=3, b=3, c=4.
    # i=2, j=0, k=0: a=4, b=0, c=2.
    # i=2, j=0, k=1: a=4, b=1, c=5.
    # i=2, j=1, k=0: a=5, b=2, c=2.
    # i=2, j=1, k=1: a=5, b=3, c=5.
    
    # This is a sparse tensor. We can try to find a rank-7 decomposition.
    # If we succeed, we can Kronecker it with a rank-8 decomposition of 1x2x4.
    # 1x2x4 is just vector-matrix mult. Rank 8 is schoolbook.
    # Is there a rank < 8 for 1x2x4?
    # n=1, m=2, p=4.
    # T[j, j*4+k, k] = 1.
    # j=0: a=0, b=k, c=k. (0,0,0), (0,1,1), (0,2,2), (0,3,3).
    # j=1: a=1, b=4+k, c=k. (1,4,0), (1,5,1), (1,6,2), (1,7,3).
    # This is 8 disjoint rank-1 tensors. Rank 8.
    # So if we find rank 7 for 3x2x2, we get 7*8=56.
    
    # Let's implement a search for rank 7 in 3x2x2.
    # Then Kronecker product with the schoolbook 1x2x4.
    
    def kronecker_decomp(U1, V1, W1, U2, V2, W2):
        """
        Given decompositions for (n1,m1,p1) and (n2,m2,p2),
        return decomposition for (n1*n2, m1*m2, p1*p2).
        """
        n1, m1, p1 = U1.shape[0], V1.shape[0] // p1, W1.shape[0] // n1
        # This is getting complicated with shapes. Let's pass dimensions.
        pass

    def build_kronecker(n1, m1, p1, U1, V1, W1, n2, m2, p2, U2, V2, W2):
        """
        Constructs U, V, W for MM_{n1*n2, m1*m2, p1*p2} from MM_{n1,m1,p1} and MM_{n2,m2,p2}.
        Index mapping:
        a = a1 * m2 + a2? No.
        a in 0..n1*n2*m1*m2 - 1.
        a = (i1, i2, j1, j2) -> i1*n2 + i2, j1*m2 + j2.
        a = (i1*n2 + i2) * (m1*m2) + (j1*m2 + j2)
        b = (j1*m2 + j2) * (p1*p2) + (k1*p2 + k2)
        c = (k1*p2 + k2) * (n1*n2) + (i1*n2 + i2)
        
        U[a, r] = U1[a1, r1] * U2[a2, r2] where r = r1 * r2 + r2?
        Let r1 be rank of 1, r2 be rank of 2. Total rank r = r1 * r2.
        r = r1 * r2 + r2_idx.
        U[a, r1*r2 + r2_idx] = U1[a1, r1] * U2[a2, r2_idx]
        """
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        r = r1 * r2
        nm = n1 * n2 * m1 * m2
        mp = m1 * m2 * p1 * p2
        pn = p1 * p2 * n1 * n2
        
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        
        for i1 in range(n1):
            for i2 in range(n2):
                for j1 in range(m1):
                    for j2 in range(m2):
                        a1 = i1 * m1 + j1
                        a2 = i2 * m2 + j2
                        a = (i1 * n2 + i2) * (m1 * m2) + (j1 * m2 + j2)
                        
                        for k1 in range(p1):
                            for k2 in range(p2):
                                b1 = j1 * p1 + k1
                                b2 = j2 * p2 + k2
                                b = (j1 * m2 + j2) * (p1 * p2) + (k1 * p2 + k2)
                                
                                c1 = k1 * n1 + i1
                                c2 = k2 * n2 + i2
                                c = (k1 * p2 + k2) * (n1 * n2) + (i1 * n2 + i2)
                                
                                for r1_idx in range(r1):
                                    for r2_idx in range(r2):
                                        r_idx = r1_idx * r2 + r2_idx
                                        U[a, r_idx] += U1[a1, r1_idx] * U2[a2, r2_idx]
                                        V[b, r_idx] += V1[b1, r1_idx] * V2[b2, r2_idx]
                                        W[c, r_idx] += W1[c1, r1_idx] * W2[c2, r2_idx]
        
        return U, V, W

    # Step 1: Find rank-7 decomposition for 3x2x2
    n_sub, m_sub, p_sub = 3, 2, 2
    nm_sub, mp_sub, pn_sub = n_sub*m_sub, m_sub*p_sub, p_sub*n_sub
    
    T_sub = np.zeros((nm_sub, mp_sub, pn_sub))
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                T_sub[i*m_sub+j, j*p_sub+k, k*n_sub+i] = 1.0

    def solve_sub(U, V, W, T_sub, nm, mp, pn, n_iters=50, tol=1e-6):
        # Reuse solver logic but with smaller dims
        def solve_U_local(U, V, W):
            U_new = np.zeros_like(U)
            r = U.shape[1]
            Phi_VW = np.einsum('br,cr->bcr', V, W).reshape(mp*pn, r)
            for a in range(nm):
                target = T_sub[a].flatten()
                U_new[a, :] = np.linalg.lstsq(Phi_VW, target, rcond=None)[0]
            return U_new

        def solve_V_local(U, V, W):
            V_new = np.zeros_like(V)
            r = V.shape[1]
            Phi_UW = np.einsum('ar,cr->acr', U, W).reshape(nm*pn, r)
            for b in range(mp):
                target = T_sub[:, b, :].flatten()
                V_new[b, :] = np.linalg.lstsq(Phi_UW, target, rcond=None)[0]
            return V_new

        def solve_W_local(U, V, W):
            W_new = np.zeros_like(W)
            r = W.shape[1]
            Phi_UV = np.einsum('ar,br->abr', U, V).reshape(nm*mp, r)
            for c in range(pn):
                target = T_sub[:, :, c].flatten()
                W_new[c, :] = np.linalg.lstsq(Phi_UV, target, rcond=None)[0]
            return W_new

        best_err = np.inf
        for _ in range(n_iters):
            U = solve_U_local(U, V, W)
            V = solve_V_local(U, V, W)
            W = solve_W_local(U, V, W)
            err = 0.0
            for a in range(nm):
                implied = (V * U[a, :]) @ W.T
                err += np.sum((implied - T_sub[a]) ** 2)
            err = np.sqrt(err)
            if err < best_err:
                best_err = err
            if err < tol:
                break
        return U, V, W, best_err

    U_sub = None
    V_sub = None
    W_sub = None
    best_sub_err = np.inf
    
    # Try to find rank 7 for 3x2x2
    for trial in range(10):
        if time.time() > deadline - 30:
            break
        U_try = np.random.randn(nm_sub, 7) * 0.5
        V_try = np.random.randn(mp_sub, 7) * 0.5
        W_try = np.random.randn(pn_sub, 7) * 0.5
        
        U_try, V_try, W_try, err = solve_sub(U_try, V_try, W_try, T_sub, nm_sub, mp_sub, pn_sub)
        if err < best_sub_err:
            best_sub_err = err
            U_sub, V_sub, W_sub = U_try, V_try, W_try
        
        if best_sub_err < 1e-5:
            break

    # Step 2: If we found a good sub-decomposition, Kronecker with 1x2x4
    if U_sub is not None and best_sub_err < 1e-4:
        # Build 1x2x4 schoolbook (rank 8)
        n2, m2, p2 = 1, 2, 4
        nm2, mp2, pn2 = n2*m2, m2*p2, p2*n2
        r2 = 8
        U2 = np.zeros((nm2, r2))
        V2 = np.zeros((mp2, r2))
        W2 = np.zeros((pn2, r2))
        col = 0
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    U2[i*m2+j, col] = 1
                    V2[j*p2+k, col] = 1
                    W2[k*n2+i, col] = 1
                    col += 1
        
        # Kronecker product
        # n=3*1=3, m=2*2=4, p=2*4=8
        U_kron, V_kron, W_kron = build_kronecker(
            n_sub, m_sub, p_sub, U_sub, V_sub, W_sub,
            n2, m2, p2, U2, V2, W2
        )
        
        # Verify
        err_kron = verify(U_kron, V_kron, W_kron, T)
        if err_kron < 1e-4:
            U_r, V_r, W_r, err_r = try_round(U_kron, V_kron, W_kron, T)
            if U_r is not None:
                best_rank = 56
                best_result = (U_r, V_r, W_r)
    
    # Fallback to original search if Kronecker didn't work or wasn't fast enough
    if best_result is None:
        # Ranks to try: 32 (lower bound), 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90
        for r in range(32, min(96, best_rank + 1)):
            if time.time() > deadline - 15:
                break
            if r >= best_rank:
                break
            
            found = False
            n_tries = 10
            for trial in range(n_tries):
                if time.time() > deadline - 15:
                    break
                
                # Different initializations
                if trial == 0:
                    U = np.random.randn(nm, r) * 0.5
                    V = np.random.randn(mp, r) * 0.5
                    W = np.random.randn(pn, r) * 0.5
                elif trial == 1:
                    # Structured init
                    U = np.zeros((nm, r))
                    V = np.zeros((mp, r))
                    W = np.random.randn(pn, r) * 0.5
                    cols_per_j = r // m
                    for j in range(m):
                        c_start = j * cols_per_j
                        c_end = c_start + cols_per_j
                        if j == m - 1:
                            c_end = r
                        n_cols = c_end - c_start
                        if n_cols > 0:
                            U[j::m, c_start:c_end] = np.random.randn(n, n_cols) * 0.5
                            V[j*p:(j+1)*p, c_start:c_end] = np.random.randn(p, n_cols) * 0.5
                elif trial == 2:
                    U = U_fb[:, :r].copy()
                    V = V_fb[:, :r].copy()
                    W = W_fb[:, :r].copy()
                else:
                    U = np.random.randn(nm, r) * 0.5
                    V = np.random.randn(mp, r) * 0.5
                    W = np.random.randn(pn, r) * 0.5
                
                U, V, W, best_err = solve_all(U, V, W, T, n_iters=50, tol=1e-6)
                
                if best_err < 1e-5:
                    U_r, V_r, W_r, err_r = try_round(U, V, W, T)
                    if U_r is not None:
                        best_rank = r
                        best_result = (U_r, V_r, W_r)
                        found = True
                        break
                
                if found:
                    break
    
    if best_result is not None:
        U, V, W = best_result
        U_list = [[float(U[i, j]) for j in range(len(U[0]))] for i in range(len(U))]
        V_list = [[float(V[i, j]) for j in range(len(V[0]))] for i in range(len(V))]
        W_list = [[float(W[i, j]) for j in range(len(W[0]))] for i in range(len(W))]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        U_list = [[float(U_fb[i, j]) for j in range(len(U_fb[0]))] for i in range(len(U_fb))]
        V_list = [[float(V_fb[i, j]) for j in range(len(V_fb[0]))] for i in range(len(V_fb))]
        W_list = [[float(W_fb[i, j]) for j in range(len(W_fb[0]))] for i in range(len(W_fb))]
        return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
