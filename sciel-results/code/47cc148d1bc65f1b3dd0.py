def _find_2x2x3_rank10(deadline):
    n, m, p = 2, 2, 3
    rank = 10
    
    # Try a structured ansatz: 7 Strassen-like terms for k=0,1 + 3 terms for k=2
    # with coupling.
    
    # Actually, let's try: use the known result that rank(M_{2,2,3}) = 10.
    # Construct it explicitly.
    
    # Here's a valid rank-10 decomposition:
    # Terms 0-6: Strassen for 2x2x2 (k=0,1)
    # Terms 7-9: 3 additional terms for k=2
    
    # Strassen for 2x2x2 (k=0,1):
    # p1 = (A00+A11)(B00+B11)
    # p2 = (A10+A11)B00
    # p3 = A00(B11-B01)  # note: B11 here means B[1,1], B01 means B[0,1]
    # p4 = A11(B10-B00)
    # p5 = (A00+A01)B11
    # p6 = (A10-A00)(B00+B01)
    # p7 = (A01-A11)(B10+B11)
    
    # C00 = p1 + p4 - p5 + p7
    # C01 = p3 + p5
    # C10 = p2 + p4
    # C11 = p1 - p2 + p3 + p6
    
    # For k=2, we need:
    # C02 = A00*B02 + A01*B12
    # C12 = A10*B02 + A11*B12
    
    # Let's add 3 terms:
    # p8 = A00*B02
    # p9 = A01*B12
    # p10 = (A10*B02 + A11*B12) -- but this is not rank 1!
    
    # So p10 can't be a single rank-1 term for C12.
    
    # Alternative: 
    # p8 = (A00+A10)*B02 -> contributes to C02 and C12
    # p9 = A01*B12 -> contributes to C02
    # p10 = A11*B12 -> contributes to C12
    # But then C02 = p8 + p9, C12 = p8 + p10
    # That's 3 terms for k=2, total 7+3=10. 
    
    # Let me verify:
    # p8 = (A00+A10)*B02: u=(1,0,1,0), v=(1,0,0,0,0,0) [B02 is at j=0,k=2], w=(0,0,0,0,1,1) [C02 and C12]
    # p9 = A01*B12: u=(0,1,0,0), v=(0,0,0,0,0,1) [B12 is at j=1,k=2], w=(0,0,0,0,1,0) [C02]
    # p10 = A11*B12: u=(0,0,0,1), v=(0,0,0,0,0,1) [B12], w=(0,0,0,0,0,1) [C12]
    
    # Check C02: p8 + p9 = (A00+A10)*B02 + A01*B12 = A00*B02 + A10*B02 + A01*B12
    # But we need C02 = A00*B02 + A01*B12. Extra term A10*B02!
    
    # That doesn't work. Let me try:
    # p8 = A00*B02: u=(1,0,0,0), v=(1,0,0,0,0,0), w=(0,0,0,0,1,0)
    # p9 = A01*B12: u=(0,1,0,0), v=(0,0,0,0,0,1), w=(0,0,0,0,1,0)
    # p10 = (A10+A11)*B12: u=(0,0,1,1), v=(0,0,0,0,0,1), w=(0,0,0,0,0,1)
    # C12 = p10 = (A10+A11)*B12 = A10*B12 + A11*B12. But we need A10*B02 + A11*B12. Extra A10*B12, missing A10*B02.
    
    # Doesn't work either.
    
    # The issue is that C02 and C12 share structure but not in a way that allows rank-1 decomposition with only 3 terms.
    
    # Let me try a different approach: mix the k=2 terms with the k=0,1 terms.
    
    # This is getting too complex for manual construction. Let me just do a fast numerical search.
    
    # Fast approach: use numpy to solve the system with a specific sparsity pattern.
    
    # Try: 10 terms, each with a specific support. Fix the supports, solve for coefficients.
    
    # Support pattern: each term r has u_r with 2 non-zeros, v_r with 2 non-zeros, w_r with 2 non-zeros.
    # That's 4 non-zero entries per term * 10 terms = 40 non-zero entries total.
    # 12 equations, 40 unknowns. Should be solvable.
    
    # Let me try a specific pattern and solve.
    
    import itertools
    
    # Define all possible 2-element subsets of {0,1,2,3} for U, {0,1,2,3,4,5} for V, {0,1,2,3,4,5} for W
    # This is too many combinations.
    
    # Alternative: just do a quick random search with 10 restarts and 100 iterations each.
    
    np.random.seed(42)
    best_err = float('inf')
    best_sol = None
    
    for restart in range(10):
        if time.time() > deadline - 20:
            break
        
        U = np.random.randn(4, rank)
        V = np.random.randn(6, rank)
        W = np.random.randn(6, rank)
        
        for it in range(100):
            # Update U
            for ij in range(4):
                i, j = divmod(ij, 2)
                M = np.zeros((rank, p))
                for r in range(rank):
                    for k in range(p):
                        M[r, k] = V[j*p+k, r] * W[k*n+i, r]
                U[ij, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
            
            # Update V
            for jk in range(6):
                j, k = divmod(jk, 3)
                M = np.zeros((rank, n*m))
                for r in range(rank):
                    for i in range(n):
                        M[r, i*m+j] = U[i*m+j, r] * W[k*n+i, r]
                V[jk, :] = np.linalg.lstsq(M, np.ones(n*m), rcond=None)[0]
            
            # Update W
            for ki in range(6):
                k, i = divmod(ki, 2)
                M = np.zeros((rank, m))
                for r in range(rank):
                    for j in range(m):
                        M[r, j] = U[i*m+j, r] * V[j*p+k, r]
                W[ki, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                        err += (val - 1.0)**2
            
            if err < best_err:
                best_err = err
                best_sol = (U.copy(), V.copy(), W.copy())
            
            if err < 1e-12:
                break
    
    if best_sol is not None and best_err < 1e-10:
        U, V, W = best_sol
        # Try to round
        for mult in [1, 2, 4, 8]:
            U_r = np.round(U * mult) / mult
            V_r = np.round(V * mult) / mult
            W_r = np.round(W * mult) / mult
            if _verify_2x2x3(U_r, V_r, W_r):
                return U_r, V_r, W_r
    
    return None, None, None
