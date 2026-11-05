import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Build the multiplication tensor T[a,b,c] where:
    # a = i*m + j, b = j*p + k, c = k*n + i
    # T[a,b,c] = 1 if the indices form a valid triple (i,j,k), else 0
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0

    def verify(U, V, W, tol=1e-9):
        """Verify exact decomposition"""
        rank = U.shape[1]
        # Check valid entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                    if abs(s - 1.0) > tol:
                        return False
        # Check invalid entries
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if j == j2 and k == k2 and i == i2:
                        continue
                    s = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
                    if abs(s) > tol:
                        return False
        return True

    def snap_to_half_int(M):
        return np.round(M * 2) / 2.0

    # --- Known rank-17 solution (Strassen-based) ---
    rank17 = 17
    U17 = np.zeros((n*m, rank17))
    V17 = np.zeros((m*p, rank17))
    W17 = np.zeros((p*n, rank17))

    # Col 0: (A00+A11)(B00+B11) -> C00, C10
    U17[0,0]=1; U17[4,0]=1
    V17[0,0]=1; V17[4,0]=1
    W17[0,0]=1; W17[3,0]=1

    # Col 1: (A10+A11)B00 -> C10, C11
    U17[3,1]=1; U17[4,1]=1
    V17[0,1]=1
    W17[1,1]=1; W17[3,1]=-1

    # Col 2: A00(B01-B11) -> C01, C11
    U17[0,2]=1
    V17[1,2]=1; V17[4,2]=-1
    W17[2,2]=1; W17[3,2]=1

    # Col 3: A11(B10-B00) -> C10, C11
    U17[4,3]=1
    V17[3,3]=1; V17[0,3]=-1
    W17[0,3]=1; W17[1,3]=1

    # Col 4: (A00+A01)B11 -> C01, C11
    U17[0,4]=1; U17[1,4]=1
    V17[4,4]=1
    W17[0,4]=-1; W17[2,4]=1

    # Col 5: (A10-A00)(B00+B01) -> C10, C11
    U17[3,5]=1; U17[0,5]=-1
    V17[0,5]=1; V17[1,5]=1
    W17[3,5]=1

    # Col 6: (A01-A11)(B10+B11) -> C01, C11
    U17[1,6]=1; U17[4,6]=-1
    V17[3,6]=1; V17[4,6]=1
    W17[0,6]=1

    # Cols 7-10: 2x1x1 for k=2
    U17[0,7]=1; V17[2,7]=1; W17[4,7]=1
    U17[1,8]=1; V17[5,8]=1; W17[4,8]=1
    U17[3,9]=1; V17[2,9]=1; W17[5,9]=1
    U17[4,10]=1; V17[5,10]=1; W17[5,10]=1

    # Cols 11-16: 2x1x3 for j=2
    U17[2,11]=1; V17[6,11]=1; W17[0,11]=1
    U17[2,12]=1; V17[7,12]=1; W17[2,12]=1
    U17[2,13]=1; V17[8,13]=1; W17[4,13]=1
    U17[5,14]=1; V17[6,14]=1; W17[1,14]=1
    U17[5,15]=1; V17[7,15]=1; W17[3,15]=1
    U17[5,16]=1; V17[8,16]=1; W17[5,16]=1

    if not verify(U17, V17, W17):
        raise ValueError("Rank 17 verification failed")

    best_U, best_V, best_W = U17.copy(), V17.copy(), W17.copy()
    best_rank = 17

    # --- Attempt rank 16 via structured search ---
    # Key idea: try to merge two of the "simple" product terms (cols 7-10 or 11-16)
    # by absorbing them into modified Strassen columns.
    
    # For the 2x1x1 block (cols 7-10), we have:
    # C02 = A00*B02 + A01*B12
    # C12 = A10*B02 + A11*B12
    # Can we express this with 3 terms instead of 4?
    # Try: (A00+A10)*B02, (A01+A11)*B12, and a correction
    # But we need to cancel cross terms A10*B02 from C02 and A01*B12 from C12
    
    # Alternative: use a different grouping. Let's try:
    # Term 1: (A00+A11)*B02 -> contributes A00*B02 + A11*B02
    # Term 2: (A01+A10)*B12 -> contributes A01*B12 + A10*B12
    # But C02 should be A00*B02 + A01*B12 (no A11*B02)
    # And C12 should be A10*B02 + A11*B12 (no A01*B12)
    # So we need correction terms. This doesn't directly work.
    
    # Let's try a completely different approach: modify one Strassen column to handle
    # part of the k=2 block.
    
    # Actually, let's try a direct algebraic approach for rank 16.
    # We'll use the fact that the tensor has a specific structure.
    
    # Try: can we reduce to rank 16 by combining the two 2x1x3 blocks?
    # Cols 11-16 handle j=2: C0k = A02*B2k, C1k = A12*B2k
    # These are 6 independent terms. Can we do better?
    # No, because each C0k and C1k for k=0,1,2 is independent.
    
    # So the reduction must come from the k in {0,1} blocks.
    # The Strassen part handles 2x2x2 in 7 terms.
    # The k=2 part handles 2x1x1 in 4 terms.
    # Total: 7 + 4 + 6 = 17.
    
    # To get 16, we need to save 1 term. The only place we might save is
    # in the 2x1x1 block (4 -> 3 terms) or by modifying Strassen.
    
    # Let's try: can we do 2x1x1 in 3 terms?
    # C02 = A00*B02 + A01*B12
    # C12 = A10*B02 + A11*B12
    # 
    # Try 3 terms:
    # T1 = (A00 + a*A10) * (B02 + b*B12)
    # T2 = (A01 + c*A11) * (B02 + d*B12)  
    # T3 = (e*A10 + f*A11) * (g*B02 + h*B12)
    # 
    # This is getting complex. Let's just try random search with the constraint
    # that we keep the Strassen structure for cols 0-6 and search over the remaining 10 columns.
    
    # Actually, let's try a more targeted approach:
    # Modify col 0 to also contribute to C02 and C12.
    # Original col 0: (A00+A11)(B00+B11) -> C00, C10
    # Modified: (A00+A11)(B00+B11+alpha*B02) -> C00, C10, and adds (A00+A11)*alpha*B02
    # The extra term (A00+A11)*alpha*B02 goes to C00 and C10, which is wrong.
    # Unless we subtract it elsewhere.
    
    # This is getting too complex for a quick fix. Let's just return the rank-17 solution.
    # The rank-17 solution is verified and correct.
    
    # Quick sanity check: ensure we haven't exceeded time
    if time.time() > deadline - 5:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    
    # Try one more thing: a small random perturbation search
    # Start from rank-17, remove one column, and try to adjust
    for col_to_remove in range(17):
        if time.time() > deadline - 10:
            break
        
        U16 = np.delete(U17, col_to_remove, axis=1)
        V16 = np.delete(V17, col_to_remove, axis=1)
        W16 = np.delete(W17, col_to_remove, axis=1)
        
        # Try a few random perturbations and snap to half-integers
        found = False
        for trial in range(100):
            if time.time() > deadline - 10:
                break
            
            # Random perturbation
            np.random.seed(42 + col_to_remove * 100 + trial)
            U_trial = U16 + 0.1 * np.random.randn(n*m, 16)
            V_trial = V16 + 0.1 * np.random.randn(m*p, 16)
            W_trial = W16 + 0.1 * np.random.randn(p*n, 16)
            
            # Snap to half-integers
            U_s = snap_to_half_int(U_trial)
            V_s = snap_to_half_int(V_trial)
            W_s = snap_to_half_int(W_trial)
            
            if verify(U_s, V_s, W_s):
                best_U, best_V, best_W = U_s, V_s, W_s
                best_rank = 16
                found = True
                break
        
        if found:
            break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
