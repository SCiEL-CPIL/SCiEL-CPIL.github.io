import numpy as np
import time
import random

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # 20s margin before 150s kill

    # Build target tensor
    M = np.zeros((n*m, m*p, p*n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                M[a, b, c] = 1

    # Schoolbook decomposition (rank 105)
    rank_schoolbook = n * m * p
    U_school = np.zeros((n*m, rank_schoolbook), dtype=int)
    V_school = np.zeros((m*p, rank_schoolbook), dtype=int)
    W_school = np.zeros((p*n, rank_schoolbook), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_school[a, col] = 1
                V_school[b, col] = 1
                W_school[c, col] = 1
                col += 1

    def verify(U, V, W):
        """Check if decomposition is exact."""
        M_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.array_equal(M_recon, M)

    def strassen_2x2x2():
        """Return Strassen decomposition for 2x2x2 as (U,V,W) with rank 7."""
        # Strassen's 7 products for 2x2x2
        # A = [[a11,a12],[a21,a22]], B = [[b11,b12],[b21,b22]]
        # C = A*B, C[i,j] = sum_k A[i,k]*B[k,j]
        # Index convention: A[i,k] -> idx = i*2+k, B[k,j] -> idx = k*2+j, C[i,j] -> idx = j*2+i (trace convention W[k*n+i,r] where n=2,p=2 so c = k*2+i... wait)
        # Actually for 2x2x2: n=2,m=2,p=2
        # a = i*m + j = i*2+j (row i, col j of A)
        # b = j*p + k = j*2+k (row j, col k of B)  
        # c = k*n + i = k*2+i (row k, col i of C) -- this is trace convention
        # So C index: row k, col i => C[k,i]
        # But standard matmul: C[i,j] = sum_k A[i,k]B[k,j]
        # With trace convention: output index is (k*n+i) meaning row=k, col=i
        # So the tensor T[a,b,c] = 1 when a=i*2+j, b=j*2+k, c=k*2+i
        # This means: for each (i,j,k), T[i*2+j, j*2+k, k*2+i] = 1
        
        # Strassen formulas for 2x2x2:
        # Let A = [[p,q],[r,s]], B = [[t,u],[v,w]]
        # Standard: C[0,0]=pt+qv, C[0,1]=pu+qw, C[1,0]=rt+sv, C[1,1]=ru+sw
        # Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        
        # Map to our indexing: A[i,k] where i=row, k=col of A (m=2 so A is 2x2)
        # a = i*m + j = i*2 + j where i in {0,1}, j in {0,1} -> A[i,j]
        # So A[0,0]=a, A[0,1]=b, A[1,0]=c, A[1,1]=d (using lowercase for entries)
        # B[k,j] where k=row of B (which is col of A), j=col of B
        # b = j*p + k = j*2 + k where j in {0,1}, k in {0,1} -> B[j,k]
        # So B[0,0]=e, B[0,1]=f, B[1,0]=g, B[1,1]=h
        # C output: c = k*n + i = k*2 + i, so C[k,i]
        # C[0,0] = k=0,i=0 -> index 0
        # C[0,1] = k=0,i=1 -> index 1
        # C[1,0] = k=1,i=0 -> index 2
        # C[1,1] = k=1,i=1 -> index 3
        
        # Strassen products (7 rank-1 terms):
        # Each term r has U[a,r], V[b,r], W[c,r]
        
        # Let me define the 7 Strassen combinations:
        # M1 = (A00+A11)(B00+B11) -> U: A00,A11 = 1; V: B00,B11 = 1; W: contributes to C
        # Actually each rank-1 term is a product of a vector in U-space, V-space, W-space
        # The 7 Strassen terms correspond to 7 specific bilinear forms
        
        # Let me just list the 7 rank-1 tensors directly:
        # Term 1: (A00+A11) * (B00+B11) -> contributes to C00, C11
        # Term 2: (A10+A11) * B00 -> contributes to C10, C11
        # Term 3: A00 * (B01-B11) -> contributes to C01, C00
        # Term 4: A11 * (B10-B00) -> contributes to C10, C00
        # Term 5: (A00+A01) * B11 -> contributes to C01, C11
        # Term 6: (A10-A00) * (B00+B01) -> contributes to C10, C11
        # Term 7: (A01-A11) * (B10+B11) -> contributes to C00, C01
        
        # This is getting complex. Let me just construct it explicitly.
        # A indices: 0=A00, 1=A01, 2=A10, 3=A11
        # B indices: 0=B00, 1=B01, 2=B10, 3=B11
        # C indices: 0=C00, 1=C01, 2=C10, 3=C11
        
        # Schoolbook for 2x2x2 has 8 terms:
        # (A00,B00,C00), (A00,B01,C01), (A01,B10,C10), (A01,B11,C11),
        # (A10,B00,C00), (A10,B01,C01), (A11,B10,C10), (A11,B11,C11)
        
        # Strassen 7 terms:
        # T1: U=[1,0,0,1] (A00,A11), V=[1,0,0,1] (B00,B11), W=[1,0,0,1] (C00,C11) -- but this gives C00+A11B11 etc
        # This approach of listing coefficients is error-prone. Let me use a verified implementation.
        
        # Verified Strassen for 2x2x2 (rank 7):
        # Using the standard Strassen decomposition
        U_s = np.zeros((4, 7), dtype=int)
        V_s = np.zeros((4, 7), dtype=int)
        W_s = np.zeros((4, 7), dtype=int)
        
        # The 7 Strassen products with their contributions:
        # I'll use a known correct set:
        # Product 1: (A00+A11)(B00+B11) -> adds to C00, C11
        #   U: A00=1, A11=1; V: B00=1, B11=1; W: C00=1, C11=1? No, each rank-1 term has one W entry
        # Actually each rank-1 term is: sum over a,b,c of U[a,r]*V[b,r]*W[c,r]
        # For term r, it contributes U[a,r]*V[b,r]*W[c,r] to M[a,b,c]
        
        # Let me define 7 terms explicitly:
        # Term 0: U=[1,0,1,0], V=[1,0,0,0], W=[0,0,1,0]  -> A00*B00*C10? Let me just use a reliable source
        # 
        # After careful thought, let me use this verified Strassen decomposition:
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # 
        # The 7 rank-1 terms (U_a, V_b, W_c) where U_a are coefficients for A entries:
        # Term 1: U=(1,0,0,1), V=(1,0,0,1), W=(1,0,0,1)  -- no this is wrong
        
        # I'll use a different approach: just verify a known decomposition
        # Actually, let me skip Strassen embedding for now and focus on a better search strategy
        
        return None  # Skip for now, focus on other strategies

    best_rank = rank_schoolbook
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()

    # Strategy: Try to reduce rank by finding linear combinations
    # For each target rank R from 104 down, try to construct a valid decomposition
    
    # Method: Start with schoolbook, try to merge terms
    # If two terms have the same W column, we can combine U and V
    # More generally, use linear algebra
    
    # Better strategy: Use iterative refinement
    # Fix U and V randomly (or from schoolbook), solve for W via least squares
    # Then fix U and W, solve for V, etc.
    
    # For exact integer solution, we need a different approach
    
    # Strategy: Try rank 104 by removing one schoolbook term and adjusting others
    # This is unlikely to work directly
    
    # Let's try a more systematic approach:
    # For R = 104, 103, ...:
    #   Try to find U,V,W of rank R that exactly decompose M
    
    # Use a coordinate descent approach:
    # 1. Initialize U,V,W from schoolbook (take first R columns)
    # 2. Iterate: fix two, solve for third exactly (if possible)
    
    # Actually, let's try a cleaner approach:
    # The tensor has structure. Let's exploit it.
    
    # For 3x5x7, the minimum rank is at most 105 (schoolbook).
    # Known bounds: For matmul, rank is between mn and... 
    # Actually for m×n×p matmul, rank is at most mn + np + mp - m - n - p + 1 in some cases
    
    # Let me try a practical search:
    # For each R from 104 down to 80:
    #   Try multiple random initializations with structured patterns
    #   Use iterative refinement
    
    # Time-budgeted search
    R_start = 104
    R_min = 50
    
    for R in range(R_start, R_min - 1, -1):
        if time.time() > deadline:
            break
        
        # Try several initializations
        n_trials = 50 if R > 90 else 20
        
        for trial in range(n_trials):
            if time.time() > deadline:
                break
            
            # Different initialization strategies
            if trial % 3 == 0:
                # Random binary
                U = np.random.randint(-1, 2, (n*m, R))
                V = np.random.randint(-1, 2, (m*p, R))
                W = np.random.randint(-1, 2, (p*n, R))
            elif trial % 3 == 1:
                # Structured: sparse random
                U = np.zeros((n*m, R), dtype=int)
                V = np.zeros((m*p, R), dtype=int)
                W = np.zeros((p*n, R), dtype=int)
                for r in range(R):
                    # Each rank-1 term touches a few entries
                    u_idx = random.sample(range(n*m), min(3, n*m))
                    v_idx = random.sample(range(m*p), min(3, m*p))
                    w_idx = random.sample(range(p*n), min(3, p*n))
                    U[u_idx, r] = random.choice([-1, 1])
                    V[v_idx, r] = random.choice([-1, 1])
                    W[w_idx, r] = random.choice([-1, 1])
            else:
                # Based on schoolbook: take R random schoolbook columns and perturb
                cols = random.sample(range(rank_schoolbook), R)
                U = U_school[:, cols].copy()
                V = V_school[:, cols].copy()
                W = W_school[:, cols].copy()
                # Small perturbation
                if random.random() < 0.5:
                    U[random.randint(0, n*m-1), random.randint(0, R-1)] += random.choice([-1, 1])
                
            # Quick check
            if verify(U, V, W):
                best_rank = R
                best_U = U
                best_V = V
                best_W = W
                break
        
        if best_rank < rank_schoolbook:
            # Found something better, keep searching for even better
            continue
        else:
            # If we couldn't find R, try lower R
            continue

    # Try one more thing: if we found rank 104, try to improve
    # by looking for linear dependencies in the current solution
    
    # Final verification
    if not verify(best_U, best_V, best_W):
        # Fallback to schoolbook
        best_U = U_school
        best_V = V_school
        best_W = W_school
        best_rank = rank_schoolbook

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
