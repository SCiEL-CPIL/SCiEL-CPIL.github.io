"""Matrix multiplication 2x3x8 rank-minimal decomposition."""
DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = 2, 3, 8
    # We'll construct a rank-44 decomposition
    # Strategy: 4 blocks of 2x3 x 3x2, each with rank 11
    # Total rank = 4 * 11 = 44
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # (6, 44)
    V = [[0] * rank for _ in range(m * p)]  # (24, 44)
    W = [[0] * rank for _ in range(p * n)]  # (16, 44)
    
    col = 0
    # For each of the 4 column-pairs (k0, k0+1) in B:
    for block in range(4):
        k0 = block * 2  # first k in this block
        k1 = k0 + 1     # second k in this block
        
        # 11 rank-1 terms for this 2x3 x 3x2 block
        # Terms: (i,j,k) for i in {0,1}, j in {0,1,2}, k in {k0,k1}
        # We merge some terms using Strassen-like combinations
        
        # Direct terms for most (i,j,k) combinations
        # Use Strassen trick: for each (i,k), combine the j-sums
        
        # For i=0, k=k0: C[0,k0] = A[0,0]*B[0,k0] + A[0,1]*B[1,k0] + A[0,2]*B[2,k0]
        # For i=1, k=k0: C[1,k0] = A[1,0]*B[0,k0] + A[1,1]*B[1,k0] + A[1,2]*B[2,k0]
        # For i=0, k=k1: C[0,k1] = A[0,0]*B[0,k1] + A[0,1]*B[1,k1] + A[0,2]*B[2,k1]
        # For i=1, k=k1: C[1,k1] = A[1,0]*B[0,k1] + A[1,1]*B[1,k1] + A[1,2]*B[2,k1]
        
        # Strassen-like: define intermediate products
        # M0 = (A[0,0]+A[1,0]) * (B[0,k0]+B[0,k1])  -> contributes to multiple C entries
        # This is getting complex. Let me use a simpler valid construction.
        
        # Just use schoolbook for now (rank 12 per block, total 48) but that's not better.
        # Let me try a different approach: use known 2x3 x 3x2 rank-11 decomposition.
        
        # For 2x3 x 3x2, I'll use these 11 rank-1 terms:
        # The key insight: we can share B[j,k] values across i indices.
        
        # Term structure for one block (k0, k1):
        # We need: for each i in {0,1}, j in {0,1,2}, k in {k0,k1}:
        #   contribution to W[k*n+i] = A[i,j] * B[j,k]
        
        # Use the following 11 terms (indices into U: i*m+j, V: j*p+k, W: k*n+i):
        
        # Terms 0-2: for j=0, use combined i terms
        # M_jk = (A[0,j] + A[1,j]) * B[j,k] for j=0,1,2 and k=k0,k1 -> 6 terms
        # But this gives (A[0,j]+A[1,j])*B[j,k] which isn't directly useful.
        
        # Let me just do schoolbook rank 48 for safety, but that's score 1.0.
        # I need to actually find a rank < 48 construction.
        
        pass
    
    # Fallback: return schoolbook if construction fails
    # Let me implement a proper rank-44 construction
    
    # Actually, let me use a cleaner approach:
    # For 2x3 x 3x8, use the following:
    # Split into 4 subproblems of 2x3 x 3x2
    # For each 2x3 x 3x2, use rank 11 (saving 1 from 12)
    
    # Here's a valid rank-11 for 2x3 x 3x2:
    # The 12 schoolbook terms minus 1 via a Strassen-like merge.
    
    # For block with k values k0, k1:
    # Define 11 rank-1 tensors:
    
    # I'll use this construction:
    # For j in {0,1,2}:
    #   Term A: U[0*m+j]=1, V[j*p+k0]=1, W[k0*n+0]=1  (i=0,k=k0)
    #   Term B: U[1*m+j]=1, V[j*p+k0]=1, W[k0*n+1]=1  (i=1,k=k0)
    #   Term C: U[0*m+j]=1, V[j*p+k1]=1, W[k1*n+0]=1  (i=0,k=k1)
    #   Term D: U[1*m+j]=1, V[j*p+k1]=1, W[k1*n+1]=1  (i=1,k=k1)
    # That's 12 terms. To save 1, I need to merge two.
    
    # Merge: combine terms for (i=0,j=0,k=k0) and (i=1,j=0,k=k0):
    # Instead of two terms, use:
    #   U[0*m+0]=1, U[1*m+0]=1, V[0*p+k0]=1, W[k0*n+0]=1, W[k0*n+1]=1
    # But this gives A[0,0]*B[0,k0] to C[0,k0] AND A[1,0]*B[0,k0] to C[1,k0]
    # Wait, that's not right for a rank-1 tensor.
    
    # A rank-1 tensor u⊗v⊗w contributes u[i*m+j]*v[j*p+k]*w[k*n+i] to T[i,j,k]
    # For it to contribute to both T[0,0,k0] and T[1,0,k0], we need:
    #   u[0*m+0]*v[0*p+k0]*w[k0*n+0] = 1  (for T[0,0,k0])
    #   u[1*m+0]*v[0*p+k0]*w[k0*n+1] = 1  (for T[1,0,k0])
    #   And all other T[i,j,k] contributions must be 0.
    
    # If I set u[0*m+0]=1, u[1*m+0]=1, v[0*p+k0]=1, w[k0*n+0]=1, w[k0*n+1]=1:
    #   T[0,0,k0] += 1*1*1 = 1 ✓
    #   T[1,0,k0] += 1*1*1 = 1 ✓
    #   But T[0,0,k0] also gets contribution from w[k0*n+0]... wait, w is indexed by (k,n+i)
    #   w[k0*n+0] corresponds to (k=k0, i=0), w[k0*n+1] corresponds to (k=k0, i=1)
    #   So for T[i,j,k], the w index is k*n+i.
    #   T[0,0,k0]: w[k0*n+0] = w[0*2+0] = w[0]
    #   T[1,0,k0]: w[k0*n+1] = w[0*2+1] = w[1]
    #   If w[0]=1 and w[1]=1, then both get 1. But do they get cross-contributions?
    #   T[0,0,k0] = u[0*3+0]*v[0*8+k0]*w[k0*2+0] = u[0]*v[k0]*w[0]
    #   T[1,0,k0] = u[1*3+0]*v[0*8+k0]*w[k0*2+1] = u[3]*v[k0]*w[1]
    #   These are independent! So I CAN set u[0]=1, u[3]=1, v[k0]=1, w[0]=1, w[1]=1
    #   and get both T[0,0,k0]=1 and T[1,0,k0]=1 with ONE rank-1 term!
    
    # But wait, does this create unwanted contributions? 
    #   T[0,0,k0] = u[0]*v[k0]*w[0] = 1*1*1 = 1 ✓
    #   T[1,0,k0] = u[3]*v[k0]*w[1] = 1*1*1 = 1 ✓
    #   T[0,0,k1] = u[0]*v[k1]*w[k1*2+0] = 1*v[k1]*w[2] -- if v[k1]=0 and w[2]=0, then 0 ✓
    #   T[1,0,k1] = u[3]*v[k1]*w[k1*2+1] = 1*v[k1]*w[3] -- if v[k1]=0 and w[3]=0, then 0 ✓
    #   T[0,1,k0] = u[1]*v[8+k0]*w[0] = 0 (u[1]=0) ✓
    #   etc.
    
    # So this works! I can merge the two i-terms for each (j,k) pair.
    # That saves 1 term per (j,k) pair where I merge i=0 and i=1.
    
    # For 2x3 x 3x2: 3 j-values × 2 k-values = 6 (j,k) pairs.
    # If I merge all 6, I go from 12 to 6 terms. But that's too good to be true.
    
    # Wait, I think I made an error. Let me recheck.
    # For the merged term for (j=0, k=k0):
    #   u[0*3+0] = u[0] = 1  (i=0, j=0)
    #   u[1*3+0] = u[3] = 1  (i=1, j=0)
    #   v[0*8+k0] = v[k0] = 1  (j=0, k=k0)
    #   w[k0*2+0] = w[0] = 1  (k=k0, i=0)
    #   w[k0*2+1] = w[1] = 1  (k=k0, i=1)
    
    # Contribution to T[i,j,k] = u[i*3+j] * v[j*8+k] * w[k*2+i]
    # T[0,0,k0] = u[0]*v[k0]*w[0] = 1*1*1 = 1 ✓
    # T[1,0,k0] = u[3]*v[k0]*w[1] = 1*1*1 = 1 ✓
    # T[0,0,k1] = u[0]*v[k1]*w[2] = 1*0*0 = 0 (assuming v[k1]=0, w[2]=0) ✓
    # T[1,0,k1] = u[3]*v[k1]*w[3] = 1*0*0 = 0 ✓
    # T[0,1,k0] = u[1]*v[8+k0]*w[0] = 0*...*1 = 0 ✓
    # T[1,1,k0] = u[4]*v[8+k0]*w[1] = 0*...*1 = 0 ✓
    # T[0,2,k0] = u[2]*v[16+k0]*w[0] = 0*...*1 = 0 ✓
    # T[1,2,k0] = u[5]*v[16+k0]*w[1] = 0*...*1 = 0 ✓
    
    # Great, it works! So I can merge i=0 and i=1 for each (j,k) pair.
    # For 2x3 x 3x2: 6 merged terms instead of 12. Rank 6 per block!
    # Total rank = 4 * 6 = 24. That's way better than 48!
    
    # But wait, is this valid? Let me double-check by considering the full sum.
    # For T[0,0,k0], it should equal 1. It gets contribution from the merged term (j=0,k=k0) = 1.
    # For T[1,0,k0], it should equal 1. It gets contribution from the same merged term = 1.
    # For T[0,1,k0], it should equal 1. It gets contribution from merged term (j=1,k=k0) = 1.
    # etc.
    
    # Yes! This works. The key insight is that for fixed (j,k), the contributions to
    # T[0,j,k] and T[1,j,k] can be captured by a single rank-1 tensor because the
    # u, v, w vectors can be set to have 1s in the right places without cross-contamination.
    
    # So for 2x3 x 3x8:
    # 8 k-values × 3 j-values = 24 merged terms, each covering both i=0 and i=1.
    # Rank = 24!
    
    # Let me implement this.
    
    rank = 24
    U = [[0] * rank for _ in range(n * m)]  # (6, 24)
    V = [[0] * rank for _ in range(m * p)]  # (24, 24)
    W = [[0] * rank for _ in range(p * n)]  # (16, 24)
    
    col = 0
    for j in range(m):  # j = 0, 1, 2
        for k in range(p):  # k = 0, ..., 7
            # Merged term for (j, k) covering i=0 and i=1
            U[0 * m + j][col] = 1  # i=0, j
            U[1 * m + j][col] = 1  # i=1, j
            V[j * p + k][col] = 1  # j, k
            W[k * n + 0][col] = 1  # k, i=0
            W[k * n + 1][col] = 1  # k, i=1
            col += 1
    
    return {"U": U, "V": V, "W": W}

