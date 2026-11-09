"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0  # safety margin before 120s limit

    def verify(U, V, W, n, m, p):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        rank = len(U[0]) if U else 0
        # Build the target tensor T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i
        T = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T.add((a, b, c))
        # Check all (a,b,c) triples
        for a in range(n * m):
            for b in range(m * p):
                s = 0.0
                for r in range(rank):
                    ua = U[a][r]
                    if ua == 0:
                        continue
                    vb = V[b][r]
                    if vb == 0:
                        continue
                    s += ua * vb * W[c_idx(a, b, r, n, m, p, W)]
                # Actually let's do it properly with full triple loop
                pass
        # More efficient: iterate over all (a,b,c)
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    s = 0.0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    expected = 1 if (a, b, c) in T else 0
                    if abs(s - expected) > 1e-9:
                        return False
        return True

    def c_idx(a, b, r, n, m, p, W):
        return 0  # placeholder, not used

    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def kron_product(U1, V1, W1, U2, V2, W2):
        """Kronecker product of two tensor decompositions.
        If T1[a1,b1,c1] = sum_r U1[a1,r]*V1[b1,r]*W1[c1,r]
        and T2[a2,b2,c2] = sum_s U2[a2,s]*V2[b2,s]*W2[c2,s]
        then T[a,b,c] = T1[a1,b1,c1]*T2[a2,b2,c2] where
        a = a1*n2 + a2, b = b1*p2 + b2, c = c1*n1' + c2
        Wait, need to be careful with indices.
        
        For matrix mult: T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i
        If we have n = n1*n2, m = m1*m2, p = p1*p2:
        i = i1*n2 + i2, j = j1*m2 + j2, k = k1*p2 + k2
        a = i*m + j = (i1*n2+i2)*(m1*m2) + (j1*m2+j2)
          = i1*n2*m1*m2 + i2*m1*m2 + j1*m2 + j2
        This is getting complex. Let me use the standard Kronecker structure.
        
        Actually for the specific case 4x4x9 = 2x2x3 ⊗ 2x2x3:
        n=4=2*2, m=4=2*2, p=9=3*3
        i = i1*2 + i2, j = j1*2 + j2, k = k1*3 + k2
        a = i*4 + j = (i1*2+i2)*4 + (j1*2+j2) = i1*8 + i2*4 + j1*2 + j2
        b = j*9 + k = (j1*2+j2)*9 + (k1*3+k2) = j1*18 + j2*9 + k1*3 + k2
        c = k*4 + i = (k1*3+k2)*4 + (i1*2+i2) = k1*12 + k2*4 + i1*2 + i2
        
        So a = a1*4 + a2 where a1 = i1*2+j1 (from 2x2 part) and a2 = i2*2+j2
        Wait: a1 = i1*2 + j1 has range 0..3, a2 = i2*2 + j2 has range 0..3
        a = i1*8 + i2*4 + j1*2 + j2 = (i1*2+j1)*4 + (i2*2+j2) = a1*4 + a2. Yes!
        
        b = j1*18 + j2*9 + k1*3 + k2
        b1 = j1*3 + k1 (from 2x2x3 V index, range 0..5), b2 = j2*3 + k2 (range 0..5)
        b = j1*18 + j2*9 + k1*3 + k2 = (j1*3+k1)*6 + (j2*3+k2) = b1*6 + b2. Yes!
        
        c = k1*12 + k2*4 + i1*2 + i2
        c1 = k1*2 + i1 (from 2x2x3 W index, range 0..5), c2 = k2*2 + i2 (range 0..5)
        c = k1*12 + k2*4 + i1*2 + i2 = (k1*2+i1)*6 + (k2*2+i2) = c1*6 + c2. Yes!
        """
        r1 = len(U1[0]) if U1 else 0
        r2 = len(U2[0]) if U2 else 0
        rank = r1 * r2
        n1 = len(U1)  # n1*m1
        m1 = len(V1) // 1  # need to figure out dimensions
        # U1 shape: (n1*m1, r1), V1 shape: (m1*p1, r1), W1 shape: (p1*n1, r1)
        # U2 shape: (n2*m2, r2), V2 shape: (m2*p2, r2), W2 shape: (p2*n2, r2)
        n1m1 = len(U1)
        m1p1 = len(V1)
        p1n1 = len(W1)
        n2m2 = len(U2)
        m2p2 = len(V2)
        p2n2 = len(W2)
        
        # Infer n1, m1, p1
        # n1*m1 = n1m1, m1*p1 = m1p1, p1*n1 = p1n1
        # For 2x2x3: n1=2, m1=2, p1=3 -> n1m1=4, m1p1=6, p1n1=6
        n1 = n1m1
        m1 = m1p1 // n1 if n1m1 != 0 else 1
        # Actually let's just pass the dimensions
        pass

    # Let's try a direct approach: construct 4x4x9 decomposition
    # using the Kronecker product of two 2x2x3 decompositions.
    
    # First, let me get a valid 2x2x3 rank-7 decomposition.
    # I'll construct it carefully.
    
    # For 2x2x3 matrix multiplication:
    # C[i,k] = sum_j A[i,j] B[j,k], i in {0,1}, j in {0,1}, k in {0,1,2}
    # Tensor: T[a,b,c] = 1 iff a=2i+j, b=2j+k, c=3k+i
    # U: (4, r), V: (6, r), W: (6, r)
    
    # Let me try to find a rank-7 decomposition by solving the system.
    # The equations are:
    # For each (a,b,c): sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    
    # There are 4*6*6 = 144 equations.
    # Unknowns: 4*r + 6*r + 6*r = 16*r (but we can fix one variable per term for gauge freedom)
    # Actually, the decomposition is not unique. We can set U[0,0]=1 WLOG (if nonzero).
    
    # This is a system of quadratic equations. Let me try a specific ansatz.
    
    # Known result: rank(2x2x3) = 7.
    # One construction: take the 4 schoolbook terms for k=0,1 and add 3 more.
    # But that's not right either.
    
    # Let me try to use the following approach:
    # The 2x2x2 rank-7 decomposition (Strassen-like) can be adapted.
    # In 2x2x2: U (4x7), V (4x7), W (4x7)
    # For 2x2x3: U (4x7), V (6x7), W (6x7)
    
    # Let me try to extend the 2x2x2 decomposition.
    # In 2x2x2, the V matrix has rows for (j,k) with k in {0,1}.
    # In 2x2x3, V has rows for (j,k) with k in {0,1,2}.
    # Similarly W has rows for (k,i) with k in {0,1,2}.
    
    # Let me try a specific construction. I'll use the following rank-7 decomposition:
    # Based on the work of Landsberg and others.
    
    # Actually, let me just try to verify a candidate and if it fails, use schoolbook.
    
    # Candidate 2x2x3 rank-7 decomposition:
    # I'll use a construction where we split one of the Strassen terms.
    
    # Let me try this specific decomposition:
    U223 = [
        [1, 0, 0, 0, 1, 0, 0],  # a=0 (i=0,j=0)
        [0, 1, 0, 0, 0, 1, 0],  # a=1 (i=0,j=1)
        [0, 0, 1, 0, 0, 0, 1],  # a=2 (i=1,j=0)
        [0, 0, 0, 1, 0, 0, 0],  # a=3 (i=1,j=1)
    ]
    V223 = [
        [1, 1, 0, 0, 0, 0, 0],  # b=0 (j=0,k=0)
        [0, 0, 1, 1, 0, 0, 0],  # b=1 (j=0,k=1)
        [0, 0, 0, 0, 1, 1, 0],  # b=2 (j=0,k=2)
        [0, 0, 0, 0, 0, 0, 1],  # b=3 (j=1,k=0)
        [0, 0, 0, 0, 0, 0, 0],  # b=4 (j=1,k=1)
        [0, 0, 0, 0, 0, 0, 0],  # b=5 (j=1,k=2)
    ]
    W223 = [
        [1, 0, 0, 0, 0, 0, 0],  # c=0 (k=0,i=0)
        [0, 1, 0, 0, 0, 0, 0],  # c=1 (k=0,i=1)
        [0, 0, 1, 0, 0, 0, 0],  # c=2 (k=1,i=0)
        [0, 0, 0, 1, 0, 0, 0],  # c=3 (k=1,i=1)
        [0, 0, 0, 0, 1, 0, 0],  # c=4 (k=2,i=0)
        [0, 0, 0, 0, 0, 1, 0],  # c=5 (k=2,i=1)
    ]
    
    # Verify this candidate
    def verify_small(U, V, W, n, m, p):
        rank = len(U[0]) if U else 0
        T = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T.add((a, b, c))
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    s = 0.0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    expected = 1 if (a, b, c) in T else 0
                    if abs(s - expected) > 1e-9:
                        return False
        return True
    
    if verify_small(U223, V223, W223, 2, 2, 3):
        # Great! Use Kronecker product to get 4x4x9
        r1 = 7
        r2 = 7
        rank = r1 * r2  # 49
        
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(36)]
        W = [[0] * rank for _ in range(36)]
        
        for r1_idx in range(r1):
            for r2_idx in range(r2):
                r = r1_idx * r2 + r2_idx
                # U: a = a1*4 + a2, where a1 in 0..3 (from 2x2x3 U), a2 in 0..3
                for a1 in range(4):
                    for a2 in range(4):
                        a = a1 * 4 + a2
                        U[a][r] = U223[a1][r1_idx] * U223[a2][r2_idx]
                # V: b = b1*6 + b2, where b1 in 0..5, b2 in 0..5
                for b1 in range(6):
                    for b2 in range(6):
                        b = b1 * 6 + b2
                        V[b][r] = V223[b1][r1_idx] * V223[b2][r2_idx]
                # W: c = c1*6 + c2, where c1 in 0..5, c2 in 0..5
                for c1 in range(6):
                    for c2 in range(6):
                        c = c1 * 6 + c2
                        W[c][r] = W223[c1][r1_idx] * W223[c2][r2_idx]
        
        if verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # If the 2x2x3 candidate failed, try another approach.
    # Let me try to construct a valid 2x2x3 decomposition more carefully.
    
    # Approach: Use the fact that we can write the 2x2x3 tensor as
    # a sum of rank-1 tensors. Let me try a different ansatz.
    
    # Try: U with entries in {0, 1, -1}, same for V and W.
    # Let me try a few more candidates.
    
    # Candidate 2:
    U223b = [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ]
    V223b = [
        [1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]
    W223b = [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
    ]
    
    if verify_small(U223b, V223b, W223b, 2, 2, 3):
        r1 = 7
        r2 = 7
        rank = r1 * r2
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(36)]
        W = [[0] * rank for _ in range(36)]
        for r1_idx in range(r1):
            for r2_idx in range(r2):
                r = r1_idx * r2 + r2_idx
                for a1 in range(4):
                    for a2 in range(4):
                        a = a1 * 4 + a2
                        U[a][r] = U223b[a1][r1_idx] * U223b[a2][r2_idx]
                for b1 in range(6):
                    for b2 in range(6):
                        b = b1 * 6 + b2
                        V[b][r] = V223b[b1][r1_idx] * V223b[b2][r2_idx]
                for c1 in range(6):
                    for c2 in range(6):
                        c = c1 * 6 + c2
                        W[c][r] = W223b[c1][r1_idx] * W223b[c2][r2_idx]
        if verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # If all candidates fail, fall back to schoolbook
    return schoolbook(n, m, p)
# EVOLVE-BLOCK-END
