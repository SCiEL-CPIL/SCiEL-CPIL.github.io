import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = 3, 3, 5
    
    # Target tensor T
    # T[I, J, K] = 1 if I=i*m+j, J=j'*p+k, K=k'*n+i and j==j', i==i', k==k'
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0
                
    def validate(U, V, W):
        """Check if U, V, W exactly decompose T"""
        try:
            R = np.einsum('ic,jc,kc->ijk', U, V, W)
            if np.allclose(R, T, atol=1e-9):
                return True
        except Exception:
            pass
        return False

    def schoolbook_decomposition():
        """Return the trivial rank 45 decomposition"""
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, column] = 1
                    V[j*p + k, column] = 1
                    W[k*n + i, column] = 1
                    column += 1
                    
        return U, V, W

    def try_rank_41():
        """
        Attempt to construct a rank 41 decomposition.
        Uses rank 23 for 3x3x3 and rank 18 for 3x3x2.
        """
        if time.time() > deadline - 10:
            return None
        
        rank_333 = 23
        rank_332 = 18
        total_rank = rank_333 + rank_332
        
        U = np.zeros((n*m, total_rank))
        V = np.zeros((m*p, total_rank))
        W = np.zeros((p*n, total_rank))
        
        # Part 1: 3x3x3 (k in 0,1,2)
        # We need the 23 terms for 3x3x3.
        # I will use a known set of 23 terms.
        # The coefficients are rational. I will use integers scaled by 2.
        
        # The 23 terms for 3x3x3 (a, b, c) where a, b, c are 9-vectors.
        # Indices: a[i*3+k], b[i*3+j], c[j*3+k]
        
        terms_333 = []
        
        # 3 diagonal terms
        terms_333.append(([1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]))
        terms_333.append(([0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]))
        terms_333.append(([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]))
        
        # 20 off-diagonal terms
        # These are derived from the 2x2x2 Strassen decomposition for each j.
        # For a fixed j, the indices i, k are from {0,1,2} \ {j}.
        # Let idx = [i1, i2] be the two indices not equal to j.
        # The 7 Strassen terms for 2x2x2 on indices idx.
        
        # Strassen terms for 2x2x2 (local indices 0,1):
        # s1: a00, b00, c00
        # s2: a11, b11, c11
        # s3: a01, b01, c10
        # s4: a10, b10, c01
        # s5: a00+a01+a10+a11, b00, c00+c10
        # s6: a00+a01, b10+b11, c10+c11
        # s7: a10+a11, b00+b01, c00+c01
        
        # We need to map local (0,1) to global (i1, i2).
        
        for j in range(3):
            idx = [i for i in range(3) if i != j]
            i1, i2 = idx
            
            # Local to global mapping for a, b, c
            # a: (i, k) -> i*3+k
            # b: (i, j) -> i*3+j
            # c: (j, k) -> j*3+k
            
            # We will generate the 7 terms for this j.
            
            # Term 1: local (0,0) -> global (i1, i1)
            a = [0]*9; b = [0]*9; c = [0]*9
            a[i1*3+i1] = 1
            b[i1*3+j] = 1
            c[j*3+i1] = 1
            terms_333.append((a, b, c))
            
            # Term 2: local (1,1) -> global (i2, i2)
            a = [0]*9; b = [0]*9; c = [0]*9
            a[i2*3+i2] = 1
            b[i2*3+j] = 1
            c[j*3+i2] = 1
            terms_333.append((a, b, c))
            
            # Term 3: local (0,1) a, (0,1) b, (1,0) c -> global (i1, i2) a, (i1, j) b, (j, i1) c?
            # Wait, Strassen term 3 is a01, b01, c10.
            # a01: i=0, k=1 -> global i1, i2
            # b01: i=0, j=1 -> global i1, j
            # c10: j=1, k=0 -> global j, i1
            a = [0]*9; b = [0]*9; c = [0]*9
            a[i1*3+i2] = 1
            b[i1*3+j] = 1
            c[j*3+i1] = 1
            terms_333.append((a, b, c))
            
            # Term 4: local (1,0) a, (1,0) b, (0,1) c
            a = [0]*9; b = [0]*9; c = [0]*9
            a[i2*3+i1] = 1
            b[i2*3+j] = 1
            c[j*3+i2] = 1
            terms_333.append((a, b, c))
            
            # Term 5: a00+a01+a10+a11, b00, c00+c10
            a = [0]*9; b = [0]*9; c = [0]*9
            a[i1*3+i1] = 1
            a[i1*3+i2] = 1
            a[i2*3+i1] = 1
            a[i2*3+i2] = 1
            b[i1*3+j] = 1
            c[j*3+i1] = 1
            c[j*3+i2] = 1
            terms_333.append((a, b, c))
            
            # Term 6: a00+a01, b10+b11, c10+c11
            a = [0]*9; b = [0]*9; c = [0]*9
            a[i1*3+i1] = 1
            a[i1*3+i2] = 1
            b[i2*3+j] = 1
            b[i2*3+j] = 1 # Wait, b10 is i=1, j=0? No, b is (i,j).
            # Local b10: i=1, j=0 (local). Global i2, j.
            # Local b11: i=1, j=1 (local). Global i2, j.
            # So b[i2*3+j] = 2?
            # Let's re-check Strassen term 6.
            # M6 = (A00+A01)(B10+B11)
            # So a = a00+a01, b = b10+b11.
            # Local b10: i=1, j=0. Global i2, j.
            # Local b11: i=1, j=1. Global i2, j.
            # So b[i2*3+j] = 2.
            b[i2*3+j] = 2
            c[j*3+i1] = 1
            c[j*3+i2] = 1
            terms_333.append((a, b, c))
            
            # Term 7: a10+a11, b00+b01, c00+c01
            a = [0]*9; b = [0]*9; c = [0]*9
            a[i2*3+i1] = 1
            a[i2*3+i2] = 1
            b[i1*3+j] = 2
            c[j*3+i1] = 1
            c[j*3+i2] = 1
            terms_333.append((a, b, c))
            
        # We have 3 + 3*7 = 24 terms.
        # But the rank is 23.
        # One term is redundant or can be merged?
        # Actually, the rank 23 decomposition is not simply the sum of 3 Strassen decompositions.
        # The diagonal terms are shared?
        # In my construction, the diagonal terms (0,0,0), (1,1,1), (2,2,2) are included in the Strassen terms?
        # For j=0, i1=1, i2=2.
        # Term 1: a[1*3+1]=1, b[1*3+0]=1, c[0*3+1]=1. This is (i=1, j=0, k=1).
        # This is NOT a diagonal term (i,k).
        # The diagonal terms are (i,i,i).
        # My Strassen terms for j=0 involve i,k in {1,2}.
        # So they don't cover the diagonal (0,0,0).
        # So I added 3 diagonal terms explicitly.
        # Total 24 terms.
        
        # To get 23, I need to merge one term.
        # Or maybe my Strassen construction is slightly off for the 3x3x3 case.
        
        # Let's just use these 24 terms for now and see if it works.
        # If it works, rank is 24+18=42.
        # 42 is still better than 45.
        
        # Add the 24 terms for 3x3x3
        for t in range(len(terms_333)):
            a, b, c = terms_333[t]
            U[:, t] = a
            V[:, t] = b
            W[:, t] = c
            
        # Part 2: 3x3x2 (k in 3,4)
        # Schoolbook: 18 terms
        col = len(terms_333)
        for i in range(3):
            for j in range(3):
                for k in [3, 4]:
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
                    
        # Verify
        if validate(U, V, W):
            return U, V, W
            
        return None

    best_result = None
    
    # Try rank 41/42 construction
    result = try_rank_41()
    if result is not None:
        best_result = result
    else:
        # Fallback to schoolbook
        U, V, W = schoolbook_decomposition()
        best_result = (U, V, W)
        
    U, V, W = best_result
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
