import numpy as np

def construct():
    n, m, p = 3, 4, 6
    
    def get_rank(n, m, p):
        # Memoization would be good but for small n,m,p it's fine
        if n == 1 and m == 1 and p == 1:
            return 1
        if n == 1 and m == 2 and p == 2:
            return 3
        if n == 2 and m == 2 and p == 2:
            return 7
        
        best = n * m * p
        
        # Split n
        for i in range(1, n):
            r = get_rank(i, m, p) + get_rank(n-i, m, p)
            if r < best:
                best = r
                
        # Split m
        for j in range(1, m):
            r = get_rank(n, j, p) + get_rank(n, m-j, p)
            if r < best:
                best = r
                
        # Split p
        for k in range(1, p):
            r = get_rank(n, m, k) + get_rank(n, m, p-k)
            if r < best:
                best = r
                
        return best

    def build(n, m, p):
        # Returns U (n*m, r), V (m*p, r), W (p*n, r)
        if n == 1 and m == 1 and p == 1:
            U = np.array([[1]])
            V = np.array([[1]])
            W = np.array([[1]])
            return U, V, W
            
        if n == 1 and m == 2 and p == 2:
            # Rank 3 decomposition for 1x2x2
            # C00 = A00 B00 + A01 B10
            # C01 = A00 B01 + A01 B11
            # M1 = A00 B00
            # M2 = A01 B11
            # M3 = (A00+A01)(B01+B10)
            # C00 = M1 + (M3 - A00 B10 - A01 B01)? No.
            # Let's use the known rank 3:
            # M1 = A00 B00
            # M2 = A01 B11
            # M3 = (A00 + A01) (B01 + B10)
            # C00 = M1 + M3 - A00 B10 - A01 B01 ??
            # This is not linear in M's only.
            # Correct Rank 3 for 1x2x2:
            # M1 = A00 B00
            # M2 = A01 B11
            # M3 = (A00 + A01) (B01 + B10)
            # C00 = M1 + A01 B10 (not in M)
            # Actually, rank 3 is:
            # M1 = A00 B00
            # M2 = A01 B10
            # M3 = (A00 + A01) B11
            # C00 = M1 + M2
            # C01 = M3 - M2
            # Check:
            # C01 = (A00+A01)B11 - A01 B10 = A00 B11 + A01 B11 - A01 B10.
            # We want A00 B01 + A01 B11.
            # This doesn't match.
            
            # Let's just use trivial rank 4 for 1x2x2 to be safe, 
            # or find the correct rank 3.
            # Rank 3 of 1x2x2 is a standard result.
            # M1 = A00 B00
            # M2 = A01 B11
            # M3 = (A00 + A01)(B01 + B10)
            # C00 = M1 + 1/2 M3 - 1/2 A00 B10 ... no.
            
            # I will use Rank 4 for 1x2x2 to avoid errors.
            # Rank 4 is trivial.
            # This might increase the total rank slightly but ensures correctness.
            # Let's re-evaluate rank with 1x2x2 = 4.
            # 1x2x3 = 4 + 2 = 6.
            # 1x2x6 = 12.
            # 1x4x6 = 24.
            # Total = 7 + 22 + 14 + 24 = 67. Still < 72.
            
            U = np.zeros((2, 4))
            V = np.zeros((4, 4))
            W = np.zeros((2, 4))
            
            # c=0: A00 B00 -> C00
            U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
            # c=1: A01 B10 -> C00
            U[1][1] = 1; V[2][1] = 1; W[0][1] = 1
            # c=2: A00 B01 -> C01
            U[0][2] = 1; V[1][2] = 1; W[1][2] = 1
            # c=3: A01 B11 -> C01
            U[1][3] = 1; V[3][3] = 1; W[1][3] = 1
            
            return U, V, W

        if n == 2 and m == 2 and p == 2:
            # Strassen Rank 7
            U = np.zeros((4, 7))
            V = np.zeros((4, 7))
            W = np.zeros((4, 7))
            
            # M1 = (A00+A11)(B00+B11)
            U[0][0] = 1; U[3][0] = 1
            V[0][0] = 1; V[3][0] = 1
            W[0][0] = 1; W[3][0] = 1 # C00, C11
            
            # M2 = (A10+A11)B00
            U[2][1] = 1; U[3][1] = 1
            V[0][1] = 1
            W[2][1] = 1 # C10 (k=0, i=1 -> 1)
            
            # M3 = A00(B01-B11)
            U[0][2] = 1
            V[1][2] = 1; V[3][2] = -1
            W[1][2] = 1 # C01 (k=1, i=0 -> 2? No, W index k*n+i. n=2. k=1, i=0 -> 2)
            # Wait, W shape is p*n. p=2, n=2.
            # W[0] -> C00 (k=0,i=0)
            # W[1] -> C01 (k=0,i=1)? No.
            # Trace convention W[k*n+i].
            # C is n x p. C[i,k].
            # W index is k*n + i.
            # C00: i=0, k=0 -> 0.
            # C01: i=0, k=1 -> 1*2+0 = 2.
            # C10: i=1, k=0 -> 0*2+1 = 1.
            # C11: i=1, k=1 -> 1*2+1 = 3.
            
            # Re-check Strassen mapping
            # M1: C00 += M1, C11 += M1
            W[0][0] = 1; W[3][0] = 1
            # M2: C10 += M2
            W[1][1] = 1
            # M3: C01 += M3
            W[2][2] = 1
            # M4: A11(B10-B00) -> C11 += M4
            U[3][3] = 1
            V[2][3] = 1; V[0][3] = -1
            W[3][3] = 1
            # M5: (A00+A01)B11 -> C01 += M5
            U[0][4] = 1; U[1][4] = 1
            V[3][4] = 1
            W[2][4] = 1
            # M6: (A10-A00)(B00+B01) -> C10 += M6
            U[2][5] = 1; U[0][5] = -1
            V[0][5] = 1; V[1][5] = 1
            W[1][5] = 1
            # M7: (A01-A11)(B10+B11) -> C11 += M7
            U[1][6] = 1; U[3][6] = -1
            V[2][6] = 1; V[3][6] = 1
            W[3][6] = 1
            
            return U, V, W

        # Recursive case
        best_rank = n * m * p
        best_split = None
        best_type = None
        
        for i in range(1, n):
            r = get_rank(i, m, p) + get_rank(n-i, m, p)
            if r < best_rank:
                best_rank = r
                best_split = i
                best_type = 'n'
                
        for j in range(1, m):
            r = get_rank(n, j, p) + get_rank(n, m-j, p)
            if r < best_rank:
                best_rank = r
                best_split = j
                best_type = 'm'
                
        for k in range(1, p):
            r = get_rank(n, m, k) + get_rank(n, m, p-k)
            if r < best_rank:
                best_rank = r
                best_split = k
                best_type = 'p'
                
        U = np.zeros((n*m, best_rank))
        V = np.zeros((m*p, best_rank))
        W = np.zeros((p*n, best_rank))
        
        if best_type == 'n':
            i = best_split
            U1, V1, W1 = build(i, m, p)
            U2, V2, W2 = build(n-i, m, p)
            
            # U1 is (i*m, r1), U2 is ((n-i)*m, r2)
            U[:i*m, :U1.shape[1]] = U1
            U[i*m:, U1.shape[1]:] = U2
            
            # V1 is (m*p, r1), V2 is (m*p, r2)
            V[:, :V1.shape[1]] = V1
            V[:, V1.shape[1]:] = V2
            
            # W1 is (p*i, r1), W2 is (p*(n-i), r2)
            # Need to map W1, W2 to W
            # W1 row k*i + i' -> W row k*n + i'
            # W2 row k*(n-i) + i' -> W row k*n + i + i'
            for k in range(p):
                for i_prime in range(i):
                    W[k*n + i_prime, :W1.shape[1]] = W1[k*i + i_prime, :W1.shape[1]]
                for i_prime in range(n-i):
                    W[k*n + i + i_prime, W1.shape[1]:] = W2[k*(n-i) + i_prime, W1.shape[1]:]
                    
        elif best_type == 'm':
            j = best_split
            U1, V1, W1 = build(n, j, p)
            U2, V2, W2 = build(n, m-j, p)
            
            # U1 is (n*j, r1), U2 is (n*(m-j), r2)
            U[:n*j, :U1.shape[1]] = U1
            U[n*j:, U1.shape[1]:] = U2
            
            # V1 is (j*p, r1), V2 is ((m-j)*p, r2)
            V[:j*p, :V1.shape[1]] = V1
            V[j*p:, V1.shape[1]:] = V2
            
            # W1 is (p*n, r1), W2 is (p*n, r2)
            W[:, :W1.shape[1]] = W1
            W[:, W1.shape[1]:] = W2
            
        elif best_type == 'p':
            k = best_split
            U1, V1, W1 = build(n, m, k)
            U2, V2, W2 = build(n, m, p-k)
            
            # U1 is (n*m, r1), U2 is (n*m, r2)
            U[:, :U1.shape[1]] = U1
            U[:, U1.shape[1]:] = U2
            
            # V1 is (m*k, r1), V2 is (m*(p-k), r2)
            V[:m*k, :V1.shape[1]] = V1
            V[m*k:, V1.shape[1]:] = V2
            
            # W1 is (k*n, r1), W2 is ((p-k)*n, r2)
            W[:k*n, :W1.shape[1]] = W1
            W[k*n:, W1.shape[1]:] = W2
            
        return U, V, W

    U, V, W = build(n, m, p)
    return {"U": U, "V": V, "W": W}
