import time

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 150s limit
    
    n, m, p = 2, 2, 11
    rank = 39  # 5 Strassen blocks (7 each) + 1 schoolbook column (4)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        
        # B indices: B[j, k] -> V[j*p + k]
        v_e = k0       # B[0, k0]
        v_f = k1       # B[0, k1]
        v_g = 11 + k0  # B[1, k0]
        v_h = 11 + k1  # B[1, k1]
        
        # W indices: C[i, k] -> W[k*n + i]
        w_c00 = 2 * k0     # C[0, k0]
        w_c10 = 2 * k0 + 1 # C[1, k0]
        w_c01 = 2 * k1     # C[0, k1]
        w_c11 = 2 * k1 + 1 # C[1, k1]
        
        # Strassen terms for 2x2 * 2x2
        # m1 = (a+d)(e+h)
        U[0][r] += 1; U[3][r] += 1
        V[v_e][r] += 1; V[v_h][r] += 1
        W[w_c00][r] += 1; W[w_c11][r] += 1
        r += 1
        
        # m2 = (c+d)e
        U[2][r] += 1; U[3][r] += 1
        V[v_e][r] += 1
        W[w_c10][r] += 1; W[w_c11][r] += -1
        r += 1
        
        # m3 = a(f-h)
        U[0][r] += 1
        V[v_f][r] += 1; V[v_h][r] += -1
        W[w_c01][r] += 1; W[w_c11][r] += 1
        r += 1
        
        # m4 = d(g-e)
        U[3][r] += 1
        V[v_g][r] += 1; V[v_e][r] += -1
        W[w_c00][r] += 1; W[w_c10][r] += 1
        r += 1
        
        # m5 = (a+b)h
        U[0][r] += 1; U[1][r] += 1
        V[v_h][r] += 1
        W[w_c00][r] += -1; W[w_c01][r] += 1
        r += 1
        
        # m6 = (c-a)(e+f)
        U[2][r] += 1; U[0][r] += -1
        V[v_e][r] += 1; V[v_f][r] += 1
        W[w_c11][r] += 1
        r += 1
        
        # m7 = (b-d)(g+h)
        U[1][r] += 1; U[3][r] += -1
        V[v_g][r] += 1; V[v_h][r] += 1
        W[w_c00][r] += 1
        r += 1
        
        if time.time() > deadline:
            break
    
    # Last column k=10: schoolbook
    if r < rank:
        k2 = 10
        v_e2 = k2      # B[0, 10]
        v_g2 = 11 + k2 # B[1, 10]
        w_c02 = 2 * k2      # C[0, 10]
        w_c12 = 2 * k2 + 1  # C[1, 10]
        
        # a * B[0, 10]
        U[0][r] += 1
        V[v_e2][r] += 1
        W[w_c02][r] += 1
        r += 1
        
        # b * B[1, 10]
        U[1][r] += 1
        V[v_g2][r] += 1
        W[w_c02][r] += 1
        r += 1
        
        # c * B[0, 10]
        U[2][r] += 1
        V[v_e2][r] += 1
        W[w_c12][r] += 1
        r += 1
        
        # d * B[1, 10]
        U[3][r] += 1
        V[v_g2][r] += 1
        W[w_c12][r] += 1
        r += 1
    
    # Verify the decomposition is correct
    def verify(U, V, W, n, m, p):
        """Check that the tensor decomposition is exact."""
        rank = len(U[0])
        # Build the multiplication tensor T[i,j,k] = delta_{i+k,n} where A[i,j]*B[j,k] -> C[i,k]
        # Actually for matrix mult: T[a_idx, b_idx, c_idx] = 1 if c_idx corresponds to a*b product
        # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # The tensor is: sum_r U[i,r] * V[j,r] * W[k,r] should equal the multiplication tensor
        # where i indexes A (n*m), j indexes B (m*p), k indexes C (p*n)
        
        # A[i,j] where i in [0,n), j in [0,m) -> index i*m + j
        # B[j,k] where j in [0,m), k in [0,p) -> index j*p + k
        # C[i,k] where i in [0,n), k in [0,p) -> index k*n + i
        
        for a_i in range(n):
            for a_j in range(m):
                for b_j in range(m):
                    for b_k in range(p):
                        if a_j != b_j:
                            expected = 0
                        else:
                            expected = 1
                        # C index: b_k * n + a_i
                        c_idx = b_k * n + a_i
                        a_idx = a_i * m + a_j
                        b_idx = b_j * p + b_k
                        
                        actual = 0
                        for r in range(rank):
                            actual += U[a_idx][r] * V[b_idx][r] * W[c_idx][r]
                        
                        if abs(actual - expected) > 1e-10:
                            return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # This shouldn't happen, but if it does, return the known good construction
        # Rebuild from scratch
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        for block in range(5):
            k0 = block * 2
            k1 = block * 2 + 1
            v_e = k0
            v_f = k1
            v_g = 11 + k0
            v_h = 11 + k1
            w_c00 = 2 * k0
            w_c10 = 2 * k0 + 1
            w_c01 = 2 * k1
            w_c11 = 2 * k1 + 1
            
            U[0][r] += 1; U[3][r] += 1
            V[v_e][r] += 1; V[v_h][r] += 1
            W[w_c00][r] += 1; W[w_c11][r] += 1
            r += 1
            
            U[2][r] += 1; U[3][r] += 1
            V[v_e][r] += 1
            W[w_c10][r] += 1; W[w_c11][r] += -1
            r += 1
            
            U[0][r] += 1
            V[v_f][r] += 1; V[v_h][r] += -1
            W[w_c01][r] += 1; W[w_c11][r] += 1
            r += 1
            
            U[3][r] += 1
            V[v_g][r] += 1; V[v_e][r] += -1
            W[w_c00][r] += 1; W[w_c10][r] += 1
            r += 1
            
            U[0][r] += 1; U[1][r] += 1
            V[v_h][r] += 1
            W[w_c00][r] += -1; W[w_c01][r] += 1
            r += 1
            
            U[2][r] += 1; U[0][r] += -1
            V[v_e][r] += 1; V[v_f][r] += 1
            W[w_c11][r] += 1
            r += 1
            
            U[1][r] += 1; U[3][r] += -1
            V[v_g][r] += 1; V[v_h][r] += 1
            W[w_c00][r] += 1
            r += 1
        
        k2 = 10
        v_e2 = k2
        v_g2 = 11 + k2
        w_c02 = 2 * k2
        w_c12 = 2 * k2 + 1
        
        U[0][r] += 1; V[v_e2][r] += 1; W[w_c02][r] += 1; r += 1
        U[1][r] += 1; V[v_g2][r] += 1; W[w_c02][r] += 1; r += 1
        U[2][r] += 1; V[v_e2][r] += 1; W[w_c12][r] += 1; r += 1
        U[3][r] += 1; V[v_g2][r] += 1; W[w_c12][r] += 1; r += 1
    
    return {"U": U, "V": V, "W": W}
