import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 100.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # Strassen terms:
        # P1 = (A11+A22)(B11+B22)
        # P2 = (A21+A22)B11
        # P3 = A11(B12-B22)
        # P4 = A22(B21-B11)
        # P5 = (A11+A12)B22
        # P6 = (A21-A11)(B11+B12)
        # P7 = (A12-A22)(B21+B22)
        #
        # C11 = P1 - P4 + P5
        # C12 = P3 + P5
        # C21 = P2 + P4
        # C22 = P1 - P3 + P7
        
        # U[i*2+j, r] for A_ij
        # V[j*2+k, r] for B_jk
        # W[k*2+i, r] for C_ik (trace convention)
        
        # Term 0: P1 = (A11+A22)(B11+B22)
        # A11: i=0,j=0; A22: i=1,j=1
        # B11: j=0,k=0; B22: j=1,k=1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        # C11: i=0,k=0 -> W[0*2+0]=W[0]; C22: i=1,k=1 -> W[1*2+1]=W[3]
        W[0][0] = 1; W[3][0] = 1
        
        # Term 1: P2 = (A21+A22)B11
        # A21: i=1,j=0; A22: i=1,j=1
        # B11: j=0,k=0
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        # C21: i=1,k=0 -> W[0*2+1]=W[1]
        W[1][1] = 1
        
        # Term 2: P3 = A11(B12-B22)
        # A11: i=0,j=0
        # B12: j=0,k=1; B22: j=1,k=1
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        # C12: i=0,k=1 -> W[1*2+0]=W[2]; C22: i=1,k=1 -> W[3]
        W[2][2] = 1; W[3][2] = -1
        
        # Term 3: P4 = A22(B21-B11)
        # A22: i=1,j=1
        # B21: j=1,k=0; B11: j=0,k=0
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        # C11: i=0,k=0 -> W[0]; C21: i=1,k=0 -> W[1]
        W[0][3] = -1; W[1][3] = 1
        
        # Term 4: P5 = (A11+A12)B22
        # A11: i=0,j=0; A12: i=0,j=1
        # B22: j=1,k=1
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        # C11: i=0,k=0 -> W[0]; C12: i=0,k=1 -> W[2]
        W[0][4] = 1; W[2][4] = 1
        
        # Term 5: P6 = (A21-A11)(B11+B12)
        # A21: i=1,j=0; A11: i=0,j=0
        # B11: j=0,k=0; B12: j=0,k=1
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        # P6 doesn't appear in C reconstruction in standard Strassen?
        # Let me verify... Actually I think I need to double-check.
        # For now, set W to 0 for term 5 since it may not be needed
        W[0][5] = 0; W[1][5] = 0; W[2][5] = 0; W[3][5] = 0
        
        # Term 6: P7 = (A12-A22)(B21+B22)
        # A12: i=0,j=1; A22: i=1,j=1
        # B21: j=1,k=0; B22: j=1,k=1
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        # C22: i=1,k=1 -> W[3]
        W[3][6] = 1
        
        return U, V, W
    
    # Verify Strassen
    U_s, V_s, W_s = strassen_2x2x2()
    # Check: sum_r U[i*2+j,r] * V[j*2+k,r] * W[k*2+i,r] should give delta(i,i')delta(j,j')delta(k,k') for the multiplication tensor
    # Actually, the decomposition means: for each output C[i,k], C[i,k] = sum_j A[i,j]*B[j,k]
    # In tensor form: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # For 2x2x2: T[i,j,k] should be 1 if this is a valid multiplication tensor coefficient
    # The multiplication tensor has T[i,j,k] = 1 for all i,j,k (it's the identity in the tensor sense)
    # Wait no. The matrix multiplication tensor M has M[i,j,k] = 1 for all i,j,k in [0,1].
    # And we need M[i,j,k] = sum_r U[i*2+j,r] * V[j*2+k,r] * W[k*2+i,r]
    
    # Let's verify
    valid = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = sum(U_s[i*2+j][r] * V_s[j*2+k][r] * W_s[k*2+i][r] for r in range(7))
                if val != 1:
                    valid = False
                    print(f"Strassen check failed: i={i},j={j},k={k}, val={val}")
    
    if not valid:
        # Fallback: use a known correct Strassen
        # Let me use a different verified formulation
        U_s = [[0]*7 for _ in range(4)]
        V_s = [[0]*7 for _ in range(4)]
        W_s = [[0]*7 for _ in range(4)]
        
        # Using a verified Strassen decomposition from literature
        # P1 = (A11+A22)(B11+B22)
        U_s[0][0]=1; U_s[3][0]=1
        V_s[0][0]=1; V_s[3][0]=1
        W_s[0][0]=1; W_s[3][0]=1
        
        # P2 = (A21+A22)B11
        U_s[2][1]=1; U_s[3][1]=1
        V_s[0][1]=1
        W_s[1][1]=1
        
        # P3 = A11(B12-B22)
        U_s[0][2]=1
        V_s[1][2]=1; V_s[3][2]=-1
        W_s[2][2]=1; W_s[3][2]=-1
        
        # P4 = A22(B21-B11)
        U_s[3][3]=1
        V_s[2][3]=1; V_s[0][3]=-1
        W_s[0][3]=-1; W_s[1][3]=1
        
        # P5 = (A11+A12)B22
        U_s[0][4]=1; U_s[1][4]=1
        V_s[3][4]=1
        W_s[0][4]=1; W_s[2][4]=1
        
        # P6 = (A21-A11)(B11+B12) -- this one is tricky
        # Let me try: it should contribute to C21 and C22
        U_s[2][5]=1; U_s[0][5]=-1
        V_s[0][5]=1; V_s[1][5]=1
        W_s[1][5]=1; W_s[3][5]=-1
        
        # P7 = (A12-A22)(B21+B22)
        U_s[1][6]=1; U_s[3][6]=-1
        V_s[2][6]=1; V_s[3][6]=1
        W_s[3][6]=1
        
        # Re-verify
        valid = True
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    val = sum(U_s[i*2+j][r] * V_s[j*2+k][r] * W_s[k*2+i][r] for r in range(7))
                    if val != 1:
                        valid = False
        
        if not valid:
            # Last resort: schoolbook for 2x2x2
            U_s = [[0]*8 for _ in range(4)]
            V_s = [[0]*8 for _ in range(4)]
            W_s = [[0]*8 for _ in range(4)]
            col = 0
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        U_s[i*2+j][col] = 1
                        V_s[j*2+k][col] = 1
                        W_s[k*2+i][col] = 1
                        col += 1
    
    # Cache for get_decomp
    decomp_cache = {}
    
    def get_decomp(n, m, p):
        """Returns U, V, W as lists of lists for n x m x p decomposition."""
        key = (n, m, p)
        if key in decomp_cache:
            return decomp_cache[key]
        
        # Base cases
        if n == 1 and m == 1 and p == 1:
            U = [[1]]
            V = [[1]]
            W = [[1]]
            decomp_cache[key] = (U, V, W)
            return U, V, W
        
        # Strassen for 2x2x2
        if n == 2 and m == 2 and p == 2:
            U, V, W = strassen_2x2x2()
            # Verify
            valid = True
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        val = sum(U[i*2+j][r] * V[j*2+k][r] * W[k*2+i][r] for r in range(len(U[0])))
                        if val != 1:
                            valid = False
            if valid:
                decomp_cache[key] = (U, V, W)
                return U, V, W
        
        # Schoolbook fallback
        best_rank = n * m * p
        best_U = [[0] * best_rank for _ in range(n * m)]
        best_V = [[0] * best_rank for _ in range(m * p)]
        best_W = [[0] * best_rank for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_U[i * m + j][col] = 1
                    best_V[j * p + k][col] = 1
                    best_W[k * n + i][col] = 1
                    col += 1
        
        # Try splitting m
        if m > 1:
            for m1 in range(1, m):
                m2 = m - m1
                U1, V1, W1 = get_decomp(n, m1, p)
                U2, V2, W2 = get_decomp(n, m2, p)
                
                r1 = len(U1[0]) if U1 else 0
                r2 = len(U2[0]) if U2 else 0
                r = r1 + r2
                
                if r < best_rank:
                    new_U = [[0] * r for _ in range(n * m)]
                    new_V = [[0] * r for _ in range(m * p)]
                    new_W = [[0] * r for _ in range(p * n)]
                    
                    for i in range(n):
                        for j in range(m1):
                            new_U[i * m + j] = U1[i * m1 + j]
                        for j in range(m2):
                            new_U[i * m + m1 + j] = U2[i * m2 + j]
                            
                    for j in range(m1):
                        for k in range(p):
                            new_V[j * p + k] = V1[j * p + k]
                    for j in range(m2):
                        for k in range(p):
                            new_V[(m1 + j) * p + k] = V2[j * p + k]
                            
                    for k in range(p):
                        for i in range(n):
                            new_W[k * n + i] = [w1 + w2 for w1, w2 in zip(W1[k * n + i], W2[k * n + i])]
                    
                    best_rank = r
                    best_U, best_V, best_W = new_U, new_V, new_W

        # Try splitting n
        if n > 1:
            for n1 in range(1, n):
                n2 = n - n1
                U1, V1, W1 = get_decomp(n1, m, p)
                U2, V2, W2 = get_decomp(n2, m, p)
                
                r1 = len(U1[0]) if U1 else 0
                r2 = len(U2[0]) if U2 else 0
                r = r1 + r2
                
                if r < best_rank:
                    new_U = [[0] * r for _ in range(n * m)]
                    new_V = [[0] * r for _ in range(m * p)]
                    new_W = [[0] * r for _ in range(p * n)]
                    
                    for i in range(n1):
                        for j in range(m):
                            new_U[i * m + j] = U1[i * m + j]
                    for i in range(n2):
                        for j in range(m):
                            new_U[(n1 + i) * m + j] = U2[i * m + j]
                            
                    for j in range(m):
                        for k in range(p):
                            new_V[j * p + k] = [v1 + v2 for v1, v2 in zip(V1[j * p + k], V2[j * p + k])]
                            
                    for k in range(p):
                        for i in range(n1):
                            new_W[k * n + i] = W1[k * n1 + i]
                        for i in range(n2):
                            new_W[k * n + n1 + i] = W2[k * n2 + i]
                            
                    best_rank = r
                    best_U, best_V, best_W = new_U, new_V, new_W

        # Try splitting p
        if p > 1:
            for p1 in range(1, p):
                p2 = p - p1
                U1, V1, W1 = get_decomp(n, m, p1)
                U2, V2, W2 = get_decomp(n, m, p2)
                
                r1 = len(U1[0]) if U1 else 0
                r2 = len(U2[0]) if U2 else 0
                r = r1 + r2
                
                if r < best_rank:
                    new_U = [[0] * r for _ in range(n * m)]
                    new_V = [[0] * r for _ in range(m * p)]
                    new_W = [[0] * r for _ in range(p * n)]
                    
                    for i in range(n):
                        for j in range(m):
                            new_U[i * m + j] = [u1 + u2 for u1, u2 in zip(U1[i * m + j], U2[i * m + j])]
                            
                    for j in range(m):
                        for k in range(p1):
                            new_V[j * p + k] = V1[j * p1 + k]
                        for k in range(p2):
                            new_V[j * p + p1 + k] = V2[j * p2 + k]
                            
                    for k in range(p1):
                        for i in range(n):
                            new_W[k * n + i] = W1[k * n + i]
                    for k in range(p2):
                        for i in range(n):
                            new_W[(p1 + k) * n + i] = W2[k * n + i]
                            
                    best_rank = r
                    best_U, best_V, best_W = new_U, new_V, new_W

        decomp_cache[key] = (best_U, best_V, best_W)
        return best_U, best_V, best_W

    U, V, W = get_decomp(n, m, p)
    
    if time.time() > deadline:
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
# EVOLVE-BLOCK-END
