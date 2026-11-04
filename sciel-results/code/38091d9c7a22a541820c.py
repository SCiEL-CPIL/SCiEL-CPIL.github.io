"""Matrix multiplication 3x4 by 4x8, rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = 3, 4, 8
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    def add_strassen(r0, r1, b0, b1, c0, c1):
        """
        Strassen 2x2x2. A rows r0,r1; A cols b0,b1; B rows b0,b1; B cols c0,c1.
        C rows c0,c1; C cols r0,r1.
        """
        # M1 = (A[r0,b0]+A[r1,b1])*(B[b0,c0]+B[b1,c1]) -> C[r0,c0]+=1, C[r1,c1]+=1
        U_terms.append((r0*m+b0, 1)); U_terms.append((r1*m+b1, 1))
        V_terms.append((b0*p+c0, 1)); V_terms.append((b1*p+c1, 1))
        W_terms.append((c0*n+r0, 1)); W_terms.append((c1*n+r1, 1))
        
        # M2 = (A[r1,b0]+A[r1,b1])*B[b0,c0] -> C[r1,c0]+=1
        U_terms.append((r1*m+b0, 1)); U_terms.append((r1*m+b1, 1))
        V_terms.append((b0*p+c0, 1))
        W_terms.append((c0*n+r1, 1))
        
        # M3 = A[r0,b0]*(B[b1,c0]-B[b1,c1]) -> C[r0,c1]+=1
        U_terms.append((r0*m+b0, 1))
        V_terms.append((b1*p+c0, 1)); V_terms.append((b1*p+c1, -1))
        W_terms.append((c1*n+r0, 1))
        
        # M4 = A[r1,b1]*(B[b0,c0]-B[b0,c1]) -> C[r0,c0]+=1, C[r1,c0]+=1
        U_terms.append((r1*m+b1, 1))
        V_terms.append((b0*p+c0, 1)); V_terms.append((b0*p+c1, -1))
        W_terms.append((c0*n+r0, 1)); W_terms.append((c0*n+r1, 1))
        
        # M5 = (A[r0,b0]+A[r0,b1])*B[b1,c1] -> C[r0,c0]-=1, C[r0,c1]+=1
        U_terms.append((r0*m+b0, 1)); U_terms.append((r0*m+b1, 1))
        V_terms.append((b1*p+c1, 1))
        W_terms.append((c0*n+r0, -1)); W_terms.append((c1*n+r0, 1))
        
        # M6 = (A[r1,b0]-A[r0,b0])*(B[b0,c0]+B[b0,c1]) -> C[r1,c1]+=1
        U_terms.append((r1*m+b0, 1)); U_terms.append((r0*m+b0, -1))
        V_terms.append((b0*p+c0, 1)); V_terms.append((b0*p+c1, 1))
        W_terms.append((c1*n+r1, 1))
        
        # M7 = (A[r0,b1]-A[r1,b1])*(B[b1,c0]+B[b1,c1]) -> C[r0,c0]+=1, C[r1,c1]+=1
        U_terms.append((r0*m+b1, 1)); U_terms.append((r1*m+b1, -1))
        V_terms.append((b1*p+c0, 1)); V_terms.append((b1*p+c1, 1))
        W_terms.append((c0*n+r0, 1)); W_terms.append((c1*n+r1, 1))
    
    def add_schoolbook(r_indices, b_indices, c_indices):
        for r in r_indices:
            for b in b_indices:
                for c in c_indices:
                    U_terms.append((r*m+b, 1))
                    V_terms.append((b*p+c, 1))
                    W_terms.append((c*n+r, 1))
    
    # Split m=4 into two groups of 2
    for b_start in [0, 2]:
        b0, b1 = b_start, b_start + 1
        # Strassen for rows 0,1 of A
        for c_start in range(0, 8, 2):
            c0, c1 = c_start, c_start + 1
            add_strassen(0, 1, b0, b1, c0, c1)
        # Schoolbook for row 2 of A
        add_schoolbook([2], [b0, b1], list(range(8)))
    
    rank = len(U_terms)
    assert rank == len(V_terms) == len(W_terms), f"Rank mismatch: {len(U_terms)}, {len(V_terms)}, {len(W_terms)}"
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for col in range(rank):
        u_idx, u_val = U_terms[col]
        v_idx, v_val = V_terms[col]
        w_idx, w_val = W_terms[col]
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
    
    return {"U": U, "V": V, "W": W}
