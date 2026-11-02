"""Block-based decomposition: Strassen 2x2x2 for (i,j) in {0,1}^2, schoolbook for rest."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # We'll build U, V, W as lists of lists, then convert to numpy at the end.
    # U: (n*m, rank) = (9, rank)
    # V: (m*p, rank) = (24, rank)
    # W: (p*n, rank) = (24, rank)
    
    U_terms = []  # Each term is a dict: row -> coeff
    V_terms = []
    W_terms = []
    
    def add_term(u_entries, v_entries, w_entries):
        """
        u_entries: list of (row, coeff)
        v_entries: list of (row, coeff)
        w_entries: list of (row, coeff)
        """
        U_terms.append(dict(u_entries))
        V_terms.append(dict(v_entries))
        W_terms.append(dict(w_entries))
    
    # ---- Strassen 2x2x2 blocks for (i,j) in {0,1}^2, k in {0,1} ----
    # For each pair of k values (k0, k0+1), we apply Strassen to the 2x2x2 sub-block
    # where i in {0,1}, j in {0,1}, k in {k0, k0+1}.
    
    # Strassen 2x2x2 decomposition:
    # For matrices A (2x2), B (2x2), C = A*B (2x2):
    # m1 = (A00+A11)*(B00+B11)
    # m2 = (A10+A11)*B00
    # m3 = A00*(B10-B11)
    # m4 = A11*(B01-B00)
    # m5 = (A00+A01)*B11
    # m6 = (A10-A00)*(B00+B10)
    # m7 = (A01-A11)*(B01+B11)
    #
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # In our tensor notation:
    # A[i,j] corresponds to U row i*m+j
    # B[j,k] corresponds to V row j*p+k
    # C[i,k] corresponds to W row k*n+i
    
    # For a 2x2x2 block with i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}:
    # The 8 schoolbook terms would be:
    # (i0,j0,k0), (i0,j0,k0+1), (i0,j0+1,k0), (i0,j0+1,k0+1)
    # (i0+1,j0,k0), (i0+1,j0,k0+1), (i0+1,j0+1,k0), (i0+1,j0+1,k0+1)
    #
    # Strassen replaces these 8 with 7 terms.
    
    def strassen_block(i0, j0, k0):
        """Add Strassen 2x2x2 decomposition for block starting at (i0, j0, k0)."""
        # m1 = (A[i0,j0]+A[i0+1,j0+1]) * (B[j0,k0]+B[j0,k0+1])
        # Wait, I need to be careful. The 2x2x2 block has:
        # A indices: (i0,j0), (i0,j0+1), (i0+1,j0), (i0+1,j0+1)
        # B indices: (j0,k0), (j0,k0+1), (j0+1,k0), (j0+1,k0+1)
        # C indices: (i0,k0), (i0,k0+1), (i0+1,k0), (i0+1,k0+1)
        
        # m1 = (A[i0,j0] + A[i0+1,j0+1]) * (B[j0,k0] + B[j0+1,k0+1])
        u1 = [(i0*m+j0, 1.0), ((i0+1)*m+(j0+1), 1.0)]
        v1 = [(j0*p+k0, 1.0), ((j0+1)*p+(k0+1), 1.0)]
        w1 = [(k0*n+i0, 1.0), ((k0+1)*n+(i0+1), 1.0)]
        add_term(u1, v1, w1)
        
        # m2 = (A[i0+1,j0] + A[i0+1,j0+1]) * B[j0,k0]
        u2 = [((i0+1)*m+j0, 1.0), ((i0+1)*m+(j0+1), 1.0)]
        v2 = [(j0*p+k0, 1.0)]
        w2 = [(k0*n+(i0+1), 1.0)]
        add_term(u2, v2, w2)
        
        # m3 = A[i0,j0] * (B[j0+1,k0] - B[j0+1,k0+1])
        u3 = [(i0*m+j0, 1.0)]
        v3 = [((j0+1)*p+k0, 1.0), ((j0+1)*p+(k0+1), -1.0)]
        w3 = [(k0*n+i0, 1.0), ((k0+1)*n+i0, 1.0)]
        add_term(u3, v3, w3)
        
        # m4 = A[i0+1,j0+1] * (B[j0,k0+1] - B[j0,k0])
        u4 = [((i0+1)*m+(j0+1), 1.0)]
        v4 = [(j0*p+(k0+1), 1.0), (j0*p+k0, -1.0)]
        w4 = [(k0*n+(i0+1), 1.0), ((k0+1)*n+(i0+1), 1.0)]
        add_term(u4, v4, w4)
        
        # m5 = (A[i0,j0] + A[i0,j0+1]) * B[j0+1,k0+1]
        u5 = [(i0*m+j0, 1.0), (i0*m+(j0+1), 1.0)]
        v5 = [((j0+1)*p+(k0+1), 1.0)]
        w5 = [(k0*n+i0, 1.0), ((k0+1)*n+i0, 1.0)]
        add_term(u5, v5, w5)
        
        # m6 = (A[i0+1,j0] - A[i0,j0]) * (B[j0,k0] + B[j0+1,k0])
        u6 = [((i0+1)*m+j0, 1.0), (i0*m+j0, -1.0)]
        v6 = [(j0*p+k0, 1.0), ((j0+1)*p+k0, 1.0)]
        w6 = [(k0*n+(i0+1), 1.0), ((k0+1)*n+(i0+1), 1.0)]
        add_term(u6, v6, w6)
        
        # m7 = (A[i0,j0+1] - A[i0+1,j0+1]) * (B[j0,k0+1] + B[j0+1,k0+1])
        u7 = [(i0*m+(j0+1), 1.0), ((i0+1)*m+(j0+1), -1.0)]
        v7 = [(j0*p+(k0+1), 1.0), ((j0+1)*p+(k0+1), 1.0)]
        w7 = [(k0*n+i0, 1.0), ((k0+1)*n+i0, 1.0)]
        add_term(u7, v7, w7)
    
    # Add Strassen blocks for all k pairs: k0 = 0, 2, 4, 6
    # Each covers k in {k0, k0+1}
    for k0 in [0, 2, 4, 6]:
        strassen_block(0, 0, k0)
    
    # Now add schoolbook terms for all (i,j,k) NOT covered by Strassen.
    # Strassen covers: i in {0,1}, j in {0,1}, all k in {0..7}
    # So we need schoolbook for: i=2 (any j,k) OR j=2 (i in {0,1}, any k)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2:
                    continue  # Already covered by Strassen
                add_term([(i*m+j, 1.0)], [(j*p+k, 1.0)], [(k*n+i, 1.0)])
    
    # Convert to dense matrices
    rank = len(U_terms)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for row, coeff in U_terms[r].items():
            U[row][r] = coeff
        for row, coeff in V_terms[r].items():
            V[row][r] = coeff
        for row, coeff in W_terms[r].items():
            W[row][r] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
