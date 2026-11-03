def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # Build decomposition using block splits
    # Split m=4 into 2+2
    # A = [A1(3x2) | A2(3x2)], B = [B1(2x7); B2(2x7)]
    # C = A1*B1 + A2*B2
    
    # For each 3x2x7 subproblem:
    # Split p=7 into 3+4
    # 3x2x3: split n=3 into 2+1 -> (2x2x3) + (1x2x3)
    #   2x2x3: split p=3 into 2+1 -> (2x2x2) + (2x2x1) = 7+4=11
    #   1x2x3: split p=3 into 2+1 -> (1x2x2) + (1x2x1) = 4+2=6
    #   Total: 17
    # 3x2x4: split p=4 into 2+2 -> 2*(3x2x2)
    #   3x2x2: split n=3 into 2+1 -> (2x2x2) + (1x2x2) = 7+4=11
    #   Total: 22
    # Grand total per 3x2x7: 17+22=39
    # Two such blocks: 78
    
    # Let me build the full decomposition explicitly
    # I'll track (U_col, V_col, W_col) for each rank-1 term
    
    terms = []  # list of (U_vec, V_vec, W_vec) where each is a dict or sparse representation
    
    def add_term(u, v, w):
        """u: dict mapping (i,j) -> val for A indices
           v: dict mapping (j,k) -> val for B indices  
           w: dict mapping (k,i) -> val for C indices"""
        terms.append((u, v, w))
    
    # Strassen 2x2x2 decomposition
    # X(2x2) * Y(2x2) = Z(2x2)
    # Z[0,0] = M1 - M2 + M3 + M4  ... wait, let me use the standard one
    # Actually for our purpose, we need the rank-7 decomposition.
    # Standard Strassen:
    # M1 = (x00+x11)*(y00+y11)
    # M2 = (x10+x11)*y00
    # M3 = x00*(y10-y11)
    # M4 = x11*(y01-y00)
    # M5 = (x00+x01)*y11
    # M6 = (x10-x00)*(y00+y10)
    # M7 = (x01-x11)*(y01+y11)
    # Z00 = M1 - M4 + M6 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M3 + M5 - M6
    
    # For 2x2x2, the Strassen decomposition gives 7 terms.
    # Let me define a helper that adds Strassen terms for a 2x2x2 block
    # at specified positions.
    
    def add_strassen_222(a_i0, a_i1, b_j0, b_j1, c_k0, c_k1):
        """Add Strassen terms for 2x2x2 block.
        a_i0, a_i1: row indices in A (2 rows)
        b_j0, b_j1: col indices in A / row indices in B (2 cols/rows)
        c_k0, c_k1: col indices in B / row indices in C (2 cols)
        """
        # M1 = (x00+x11)*(y00+y11)
        u1 = {(a_i0,b_j0):1, (a_i1,b_j1):1}
        v1 = {(b_j0,c_k0):1, (b_j1,c_k1):1}
        w1 = {(c_k0,a_i0):1, (c_k1,a_i1):-1, (c_k0,a_i1):1, (c_k1,a_i0):1}
        # Wait, I need to be more careful. Let me use the standard Strassen.
        
        # Let me redefine. For X(2x2)*Y(2x2)=Z(2x2):
        # x_ij are at A positions (a_i0+a, b_j0+b) for a,b in {0,1}
        # y_jk are at B positions (b_j0+j, c_k0+k) for j,k in {0,1}
        # z_ik are at C positions (c_k0+k, a_i0+i) for k,i in {0,1}
        
        # M1 = (x00+x11)*(y00+y11)
        u = {(a_i0,b_j0):1, (a_i1,b_j1):1}
        v = {(b_j0,c_k0):1, (b_j1,c_k1):1}
        # z00 += M1, z11 += M1, z01 -= M1... no wait
        # Z00 = M1 - M4 + M6 + M7
        # Z01 = M3 + M5
        # Z10 = M2 + M4
        # Z11 = M1 - M3 + M5 - M6
        
        w = {(c_k0,a_i0):1, (c_k1,a_i1):1}  # M1 contributes to Z00 and Z11
        add_term(u, v, w)
        
        # M2 = (x10+x11)*y00
        u = {(a_i1,b_j0):1, (a_i1,b_j1):1}
        v = {(b_j0,c_k0):1}
        w = {(c_k0,a_i1):1}  # M2 contributes to Z10
        add_term(u, v, w)
        
        # M3 = x00*(y10-y11)
        u = {(a_i0,b_j0):1}
        v = {(b_j1,c_k0):1, (b_j1,c_k1):-1}
        w = {(c_k0,a_i0):1, (c_k1,a_i0):1}  # M3 contributes to Z01
        add_term(u, v, w)
        
        # M4 = x11*(y01-y00)
        u = {(a_i1,b_j1):1}
        v = {(b_j0,c_k1):1, (b_j0,c_k0):-1}
        w = {(c_k1,a_i1):1, (c_k0,a_i1):1}  # M4 contributes to Z10 and Z11
        # Z10 += M4, Z11 -= M4
        w = {(c_k1,a_i1):1, (c_k0,a_i1):-1}
        add_term(u, v, w)
        
        # M5 = (x00+x01)*y11
        u = {(a_i0,b_j0):1, (a_i0,b_j1):1}
        v = {(b_j1,c_k1):1}
        w = {(c_k1,a_i0):1}  # M5 contributes to Z01
        add_term(u, v, w)
        
        # M6 = (x10-x00)*(y00+y10)
        u = {(a_i1,b_j0):1, (a_i0,b_j0):-1}
        v = {(b_j0,c_k0):1, (b_j1,c_k0):1}
        w = {(c_k0,a_i0):1}  # M6 contributes to Z00
        add_term(u, v, w)
        
        # M7 = (x01-x11)*(y01+y11)
        u = {(a_i0,b_j1):1, (a_i1,b_j1):-1}
        v = {(b_j0,c_k1):1, (b_j1,c_k1):1}
        w = {(c_k1,a_i0):1}  # M7 contributes to Z00
        add_term(u, v, w)
    
    # This is getting very complex. Let me take a different approach.
    # I'll directly enumerate all the rank-1 terms by building the decomposition
    # step by step, being very explicit about indices.
    
    # Let me restart with a cleaner approach.
    # I'll build the full U, V, W matrices directly.
    
    n, m, p = 3, 4, 7
    
    # Step 1: Split m=4 into 2+2
    # A rows: 0,1,2 ; A cols: 0,1,2,3
    # B rows: 0,1,2,3 ; B cols: 0,1,2,3,4,5,6
    # C rows: 0,1,2,3,4,5,6 ; C cols: 0,1,2
    # (C is 7x3 in the W[k*n+i,r] convention: W[k*n+i] where k is row of C (0..6), i is col of C (0..2))
    
    # Wait, the convention is W[k*n+i,r] where the output C has dimensions p×n = 7×3.
    # C[i,j] for i in 0..2 (rows of A), j in 0..6 (cols of B)
    # W index: k*n+i where k=j (col of B / row of C in standard), i=row of A
    # So W[j*n+i, r] corresponds to C[i,j]
    
    # Let me verify: "output factor follows the official trace convention W[k*n+i,r]"
    # C[i,j] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait no. Let me re-read: "W[k*n+i,r]"
    # The standard tensor decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Where T[i,j,k] is the multiplication tensor: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 if this is a valid triple, else 0.
    # U has shape (n*m, rank) -> U[i*m+j, r]
    # V has shape (m*p, rank) -> V[j*p+k, r]  
    # W has shape (p*n, rank) -> W[k*n+i, r]
    
    # OK so:
    # C[i,k] = sum_j A[i,j] * B[j,k] = sum_r (sum_j U[i*m+j,r]*V[j*p+k,r]) * W[k*n+i,r]
    # No wait, that's not right either. The tensor decomposition is:
    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # And T[i,j,k] = 1 for all valid (i,j,k) in the multiplication tensor.
    
    # Actually, the matrix multiplication tensor T has T[i,j,k] = 1 for all i,j,k
    # (it's the all-ones tensor of size n×m×p). The decomposition means:
    # For all i,j,k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # And for invalid triples... wait, there are no invalid triples in the mult tensor.
    
    # Hmm, actually the matrix multiplication tensor IS the all-ones tensor.
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # The rank is the minimum r such that T can be written as sum of r rank-1 tensors.
    
    # So I need: for all i,j,k: sum_{r=0}^{R-1} U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    # Let me build this up.
    
    # I'll use a list of (U_dict, V_dict, W_dict) where each dict maps index->value.
    # Then at the end, I'll convert to matrices.
    
    terms = []
    
    def add(u, v, w):
        terms.append((u, v, w))
    
    # Strassen 2x2x2 at A rows [i0,i1], B cols [j0,j1], C cols [k0,k1]
    # A indices: (i0,j0),(i0,j1),(i1,j0),(i1,j1)
    # B indices: (j0,k0),(j0,k1),(j1,k0),(j1,k1)
    # C indices: (k0,i0),(k0,i1),(k1,i0),(k1,i1)
    
    def strassen222(i0, i1, j0, j1, k0, k1):
        # M1 = (x00+x11)*(y00+y11)
        u = {i0*m+j0:1, i1*m+j1:1}
        v = {j0*p+k0:1, j1*p+k1:1}
        w = {k0*n+i0:1, k1*n+i1:1}  # contributes to C[i0,k0] and C[i1,k1]
        add(u, v, w)
        
        # M2 = (x10+x11)*y00
        u = {i1*m+j0:1, i1*m+j1:1}
        v = {j0*p+k0:1}
        w = {k0*n+i1:1}  # contributes to C[i1,k0]
        add(u, v, w)
        
        # M3 = x00*(y10-y11)
        u = {i0*m+j0:1}
        v = {j1*p+k0:1, j1*p+k1:-1}
        w = {k0*n+i0:1, k1*n+i0:1}  # contributes to C[i0,k0] and C[i0,k1]
        add(u, v, w)
        
        # M4 = x11*(y01-y00)
        u = {i1*m+j1:1}
        v = {j0*p+k1:1, j0*p+k0:-1}
        w = {k1*n+i1:1, k0*n+i1:-1}  # contributes to C[i1,k1] and C[i1,k0]
        add(u, v, w)
        
        # M5 = (x00+x01)*y11
        u = {i0*m+j0:1, i0*m+j1:1}
        v = {j1*p+k1:1}
        w = {k1*n+i0:1}  # contributes to C[i0,k1]
        add(u, v, w)
        
        # M6 = (x10-x00)*(y00+y10)
        u = {i1*m+j0:1, i0*m+j0:-1}
        v = {j0*p+k0:1, j1*p+k0:1}
        w = {k0*n+i0:1}  # contributes to C[i0,k0]
        add(u, v, w)
        
        # M7 = (x01-x11)*(y01+y11)
        u = {i0*m+j1:1, i1*m+j1:-1}
        v = {j0*p+k1:1, j1*p+k1:1}
        w = {k1*n+i0:1}  # contributes to C[i0,k1]
        add(u, v, w)
    
    # Schoolbook 2x2x1 at A rows [i0,i1], B col j, C col k
    # C[i0,k] += A[i0,j]*B[j,k], C[i1,k] += A[i1,j]*B[j,k]
    # Rank 4: each (i,j,k) triple is separate
    def schoolbook_221(i0, i1, j, k):
        for i in [i0, i1]:
            u = {i*m+j:1}
            v = {j*p+k:1}
            w = {k*n+i:1}
            add(u, v, w)
    
    # Schoolbook 1x2x2 at A row i, B cols [j0,j1], C cols [k0,k1]
    # C[i,k0] += A[i,j0]*B[j0,k0] + A[i,j1]*B[j1,k0]
    # C[i,k1] += A[i,j0]*B[j0,k1] + A[i,j1]*B[j1,k1]
    # Rank 4
    def schoolbook_122(i, j0, j1, k0, k1):
        for j in [j0, j1]:
            for k in [k0, k1]:
                u = {i*m+j:1}
                v = {j*p+k:1}
                w = {k*n+i:1}
                add(u, v, w)
    
    # Schoolbook 1x2x1
    def schoolbook_121(i, j, k):
        u = {i*m+j:1}
        v = {j*p+k:1}
        w = {k*n+i:1}
        add(u, v, w)
    
    # Now build the decomposition:
    # Split m=4 into [0,1] and [2,3]
    
    # Part 1: A rows 0,1 (top 2 rows) x B cols 0,1 (first 2 of middle)
    # This gives C contributions to all 7 output cols for rows 0,1
    # A[0:2, 0:2] x B[0:2, :] -> C[0:2, :]
    
    # For C[0:2, 0:3] (3 output cols): use Strassen 2x2x2 + 2x2x1
    strassen222(0, 1, 0, 1, 0, 1)  # C[0:2, 0:2]
    schoolbook_221(0, 1, 0, 2)      # C[0:2, 2] from B col 0
    schoolbook_221(0, 1, 1, 2)      # C[0:2, 2] from B col 1
    
    # For C[0:2, 3:7] (4 output cols): use 2*(2x2x2)
    strassen222(0, 1, 0, 1, 3, 4)   # C[0:2, 3:5]
    strassen222(0, 1, 0, 1, 5, 6)   # C[0:2, 5:7]
    
    # Part 2: A rows 0,1 x B cols 2,3
    # A[0:2, 2:4] x B[2:4, :] -> C[0:2, :]
    
    # For C[0:2, 0:3]:
    strassen222(0, 1, 2, 3, 0, 1)
    schoolbook_221(0, 1, 2, 2)
    schoolbook_221(0, 1, 3, 2)
    
    # For C[0:2, 3:7]:
    strassen222(0, 1, 2, 3, 3, 4)
    strassen222(0, 1, 2, 3, 5, 6)
    
    # Part 3: A row 2 (bottom row) x B cols 0,1
    # A[2, 0:2] x B[0:2, :] -> C[2, :]
    
    # For C[2, 0:3]: 1x2x3
    # Split into 1x2x2 + 1x2x1
    schoolbook_122(2, 0, 1, 0, 1)   # C[2, 0:2]
    schoolbook_121(2, 0, 2)          # C[2, 2] from B col 0
    schoolbook_121(2, 1, 2)          # C[2, 2] from B col 1
    
    # For C[2, 3:7]: 1x2x4
    # Split into 2*(1x2x2)
    schoolbook_122(2, 0, 1, 3, 4)   # C[2, 3:5]
    schoolbook_122(2, 0, 1, 5, 6)   # C[2, 5:7]
    
    # Part 4: A row 2 x B cols 2,3
    # A[2, 2:4] x B[2:4, :] -> C[2, :]
    
    # For C[2, 0:3]:
    schoolbook_122(2, 2, 3, 0, 1)
    schoolbook_121(2, 2, 2)
    schoolbook_121(2, 3, 2)
    
    # For C[2, 3:7]:
    schoolbook_122(2, 2, 3, 3, 4)
    schoolbook_122(2, 2, 3, 5, 6)
    
    # Count terms:
    # Part 1: 7 + 2 + 2 + 7 + 7 = 25
    # Part 2: 7 + 2 + 2 + 7 + 7 = 25
    # Part 3: 4 + 1 + 1 + 4 + 4 = 14
    # Part 4: 4 + 1 + 1 + 4 + 4 = 14
    # Total: 25 + 25 + 14 + 14 = 78
    
    # Now convert to matrices
    R = len(terms)
    U = [[0]*R for _ in range(n*m)]
    V = [[0]*R for _ in range(m*p)]
    W = [[0]*R for _ in range(p*n)]
    
    for r, (u, v, w) in enumerate(terms):
        for idx, val in u.items():
            U[idx][r] = val
        for idx, val in v.items():
            V[idx][r] = val
        for idx, val in w.items():
            W[idx][r] = val
    
    return {"U": U, "V": V, "W": W}
