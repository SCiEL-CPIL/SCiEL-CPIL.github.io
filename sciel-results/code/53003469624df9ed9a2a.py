"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=3, m=4, p=8
    
    # We'll build U, V, W lists and append columns as we go
    U_cols = []  # Each entry is a list of length n*m=12
    V_cols = []  # Each entry is a list of length m*p=32
    W_cols = []  # Each entry is a list of length p*n=24
    
    def add_term(u_vals, v_vals, w_vals):
        """Add one rank-1 term."""
        U_cols.append(u_vals)
        V_cols.append(v_vals)
        W_cols.append(w_vals)
    
    # For each output block b (0-3) and inner group g (0-1):
    #   A_slice: rows 0-2, cols 2g..2g+1
    #   B_slice: rows 2g..2g+1, cols 2b..2b+1
    #   C_slice: rows 0-2, cols 2b..2b+1
    
    for b in range(4):  # output block
        for g in range(2):  # inner group
            # A_slice is 3x2: rows 0-2, cols 2g, 2g+1
            # B_slice is 2x2: rows 2g, 2g+1, cols 2b, 2b+1
            # C_slice is 3x2: rows 0-2, cols 2b, 2b+1
            
            # Part 1: Row 0 of A_slice (1x2) times B_slice (2x2) -> 1x2
            # Schoolbook: 4 terms
            # C[0, 2b]   = A[0,2g]*B[2g,2b]   + A[0,2g+1]*B[2g+1,2b]
            # C[0, 2b+1] = A[0,2g]*B[2g,2b+1] + A[0,2g+1]*B[2g+1,2b+1]
            
            # Term 1: A[0,2g] * B[2g,2b]
            u = [0]*12; v = [0]*32; w = [0]*24
            u[0*m + 2*g] = 1  # A row 0, col 2g
            v[(2*g)*p + 2*b] = 1  # B row 2g, col 2b
            w[(2*b)*n + 0] = 1  # C row 0, col 2b -> W[k*n+i] where k=2b, i=0
            add_term(u, v, w)
            
            # Term 2: A[0,2g+1] * B[2g+1,2b]
            u = [0]*12; v = [0]*32; w = [0]*24
            u[0*m + 2*g+1] = 1
            v[(2*g+1)*p + 2*b] = 1
            w[(2*b)*n + 0] = 1
            add_term(u, v, w)
            
            # Term 3: A[0,2g] * B[2g,2b+1]
            u = [0]*12; v = [0]*32; w = [0]*24
            u[0*m + 2*g] = 1
            v[(2*g)*p + 2*b+1] = 1
            w[(2*b+1)*n + 0] = 1
            add_term(u, v, w)
            
            # Term 4: A[0,2g+1] * B[2g+1,2b+1]
            u = [0]*12; v = [0]*32; w = [0]*24
            u[0*m + 2*g+1] = 1
            v[(2*g+1)*p + 2*b+1] = 1
            w[(2*b+1)*n + 0] = 1
            add_term(u, v, w)
            
            # Part 2: Rows 1-2 of A_slice (2x2) times B_slice (2x2) -> 2x2
            # Strassen: 7 terms
            # Let A2 = A_slice[1:3, :], B2 = B_slice
            # A2[0,0] = A[1,2g], A2[0,1] = A[1,2g+1]
            # A2[1,0] = A[2,2g], A2[1,1] = A[2,2g+1]
            # B2[0,0] = B[2g,2b], B2[0,1] = B[2g,2b+1]
            # B2[1,0] = B[2g+1,2b], B2[1,1] = B[2g+1,2b+1]
            # C2[0,0] = C[1,2b], C2[0,1] = C[1,2b+1]
            # C2[1,0] = C[2,2b], C2[1,1] = C[2,2b+1]
            
            # Strassen multipliers:
            # M1 = (A2[1,1]) * (B2[0,0] - B2[1,0])
            # M2 = (A2[0,0] + A2[1,1]) * (B2[1,0])
            # M3 = (A2[0,0] - A2[1,1]) * (B2[0,1])
            # M4 = (A2[0,0]) * (B2[0,0])
            # M5 = (A2[1,1]) * (B2[0,1] + B2[1,1])
            # M6 = (A2[0,0] + A2[1,0]) * (B2[1,1])
            # M7 = (A2[1,0] - A2[0,0]) * (B2[0,0] + B2[0,1])
            
            # C2[0,0] = M4 + M5 - M2 + M6
            # C2[0,1] = M1 + M5
            # C2[1,0] = M2 + M4
            # C2[1,1] = M1 - M3 + M4 + M7
            
            # M1: A[2,2g+1] * (B[2g,2b] - B[2g+1,2b])
            u = [0]*12; v = [0]*32; w = [0]*24
            u[2*m + 2*g+1] = 1
            v[(2*g)*p + 2*b] = 1
            v[(2*g+1)*p + 2*b] = -1
            w[(2*b)*n + 1] = 1   # C2[0,0] -> C[1,2b]
            w[(2*b+1)*n + 1] = 1 # C2[0,1] -> C[1,2b+1]
            w[(2*b)*n + 2] = 0   # C2[1,0]
            w[(2*b+1)*n + 2] = 1 # C2[1,1]
            add_term(u, v, w)
            
            # M2: (A[1,2g] + A[2,2g+1]) * B[2g+1,2b]
            u = [0]*12; v = [0]*32; w = [0]*24
            u[1*m + 2*g] = 1
            u[2*m + 2*g+1] = 1
            v[(2*g+1)*p + 2*b] = 1
            w[(2*b)*n + 1] = -1  # -M2 in C2[0,0]
            w[(2*b+1)*n + 1] = 0
            w[(2*b)*n + 2] = 1   # M2 in C2[1,0]
            w[(2*b+1)*n + 2] = 0
            add_term(u, v, w)
            
            # M3: (A[1,2g] - A[2,2g+1]) * B[2g,2b+1]
            u = [0]*12; v = [0]*32; w = [0]*24
            u[1*m + 2*g] = 1
            u[2*m + 2*g+1] = -1
            v[(2*g)*p + 2*b+1] = 1
            w[(2*b)*n + 1] = 0
            w[(2*b+1)*n + 1] = 0
            w[(2*b)*n + 2] = 0
            w[(2*b+1)*n + 2] = -1 # -M3 in C2[1,1]
            add_term(u, v, w)
            
            # M4: A[1,2g] * B[2g,2b]
            u = [0]*12; v = [0]*32; w = [0]*24
            u[1*m + 2*g] = 1
            v[(2*g)*p + 2*b] = 1
            w[(2*b)*n + 1] = 1   # M4 in C2[0,0]
            w[(2*b+1)*n + 1] = 0
            w[(2*b)*n + 2] = 1   # M4 in C2[1,0]
            w[(2*b+1)*n + 2] = 1 # M4 in C2[1,1]
            add_term(u, v, w)
            
            # M5: A[2,2g+1] * (B[2g,2b+1] + B[2g+1,2b+1])
            u = [0]*12; v = [0]*32; w = [0]*24
            u[2*m + 2*g+1] = 1
            v[(2*g)*p + 2*b+1] = 1
            v[(2*g+1)*p + 2*b+1] = 1
            w[(2*b)*n + 1] = 1   # M5 in C2[0,0]
            w[(2*b+1)*n + 1] = 1 # M5 in C2[0,1]
            w[(2*b)*n + 2] = 0
            w[(2*b+1)*n + 2] = 0
            add_term(u, v, w)
            
            # M6: (A[1,2g] + A[2,2g]) * B[2g+1,2b+1]
            u = [0]*12; v = [0]*32; w = [0]*24
            u[1*m + 2*g] = 1
            u[2*m + 2*g] = 1
            v[(2*g+1)*p + 2*b+1] = 1
            w[(2*b)*n + 1] = 1   # M6 in C2[0,0]
            w[(2*b+1)*n + 1] = 0
            w[(2*b)*n + 2] = 0
            w[(2*b+1)*n + 2] = 0
            add_term(u, v, w)
            
            # M7: (A[2,2g] - A[1,2g]) * (B[2g,2b] + B[2g,2b+1])
            u = [0]*12; v = [0]*32; w = [0]*24
            u[2*m + 2*g] = 1
            u[1*m + 2*g] = -1
            v[(2*g)*p + 2*b] = 1
            v[(2*g)*p + 2*b+1] = 1
            w[(2*b)*n + 1] = 0
            w[(2*b+1)*n + 1] = 0
            w[(2*b)*n + 2] = 0
            w[(2*b+1)*n + 2] = 1 # M7 in C2[1,1]
            add_term(u, v, w)
    
    rank = len(U_cols)
    
    # Convert lists of columns to lists of rows
    U = [[U_cols[c][r] for c in range(rank)] for r in range(n*m)]
    V = [[V_cols[c][r] for c in range(rank)] for r in range(m*p)]
    W = [[W_cols[c][r] for c in range(rank)] for r in range(p*n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
