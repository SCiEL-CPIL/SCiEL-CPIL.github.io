"""Matrix multiplication 3x4x8 with Strassen-based decomposition."""
DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = 3, 4, 8  # A: n x m, B: m x p, C: n x p
    # W index: W[k*n + i, r] where i is row of C, k is col of C
    
    U_cols = []  # Each entry is a list of length n*m = 12
    V_cols = []  # Each entry is a list of length m*p = 32
    W_cols = []  # Each entry is a list of length p*n = 24
    
    def add_term(u, v, w):
        """Add one rank-1 term."""
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    def zeros(size):
        return [0] * size
    
    def add_strassen_2x2x2(i0, j0, k0):
        """Add Strassen decomposition for 2x2x2 block."""
        # C[i0:i0+2, k0:k0+2] = A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        # A indices: row i in {i0, i0+1}, col j in {j0, j0+1}
        # B indices: row j in {j0, j0+1}, col k in {k0, k0+1}
        # C indices: row i in {i0, i0+1}, col k in {k0, k0+1}
        
        # M1 = (A[i0,j0]+A[i0+1,j0+1]) * (B[j0,k0]+B[j0+1,k0+1])
        u = zeros(12); v = zeros(32); w = zeros(24)
        u[i0*m + j0] = 1
        u[(i0+1)*m + (j0+1)] = 1
        v[j0*p + k0] = 1
        v[(j0+1)*p + (k0+1)] = 1
        w[k0*n + i0] = 1        # C11
        w[(k0+1)*n + (i0+1)] = 1  # C22
        add_term(u, v, w)
        
        # M2 = (A[i0+1,j0]+A[i0+1,j0+1]) * B[j0,k0]
        u = zeros(12); v = zeros(32); w = zeros(24)
        u[(i0+1)*m + j0] = 1
        u[(i0+1)*m + (j0+1)] = 1
        v[j0*p + k0] = 1
        w[k0*n + (i0+1)] = 1      # C21
        w[(k0+1)*n + (i0+1)] = -1  # C22
        add_term(u, v, w)
        
        # M3 = A[i0,j0] * (B[j0+1,k0]-B[j0+1,k0+1])
        u = zeros(12); v = zeros(32); w = zeros(24)
        u[i0*m + j0] = 1
        v[(j0+1)*p + k0] = 1
        v[(j0+1)*p + (k0+1)] = -1
        w[(k0+1)*n + i0] = 1     # C12
        w[(k0+1)*n + (i0+1)] = 1  # C22
        add_term(u, v, w)
        
        # M4 = A[i0+1,j0+1] * (B[j0,k0+1]-B[j0,k0])
        u = zeros(12); v = zeros(32); w = zeros(24)
        u[(i0+1)*m + (j0+1)] = 1
        v[j0*p + (k0+1)] = 1
        v[j0*p + k0] = -1
        w[k0*n + i0] = 1       # C11
        w[k0*n + (i0+1)] = 1   # C21
        add_term(u, v, w)
        
        # M5 = (A[i0,j0]+A[i0,j0+1]) * B[j0+1,k0+1]
        u = zeros(12); v = zeros(32); w = zeros(24)
        u[i0*m + j0] = 1
        u[i0*m + (j0+1)] = 1
        v[(j0+1)*p + (k0+1)] = 1
        w[k0*n + i0] = -1       # C11
        w[(k0+1)*n + i0] = 1    # C12
        add_term(u, v, w)
        
        # M6 = (A[i0+1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k0+1])
        u = zeros(12); v = zeros(32); w = zeros(24)
        u[(i0+1)*m + j0] = 1
        u[i0*m + j0] = -1
        v[j0*p + k0] = 1
        v[j0*p + (k0+1)] = 1
        w[(k0+1)*n + (i0+1)] = 1  # C22
        add_term(u, v, w)
        
        # M7 = (A[i0,j0+1]-A[i0+1,j0+1]) * (B[j0+1,k0]+B[j0+1,k0+1])
        u = zeros(12); v = zeros(32); w = zeros(24)
        u[i0*m + (j0+1)] = 1
        u[(i0+1)*m + (j0+1)] = -1
        v[(j0+1)*p + k0] = 1
        v[(j0+1)*p + (k0+1)] = 1
        w[k0*n + i0] = 1  # C11
        add_term(u, v, w)
    
    def add_schoolbook_1x2x2(i0, j0, k0):
        """Add schoolbook for 1x2x2 block."""
        for dj in range(2):
            for dk in range(2):
                u = zeros(12); v = zeros(32); w = zeros(24)
                u[i0*m + (j0+dj)] = 1
                v[(j0+dj)*p + (k0+dk)] = 1
                w[(k0+dk)*n + i0] = 1
                add_term(u, v, w)
    
    # Top-left: C[0:2, 0:4] = A[0:2, 0:4] * B[0:4, 0:4]
    for j0 in [0, 2]:
        for k0 in [0, 2]:
            add_strassen_2x2x2(0, j0, k0)
    
    # Top-right: C[0:2, 4:8] = A[0:2, 0:4] * B[0:4, 4:8]
    for j0 in [0, 2]:
        for k0 in [4, 6]:
            add_strassen_2x2x2(0, j0, k0)
    
    # Bottom-left: C[2, 0:4] = A[2, 0:4] * B[0:4, 0:4]
    for j0 in [0, 2]:
        for k0 in [0, 2]:
            add_schoolbook_1x2x2(2, j0, k0)
    
    # Bottom-right: C[2, 4:8] = A[2, 0:4] * B[0:4, 4:8]
    for j0 in [0, 2]:
        for k0 in [4, 6]:
            add_schoolbook_1x2x2(2, j0, k0)
    
    rank = len(U_cols)
    U = [list(row) for row in zip(*U_cols)] if U_cols else []
    V = [list(row) for row in zip(*V_cols)] if V_cols else []
    W = [list(row) for row in zip(*W_cols)] if W_cols else []
    
    return {"U": U, "V": V, "W": W}
