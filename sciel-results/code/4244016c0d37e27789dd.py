"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 5, 7
    rank = 0
    U = []
    V = []
    W = []
    
    def add_term(u_idx, v_idx, w_idx, val_u=1, val_v=1, val_w=1):
        nonlocal rank
        U.append([0] * (rank + 1))
        V.append([0] * (rank + 1))
        W.append([0] * (rank + 1))
        
        # Pad existing rows
        for i in range(rank):
            U[i].append(0)
            V[i].append(0)
            W[i].append(0)
            
        U[u_idx][rank] = val_u
        V[v_idx][rank] = val_v
        W[w_idx][rank] = val_w
        rank += 1

    def add_strassen(i0, j0, k0):
        # i0, j0, k0 are start indices for 2x2x2 block
        # A[i0:i0+2, j0:j0+2], B[j0:j0+2, k0:k0+2], C[i0:i0+2, k0:k0+2]
        
        # M1 = (A11+A22)(B11+B22)
        add_term(i0*m + j0, j0*p + k0, k0*n + i0, 1, 1, 1)
        add_term((i0+1)*m + (j0+1), (j0+1)*p + (k0+1), (k0+1)*n + (i0+1), 1, 1, 1)
        # Wait, Strassen combines inputs. We need to represent the sums in U and V.
        # U row for A11: val 1. U row for A22: val 1.
        # V row for B11: val 1. V row for B22: val 1.
        # W row for C11: val 1. W row for C22: val 1.
        # But C11 gets M1 + ...
        # So for M1:
        # U: A11(1), A22(1)
        # V: B11(1), B22(1)
        # W: C11(1), C22(1)
        # This is 2 terms in U, 2 in V, 2 in W?
        # No, one column in U, V, W.
        # U[i0*m+j0] = 1
        # U[(i0+1)*m+j0+1] = 1
        # V[j0*p+k0] = 1
        # V[(j0+1)*p+k0+1] = 1
        # W[k0*n+i0] = 1
        # W[(k0+1)*n+i0+1] = 1
        
        # Let's implement a generic add_mult(a_idx, b_idx, c_idx, val)
        # where a_idx is (i,j), b_idx is (j,k), c_idx is (k,i)
        
        pass

    # Re-implementing with a more flexible structure
    # We will build the matrices as lists of lists, then convert to dense if needed?
    # The output format is {'U': U, 'V': V, 'W': W} with shapes.
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    
    # Let's use a list of tuples for each column: (u_indices, u_vals, v_indices, v_vals, w_indices, w_vals)
    cols = []
    
    def add_col(u_dict, v_dict, w_dict):
        cols.append((u_dict, v_dict, w_dict))

    def add_mult(i, j, k, val=1):
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        add_col({u_idx: val}, {v_idx: val}, {w_idx: val})

    def add_strassen_block(i0, j0, k0):
        # M1 = (A11+A22)(B11+B22)
        add_col({i0*m+j0: 1, (i0+1)*m+j0+1: 1}, 
                {j0*p+k0: 1, (j0+1)*p+k0+1: 1}, 
                {k0*n+i0: 1, (k0+1)*n+i0+1: 1})
                
        # M2 = (A21+A22)B11
        add_col({(i0+1)*m+j0: 1, (i0+1)*m+j0+1: 1}, 
                {j0*p+k0: 1}, 
                {(k0)*n+i0+1: 1}) # C21
        
        # M3 = A11(B12-B22)
        add_col({i0*m+j0: 1}, 
                {j0*p+k0+1: 1, (j0+1)*p+k0+1: -1}, 
                {k0*n+i0: 1}) # C12
        
        # M4 = A22(B21-B11)
        add_col({(i0+1)*m+j0+1: 1}, 
                {(j0+1)*p+k0: 1, j0*p+k0: -1}, 
                {k0*n+i0+1: 1}) # C21
        
        # M5 = (A11+A12)B22
        add_col({i0*m+j0: 1, i0*m+j0+1: 1}, 
                {(j0+1)*p+k0+1: 1}, 
                {k0*n+i0: 1}) # C11
        
        # M6 = (A21-A11)(B11+B12)
        add_col({(i0+1)*m+j0: 1, i0*m+j0: -1}, 
                {j0*p+k0: 1, j0*p+k0+1: 1}, 
                {(k0+1)*n+i0+1: 1}) # C22
        
        # M7 = (A12-A22)(B21+B22)
        add_col({i0*m+j0+1: 1, (i0+1)*m+j0+1: -1}, 
                {(j0+1)*p+k0: 1, (j0+1)*p+k0+1: 1}, 
                {k0*n+i0: 1}) # C11

    # Define Strassen blocks
    # I1: 0,1. J1: 0,1. J2: 2,3. K1: 0,1. K2: 2,3. K3: 4,5.
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    add_strassen_block(0, 2, 0)
    add_strassen_block(0, 2, 2)
    add_strassen_block(0, 2, 4)

    # Schoolbook for the rest
    # I2: 2. J3: 4. K4: 6.
    # All (i,j,k) not covered by Strassen blocks.
    # Strassen covers i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5}.
    # Specifically:
    # i=0,1; j=0,1; k=0,1
    # i=0,1; j=0,1; k=2,3
    # i=0,1; j=0,1; k=4,5
    # i=0,1; j=2,3; k=0,1
    # i=0,1; j=2,3; k=2,3
    # i=0,1; j=2,3; k=4,5
    
    covered = set()
    for i0 in [0]:
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                for di in [0, 1]:
                    for dj in [0, 1]:
                        for dk in [0, 1]:
                            covered.add((i0+di, j0+dj, k0+dk))

    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    add_mult(i, j, k, 1)

    # Construct U, V, W matrices
    rank = len(cols)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        u_dict, v_dict, w_dict = cols[c]
        for idx, val in u_dict.items():
            U[idx][c] = val
        for idx, val in v_dict.items():
            V[idx][c] = val
        for idx, val in w_dict.items():
            W[idx][c] = val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
