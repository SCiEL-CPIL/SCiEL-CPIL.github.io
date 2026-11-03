"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We are constructing a rank 66 decomposition.
    # U shape: (n*m, rank) = (12, 66)
    # V shape: (m*p, rank) = (24, 66)
    # W shape: (p*n, rank) = (18, 66)
    
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val
        col += 1

    # Iterate over the 2 splits of the inner dimension (m=4)
    # j_start in {0, 2}
    for j_start in [0, 2]:
        c0 = j_start
        c1 = j_start + 1
        
        # Iterate over the 3 splits of the output dimension (p=6)
        # k_start in {0, 2, 4}
        for k_start in [0, 2, 4]:
            k0 = k_start
            k1 = k_start + 1
            
            # Rows in A and C are 0, 1, 2
            r0 = 0
            r1 = 1
            r2 = 2
            
            # --- Strassen part for rows r0, r1 ---
            # M1 = A[r0,c0] B[c0,k0] -> C[r0,k0]
            # U[r0*m+c0], V[c0*p+k0], W[k0*n+r0]
            add_term(r0*m + c0, c0*p + k0, k0*n + r0)
            
            # M2 = A[r1,c0] (B[c0,k1] + B[c1,k1]) -> C[r0,k1]
            # U[r1*m+c0]=1, V[c0*p+k1]=1, V[c1*p+k1]=1, W[k1*n+r0]=1
            # We need to add to the same column
            c_idx = col
            U[r1*m + c0][c_idx] = 1
            V[c0*p + k1][c_idx] = 1
            V[c1*p + k1][c_idx] = 1
            W[k1*n + r0][c_idx] = 1
            col += 1
            
            # M3 = (A[r0,c0] + A[r0,c1]) B[c1,k0] -> C[r1,k0]
            c_idx = col
            U[r0*m + c0][c_idx] = 1
            U[r0*m + c1][c_idx] = 1
            V[c1*p + k0][c_idx] = 1
            W[k0*n + r1][c_idx] = 1
            col += 1
            
            # M4 = A[r1,c1] B[c1,k1] -> C[r1,k1]
            add_term(r1*m + c1, c1*p + k1, k1*n + r1)
            
            # M5 = (A[r0,c0] + A[r1,c0]) (B[c0,k1] - B[c1,k0]) -> C[r0,k1]
            c_idx = col
            U[r0*m + c0][c_idx] = 1
            U[r1*m + c0][c_idx] = 1
            V[c0*p + k1][c_idx] = 1
            V[c1*p + k0][c_idx] = -1
            W[k1*n + r0][c_idx] = 1
            col += 1
            
            # M6 = (A[r0,c1] - A[r0,c0]) (B[c1,k0] + B[c1,k1]) -> C[r1,k0] (with sign -1 in C)
            # C[r1,k0] += -M6
            c_idx = col
            U[r0*m + c1][c_idx] = 1
            U[r0*m + c0][c_idx] = -1
            V[c1*p + k0][c_idx] = 1
            V[c1*p + k1][c_idx] = 1
            W[k0*n + r1][c_idx] = -1
            col += 1
            
            # M7 = (A[r1,c1] - A[r0,c1]) (B[c0,k0] + B[c1,k0]) -> C[r0,k0]
            c_idx = col
            U[r1*m + c1][c_idx] = 1
            U[r0*m + c1][c_idx] = -1
            V[c0*p + k0][c_idx] = 1
            V[c1*p + k0][c_idx] = 1
            W[k0*n + r0][c_idx] = 1
            col += 1
            
            # --- Schoolbook part for row r2 ---
            # C[r2,k0] = A[r2,c0]B[c0,k0] + A[r2,c1]B[c1,k0]
            add_term(r2*m + c0, c0*p + k0, k0*n + r2)
            add_term(r2*m + c1, c1*p + k0, k0*n + r2)
            
            # C[r2,k1] = A[r2,c0]B[c0,k1] + A[r2,c1]B[c1,k1]
            add_term(r2*m + c0, c0*p + k1, k1*n + r2)
            add_term(r2*m + c1, c1*p + k1, k1*n + r2)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
