import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Build the multiplication tensor T[i,j,k] = 1 for all i,j,k
    # C[i,k] = sum_j A[i,j]*B[j,k]
    
    # Strategy: Use structured block decomposition
    # Split m=5 into m1=1, m2=4
    # For m1=1: rank = n*m1*p = 2*1*6 = 12
    # For m2=4: Use Strassen blocks of size 2x2x2
    #   Split n=2, m2=4, p=6 into blocks
    #   For each 2x2 block in m2 and 2x2 block in p:
    #     Use Strassen 2x2x2 (rank 7)
    #   Number of blocks: (m2/2) * (p/2) = 2 * 3 = 6 blocks
    #   Rank for m2 part: 6 * 7 = 42
    # Total rank: 12 + 42 = 54
    
    # Let's build the decomposition
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_dict, v_dict, w_dict):
        """Add a rank-1 term to the decomposition."""
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        
        for (i, j), coeff in u_dict.items():
            u_vec[i * m + j] = coeff
        for (j, k), coeff in v_dict.items():
            v_vec[j * p + k] = coeff
        for (k, i), coeff in w_dict.items():
            w_vec[k * n + i] = coeff
        
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # Part 1: m=1 (first column of A, first row of B)
    # A[0,0]*B[0,k] -> C[0,k] for k=0..5
    # A[1,0]*B[0,k] -> C[1,k] for k=0..5
    for k in range(p):
        # Term for A[0,0]*B[0,k] -> C[0,k]
        add_term(
            {(0, 0): 1},
            {(0, k): 1},
            {(k, 0): 1}
        )
        # Term for A[1,0]*B[0,k] -> C[1,k]
        add_term(
            {(1, 0): 1},
            {(0, k): 1},
            {(k, 1): 1}
        )
    
    # Part 2: m=4 (columns 1-4 of A, rows 1-4 of B)
    # Use Strassen 2x2x2 blocks
    # A block: rows 0-1, cols j0-j0+1 (where j0 in {1,3})
    # B block: rows j0-j0+1, cols k0-k0+1 (where k0 in {0,2,4})
    # C block: rows 0-1, cols k0-k0+1
    
    for j0 in [1, 3]:  # Starting columns for A block (1, 3)
        for k0 in [0, 2, 4]:  # Starting columns for B block (0, 2, 4)
            # A indices: (0, j0), (0, j0+1), (1, j0), (1, j0+1)
            # B indices: (j0, k0), (j0, k0+1), (j0+1, k0), (j0+1, k0+1)
            # C indices: (0, k0), (0, k0+1), (1, k0), (1, k0+1)
            
            # Strassen 2x2x2 decomposition:
            # Let A_block = [[a, b], [c, d]] where:
            #   a = A[0, j0], b = A[0, j0+1], c = A[1, j0], d = A[1, j0+1]
            # Let B_block = [[x, y], [u, v]] where:
            #   x = B[j0, k0], y = B[j0, k0+1], u = B[j0+1, k0], v = B[j0+1, k0+1]
            # C_block = [[C00, C01], [C10, C11]] where:
            #   C00 = C[0, k0], C01 = C[0, k0+1], C10 = C[1, k0], C11 = C[1, k0+1]
            
            # m1 = (a+d)(x+u)
            add_term(
                {(0, j0): 1, (1, j0+1): 1},
                {(j0, k0): 1, (j0+1, k0): 1},
                {(k0, 0): 1, (k0, 1): 1}
            )
            
            # m2 = (c+d)x
            add_term(
                {(1, j0): 1, (1, j0+1): 1},
                {(j0, k0): 1},
                {(k0, 1): 1}
            )
            
            # m3 = a(y-u)
            add_term(
                {(0, j0): 1},
                {(j0, k0+1): 1, (j0+1, k0): -1},
                {(k0+1, 0): 1}
            )
            
            # m4 = d(v-u)
            add_term(
                {(1, j0+1): 1},
                {(j0+1, k0+1): 1, (j0+1, k0): -1},
                {(k0+1, 1): 1}
            )
            
            # m5 = (a+b)v
            add_term(
                {(0, j0): 1, (0, j0+1): 1},
                {(j0+1, k0+1): 1},
                {(k0+1, 0): 1, (k0+1, 1): 1}
            )
            
            # m6 = (c-a)(x+u)
            add_term(
                {(1, j0): 1, (0, j0): -1},
                {(j0, k0): 1, (j0+1, k0): 1},
                {(k0, 1): 1}
            )
            
            # m7 = (b-d)(y+v)
            add_term(
                {(0, j0+1): 1, (1, j0+1): -1},
                {(j0, k0+1): 1, (j0+1, k0+1): 1},
                {(k0+1, 0): 1, (k0+1, 1): 1}
            )
    
    # Now we have a decomposition. Let's verify it.
    rank = len(U_cols)
    
    # Convert to numpy arrays
    U = np.array(U_cols, dtype=float).T  # Shape: (n*m, rank)
    V = np.array(V_cols, dtype=float).T  # Shape: (m*p, rank)
    W = np.array(W_cols, dtype=float).T  # Shape: (p*n, rank)
    
    # Verify the decomposition
    # T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    T_recon = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for t in range(rank):
                    T_recon[i, j, k] += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    
    # Check if T_recon matches the target tensor (all ones)
    T_target = np.ones((n, m, p))
    if not np.allclose(T_recon, T_target, atol=1e-10):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    
    # Convert to lists for the return interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
