"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]
N, M, P = 2, 3, 6
RANK_SCHOOLBOOK = N * M * P  # 36

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix
    by a 3x6 matrix.
    
    Strategy:
    Use the decomposition M_{2,3,6} = M_{2,3,2} \otimes M_{1,1,3}.
    Rank(M_{1,1,3}) = 1.
    Rank(M_{2,3,2}) = Rank(M_{2,1,2}) + Rank(M_{2,2,2}) = 1 + 7 = 8.
    Total rank = 8 * 1 = 8.
    """
    t_start = time.time()
    deadline = t_start + 110.0

    # Helper to create Strassen matrices for M_{2,2,2}
    # U, V, W are 4x7
    def get_strassen():
        # Standard Strassen matrices
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A11 + A22)(B11 + B22)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A21 + A22)B11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = 1
        
        # M3 = A11(B21 - B22)
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[0, 2] = 1; W[1, 2] = 1
        
        # M4 = A22(B11 - B12)
        U[3, 3] = 1
        V[0, 3] = 1; V[1, 3] = -1
        W[2, 3] = 1; W[3, 3] = 1
        
        # M5 = (A11 + A12)B22
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = 1; W[3, 4] = 1
        
        # M6 = (A21 - A11)(B11 + B12)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[2, 5] = 1; W[1, 5] = 1
        
        # M7 = (A12 - A22)(B21 + B22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[1, 6] = 1; W[2, 6] = 1
        
        return U, V, W

    # 1. Construct M_{2,3,2} with rank 8
    # n=2, m=3, p=2
    # U_dim = 6, V_dim = 6, W_dim = 4
    rank_232 = 8
    U_232 = np.zeros((6, rank_232))
    V_232 = np.zeros((6, rank_232))
    W_232 = np.zeros((4, rank_232))
    
    # Part 1: M_{2,1,2} (j=0), rank 1
    # U: i*3+0 -> i (0,1). Indices 0, 3.
    # V: 0*2+k -> k (0,1). Indices 0, 1.
    # W: k*2+i. Indices 0,1,2,3.
    U_232[0, 0] = 1
    U_232[3, 0] = 1
    V_232[0, 0] = 1
    V_232[1, 0] = 1
    W_232[0, 0] = 1
    W_232[1, 0] = 1
    W_232[2, 0] = 1
    W_232[3, 0] = 1
    
    # Part 2: M_{2,2,2} (j=1,2), rank 7 (Strassen)
    # Offset in U: j=1,2. U index = i*3+j.
    # i=0: 3,4. i=1: 6,7? No, i in {0,1}, j in {1,2}.
    # i=0, j=1 -> 1. i=0, j=2 -> 2.
    # i=1, j=1 -> 4. i=1, j=2 -> 5.
    # Offset in V: j=1,2. V index = j*2+k.
    # j=1: 2,3. j=2: 4,5.
    # Offset in W: k=0,1. W index = k*2+i.
    # k=0: 0,1. k=1: 2,3.
    
    U_S, V_S, W_S = get_strassen()
    
    for r in range(7):
        col = 1 + r # Start at column 1
        
        # Map U_S
        for i_prime in range(2):
            for j_prime in range(2):
                # Local U index: i_prime * 2 + j_prime
                # Global U index: i_prime * 3 + (j_prime + 1)
                u_idx_global = i_prime * 3 + (j_prime + 1)
                u_idx_local = i_prime * 2 + j_prime
                U_232[u_idx_global, col] = U_S[u_idx_local, r]
                
        # Map V_S
        for j_prime in range(2):
            for k_prime in range(2):
                # Local V index: j_prime * 2 + k_prime
                # Global V index: (j_prime + 1) * 2 + k_prime
                v_idx_global = (j_prime + 1) * 2 + k_prime
                v_idx_local = j_prime * 2 + k_prime
                V_232[v_idx_global, col] = V_S[v_idx_local, r]
                
        # Map W_S
        for k_prime in range(2):
            for i_prime in range(2):
                # Local W index: k_prime * 2 + i_prime
                # Global W index: k_prime * 2 + i_prime
                w_idx_global = k_prime * 2 + i_prime
                w_idx_local = k_prime * 2 + i_prime
                W_232[w_idx_global, col] = W_S[w_idx_local, r]

    # 2. Construct M_{1,1,3} with rank 1
    # n=1, m=1, p=3
    # U: 1x1, V: 1x3, W: 3x1
    rank_113 = 1
    U_113 = np.ones((1, rank_113))
    V_113 = np.ones((3, rank_113))
    W_113 = np.ones((3, rank_113))
    
    # 3. Kronecker Product
    # M_{2,3,6} = M_{2,3,2} \otimes M_{1,1,3}
    # U_total = U_232 \otimes U_113
    # V_total = V_232 \otimes V_113
    # W_total = W_232 \otimes W_113
    
    U_final = np.kron(U_232, U_113)
    V_final = np.kron(V_232, V_113)
    W_final = np.kron(W_232, W_113)
    
    # Verify shapes
    # U: 6 * 1 = 6. Rank 8 * 1 = 8.
    # V: 6 * 3 = 18. Rank 8 * 1 = 8.
    # W: 4 * 3 = 12. Rank 8 * 1 = 8.
    
    # Convert to lists for return
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
# EVOLVE-BLOCK-END
