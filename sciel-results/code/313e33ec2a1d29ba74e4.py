"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from scipy import linalg

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 21
    pn = p * n  # 14
    
    # Precompute target tensor
    target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0

    def verify(U, V, W, tol=1e-6):
        """Check if the decomposition is exact and has integer/half-integer entries."""
        r = U.shape[1]
        R = np.zeros((nm, mp, pn))
        for c in range(r):
            R += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        
        if np.max(np.abs(R - target)) > tol:
            return False
        
        # Check integer/half-integer
        for mat in [U, V, W]:
            for val in mat:
                if not (abs(val * 2 - round(val * 2)) < tol):
                    return False
        return True

    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def build_block_decomposition():
        """
        Constructs a rank 39 decomposition using block splitting.
        Splits m=3 into {0,1} and {2}.
        Splits p=7 into {0,1,2,3} and {4,5,6}.
        
        Blocks:
        1. M_{2,2,4} (j in 0,1; k in 0..3) -> Rank 14 (2x Strassen)
        2. M_{2,2,3} (j in 0,1; k in 4..6) -> Rank 11 (1 Strassen + 1 Schoolbook)
        3. M_{2,1,4} (j in 2; k in 0..3)   -> Rank 8  (Schoolbook)
        4. M_{2,1,3} (j in 2; k in 4..6)   -> Rank 6  (Schoolbook)
        """
        U_list = []
        V_list = []
        W_list = []
        
        def add_strassen_block(i_start, j_start, k_start, n_sub, m_sub, p_sub):
            """
            Adds a Strassen decomposition for an n_sub x m_sub x p_sub block?
            Strassen is for 2x2x2.
            Here we use it for 2x2x2 sub-blocks within the larger block.
            """
            pass

        def add_block_UVW(i_indices, j_indices, k_indices, U_block, V_block, W_block):
            """
            Maps a local decomposition to global indices.
            i_indices: list of global i's
            j_indices: list of global j's
            k_indices: list of global k's
            U_block: local U (n_sub*m_sub, r)
            V_block: local V (m_sub*p_sub, r)
            W_block: local W (p_sub*n_sub, r)
            """
            n_sub = len(i_indices)
            m_sub = len(j_indices)
            p_sub = len(k_indices)
            r = U_block.shape[1]
            
            U_g = np.zeros((nm, r))
            V_g = np.zeros((mp, r))
            W_g = np.zeros((pn, r))
            
            for loc_i, glob_i in enumerate(i_indices):
                for loc_j, glob_j in enumerate(j_indices):
                    loc_idx = loc_i * m_sub + loc_j
                    glob_idx = glob_i * m + glob_j
                    U_g[glob_idx, :] = U_block[loc_idx, :]
                    
            for loc_j, glob_j in enumerate(j_indices):
                for loc_k, glob_k in enumerate(k_indices):
                    loc_idx = loc_j * p_sub + loc_k
                    glob_idx = glob_j * p + glob_k
                    V_g[glob_idx, :] = V_block[loc_idx, :]
                    
            for loc_k, glob_k in enumerate(k_indices):
                for loc_i, glob_i in enumerate(i_indices):
                    loc_idx = loc_k * n_sub + loc_i
                    glob_idx = glob_k * n + glob_i
                    W_g[glob_idx, :] = W_block[loc_idx, :]
                    
            U_list.append(U_g)
            V_list.append(V_g)
            W_list.append(W_g)

        # Define Strassen decomposition for 2x2x2
        def strassen_222():
            # U, V, W for 2x2x2 multiplication
            # C = A B
            # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
            # m1 = (a+d)(e+h)
            # m2 = (c+d)e
            # m3 = a(f-h)
            # m4 = d(g-e)
            # m5 = (a+b)h
            # m6 = (c-d)(f+g)
            # m7 = (b-d)(e+g)
            # C11 = m1 + m4 - m5 + m7
            # C12 = m3 + m5
            # C21 = m2 + m4
            # C22 = m1 - m2 + m3 + m6
            
            U = np.array([
                [1, 0, 1, 0, 1, 0, 1], # a
                [0, 0, 0, 0, 1, 0, -1], # b
                [0, 1, 0, 0, 0, 1, 0], # c
                [1, 1, 0, 1, 0, -1, -1] # d
            ])
            V = np.array([
                [1, 1, 0, 0, 0, 0, 0], # e
                [0, 0, 1, 0, 0, 0, 0], # f
                [0, 1, 0, 1, 0, 1, 1], # g
                [1, 0, -1, -1, 1, 0, 0] # h
            ])
            W = np.array([
                [1, 0, 0, 0, -1, 0, 1], # C11
                [0, 0, 1, 0, 1, 0, 0], # C12
                [0, 1, 0, 1, 0, 0, 0], # C21
                [1, -1, 1, 0, 0, 1, 0] # C22
            ])
            return U, V, W

        def schoolbook_block(n_sub, m_sub, p_sub):
            r = n_sub * m_sub * p_sub
            U = np.zeros((n_sub * m_sub, r))
            V = np.zeros((m_sub * p_sub, r))
            W = np.zeros((p_sub * n_sub, r))
            col = 0
            for i in range(n_sub):
                for j in range(m_sub):
                    for k in range(p_sub):
                        U[i * m_sub + j, col] = 1
                        V[j * p_sub + k, col] = 1
                        W[k * n_sub + i, col] = 1
                        col += 1
            return U, V, W

        # Block 1: M_{2,2,4} (i:0,1, j:0,1, k:0,1,2,3)
        # Split k into 0,1 and 2,3
        # M_{2,2,2} for k=0,1
        U_s1, V_s1, W_s1 = strassen_222()
        add_block_UVW([0, 1], [0, 1], [0, 1], U_s1, V_s1, W_s1)
        # M_{2,2,2} for k=2,3
        U_s2, V_s2, W_s2 = strassen_222()
        add_block_UVW([0, 1], [0, 1], [2, 3], U_s2, V_s2, W_s2)

        # Block 2: M_{2,2,3} (i:0,1, j:0,1, k:4,5,6)
        # Split k into 4 and 5,6
        # M_{2,2,1} for k=4
        U_s3, V_s3, W_s3 = schoolbook_block(2, 2, 1)
        add_block_UVW([0, 1], [0, 1], [4], U_s3, V_s3, W_s3)
        # M_{2,2,2} for k=5,6
        U_s4, V_s4, W_s4 = strassen_222()
        add_block_UVW([0, 1], [0, 1], [5, 6], U_s4, V_s4, W_s4)

        # Block 3: M_{2,1,4} (i:0,1, j:2, k:0,1,2,3)
        U_s5, V_s5, W_s5 = schoolbook_block(2, 1, 4)
        add_block_UVW([0, 1], [2], [0, 1, 2, 3], U_s5, V_s5, W_s5)

        # Block 4: M_{2,1,3} (i:0,1, j:2, k:4,5,6)
        U_s6, V_s6, W_s6 = schoolbook_block(2, 1, 3)
        add_block_UVW([0, 1], [2], [4, 5, 6], U_s6, V_s6, W_s6)

        U = np.hstack(U_list)
        V = np.hstack(V_list)
        W = np.hstack(W_list)
        
        return U, V, W

    U, V, W = build_block_decomposition()
    
    if verify(U, V, W):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback to schoolbook if verification fails
    U, V, W = schoolbook()
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
