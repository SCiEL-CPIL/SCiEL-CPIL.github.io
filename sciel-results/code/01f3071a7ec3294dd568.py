import numpy as np
import time

n, m, p = 2, 5, 6
nm, mp, pn = n * m, m * p, p * n

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit

    def check_decomposition(U, V, W, tol=1e-9):
        """Check if U, V, W form a valid decomposition."""
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def get_strassen_decomposition():
        """
        Construct the Strassen-based decomposition of rank 54.
        This splits the 5 middle indices into pairs (0,1), (2,3) and single (4).
        For each pair, we use a 2x2 Strassen block (rank 7).
        For the single, we use a naive 2x1x2 block (rank 4).
        Total rank: 3 * 7 + 3 * 4 = 21 + 12 = 33? No.
        
        Wait, the dimensions are:
        A: 2x5 (i in 0..1, j in 0..4)
        B: 5x6 (j in 0..4, k in 0..5)
        C: 2x6 (i in 0..1, k in 0..5)
        
        The inspiration code splits j into pairs (0,1), (2,3) and single (4).
        And splits k into pairs (0,1), (2,3), (4,5).
        
        So for each j-pair (2 of them) and k-pair (3 of them), we have a 2x2x2 block.
        Number of blocks: 2 * 3 = 6 blocks of size 2x2x2.
        Each 2x2x2 block has rank 7 (Strassen).
        Total rank = 6 * 7 = 42?
        
        Let's re-read the inspiration code carefully.
        It calls:
        add_strassen_block(0, 0, 0)  # j=0,1; k=0,1
        add_strassen_block(0, 0, 2)  # j=0,1; k=2,3
        add_strassen_block(0, 0, 4)  # j=0,1; k=4,5
        add_strassen_block(0, 2, 0)  # j=2,3; k=0,1
        add_strassen_block(0, 2, 2)  # j=2,3; k=2,3
        add_strassen_block(0, 2, 4)  # j=2,3; k=4,5
        
        That's 6 Strassen blocks. Rank = 6 * 7 = 42.
        
        Then it calls:
        add_naive_2112_block(0, 4, 0)  # j=4; k=0,1
        add_naive_2112_block(0, 4, 2)  # j=4; k=2,3
        add_naive_2112_block(0, 4, 4)  # j=4; k=4,5
        
        That's 3 naive blocks. Each naive block is 2x1x2, which has rank 4 (schoolbook).
        Total rank = 3 * 4 = 12.
        
        Total rank = 42 + 12 = 54.
        This matches the inspiration's score of 1.1111 (which implies rank 54).
        """
        rank = 54
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        
        def add_strassen_block(i_base, j_base, k_base):
            nonlocal col
            # M0: (A00+A11)(B00+B11)
            U[i_base * m + j_base, col] += 1
            U[(i_base + 1) * m + (j_base + 1), col] += 1
            V[j_base * p + k_base, col] += 1
            V[(j_base + 1) * p + (k_base + 1), col] += 1
            W[k_base * n + i_base, col] += 1
            W[(k_base + 1) * n + (i_base + 1), col] += 1
            col += 1
            
            # M1: (A10+A11)B00
            U[(i_base + 1) * m + j_base, col] += 1
            U[(i_base + 1) * m + (j_base + 1), col] += 1
            V[j_base * p + k_base, col] += 1
            W[k_base * n + (i_base + 1), col] += 1
            W[(k_base + 1) * n + (i_base + 1), col] -= 1
            col += 1
            
            # M2: A00(B01-B11)
            U[i_base * m + j_base, col] += 1
            V[j_base * p + (k_base + 1), col] += 1
            V[(j_base + 1) * p + (k_base + 1), col] -= 1
            W[(k_base + 1) * n + i_base, col] += 1
            W[(k_base + 1) * n + (i_base + 1), col] += 1
            col += 1
            
            # M3: A11(B10-B00)
            U[(i_base + 1) * m + (j_base + 1), col] += 1
            V[(j_base + 1) * p + k_base, col] += 1
            V[j_base * p + k_base, col] -= 1
            W[k_base * n + i_base, col] += 1
            W[k_base * n + (i_base + 1), col] += 1
            col += 1
            
            # M4: (A00+A01)B11
            U[i_base * m + j_base, col] += 1
            U[i_base * m + (j_base + 1), col] += 1
            V[(j_base + 1) * p + (k_base + 1), col] += 1
            W[k_base * n + i_base, col] -= 1
            W[(k_base + 1) * n + i_base, col] += 1
            col += 1
            
            # M5: (A10-A00)(B00+B01)
            U[(i_base + 1) * m + j_base, col] += 1
            U[i_base * m + j_base, col] -= 1
            V[j_base * p + k_base, col] += 1
            V[j_base * p + (k_base + 1), col] += 1
            W[(k_base + 1) * n + (i_base + 1), col] += 1
            col += 1
            
            # M6: (A01-A11)(B10+B11)
            U[i_base * m + (j_base + 1), col] += 1
            U[(i_base + 1) * m + (j_base + 1), col] -= 1
            V[(j_base + 1) * p + k_base, col] += 1
            V[(j_base + 1) * p + (k_base + 1), col] += 1
            W[k_base * n + i_base, col] += 1
            col += 1

        def add_naive_2112_block(i_base, j_base, k_base):
            nonlocal col
            # 1: A00 B00 -> C00
            U[i_base * m + j_base, col] += 1
            V[j_base * p + k_base, col] += 1
            W[k_base * n + i_base, col] += 1
            col += 1
            
            # 2: A00 B01 -> C01
            U[i_base * m + j_base, col] += 1
            V[j_base * p + (k_base + 1), col] += 1
            W[(k_base + 1) * n + i_base, col] += 1
            col += 1
            
            # 3: A10 B00 -> C10
            U[(i_base + 1) * m + j_base, col] += 1
            V[j_base * p + k_base, col] += 1
            W[k_base * n + (i_base + 1), col] += 1
            col += 1
            
            # 4: A10 B01 -> C11
            U[(i_base + 1) * m + j_base, col] += 1
            V[j_base * p + (k_base + 1), col] += 1
            W[(k_base + 1) * n + (i_base + 1), col] += 1
            col += 1

        # Apply Strassen blocks for j pairs (0,1) and (2,3)
        # and k pairs (0,1), (2,3), (4,5)
        add_strassen_block(0, 0, 0)
        add_strassen_block(0, 0, 2)
        add_strassen_block(0, 0, 4)
        
        add_strassen_block(0, 2, 0)
        add_strassen_block(0, 2, 2)
        add_strassen_block(0, 2, 4)
        
        # Apply naive blocks for j=4 and k pairs (0,1), (2,3), (4,5)
        add_naive_2112_block(0, 4, 0)
        add_naive_2112_block(0, 4, 2)
        add_naive_2112_block(0, 4, 4)
        
        if col != rank:
            # Fallback to schoolbook if construction failed
            return get_schoolbook()
            
        return U, V, W

    def get_schoolbook():
        """Return the schoolbook decomposition with rank n*m*p."""
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

    # Primary strategy: Use the known Strassen decomposition of rank 54
    U, V, W = get_strassen_decomposition()
    
    # Verify the decomposition
    if check_decomposition(U, V, W, tol=1e-9):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook (Rank 60)
    U, V, W = get_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
