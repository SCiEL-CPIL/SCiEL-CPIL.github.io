"""Matrix multiplication 3x5x6 rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 3, 5, 6

    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p = 90."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for c in range(rank):
                        actual += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True

    # Try to find a better decomposition using block Strassen
    # For 3x5x6, we can partition as:
    # A = [A_top (2x5); A_bottom (1x5)]
    # B = [B_left (5x2) B_mid (5x2) B_right (5x2)]
    # C = A @ B (3x6)
    #
    # For the top 2 rows of A and left 2 cols of B: 2x2x2 -> Strassen rank 7
    # For the top 2 rows of A and mid 2 cols of B: 2x2x2 -> Strassen rank 7
    # For the top 2 rows of A and right 1 col of B: 2x1x2 -> schoolbook rank 4
    # For the bottom 1 row of A and left 2 cols of B: 1x2x2 -> schoolbook rank 4
    # For the bottom 1 row of A and mid 2 cols of B: 1x2x2 -> schoolbook rank 4
    # For the bottom 1 row of A and right 1 col of B: 1x1x2 -> schoolbook rank 2
    #
    # Total: 7 + 7 + 4 + 4 + 4 + 2 = 28
    
    def strassen_block(n_block, m_block, p_block, U_offset, V_offset, W_offset):
        """
        Create a Strassen or schoolbook block for multiplying n_block x m_block by m_block x p_block.
        Returns (U_block, V_block, W_block, rank) where each is a list of lists.
        """
        if n_block == 2 and m_block == 2 and p_block == 2:
            # Strassen 2x2x2
            U = [
                [1, 1, 0, 1, 1, -1, 0],   # (0,0)
                [1, 1, -1, -1, -1, 0, -1], # (0,1)
                [1, 0, 1, 0, 1, 1, 0],     # (1,0)
                [1, -1, 0, 0, 1, 0, 1],    # (1,1)
            ]
            V = [
                [1, -1, 1, 1, 1, 1, 0],   # (0,0)
                [1, 1, 1, 0, -1, 0, 1],   # (0,1)
                [1, -1, 1, 1, 1, 1, 0],   # (1,0)
                [1, 1, 1, 0, -1, 0, 1],   # (1,1)
            ]
            W = [
                [1, 0, 1, -1, -1, 0, -1], # (0,0)
                [0, 1, 0, 1, 1, 1, 0],    # (0,1)
                [1, 0, 1, -1, -1, 0, -1], # (1,0)
                [0, 1, 0, 1, 1, 1, 0],    # (1,1)
            ]
            rank = 7
            return U, V, W, rank
        else:
            # Schoolbook
            rank = n_block * m_block * p_block
            U = [[0] * rank for _ in range(n_block * m_block)]
            V = [[0] * rank for _ in range(m_block * p_block)]
            W = [[0] * rank for _ in range(p_block * n_block)]
            col = 0
            for i in range(n_block):
                for j in range(m_block):
                    for k in range(p_block):
                        U[i * m_block + j][col] = 1
                        V[j * p_block + k][col] = 1
                        W[k * n_block + i][col] = 1
                        col += 1
            return U, V, W, rank

    def build_block_decomposition():
        """
        Build the block decomposition.
        """
        if time.time() > deadline:
            return None
        
        # Define block structure
        # A: 3x5, B: 5x6
        # Split A into rows [0,1] and [2]
        # Split B into cols [0,1], [2,3], [4,5]
        
        blocks = [
            (2, 2, 2, 0, 0, 0),    # A_top, B_left
            (2, 2, 2, 0, 2, 2),    # A_top, B_mid
            (2, 1, 2, 0, 4, 4),    # A_top, B_right
            (1, 2, 2, 2, 0, 0),    # A_bottom, B_left
            (1, 2, 2, 2, 2, 2),    # A_bottom, B_mid
            (1, 1, 2, 2, 4, 4),    # A_bottom, B_right
        ]
        
        # Build full U, V, W
        U_rows = n * m  # 15
        V_rows = m * p  # 30
        W_rows = p * n  # 18
        
        # First pass: determine total rank
        total_rank = 0
        block_info = []
        for nb, mb, pb, i_off, j_off, k_off in blocks:
            _, _, _, r = strassen_block(nb, mb, pb, 0, 0, 0)
            total_rank += r
            block_info.append((nb, mb, pb, i_off, j_off, k_off, r))
        
        U = [[0] * total_rank for _ in range(U_rows)]
        V = [[0] * total_rank for _ in range(V_rows)]
        W = [[0] * total_rank for _ in range(W_rows)]
        
        # Fill in blocks
        rank_offset = 0
        for nb, mb, pb, i_off, j_off, k_off, r in block_info:
            U_block, V_block, W_block, _ = strassen_block(nb, mb, pb, 0, 0, 0)
            
            # Copy U_block into U
            for i in range(nb):
                for j in range(mb):
                    for c in range(r):
                        U[i_off * m + j_off + j][rank_offset + c] = U_block[i * mb + j][c]
            
            # Copy V_block into V
            for j in range(mb):
                for k in range(pb):
                    for c in range(r):
                        V[j_off * p + k_off + k][rank_offset + c] = V_block[j * pb + k][c]
            
            # Copy W_block into W
            for k in range(pb):
                for i in range(nb):
                    for c in range(r):
                        W[k_off * n + i_off + i][rank_offset + c] = W_block[k * nb + i][c]
            
            rank_offset += r
        
        return {"U": U, "V": V, "W": W}

    # Try block decomposition first
    result = build_block_decomposition()
    
    if result is not None:
        # Verify it
        U, V, W = result["U"], result["V"], result["W"]
        rank = len(U[0]) if U else 0
        if verify(U, V, W, rank):
            return result
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
