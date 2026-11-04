"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 6
    rank = 12 * 7  # 12 block multiplications, each rank 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen components for 2x2x2
    # Local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # U_local, V_local, W_local are length 4
    strassen_parts = [
        # M1
        ([0, 0, 1, 1], [1, -1, 0, 0], [0, 1, 0, 0]),
        # M2
        ([0, 0, 1, 1], [0, 1, 0, 1], [-1, 1, 0, 0]),
        # M3
        ([1, 0, -1, 0], [1, 0, 1, 0], [0, 0, 1, 1]),
        # M4
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 1, 0]),
        # M5
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M6
        ([1, 1, -1, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
        # M7
        ([-1, 0, 1, 0], [1, -1, 0, 0], [0, 0, 0, -1]),
    ]
    
    col = 0
    # Iterate over block multiplications
    # I: row block of A (0,1), J: col block of B (0,1,2), K: sum block (0,1)
    for I in range(2):
        for J in range(3):
            for K in range(2):
                # A block is A[I, K]
                # B block is B[K, J]
                # C block is C[I, J]
                
                # Define local to global index maps
                # A: local (i, j) -> global (2I+i, 2K+j) -> index (2I+i)*4 + (2K+j)
                # B: local (k, l) -> global (2K+k, 2J+l) -> index (2K+k)*6 + (2J+l)
                # W: local (i, k) -> global row (2I+i), col (2J+k) -> index (2J+k)*4 + (2I+i)
                
                a_indices = []
                for li in range(2):
                    for lj in range(2):
                        gr = 2 * I + li
                        gc = 2 * K + lj
                        a_indices.append(gr * m + gc)
                        
                b_indices = []
                for lk in range(2):
                    for ll in range(2):
                        gr = 2 * K + lk
                        gc = 2 * J + ll
                        b_indices.append(gr * p + gc)
                        
                w_indices = []
                for li in range(2):
                    for lk in range(2):
                        gr = 2 * I + li
                        gc = 2 * J + lk
                        # W index is gc * n + gr
                        w_indices.append(gc * n + gr)
                        
                # Add each Strassen component
                for u_vec, v_vec, w_vec in strassen_parts:
                    for idx in range(4):
                        U[a_indices[idx]][col] += u_vec[idx]
                        V[b_indices[idx]][col] += v_vec[idx]
                        W[w_indices[idx]][col] += w_vec[idx]
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
