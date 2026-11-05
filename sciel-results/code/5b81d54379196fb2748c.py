"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    def add_term(u_indices, v_indices, w_indices, sign=1):
        """Add a rank-1 term. u_indices: list of (row, col) for U, etc."""
        col = len(W_rows)
        u_row = [0] * (len(U_rows) + 1)
        v_row = [0] * (len(V_rows) + 1)
        w_row = [0] * (len(W_rows) + 1)
        
        # We'll build these differently - accumulate columns
        return col
    
    # Instead, let's build column by column
    U_cols = []  # Each is a dict mapping row->value
    V_cols = []
    W_cols = []
    
    def add_rank1(u_entries, v_entries, w_entries):
        """u_entries, v_entries, w_entries: list of (row, value)"""
        U_cols.append(dict(u_entries))
        V_cols.append(dict(v_entries))
        W_cols.append(dict(w_entries))
    
    # Partition: i: 1+2, j: 2+2, k: 3+4
    # Block (i_slice, j_slice, k_slice)
    
    # Helper for 2x2x2 Strassen block
    # i0,i1 are the two i-indices, j0,j1 are the two j-indices, k0,k1 are the two k-indices
    def strassen_222(i0, i1, j0, j1, k0, k1):
        # M1 = (A[i0,j0]+A[i1,j1])(B[j0,k0]+B[j1,k1])
        add_rank1(
            [(i0*m+j0, 1), (i1*m+j1, 1)],
            [(j0*p+k0, 1), (j1*p+k1, 1)],
            [(k0*n+i0, 1), (k1*n+i1, 1)]
        )
        # M2 = (A[i1,j0]+A[i1,j1])B[j0,k0]
        add_rank1(
            [(i1*m+j0, 1), (i1*m+j1, 1)],
            [(j0*p+k0, 1)],
            [(k0*n+i1, 1)]
        )
        # M3 = A[i0,j0](B[j1,k0]-B[j0,k0])
        add_rank1(
            [(i0*m+j0, 1)],
            [(j1*p+k0, 1), (j0*p+k0, -1)],
            [(k0*n+i0, 1)]
        )
        # M4 = A[i1,j1](B[j0,k1]-B[j1,k1])
        add_rank1(
            [(i1*m+j1, 1)],
            [(j0*p+k1, 1), (j1*p+k1, -1)],
            [(k1*n+i1, 1)]
        )
        # M5 = (A[i0,j0]+A[i0,j1])B[j1,k1]
        add_rank1(
            [(i0*m+j0, 1), (i0*m+j1, 1)],
            [(j1*p+k1, 1)],
            [(k1*n+i0, 1)]
        )
        # M6 = (A[i1,j0]-A[i0,j0])(B[j0,k0]+B[j0,k1])
        add_rank1(
            [(i1*m+j0, 1), (i0*m+j0, -1)],
            [(j0*p+k0, 1), (j0*p+k1, 1)],
            [(k1*n+i1, 1)]
        )
        # M7 = (A[i0,j1]-A[i1,j1])(B[j1,k0]+B[j1,k1])
        add_rank1(
            [(i0*m+j1, 1), (i1*m+j1, -1)],
            [(j1*p+k0, 1), (j1*p+k1, 1)],
            [(k0*n+i0, 1)]
        )
    
    # Helper for 1x2x2 block (just schoolbook, rank 4)
    def school_122(i0, j0, j1, k0, k1):
        for j in [j0, j1]:
            for k in [k0, k1]:
                add_rank1(
                    [(i0*m+j, 1)],
                    [(j*p+k, 1)],
                    [(k*n+i0, 1)]
                )
    
    # Helper for 1x2x1 block (rank 2)
    def school_121(i0, j0, j1, k0):
        for j in [j0, j1]:
            add_rank1(
                [(i0*m+j, 1)],
                [(j*p+k0, 1)],
                [(k0*n+i0, 1)]
            )
    
    # Helper for 2x2x1 block (rank 4, schoolbook)
    def school_221(i0, i1, j0, j1, k0):
        for i in [i0, i1]:
            for j in [j0, j1]:
                add_rank1(
                    [(i*m+j, 1)],
                    [(j*p+k0, 1)],
                    [(k0*n+i, 1)]
                )
    
    # Block 1: i=[0], j=[0,1], k=[0,1,2] -> 1x2x3
    # Split k into [0,1] and [2]
    school_122(0, 0, 1, 0, 1)  # 1x2x2, rank 4
    school_121(0, 0, 1, 2)      # 1x2x1, rank 2
    
    # Block 2: i=[0], j=[2,3], k=[0,1,2] -> 1x2x3
    school_122(0, 2, 3, 0, 1)  # 1x2x2, rank 4
    school_121(0, 2, 3, 2)      # 1x2x1, rank 2
    
    # Block 3: i=[1,2], j=[0,1], k=[0,1,2] -> 2x2x3
    # Split k into [0,1] and [2]
    strassen_222(1, 2, 0, 1, 0, 1)  # 2x2x2, rank 7
    school_221(1, 2, 0, 1, 2)        # 2x2x1, rank 4
    
    # Block 4: i=[1,2], j=[2,3], k=[0,1,2] -> 2x2x3
    strassen_222(1, 2, 2, 3, 0, 1)  # 2x2x2, rank 7
    school_221(1, 2, 2, 3, 2)        # 2x2x1, rank 4
    
    # Block 5: i=[0], j=[0,1], k=[3,4,5,6] -> 1x2x4
    school_122(0, 0, 1, 3, 4)  # 1x2x2, rank 4
    school_122(0, 0, 1, 5, 6)  # 1x2x2, rank 4
    
    # Block 6: i=[0], j=[2,3], k=[3,4,5,6] -> 1x2x4
    school_122(0, 2, 3, 3, 4)  # 1x2x2, rank 4
    school_122(0, 2, 3, 5, 6)  # 1x2x2, rank 4
    
    # Block 7: i=[1,2], j=[0,1], k=[3,4,5,6] -> 2x2x4
    strassen_222(1, 2, 0, 1, 3, 4)  # 2x2x2, rank 7
    strassen_222(1, 2, 0, 1, 5, 6)  # 2x2x2, rank 7
    
    # Block 8: i=[1,2], j=[2,3], k=[3,4,5,6] -> 2x2x4
    strassen_222(1, 2, 2, 3, 3, 4)  # 2x2x2, rank 7
    strassen_222(1, 2, 2, 3, 5, 6)  # 2x2x2, rank 7
    
    rank = len(W_cols)
    
    # Convert to dense matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col in range(rank):
        for row, val in U_cols[col].items():
            U[row][col] = val
        for row, val in V_cols[col].items():
            V[row][col] = val
        for row, val in W_cols[col].items():
            W[row][col] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

