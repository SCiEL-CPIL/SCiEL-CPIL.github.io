"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Block decomposition of 5x5 matrix multiplication using 2+2+1 partition.
    Uses Strassen (rank 7) for 2x2x2 sub-problems, schoolbook for others.
    Total rank: 117
    """
    n, m, p = DIMENSIONS
    
    U_entries = []  # list of dicts: index -> coeff
    V_entries = []
    W_entries = []
    
    def add_term(u_sp, v_sp, w_sp):
        U_entries.append(u_sp)
        V_entries.append(v_sp)
        W_entries.append(w_sp)
    
    def strassen_block(i0, j0, k0):
        a00 = i0*m + j0
        a01 = i0*m + j0 + 1
        a10 = (i0+1)*m + j0
        a11 = (i0+1)*m + j0 + 1
        b00 = j0*p + k0
        b01 = j0*p + k0 + 1
        b10 = (j0+1)*p + k0
        b11 = (j0+1)*p + k0 + 1
        c00 = k0*n + i0
        c01 = (k0+1)*n + i0
        c10 = k0*n + (i0+1)
        c11 = (k0+1)*n + (i0+1)
        
        add_term({a00:1, a11:1}, {b00:1, b11:1}, {c00:1, c11:1})
        add_term({a10:1, a11:1}, {b00:1}, {c10:1})
        add_term({a00:1}, {b01:1, b11:-1}, {c01:1})
        add_term({a11:1}, {b10:1, b00:-1}, {c00:1, c10:1})
        add_term({a00:1, a01:1}, {b11:1}, {c00:-1, c01:1})
        add_term({a10:1, a00:-1}, {b00:1, b01:1}, {c11:1})
        add_term({a01:1, a11:-1}, {b10:1, b11:1}, {c00:1, c11:1})
    
    def schoolbook_block(i0, ni, j0, mj, k0, pk):
        for di in range(ni):
            for dk in range(pk):
                for dj in range(mj):
                    a_idx = (i0+di)*m + (j0+dj)
                    b_idx = (j0+dj)*p + (k0+dk)
                    c_idx = (k0+dk)*n + (i0+di)
                    add_term({a_idx:1}, {b_idx:1}, {c_idx:1})
    
    block_sizes = [2, 2, 1]
    offsets = [0, 2, 4, 5]
    
    for ib in range(3):
        for jb in range(3):
            ni = block_sizes[ib]
            pk = block_sizes[jb]
            i0 = offsets[ib]
            k0 = offsets[jb]
            for kb in range(3):
                mk = block_sizes[kb]
                j0 = offsets[kb]
                if ni == 2 and pk == 2 and mk == 2:
                    strassen_block(i0, j0, k0)
                else:
                    schoolbook_block(i0, ni, j0, mk, k0, pk)
    
    rank = len(U_entries)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for idx, coeff in U_entries[c].items():
            U[idx][c] = coeff
        for idx, coeff in V_entries[c].items():
            V[idx][c] = coeff
        for idx, coeff in W_entries[c].items():
            W[idx][c] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
