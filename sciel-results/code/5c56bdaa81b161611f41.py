"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Partition: n=2+2, m=3+2, p=3+3
    # All subproblems use schoolbook decomposition.
    # Total rank: 60
    
    rank_total = 60
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    col = 0
    
    def add_schoolbook(ni, mj, pk, ni_start, mj_start, pk_start):
        nonlocal col
        for a in range(ni):
            for b in range(mj):
                for c in range(pk):
                    U[ni_start + a][col] = 1
                    V[mj_start + b][col] = 1
                    # W index: column * n + row = c_global * n + a_global
                    W[pk_start + c][col] = 1  # This is wrong, need to fix
                    col += 1
    
    # The W indexing: W[k*n+i, r] where k is column index, i is row index of C
    # So for subproblem at (ni_start, mj_start, pk_start), 
    # local (a, b, c) maps to:
    # U[ni_start + a, col]
    # V[mj_start + b, col]  
    # W[(pk_start + c) * n + (ni_start + a), col]
    
    # Let me redo this properly.
    
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    col = 0
    
    ni_sizes = [2, 2]
    mj_sizes = [3, 2]
    pk_sizes = [3, 3]
    ni_starts = [0, 2]
    mj_starts = [0, 3]
    pk_starts = [0, 3]
    
    for ii in range(2):
        for jj in range(2):
            for kk in range(2):
                ni = ni_sizes[ii]
                mj = mj_sizes[jj]
                pk = pk_sizes[kk]
                for a in range(ni):
                    for b in range(mj):
                        for c in range(pk):
                            U[ni_start + a][col] = 1
                            V[mj_start + b][col] = 1
                            W[(pk_start + c) * n + (ni_start + a)][col] = 1
                            col += 1
    
    assert col == 60, f"Expected 60, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

