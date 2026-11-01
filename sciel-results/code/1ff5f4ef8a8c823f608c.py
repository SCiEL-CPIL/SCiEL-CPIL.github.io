"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Returns a rank-7 tensor decomposition for 2x2 matrix multiplication.
    
    Uses a tabular representation of Strassen's algorithm to define the
    decomposition factors. This structure separates the definition of the
    scalar products from the matrix assembly logic.
    """
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit
    
    n, m, p = 2, 2, 2
    rank = 7
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Definition of the 7 rank-1 tensors.
    # Each entry is a tuple: (u_indices, v_indices, w_indices)
    # where each component is a list of (index, value) pairs.
    # This is a structural change: data-driven decomposition definition.
    
    # Index mapping for W (based on task convention W[k*n+i]):
    # k is col, i is row? Or k is row, i is col?
    # Previous analysis: Code uses W[1] for c10 and W[2] for c01.
    # Index 1 = 0*2+1 (if k=0, i=1) -> c10 (row 1, col 0)
    # Index 2 = 1*2+0 (if k=1, i=0) -> c01 (row 0, col 1)
    # This implies Index = k*n + i where k is COL and i is ROW.
    # So W index for c_{r,c} is c*n + r.
    
    # c00: idx 0
    # c01: idx 2
    # c10: idx 1
    # c11: idx 3
    
    # Strassen components:
    # m0: (a00+a11)(b00+b11) -> c00(+1), c11(+1)
    # m1: (a10+a11)b00     -> c10(+1), c11(-1)
    # m2: a00(b01-b11)      -> c01(+1), c11(+1)
    # m3: a11(b10-b00)      -> c00(+1), c10(+1)
    # m4: (a00+a01)b11      -> c00(-1), c01(+1)
    # m5: (a10-a00)(b00+b01) -> c11(+1)
    # m6: (a01-a11)(b10+b11) -> c00(+1)
    
    decompositions = [
        # m0
        {
            'U': [(0, 1), (3, 1)],
            'V': [(0, 1), (3, 1)],
            'W': [(0, 1), (3, 1)]
        },
        # m1
        {
            'U': [(2, 1), (3, 1)],
            'V': [(0, 1)],
            'W': [(1, 1), (3, -1)]
        },
        # m2
        {
            'U': [(0, 1)],
            'V': [(1, 1), (3, -1)],
            'W': [(2, 1), (3, 1)]
        },
        # m3
        {
            'U': [(3, 1)],
            'V': [(2, 1), (0, -1)],
            'W': [(0, 1), (1, 1)]
        },
        # m4
        {
            'U': [(0, 1), (1, 1)],
            'V': [(3, 1)],
            'W': [(0, -1), (2, 1)]
        },
        # m5
        {
            'U': [(2, 1), (0, -1)],
            'V': [(0, 1), (1, 1)],
            'W': [(3, 1)]
        },
        # m6
        {
            'U': [(1, 1), (3, -1)],
            'V': [(2, 1), (3, 1)],
            'W': [(0, 1)]
        }
    ]
    
    for r, comp in enumerate(decompositions):
        for idx, val in comp['U']:
            U[idx][r] = val
        for idx, val in comp['V']:
            V[idx][r] = val
        for idx, val in comp['W']:
            W[idx][r] = val
            
    if time.time() > deadline:
        pass
        
    return {
        "U": U,
        "V": V,
        "W": W
    }

# EVOLVE-BLOCK-END
