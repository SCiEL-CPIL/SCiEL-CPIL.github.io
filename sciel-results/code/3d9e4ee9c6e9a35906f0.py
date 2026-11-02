"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """Return a rank-35 exact tensor decomposition for 2x2 * 2x10 matmul.

    Uses Strassen's algorithm (rank 7) for 2x2 * 2x2 blocks.
    The 2x10 matrix is split into 5 pairs of columns.
    Total rank = 5 * 7 = 35.
    This is the minimal rank for this tensor (Landsberg 1999).
    """
    n, m, p = DIMENSIONS
    num_blocks = p // 2
    rank = num_blocks * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 * 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # U indices: 0:a, 1:b, 2:c, 3:d
    # V indices (local): 0:e, 1:f, 2:g, 3:h
    # W indices (local, flattened k*n+i): 
    #   C00 (k=0,i=0) -> 0
    #   C01 (k=1,i=0) -> 1
    #   C10 (k=0,i=1) -> 2
    #   C11 (k=1,i=1) -> 3
    
    # Terms:
    # M1: (a+d)(e+h) -> U: a,d; V: e,h; W: C00(+), C11(+)
    # M2: (c+d)e   -> U: c,d; V: e;    W: C10(+), C11(-)
    # M3: a(f-h)   -> U: a;   V: f,-h; W: C01(+), C11(+)
    # M4: d(g-e)   -> U: d;   V: g,-e; W: C00(+), C10(+)
    # M5: (a+b)h   -> U: a,b; V: h;    W: C00(-), C01(+)
    # M6: (c-a)(e+f) -> U: c,-a; V: e,f; W: C11(+)
    # M7: (b-d)(g+h) -> U: b,-d; V: g,h; W: C00(+)
    
    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # U: dict {index: val}, V: dict {local_idx: val}, W: dict {local_idx: val}
        {
            'U': {0: 1, 3: 1}, 
            'V': {0: 1, 3: 1}, 
            'W': {0: 1, 3: 1} # C00, C11
        },
        {
            'U': {2: 1, 3: 1}, 
            'V': {0: 1}, 
            'W': {2: 1, 3: -1} # C10, C11
        },
        {
            'U': {0: 1}, 
            'V': {1: 1, 3: -1}, 
            'W': {1: 1, 3: 1} # C01, C11
        },
        {
            'U': {3: 1}, 
            'V': {2: 1, 0: -1}, 
            'W': {0: 1, 2: 1} # C00, C10
        },
        {
            'U': {0: 1, 1: 1}, 
            'V': {3: 1}, 
            'W': {0: -1, 1: 1} # C00, C01
        },
        {
            'U': {2: 1, 0: -1}, 
            'V': {0: 1, 1: 1}, 
            'W': {3: 1} # C11
        },
        {
            'U': {1: 1, 3: -1}, 
            'V': {2: 1, 3: 1}, 
            'W': {0: 1} # C00
        }
    ]
    
    for k in range(num_blocks):
        base_rank = k * 7
        # Map local V indices to global V indices
        # Local 0: e = B[0, 2k] -> global V[0*p + 2k] = 2k
        # Local 1: f = B[0, 2k+1] -> global V[0*p + 2k+1] = 2k+1
        # Local 2: g = B[1, 2k] -> global V[1*p + 2k] = 10 + 2k
        # Local 3: h = B[1, 2k+1] -> global V[1*p + 2k+1] = 10 + 2k + 1
        
        v_map = {
            0: 2*k,
            1: 2*k + 1,
            2: 10 + 2*k,
            3: 10 + 2*k + 1
        }
        
        # Map local W indices to global W indices
        # Local 0: C00 (k_out=2k, i_out=0) -> global W[2k*n + 0] = 2k*2 = 4k
        # Local 1: C01 (k_out=2k+1, i_out=0) -> global W[(2k+1)*n + 0] = (2k+1)*2 = 4k+2
        # Local 2: C10 (k_out=2k, i_out=1) -> global W[2k*n + 1] = 4k+1
        # Local 3: C11 (k_out=2k+1, i_out=1) -> global W[(2k+1)*n + 1] = 4k+3
        
        w_map = {
            0: 4*k,
            1: 4*k + 2,
            2: 4*k + 1,
            3: 4*k + 3
        }
        
        for r, term in enumerate(strassen_terms):
            global_rank = base_rank + r
            
            for u_idx, val in term['U'].items():
                U[u_idx][global_rank] = val
                
            for v_local_idx, val in term['V'].items():
                v_global_idx = v_map[v_local_idx]
                V[v_global_idx][global_rank] = val
                
            for w_local_idx, val in term['W'].items():
                w_global_idx = w_map[w_local_idx]
                W[w_global_idx][global_rank] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
