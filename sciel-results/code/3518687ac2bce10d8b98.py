"""Program for 2x2 by 2x10 matrix multiplication tensor decomposition."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=10
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for A (2x2)
    # A = [[a, b], [c, d]] -> indices 0, 1, 2, 3
    # B_g = [[p, q], [r, s]] -> local indices 0, 1, 2, 3
    # Global B indices for group g:
    # p: 0*p + 2g = 2g
    # q: 0*p + 2g+1 = 2g+1
    # r: 1*p + 2g = 10+2g
    # s: 1*p + 2g+1 = 10+2g+1
    
    # Global C indices for group g:
    # C00: 2g*n + 0 = 4g
    # C10: 2g*n + 1 = 4g+1
    # C01: (2g+1)*n + 0 = 4g+2
    # C11: (2g+1)*n + 1 = 4g+3
    
    # Strassen products M1..M7
    # u: coeffs for A
    # v_local: coeffs for B_g (p, q, r, s)
    # w_contribs: dict mapping C_local_idx (0:C00, 1:C10, 2:C01, 3:C11) to coeff
    strassen_terms = [
        {
            'u': [1, 0, 0, 1],
            'v_local': [1, 0, 1, 0], # p+r
            'w': {0: 1, 3: 1} # C00 += M1, C11 += M1
        },
        {
            'u': [0, 0, 1, 1],
            'v_local': [1, 0, 0, 0], # p
            'w': {1: 1, 3: -1} # C10 += M2, C11 -= M2
        },
        {
            'u': [1, 0, 0, 0],
            'v_local': [0, -1, 1, 0], # r-s
            'w': {2: 1, 3: 1} # C01 += M3, C11 += M3
        },
        {
            'u': [0, 0, 0, 1],
            'v_local': [0, 1, -1, 0], # q-p
            'w': {0: 1, 1: 1} # C00 += M4, C10 += M4
        },
        {
            'u': [1, 1, 0, 0],
            'v_local': [0, 0, 0, 1], # s
            'w': {0: -1, 2: 1} # C00 -= M5, C01 += M5
        },
        {
            'u': [0, 0, 1, -1],
            'v_local': [1, 1, 0, 0], # p+q
            'w': {3: 1} # C11 += M6
        },
        {
            'u': [0, 1, 0, -1],
            'v_local': [0, 0, 1, 1], # r+s
            'w': {0: 1} # C00 += M7
        }
    ]
    
    # Map local B indices to global V indices for group g
    def get_v_indices(g):
        return [2*g, 2*g+1, 10+2*g, 10+2*g+1]
        
    # Map local C indices to global W indices for group g
    def get_w_indices(g):
        return [4*g, 4*g+1, 4*g+2, 4*g+3]
        
    col = 0
    for g in range(5):
        v_inds = get_v_indices(g)
        w_inds = get_w_indices(g)
        
        for term in strassen_terms:
            # Fill U
            for i in range(4):
                if term['u'][i] != 0:
                    U[i][col] = term['u'][i]
                    
            # Fill V
            for i in range(4):
                if term['v_local'][i] != 0:
                    V[v_inds[i]][col] = term['v_local'][i]
                    
            # Fill W
            for c_local, coeff in term['w'].items():
                W[w_inds[c_local]][col] = coeff
                
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
