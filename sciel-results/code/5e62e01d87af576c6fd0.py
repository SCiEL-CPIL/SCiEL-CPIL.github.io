"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # n=2, m=4, p=8
    # U size: 8
    # V size: 32
    # W size: 16
    
    # Rank 56
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2 multiplication
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # Output:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Define the 7 terms as (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs: dict { (i,j): coeff }
    # B_coeffs: dict { (i,j): coeff }
    # C_coeffs: dict { (i,j): coeff }
    
    terms = []
    
    # M1
    terms.append((
        {(0,0): 1, (1,1): 1}, 
        {(0,0): 1, (1,1): 1}, 
        {(0,0): 1, (1,1): 1}
    ))
    # M2
    terms.append((
        {(1,0): 1, (1,1): 1}, 
        {(0,0): 1}, 
        {(1,0): 1}
    ))
    # M3
    terms.append((
        {(0,0): 1}, 
        {(0,1): 1, (1,1): -1}, 
        {(0,1): 1}
    ))
    # M4
    terms.append((
        {(1,1): 1}, 
        {(1,0): 1, (0,0): -1}, 
        {(0,0): 1, (1,0): 1}
    ))
    # M5
    terms.append((
        {(0,0): 1, (0,1): 1}, 
        {(1,1): 1}, 
        {(0,0): -1, (0,1): 1}
    ))
    # M6
    terms.append((
        {(1,0): 1, (0,0): -1}, 
        {(0,0): 1, (0,1): 1}, 
        {(1,1): 1}
    ))
    # M7
    terms.append((
        {(0,1): 1, (1,1): -1}, 
        {(1,0): 1, (1,1): 1}, 
        {(0,0): 1, (1,0): 1}
    ))
    
    # Iterate over sub-problems
    # j_start in [0, 2]
    # k_start in [0, 2, 4, 6]
    
    for j_start in [0, 2]:
        j1 = j_start
        j2 = j_start + 1
        
        for k_start in [0, 2, 4, 6]:
            k1 = k_start
            k2 = k_start + 1
            
            # Map local indices to global indices
            # A local (i, j) -> global (i, j_start + j)
            # B local (j, k) -> global (j_start + j, k_start + k)
            # C local (i, k) -> global (i, k_start + k)
            
            for a_coeffs, b_coeffs, c_coeffs in terms:
                # U
                for (li, lj), coeff in a_coeffs.items():
                    gi = li
                    gj = j_start + lj
                    U[gi * m + gj][col] = coeff
                
                # V
                for (lj, lk), coeff in b_coeffs.items():
                    gj = j_start + lj
                    gk = k_start + lk
                    V[gj * p + gk][col] = coeff
                
                # W
                for (li, lk), coeff in c_coeffs.items():
                    gi = li
                    gk = k_start + lk
                    W[gk * n + gi][col] = coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
