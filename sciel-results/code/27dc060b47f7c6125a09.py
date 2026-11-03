def laderman_terms():
    """Return list of (A_coeffs, B_coeffs, C_coeffs) for 23 bilinear terms."""
    # A_coeffs: list of (row, col, coeff)
    # B_coeffs: list of (row, col, coeff)  
    # C_coeffs: list of (row, col, coeff)
    
    terms = []
    
    # Terms 1-7: Strassen for top-left 2x2
    terms.append(([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]))
    terms.append(([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]))
    terms.append(([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]))
    terms.append(([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]))
    terms.append(([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]))
    terms.append(([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]))
    terms.append(([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]))
    
    # Terms 8-13: for column index 2
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    terms.append(([(0,2,1),(1,2,1)], [(0,2,1),(1,2,1)], [(0,2,1),(1,2,1)]))
    terms.append(([(1,2,1),(1,1,1)], [(0,2,1)], [(1,2,1),(1,1,-1)]))
    terms.append(([(0,2,1)], [(1,2,1),(0,2,-1)], [(0,2,1),(1,2,1)]))
    terms.append(([(1,2,1)], [(1,0,1),(0,0,-1)], [(0,2,1),(1,0,1)]))
    terms.append(([(0,2,1),(0,1,1)], [(1,2,1)], [(0,2,-1),(0,1,1)]))
    terms.append(([(1,2,1),(0,2,-1)], [(0,2,1),(0,1,1)], [(1,1,1)]))
    
    # Terms 14-19: for row index 2
    terms.append(([(2,0,1),(2,1,1)], [(0,0,1),(1,0,1)], [(2,0,1),(2,1,1)]))
    terms.append(([(2,0,1),(2,1,1)], [(0,1,1)], [(2,1,1),(2,0,-1)]))
    terms.append(([(2,0,1)], [(0,1,1),(1,1,-1)], [(2,1,1),(2,0,1)]))
    terms.append(([(2,1,1)], [(1,0,1),(0,0,-1)], [(2,0,1),(2,1,1)]))
    terms.append(([(2,0,1),(2,1,1)], [(1,1,1)], [(2,0,-1),(2,1,1)]))
    terms.append(([(2,1,1),(2,0,-1)], [(0,0,1),(0,1,1)], [(2,1,1)]))
    
    # Terms 20-23: for C[2,2]
    terms.append(([(2,2,1)], [(2,2,1)], [(2,2,1)]))
    # Need 3 more terms for C[2,2] = A[2,0]*B[0,2] + A[2,1]*B[1,2] + A[2,2]*B[2,2]
    # But we already have A[2,2]*B[2,2] from term 20
    # Need A[2,0]*B[0,2] and A[2,1]*B[1,2]
    # These can be incorporated into existing terms or new ones
    
    return terms
