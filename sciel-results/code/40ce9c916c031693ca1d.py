def get_strassen_2x2_terms():
    # Returns list of (U_rows, V_rows, W_rows) for 2x2x2
    # Indices: 0,1,2,3 for 2x2
    # U: 4x7, V: 4x7, W: 4x7
    terms = []
    
    # M1
    terms.append(([0, 3], [0, 3], [0, 3]))
    # M2
    terms.append(([2, 3], [0], [2]))
    # M3
    terms.append(([0], [1, 3], [1])) # V: 1, -3? No, signs are in coefficients
    # Actually, the "terms" in tensor decomposition are rank-1 tensors.
    # Strassen uses linear combinations.
    # The tensor T = sum c_k U_k \otimes V_k \otimes W_k.
    # For Strassen, the coefficients are +/- 1.
    
    # Let's just use the explicit 16 terms for 2x2x2 schoolbook? No, rank 7.
    # The 7 terms are:
    # 1. U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1]  (M1) -> Coeff 1
    # 2. U=[0,0,1,1], V=[1,0,0,0], W=[0,0,1,0]  (M2) -> Coeff 1
    # 3. U=[1,0,0,0], V=[0,1,0,-1], W=[0,1,0,0] (M3) -> Coeff 1
    # 4. U=[0,0,0,1], V=[0,0,1,-1], W=[1,0,1,0] (M4) -> Coeff 1
    # 5. U=[1,1,0,0], V=[0,0,0,1], W=[0,1,-1,0] (M5) -> Coeff 1
    # 6. U=[0,-1,1,0], V=[1,1,0,0], W=[0,0,0,1] (M6) -> Coeff 1
    # 7. U=[0,1,-1,0], V=[0,0,1,1], W=[1,0,0,-1] (M7) -> Coeff 1
    
    # Note: The vectors above are the rows of U, V, W for the 7 multiplications.
    # U_k is a vector of length 4.
    
    return terms
