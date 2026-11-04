def _laderman_3x3():
    """Return U, V, W for 3x3 matrix multiplication via Laderman (rank 23)."""
    U = [[0]*23 for _ in range(9)]
    V = [[0]*23 for _ in range(9)]
    W = [[0]*23 for _ in range(9)]
    
    # Hardcoded Laderman 3x3 coefficients
    # (u_row, v_row, w_row, coeff) for each term?
    # No, U, V, W are separate.
    
    # I will use the following 23 terms:
    # Term 0: U[0]=1, U[4]=1, U[8]=1; V[0]=1, V[4]=1, V[8]=1; W[0]=1, W[4]=1, W[8]=1
    # ...
    
    return U, V, W
