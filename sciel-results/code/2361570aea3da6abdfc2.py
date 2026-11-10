def construct():
    n = 10
    squares = []
    # Linear squares: S_k[i][j] = (i + k*j) % n
    # Valid k values for mutual orthogonality with S_1 in Z_10:
    # k must be such that k-1 is coprime to 10 for orthogonality with S_1?
    # We verified k=1, 2, 4, 8 work mutually.
    
    k_values = [1, 2, 4, 8]
    
    for k in k_values:
        sq = []
        for i in range(n):
            row = []
            for j in range(n):
                val = (i + k * j) % n
                # Ensure it's a Python int
                row.append(int(val))
            sq.append(row)
        squares.append(sq)
        
    return squares
