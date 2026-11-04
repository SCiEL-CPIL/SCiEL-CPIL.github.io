def construct():
    n, m, p = 2, 4, 5
    # Split p into 3+2
    # M_{2,4,3} + M_{2,4,2}
    
    # M_{2,4,2}: split m into 2+2, use Strassen for each M_{2,2,2}
    # M_{2,4,3}: split m into 2+2, use schoolbook for each M_{2,2,3}
    
    # Let's build the full decomposition
    
    U = []
    V = []
    W = []
    
    # Part 1: M_{2,4,2} (k=0,1), using Strassen for j=0,1 and j=2,3
    # For j=0,1 (M_{2,2,2} with Strassen):
    # U indices: i*4+j for i=0,1, j=0,1 → 0,1,4,5
    # V indices: j*5+k for j=0,1, k=0,1 → 0,1,5,6
    # W indices: k*2+i for k=0,1, i=0,1 → 0,1,2,3
    
    # Strassen terms for 2x2x2:
    # I'll use the correct Strassen decomposition
    
    # Actually, let me just use schoolbook for everything to get rank 40,
    # then try to improve.
    
    # Schoolbook:
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
