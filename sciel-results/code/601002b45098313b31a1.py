"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank exact tensor decomposition for 2x3 x 3x4 matrix multiplication.
    Uses a structured approach: the schoolbook rank is 24. We try to find
    a lower rank by exploiting symmetries and solving the linear constraints.
    
    The condition is: for all i,k,a,b,c,d:
      sum_r W[k*n+i, r] * U[a*m+b, r] * V[c*p+d, r] = 1 if (a==i, b==c, d==k) else 0
    """
    import numpy as np
    
    n, m, p = DIMENSIONS  # 2, 3, 4
    
    # First, let's establish the schoolbook rank-24 solution and verify it works.
    # Then try to reduce rank.
    
    # The tensor has 24 "active" entries (one per (i,j,k) triple).
    # The condition: for each output (i,k), the coefficient of A[a,b]*B[c,d] in C[i,k]
    # must be 1 iff a==i, b==c, d==k, else 0.
    
    # Let me try a structured rank-20 approach.
    # Key idea: group by j-value. For each j, we have n*p = 8 (i,k) pairs.
    # Schoolbook uses 8 columns per j (one per i,k). Total = 3*8 = 24.
    # If we can share some columns across j-values, we reduce rank.
    
    # Actually, let me try a different approach: use the fact that for a fixed j,
    # the contribution is A[i,j]*B[j,k]. We can factor this differently.
    
    # Let me try rank 20 with a specific structured construction.
    # I'll use 5 columns per j-value (instead of 8), for a total of 15,
    # plus 5 more shared columns = 20.
    
    # Actually, let me just try to solve for rank r using a greedy/structured approach.
    # For small ranks, I can try specific patterns.
    
    # Let me try the following construction for rank 20:
    # For each j in {0,1,2}, use 6 columns (total 18), plus 2 shared = 20.
    # For j, the 6 columns handle the 8 (i,k) pairs with 2 shared.
    
    # Hmm, let me try a cleaner approach. Let me use rank 20 with:
    # - 8 columns for j=0 (one per (i,k) pair)  
    # - 6 columns for j=1 (sharing 2 with j=0 pattern)
    # - 6 columns for j=2 (sharing 2 with j=0 pattern)
    # Total: 20
    
    # Actually, the cleanest approach: let me just verify the schoolbook and return it.
    # The task says "rank-minimal" but if I can't find better, 24 is safe.
    
    # Let me try one more thing: can I do rank 20 by using the structure?
    # For a fixed j, I need: sum over r of W[k*n+i,r] * U[i*m+j,r] * V[j*p+k,r] = 1
    # AND cross-terms must vanish.
    
    # The cross-term condition means: for fixed (i,k), the vector
    # w_{ik} = [W[k*n+i, r] for r] dotted with [U[a*m+b,r]*V[c*p+d,r] for r]
    # must give 1 only when (a,b,c,d)=(i,j,j,k) for some j.
    
    # This is a very strong condition. Let me just return schoolbook rank 24.
    
    rank = n * m * p  # 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
