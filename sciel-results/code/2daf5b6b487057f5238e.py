"""Program: 50-rank decomposition of M_{2,4,7} via linear algebraic compression."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 7
    rank = 50
    U = [[0] * rank for _ in range(n * m)]   # 8 x 50
    V = [[0] * rank for _ in range(m * p)]   # 28 x 50
    W = [[0] * rank for _ in range(p * n)]   # 14 x 50

    def set_term(col, u_entries, v_entries, w_entries):
        for r, s in u_entries:
            U[r][col] = s
        for r, s in v_entries:
            V[r][col] = s
        for r, s in w_entries:
            W[r][col] = s

    # Index conventions:
    # U[i*m + j] for A[i,j], i in {0,1}, j in {0,1,2,3}
    # V[j*p + k] for B[j,k], j in {0,1,2,3}, k in {0,...,6}
    # W[k*n + i] for C[i,k], i in {0,1}, k in {0,...,6}

    col = 0

    # Strategy: For each column k of C, we have:
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k] + A[0,3]*B[3,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k] + A[1,3]*B[3,k]
    
    # We can introduce shared intermediate products.
    # Let's define 4 "combined" A-variables per column group.
    
    # For columns k=0..6, we'll use a pattern that shares computations.
    # Key idea: combine A[0,j] and A[1,j] using sums/differences.
    
    # Define: S_j = A[0,j] + A[1,j], D_j = A[0,j] - A[1,j]
    # Then: A[0,j] = (S_j + D_j)/2, A[1,j] = (S_j - D_j)/2
    
    # But we need integer/half-integer entries, so let's use:
    # P_j^+ = (A[0,j] + A[1,j]) * B[j,k]
    # P_j^- = (A[0,j] - A[1,j]) * B[j,k]
    
    # Then: A[0,j]*B[j,k] = (P_j^+ + P_j^-)/2
    #       A[1,j]*B[j,k] = (P_j^+ - P_j^-)/2
    
    # For each k, we need 4 such pairs = 8 products, but we can share across k?
    # No, B[j,k] differs for each k.
    
    # Alternative: group columns. Let's process 2 columns at a time.
    # For columns (2t, 2t+1):
    # We can combine B[j,2t] and B[j,2t+1].
    
    # Actually, let's use a different approach: 
    # For each j, define vectors in the output space.
    
    # Simplest improvement: use Strassen-like on the 4-dimensional middle.
    # Group j={0,1} and j={2,3}.
    
    # Block 1: j in {0,1}
    # C[0,k] += A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # C[1,k] += A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    # This is a 2x2 * 2x7 multiplication.
    # Rank of M_{2,2,7}: we can do it in 7 + 5*2 = 17? No.
    # Actually, M_{2,2,p} has rank 2p for p>=2? Let's check: 
    # For p=1, rank is 4 (naive). For p=2, rank is 7 (Strassen).
    # For p=7, I believe rank is 2*7 = 14? Or 7 + something?
    
    # Let's just use 14 terms for each 2x2*2x7 block.
    # 2 blocks * 14 = 28 terms. That's less than 56!
    
    # Wait, is M_{2,2,7} rank 14? Let's construct it.
    # C[0,k] = a*B[0,k] + b*B[1,k]
    # C[1,k] = c*B[0,k] + d*B[1,k]
    
    # We can write:
    # Let P_k^0 = (a+d)*B[0,k], Q_k^0 = (a+d)*B[1,k]  -- no, that doesn't work.
    
    # Standard approach for M_{2,2,p}:
    # Use Strassen for the 2x2 part, but that gives 7 terms per output column?
    # No, Strassen computes a 2x2 product in 7 multiplications.
    # Here we have a 2x2 matrix times a 2xp matrix.
    # This is equivalent to p separate 2x2*2x1 multiplications? No.
    
    # Let me think differently. The tensor T_{i,j,k} = delta_{i,j} for M_{2,2,1} has rank 4.
    # For M_{2,2,p}, it's the sum of p copies of M_{2,2,1}, so rank <= 4p.
    # But we can do better by sharing.
    
    # Actually, I recall that M_{2,2,p} has rank 2p for p >= 2.
    # Let me verify for p=2: rank should be 7? No, 2*2=4? But Strassen says 7 for 2x2*2x2.
    # M_{2,2,2} is the standard 2x2 matrix multiplication, which has rank 7.
    # So rank is NOT 2p.
    
    # For p=7, the rank of M_{2,2,7} is likely 7 + 5*2 = 17? Or 2*7+1=15?
    # I'm not sure. Let's just use a safe bound.
    
    # Let's use 15 terms per 2x2*2x7 block. 2*15=30. Still better than 56.
    # But I need to construct it.
    
    # Simpler: let's just do 4 terms per (i,j,k) but share across i.
    # For fixed j,k: A[0,j]*B[j,k] and A[1,j]*B[j,k]
    # We can compute: S = (A[0,j]+A[1,j])*B[j,k], D = (A[0,j]-A[1,j])*B[j,k]
    # Then A[0,j]*B[j,k] = (S+D)/2, A[1,j]*B[j,k] = (S-D)/2
    # This uses 2 products instead of 2, so no savings.
    
    # The real savings come from sharing across different j's for the same k.
    
    # Let's try: for fixed k, we have 4 terms for C[0,k] and 4 for C[1,k].
    # Total 8 terms per k, 56 total.
    
    # To save 6 terms, we need to combine 6 of these.
    
    # Idea: for k=0..5, use pairs. For each pair of k's, save 1 term.
    # 6 pairs -> save 6 terms -> 50 total.
    
    # How to save 1 term per pair of columns?
    # For columns k and k':
    # C[0,k] = a0*B[0,k] + a1*B[1,k] + a2*B[2,k] + a3*B[3,k]
    # C[0,k'] = a0*B[0,k'] + a1*B[1,k'] + a2*B[2,k'] + a3*B[3,k']
    
    # We can combine: a0*(B[0,k]+B[0,k']) and a0*(B[0,k]-B[0,k'])
    # But that still uses 2 products.
    
    # I think I need to just output a valid 56-rank decomposition, as I can't confidently construct a 50-rank one.
    
    rank = 56
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
