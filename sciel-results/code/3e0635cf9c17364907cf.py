"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # We'll construct a decomposition using a structured pattern.
    # Start with schoolbook and try to optimize by merging terms.
    
    # For now, use a rank-100 decomposition based on sharing.
    # Each term is (u_row, v_row, w_row, coeff)
    
    # Let's use a different approach: exploit the structure.
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # We can write:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    
    # Use the identity: a*b = ((a+b)^2 - a^2 - b^2)/2
    # But that's for scalars, not directly applicable.
    
    # Let's try a simple rank reduction: for each i, compute S[i] = A[i,0]+A[i,1]+A[i,2]+A[i,3]
    # and T[k] = B[0,k]+B[1,k]+B[2,k]+B[3,k]
    # Then S[i]*T[k] = sum_j sum_l A[i,j]*B[l,k]
    # So C[i,k] = S[i]*T[k] - sum_{j!=l} A[i,j]*B[l,k]
    
    # This doesn't directly help. Let me just use a known good decomposition.
    
    # I'll implement a rank-96 decomposition using a pattern.
    rank = 96
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each (i, j, k) triple, we have a term. But we'll merge some.
    # Actually, let me just do a slightly better schoolbook.
    
    # Use the fact that for fixed i, the 4 terms in j can be handled with 4 rank-1 terms,
    # but we can share across k.
    
    # Simple approach: keep schoolbook but with a different ordering.
    # This won't improve rank, so let me try something else.
    
    # Let me try to find a decomposition with rank < 112 using a greedy method.
    # But that's complex. For now, return schoolbook.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
