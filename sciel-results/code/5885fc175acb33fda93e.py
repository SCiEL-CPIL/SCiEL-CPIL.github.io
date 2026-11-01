"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-minimal exact tensor decomposition for 2x2 matrix multiplication.
    Uses Strassen's algorithm (rank 7, optimal).
    Output factor follows trace convention W[k*n+i, r].
    
    The decomposition satisfies:
    T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    where T[i, j, k] is 1 if (i,j,k) corresponds to a non-zero entry in the matmul tensor, else 0.
    For 2x2 matmul, the tensor has 8 non-zero entries (schoolbook rank 8), but Strassen compresses this to rank 7.
    
    Indices:
    n, m, p = 2, 2, 2
    U shape: (n*m, rank) = (4, 7)
    V shape: (m*p, rank) = (4, 7)
    W shape: (p*n, rank) = (4, 7)
    
    U row index: i*m + j
    V row index: j*p + k
    W row index: k*n + i
    """
    n, m, p = DIMENSIONS
    rank = 7
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 products:
    # Let A = [[a, b], [c, d]] and B = [[e, f], [g, h]]
    # A indices in U: a->0, b->1, c->2, d->3
    # B indices in V: e->0, f->1, g->2, h->3
    # C indices in W: C00->0 (k=0,i=0), C10->1 (k=0,i=1), C01->2 (k=1,i=0), C11->3 (k=1,i=1)
    
    # M1 = (a + d) * (e + h)
    # U: a(1), d(1) -> U[0][0]=1, U[3][0]=1
    # V: e(1), h(1) -> V[0][0]=1, V[3][0]=1
    # W: C00(+1), C11(+1) -> W[0][0]=1, W[3][0]=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (c + d) * e
    # U: c(1), d(1) -> U[2][1]=1, U[3][1]=1
    # V: e(1) -> V[0][1]=1
    # W: C10(+1), C11(-1) -> W[1][1]=1, W[3][1]=-1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3 = a * (f - h)
    # U: a(1) -> U[0][2]=1
    # V: f(1), h(-1) -> V[1][2]=1, V[3][2]=-1
    # W: C01(+1), C11(+1) -> W[2][2]=1, W[3][2]=1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = d * (g - e)
    # U: d(1) -> U[3][3]=1
    # V: g(1), e(-1) -> V[2][3]=1, V[0][3]=-1
    # W: C00(+1), C10(+1) -> W[0][3]=1, W[1][3]=1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (a + b) * h
    # U: a(1), b(1) -> U[0][4]=1, U[1][4]=1
    # V: h(1) -> V[3][4]=1
    # W: C00(-1), C01(+1) -> W[0][4]=-1, W[2][4]=1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (c - a) * (e + f)
    # U: c(1), a(-1) -> U[2][5]=1, U[0][5]=-1
    # V: e(1), f(1) -> V[0][5]=1, V[1][5]=1
    # W: C11(+1) -> W[3][5]=1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (b - d) * (g + h)
    # U: b(1), d(-1) -> U[1][6]=1, U[3][6]=-1
    # V: g(1), h(1) -> V[2][6]=1, V[3][6]=1
    # W: C00(+1) -> W[0][6]=1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    # Optional: Verification step to ensure correctness.
    # This is cheap (4*4*7 operations) and ensures we don't return an invalid decomposition.
    # The tensor T[i,j,k] for matmul is 1 if the product involves A[i,j] and B[j,k] to form C[i,k].
    # Specifically, C[i,k] = sum_j A[i,j]B[j,k].
    # So T[i,j,k] = 1 if k is the output column and i is the output row, and j is the contraction index.
    # Wait, the tensor definition is usually T_{ijk} such that C_{ik} = sum_j A_{ij} B_{jk} T_{ijk}? 
    # No, the standard tensor for matmul is T such that C_{ik} = sum_{j} A_{ij} B_{jk}.
    # In the decomposition context, we are decomposing the tensor of the operation.
    # The evaluator checks if sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] equals the expected tensor value.
    # For 2x2 matmul, the expected tensor value for indices (i,j,k) is 1 if the term A[i,j]B[j,k] contributes to C[i,k].
    # Actually, the tensor is defined on the indices of the three matrices.
    # The value is 1 if the combination of indices (i from A, j from A/B, k from B) forms a valid multiplication term.
    # Specifically, for each i, j, k, the term A[i,j] * B[j,k] appears in C[i,k].
    # So T[i,j,k] = 1 for all i,j,k? No.
    # Let's check the evaluator logic implied by "exact tensor decomposition".
    # Usually, the tensor T has entries T_{i,j,k} = 1 if the product A_{i,j} B_{j,k} is part of the sum for C_{i,k}.
    # In 2x2 matmul, every combination i, j, k where j is the contraction index is valid.
    # i in {0,1}, j in {0,1}, k in {0,1}.
    # There are 2*2*2 = 8 terms.
    # The decomposition must satisfy: sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = 1 for all i,j,k.
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if val != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook rank 8 if Strassen fails verification (should not happen)
        rank = 8
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
