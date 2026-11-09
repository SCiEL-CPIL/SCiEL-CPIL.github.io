"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    # Decompose 2x3 * 3x9 into three 2x3 * 3x3 block multiplications.
    # Use rank-9 decomposition per block (total rank 27).
    # For T_{2,3,3}: C[i,k] = sum_j A[i,j]*B[j,k]
    # Rank-9 construction: for each j, use 3 terms to handle the outer product A[:,j] outer B[j,:]
    # Each term r for fixed j contributes: U has 1 at (i,j) for one i, V has 1 at (j,k) for one k, W has 1 at (k*n+i)
    # But that's still 18 per block. Instead, use a smarter decomposition.
    # 
    # Known: rank(T_{2,3,3}) = 9. Construction:
    # For j=0,1,2 and i=0,1, create terms where:
    # Term (j,i): U[i*m+j]=1, V[j*p+k] = 1 for all k in block, W[k*n+i] = 1 for all k
    # This doesn't work because V and W would need to be all-ones.
    #
    # Alternative: use the fact that we can combine the k dimension.
    # For each (i,j), create one term: U[i*m+j]=1, V[j*p + k_global]=1 for a specific k, W[k_global*n+i]=1
    # This is the schoolbook approach with rank 18 per block.
    #
    # For rank 9, we need to share structure. Let's use:
    # For each j, create 3 terms (one per k) but combine the i dimension using linearity.
    # Actually, let's just use rank 18 per block for now (total 54) and verify correctness.
    # If rank 9 is possible, it requires a non-trivial decomposition.
    
    n, m, p = DIMENSIONS
    num_blocks = p // 3
    p_block = 3
    
    cols_U = []
    cols_V = []
    cols_W = []
    
    for b in range(num_blocks):
        for i in range(n):
            for j in range(m):
                for k in range(p_block):
                    u_idx = i * m + j
                    v_idx = j * p + (b * p_block + k)
                    w_idx = (b * p_block + k) * n + i
                    
                    col_u = [0] * (n * m)
                    col_v = [0] * (m * p)
                    col_w = [0] * (p * n)
                    
                    col_u[u_idx] = 1
                    col_v[v_idx] = 1
                    col_w[w_idx] = 1
                    
                    cols_U.append(col_u)
                    cols_V.append(col_v)
                    cols_W.append(col_w)
                    
    rank = len(cols_U)
    
    U = [[cols_U[c][r] for c in range(rank)] for r in range(n * m)]
    V = [[cols_V[c][r] for c in range(rank)] for r in range(m * p)]
    W = [[cols_W[c][r] for c in range(rank)] for r in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
