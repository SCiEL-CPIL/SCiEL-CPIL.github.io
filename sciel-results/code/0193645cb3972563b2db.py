<<<<<<< SEARCH
def _try_block_structure(n, m, p, deadline):
    """
    Try to exploit block structure. Split the 3x7 into blocks and use
    a tensor product / Kronecker-like decomposition.
    
    Idea: 3x3 * 3x7 = 3x3 * (3x2 + 3x1) roughly, but that's not quite right.
    
    Alternative: Use the fact that we can share the "3" dimension.
    Write C = A * B where A is 3x3, B is 3x7.
    
    For each column block of B (say 2 columns at a time), we can use
    a 3x3 * 3x2 multiplication which has rank at most 3*3*2 = 18,
    but potentially less.
    
    7 = 2+2+2+1, so 4 blocks: three 3x2 and one 3x1.
    3x2 mult: rank <= 18, 3x1 mult: rank <= 9.
    Total: 3*18 + 9 = 63. Same as schoolbook.
    
    But if we can share computation across blocks, we might do better.
    """
    pass
=======
def _try_sparse_structured(n, m, p, deadline, seed=42):
    """
    Try structured sparse patterns for U, V, W.
    Key insight: for rank r, we need the tensor decomposition to be exact.
    Try patterns where each column of U, V, W has specific sparsity.
    
    Strategy: Use a parametric family where columns are indexed by tuples.
    For 3x3 * 3x7, try rank = 3*7 = 21 (the lower bound).
    Structure: column r corresponds to (i, k) pair, with U depending on i,
    V depending on k, W depending on (i,k).
    """
    rng = random.Random(seed)
    
    # Try rank = 21 first (the theoretical lower bound)
    # Structure: 21 columns, indexed by (i, k) where i in 0..2, k in 0..6
    # U[i*m+j, (i',k')] = delta(i,i') * u_j  (depends on j)
    # V[j*p+k, (i',k')] = delta(k,k') * v_j  (depends on j)
    # W[k'*n+i, (i',k')] = w_{i,i'}
    
    # This gives: sum_{i',k'} delta(i,i')*u_j * delta(k,k')*v_j * w_{i,i'}
    # For term (i,j,k): only i'=i, k'=k contribute
    # = u_j * v_j * w_{i,i} = 1 for all i,j,k
    # So we need u_j * v_j * w_{i,i} = 1 for all i,j
    # This means w_{i,i} = 1/(u_j*v_j) which must be same for all j
    # So u_j * v_j = constant for all j, and w_{i,i} = 1/that constant
    
    # Let u_j = 1, v_j = 1, w_{i,i} = 1
    # Then U[i*m+j, (i',k')] = delta(i,i') for all j
    # V[j*p+k, (i',k')] = delta(k,k') for all j
    # W[k'*n+i, (i',k')] = delta(i,i') if k'=k else 0... wait, W index is (k'*n+i)
    
    # Actually W[k'*n+i, (i',k')] should depend on i and i'
    # Let's try: W[k'*n+i, (i',k')] = delta(i, i') * delta(k, k')
    # Then term (i,j,k): sum over (i',k') where i'=i, k'=k: u_j * v_j * 1 = u_j*v_j
    # Need u_j * v_j = 1 for all j. Set u_j=1, v_j=1.
    
    # So: U[i*m+j, (i',k')] = delta(i,i')
    #     V[j*p+k, (i',k')] = delta(k,k')
    #     W[k'*n+i, (i',k')] = delta(i,i') * delta(k,k')
    
    # Check: for (i,j,k), sum over (i',k'):
    # U[i*m+j, (i',k')] * V[j*p+k, (i',k')] * W[k*n+i, (i',k')]
    # = sum over (i',k') where i'=i AND k'=k: 1 * 1 * delta(i,i)*delta(k,k)
    # = 1 * 1 * 1 * 1 = 1. Yes!
    
    # This gives a rank-21 decomposition!
    r = 21
    U = [[Fraction(0)] * r for _ in range(n*m)]
    V = [[Fraction(0)] * r for _ in range(m*p)]
    W = [[Fraction(0)] * r for _ in range(p*n)]
    
    for i in range(n):
        for k in range(p):
            col = i * p + k  # column index for (i,k)
            for j in range(m):
                U[i*m+j][col] = Fraction(1)
                V[j*p+k][col] = Fraction(1)
            W[k*n+i][col] = Fraction(1)
    
    if _verify_exact(U, V, W, n, m, p):
        return (U, V, W)
    
    return None
>>>>>>> REPLACE

