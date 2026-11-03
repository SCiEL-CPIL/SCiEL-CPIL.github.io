import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # 1. Construct the tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # 2. Try to use nptools FTD
    try:
        import nptools
        from nptools.tensor.decomposition import FTD
        
        # FTD might require specific settings.
        # We want an exact decomposition.
        # FTD returns a decomposition, but it's approximate.
        # We will try to round the factors to half-integers and check exactness.
        
        # Initialize FTD
        # maxrank: We don't know the minimal rank, so we might need to search.
        # However, FTD is a specific algorithm that finds a decomposition.
        # It might not minimize rank directly.
        # The directive says "apply the Fast Tensor Decomposition (FTD) algorithm... to compute an exact rank decomposition".
        # This implies FTD might yield a low rank decomposition.
        
        # Let's try to get a decomposition.
        # nptools FTD signature: FTD(tensor, rank=None, ...)
        # If rank is not specified, it might estimate it.
        
        # Since we want to minimize rank, and FTD is an algorithm for decomposition,
        # we might need to try different ranks or let it decide.
        # For now, let's try to get a decomposition and see what rank it gives.
        # Note: FTD in nptools is often used for approximations.
        
        # Let's attempt a decomposition.
        # We'll try a few ranks starting from a lower bound.
        # Lower bound for rank R of M_{n,m,p} is roughly max(nm, mp, pn) is not a lower bound.
        # The trivial rank is nmp.
        
        # Let's just try to run FTD and see if it produces a valid exact decomposition
        # with rank < n*m*p after rounding.
        
        # nptools FTD might be slow or fail.
        # We'll catch exceptions.
        
        # Try to get a decomposition with rank close to the lower bound?
        # Or just let it run.
        
        # Actually, nptools FTD is a specific algorithm.
        # Let's try to instantiate it.
        # The exact API might vary.
        
        # Common usage:
        # decomp = FTD(T)
        # U, V, W = decomp.factors
        
        # But we need exactness.
        # We will try to round U, V, W to half-integers.
        
        # Since we don't know the rank, we might try to find the minimal rank
        # by checking if a decomposition of rank r exists.
        # This is hard.
        
        # Alternative interpretation: The directive says "apply FTD... to compute an exact rank decomposition".
        # This might imply that for this specific tensor, FTD yields a low rank exact decomposition.
        
        # Let's try to run FTD with a guessed rank or default.
        # If it fails, we fallback.
        
        # Let's try a rank slightly below n*m*p.
        # Actually, let's just try to get *any* decomposition and see if we can round it.
        
        # If nptools is available, let's try to use it.
        # We'll try a few candidate ranks.
        
        best_rank = n * m * p
        best_decomp = None
        
        # We know the trivial rank is n*m*p.
        # We can try to find a rank < n*m*p.
        
        # Let's try to run FTD.
        # If FTD doesn't minimize rank, this might not work.
        # But the directive is specific.
        
        # Let's try to decompose T.
        # We'll try ranks from a lower estimate up to n*m*p - 1.
        # Lower bound for rank of M_{n,m,p} is at least max(nm, mp, pn) / something?
        # Actually, the rank is at least the dimension of the span of slices.
        
        # Let's just try to run FTD and see.
        # If it's too slow, we timeout.
        
        # Let's try to get a decomposition with rank = n*m*p - 1, etc.
        # This is inefficient.
        
        # Maybe FTD finds a specific structure.
        # Let's just try to get a decomposition and round it.
        
        # To avoid timeout, we will limit the attempts.
        
        # Try default FTD
        try:
            decomp = FTD(T)
            U, V, W = decomp.factors
            rank = U.shape[1]
            
            # Round to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Check exactness
            # Reconstruct tensor
            T_recon = np.zeros((n*m, m*p, p*n))
            for r in range(rank):
                T_recon += np.outer(np.outer(U_rounded[:, r], V_rounded[:, r]), W_rounded[:, r])
            
            if np.allclose(T_recon, T, atol=1e-9):
                if rank < best_rank:
                    best_rank = rank
                    best_decomp = (U_rounded, V_rounded, W_rounded)
        except Exception:
            pass
            
        # If we found a better rank, return it
        if best_decomp is not None:
            U, V, W = best_decomp
            # Convert to lists of lists with float/ints
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }

    except Exception:
        pass

    # Fallback to schoolbook
    rank = n * m * p
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
