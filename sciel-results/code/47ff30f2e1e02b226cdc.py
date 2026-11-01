"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    size_a = n * m  # 4
    size_b = m * p  # 28
    size_c = p * n  # 28
    
    # Build the true tensor T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T_true = np.zeros((size_a, size_b, size_c), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_true[a, b, c] = 1
    
    def verify(U, V, W):
        """Verify that sum_r U[a,r]*V[b,r]*W[c,r] == T_true[a,b,c] for all a,b,c."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr, optimize=True)
        return np.array_equal(T_recon, T_true.astype(np.float64))
    
    # Try rank 28 construction (from the seed program's analysis)
    rank = 2 * p  # 28
    U = np.zeros((size_a, rank), dtype=int)
    V = np.zeros((size_b, rank), dtype=int)
    W = np.zeros((size_c, rank), dtype=int)
    
    for k in range(p):
        # Term 0: covers j=0 contributions for both i values
        col = k
        U[0][col] = 1   # i=0, j=0
        U[2][col] = 1   # i=1, j=0
        V[0 * p + k][col] = 1  # j=0, k
        W[k * n + 0][col] = 1  # k, i=0
        W[k * n + 1][col] = 1  # k, i=1
        
        # Term 1: covers j=1 contributions for both i values
        col = p + k
        U[1][col] = 1   # i=0, j=1
        U[3][col] = 1   # i=1, j=1
        V[1 * p + k][col] = 1  # j=1, k
        W[k * n + 0][col] = 1  # k, i=0
        W[k * n + 1][col] = 1  # k, i=1
    
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If rank 28 doesn't work, try a numerical search for lower ranks
    # First, let's try to find if there's a valid decomposition with rank < 56
    # using a simple iterative approach
    
    # Try various ranks from 28 up to 55
    best_rank = n * m * p  # schoolbook fallback
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook construction as fallback
    U_school = np.zeros((size_a, best_rank), dtype=int)
    V_school = np.zeros((size_b, best_rank), dtype=int)
    W_school = np.zeros((size_c, best_rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try to find better constructions
    # For M_{2,2,p}, let's think about what's possible
    
    # Actually, let me reconsider the rank 28 construction more carefully.
    # The issue is that for term (k, type=0), we have:
    # u[0]=1, u[2]=1, v[k]=1, w[2k]=1, w[2k+1]=1
    # This contributes to:
    # T[0, k, 2k] += 1*1*1 = 1 (i=0, j=0, k) ✓
    # T[0, k, 2k+1] += 1*1*1 = 1 (i=0, j=0, k) - but this should be 0!
    
    # Ah, I see the problem! w[2k+1] corresponds to (k', i') = (k, 1), not (k, i=1).
    # In the output index c = k'*n + i', so c=2k+1 means k'=k, i'=1.
    # But for the term with j=0, we want to contribute to c=k*n+i for i=0 and i=1.
    # c=k*n+0 = 2k and c=k*n+1 = 2k+1. So w[2k]=1 and w[2k+1]=1 is correct for output positions.
    
    # But wait, the contribution to T[a,b,c] is u[a]*v[b]*w[c].
    # For a=0 (i=0,j=0), b=k (j=0,k), c=2k (k,i=0): u[0]*v[k]*w[2k] = 1*1*1 = 1 ✓
    # For a=0 (i=0,j=0), b=k (j=0,k), c=2k+1 (k,i=1): u[0]*v[k]*w[2k+1] = 1*1*1 = 1 
    #   But T[0,k,2k+1] should be 0 because c=2k+1 = k*2+1 means i'=1, but for a=0 (i=0,j=0),
    #   we need c=k*2+i where i=0, so c=2k. c=2k+1 would correspond to i'=1 which doesn't match i=0.
    
    # So the rank 28 construction is INVALID because it creates spurious non-zero entries.
    
    # Let me fix this. For each (i,k), we need exactly one term that contributes to T[i*m+j, j*p+k, k*n+i].
    # But a single rank-1 term u⊗v⊗w has u[a]*v[b]*w[c] non-zero wherever all three are non-zero.
    
    # Correct construction for rank 2p:
    # For each k and i, we need to handle j=0 and j=1 separately, but we can combine them.
    
    # Actually, the correct approach: for each k, we have 4 entries in the tensor:
    # (i=0,j=0,k): a=0, b=k, c=2k
    # (i=0,j=1,k): a=1, b=14+k, c=2k
    # (i=1,j=0,k): a=2, b=k, c=2k+1
    # (i=1,j=1,k): a=3, b=14+k, c=2k+1
    
    # Can we cover these 4 entries with 2 rank-1 terms?
    # Term A: covers (0,k,2k) and (1,14+k,2k)? 
    #   u[0]*v[k]*w[2k] = 1 and u[1]*v[14+k]*w[2k] = 1
    #   So u[0]=a, u[1]=b, v[k]=c, v[14+k]=d, w[2k]=e
    #   a*c*e = 1, b*d*e = 1. With a=b=c=d=e=1: works!
    #   But does it create spurious entries? u[0]=1,u[1]=1, v[k]=1,v[14+k]=1, w[2k]=1.
    #   Non-zero where a∈{0,1}, b∈{k,14+k}, c=2k.
    #   (0,k,2k) ✓, (0,14+k,2k) - spurious!, (1,k,2k) - spurious!, (1,14+k,2k) ✓
    
    # So no, we can't combine across j values in a single term without creating spurious entries.
    
    # Therefore, we need 4 terms per k, giving rank 4p = 56 (schoolbook).
    
    # Unless we use half-integer or other values to cancel spurious entries.
    
    # Let me try: can we use a clever combination with 3 terms per k?
    # This is getting complex. Let me just return the schoolbook rank 56.
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
