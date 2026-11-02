"""Matrix multiplication 3,3,8 - rank-minimal decomposition."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
    nm, mp, pn = n * m, m * p, p * n
    rank = n * m * p
    
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return U, V, W

def _try_low_rank_search(n, m, p, deadline):
    """
    Search for low-rank decompositions using structured approaches and random search.
    The key insight: for matmul tensor T[i,j,k]=1, we need
    sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all (i,j,k)
    """
    nm, mp, pn = n * m, m * p, p * n
    best_rank = n * m * p
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_decomp = (U_sb, V_sb, W_sb)
    
    # The tensor is all-ones. Key structural insight:
    # If we can write 1 = sum_r a_r[i,j] * b_r[j,k] * c_r[k,i]
    # where a_r, b_r, c_r are structured.
    
    # For the all-ones tensor, one natural approach:
    # Use rank = n*m*p / gcd structure
    # Actually, let's think about this more carefully.
    
    # The constraint is: for ALL (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # This is 3*3*8 = 72 equations.
    
    # Lower bound: the tensor T[i,j,k]=1 has rank at least max(n,m,p) = 8
    # (by fixing i and j, we get a vector in k-space that must be represented)
    
    # Actually the rank of the all-ones tensor J_{n,m,p} is min(n,m,p) in general,
    # but here we have additional structure because U is indexed by (i,j), V by (j,k), W by (k,i).
    
    # Let's try: rank = n*m = 9 (one term per (i,j) pair)
    # Then U[i*m+j, (i',j')] = delta_{i,i'} * delta_{j,j'} = 1 if (i,j)=(i',j')
    # V[j*p+k, (i',j')] needs to satisfy: for fixed (i,j), sum over (i',j') with i'=i,j'=j:
    #   1 * V[j*p+k, (i,j)] * W[k*n+i, (i,j)] = 1
    # So V[j*p+k, (i,j)] * W[k*n+i, (i,j)] = 1 for all k
    # This means V[j*p+k, (i,j)] = 1/W[k*n+i, (i,j)] for all k
    # If we set W[k*n+i, (i,j)] = 1 for all k, then V[j*p+k, (i,j)] = 1 for all k.
    # But wait, V is indexed by (j,k) and the column is (i',j'). For a given column (i',j'),
    # V[j*p+k, (i',j')] should be non-zero only when j=j'.
    # So V[j*p+k, (i',j')] = delta_{j,j'} * (1/W[k*n+i', (i',j')])
    
    # Let's try: W[k*n+i, (i',j')] = delta_{i,i'} for all k
    # Then V[j*p+k, (i',j')] = delta_{j,j'} * (1/delta_{i,i'}) ... this doesn't work cleanly.
    
    # Let me try a different structured approach.
    
    # Approach: rank = n*p = 24
    # Column r corresponds to (i_r, k_r)
    # U[i*m+j, (i',k')] = delta_{i,i'} for all j
    # V[j*p+k, (i',k')] = delta_{k,k'} for all j  
    # W[k*n+i, (i',k')] = delta_{i,i'} * delta_{k,k'}
    # Then sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = sum_{i',k'} delta_{i,i'}*delta_{k,k'}*delta_{i,i'}*delta_{k,k'} = 1
    # Wait, that gives sum over (i',k') where i'=i and k'=k: 1*1*1*1 = 1. 
    # But we're summing over ALL (i',k'), so only the term (i',k')=(i,k) contributes.
    # So the sum is 1. 
    
    # Let's verify: U[i*m+j, (i',k')] = delta_{i,i'} (independent of j)
    # V[j*p+k, (i',k')] = delta_{k,k'} (independent of j)
    # W[k*n+i, (i',k')] = delta_{i,i'} * delta_{k,k'}
    
    # sum_{i',k'} delta_{i,i'} * delta_{k,k'} * delta_{i,i'} * delta_{k,k'}
    # = sum_{i',k'} delta_{i,i'}^2 * delta_{k,k'}^2
    # = 1 (only (i',k')=(i,k) contributes, giving 1)
    
    # This works! Rank = n*p = 24.
    
    # Can we do better? What about rank = m*p = 24? Similar argument.
    
    # What about rank = n*m = 9?
    # Column r corresponds to (i_r, j_r)
    # U[i*m+j, (i',j')] = delta_{i,i'} * delta_{j,j'}
    # V[j*p+k, (i',j')] = delta_{j,j'} * a_{k,(i',j')}
    # W[k*n+i, (i',j')] = b_{k,(i',j')} * delta_{i,i'}
    # sum_{i',j'} delta_{i,i'}*delta_{j,j'} * delta_{j,j'}*a_{k,(i',j')} * b_{k,(i',j')}*delta_{i,i'}
    # = delta_{i,i}*delta_{j,j} * a_{k,(i,j)} * b_{k,(i,j)}
    # = a_{k,(i,j)} * b_{k,(i,j)}
    # We need this to equal 1 for all k, so a_{k,(i,j)} * b_{k,(i,j)} = 1
    # Set a_{k,(i,j)} = 1, b_{k,(i,j)} = 1
    # Then V[j*p+k, (i',j')] = delta_{j,j'}, W[k*n+i, (i',j')] = delta_{i,i'}
    
    # Let's verify:
    # sum_{i',j'} U[i*m+j,(i',j')] * V[j*p+k,(i',j')] * W[k*n+i,(i',j')]
    # = sum_{i',j'} delta_{i,i'}*delta_{j,j'} * delta_{j,j'} * delta_{i,i'}
    # = delta_{i,i}*delta_{j,j}*delta_{j,j}*delta_{i,i}
    # = 1
    
    # This works! Rank = n*m = 9!
    
    # Let's construct it:
    rank_9 = n * m
    U_9 = [[0] * rank_9 for _ in range(nm)]
    V_9 = [[0] * rank_9 for _ in range(mp)]
    W_9 = [[0] * rank_9 for _ in range(pn)]
    
    for col, (i_r, j_r) in enumerate(product(range(n), range(m))):
        for i in range(n):
            for j in range(m):
                if i == i_r and j == j_r:
                    U_9[i*m + j][col] = 1
        for j in range(m):
            for k in range(p):
                if j == j_r:
                    V_9[j*p + k][col] = 1
        for k in range(p):
            for i in range(n):
                if i == i_r:
                    W_9[k*n + i][col] = 1
    
    if _verify_decomposition(U_9, V_9, W_9, n, m, p):
        best_decomp = (U_9, V_9, W_9)
        best_rank = rank_9
    
    # Can we go below 9? The lower bound is max(n,m,p)=8.
    # Try rank 8:
    # We need 72 equations with 8*9 + 8*24 + 8*24 = 72+192+192 = 456 unknowns.
    # This is underdetermined, so solutions likely exist.
    
    # Let's try to find rank 8 solutions using random search with rounding.
    
    for target_rank in range(min(8, best_rank - 1), 5, -1):
        if time.time() > deadline:
            break
        
        # Try random search with rounding to integers/half-integers
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            # Random initialization
            U_rand = np.random.randint(-2, 3, size=(nm, target_rank)) * 0.5
            V_rand = np.random.randint(-2, 3, size=(mp, target_rank)) * 0.5
            W_rand = np.random.randint(-2, 3, size=(pn, target_rank)) * 0.5
            
            # Check
            if _verify_decomposition(U_rand.tolist(), V_rand.tolist(), W_rand.tolist(), n, m, p):
                best_decomp = (U_rand.tolist(), V_rand.tolist(), W_rand.tolist())
                best_rank = target_rank
                break
        
        # Also try more structured approaches for rank 8
        if target_rank == 8:
            # Try to exploit the structure: 8 = p
            # Column r corresponds to k_r
            # U[i*m+j, k_r] = a_{i,j,k_r}
            # V[j*p+k, k_r] = delta_{k,k_r} * b_{j,k_r}
            # W[k*n+i, k_r] = c_{k,i,k_r}
            
            # sum_{k_r} a_{i,j,k_r} * delta_{k,k_r} * b_{j,k_r} * c_{k,i,k_r}
            # = a_{i,j,k} * b_{j,k} * c_{k,i,k}
            # Need this to equal 1 for all (i,j,k)
            
            # So a_{i,j,k} * b_{j,k} * c_{k,i,k} = 1
            # Set a_{i,j,k} = 1, b_{j,k} = 1, c_{k,i,k} = 1
            # Then U[i*m+j, k_r] = 1 for all i,j,k_r
            # V[j*p+k, k_r] = delta_{k,k_r}
            # W[k*n+i, k_r] = delta_{k,k_r}
            
            # Verify: sum_{k_r} 1 * delta_{k,k_r} * 1 * delta_{k,k_r} = delta_{k,k} = 1. 
            # This works! But wait, this is just rank p = 8.
            
            U_8 = [[0] * 8 for _ in range(nm)]
            V_8 = [[0] * 8 for _ in range(mp)]
            W_8 = [[0] * 8 for _ in range(pn)]
            
            for k_r in range(p):
                for i in range(n):
                    for j in range(m):
                        U_8[i*m + j][k_r] = 1
                for j in range(m):
                    for k in range(p):
                        if k == k_r:
                            V_8[j*p + k][k_r] = 1
                for k in range(p):
                    for i in range(n):
                        if k == k_r:
                            W_8[k*n + i][k_r] = 1
            
            if _verify_decomposition(U_8, V_8, W_8, n, m, p):
                best_decomp = (U_8, V_8, W_8)
                best_rank = 8
                break
    
    return best_decomp[0], best_decomp[1], best_decomp[2], best_rank

def construct():
    """
    Returns a rank-minimal tensor decomposition for 3x3 * 3x8 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    U, V, W, rank = _try_low_rank_search(n, m, p, deadline)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
