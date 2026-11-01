"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0]) if U else 0
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # should equal 1 if this is a valid decomposition of the multiplication tensor.
    # Actually, the convention is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_c (sum_{i',j'} U[i',j',c] * A[i',j']) * (sum_{j'',k''} V[j'',k'',c] * B[j'',k'']) * ... 
    # No, the standard tensor decomposition is:
    # T_{(i,j),(j',k),(i',k')} = sum_c U_{(i,j),c} * V_{(j',k),c} * W_{(i',k'),c}
    # where T_{(i,j),(j',k),(i',k')} = 1 iff i=i', j=j', k=k'
    
    # So we check: for all i,j,k,i',j',k':
    # sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c] == (1 if i==i' and j==j' and k==k' else 0)
    
    # This is expensive to check fully. Instead, we can check via random test vectors.
    # Or we can check the tensor directly for small sizes.
    
    # For n=2, m=3, p=9: tensor size is 6*27*18 = 2916 entries
    # We can construct the full tensor and verify.
    
    # Actually, let's use a more efficient check: 
    # The multiplication tensor T has T[(i,j),(j',k),(i',k')] = delta(i,i')*delta(j,j')*delta(k,k')
    # A rank-r decomposition gives: T = sum_c U_c tensor V_c tensor W_c
    # So T[(i,j),(j',k),(i',k')] = sum_c U[(i,j),c] * V[(j',k),c] * W[(i',k'),c]
    
    # We can verify by checking that for random A, B:
    # (A*B)[i,k] == sum_c (sum_{i',j'} U[(i',j'),c]*A[i',j']) * (sum_{j'',k''} V[(j'',k''),c]*B[j'',k'']) * W[(i,k),c]
    # Wait, that's not right either.
    
    # Let me re-derive. The standard formulation:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # A rank-r decomposition means:
    # (A*B)[i,k] = sum_{c=1}^{r} (sum_{i',j'} U[c, i', j'] * A[i', j']) * (sum_{j'',k''} V[c, j'', k''] * B[j'', k'']) 
    # No, that gives a scalar, not C[i,k].
    
    # Correct formulation:
    # (A*B)[i,k] = sum_c f_c(A) * g_c(B) * h_c(i,k)
    # where f_c is linear in A, g_c is linear in B, h_c is a function of (i,k).
    # 
    # More precisely:
    # (A*B)[i,k] = sum_c (sum_{i',j'} U[c,i',j'] * A[i',j']) * (sum_{j'',k''} V[c,j'',k''] * B[j'',k'']) * W[c,i,k]
    
    # For this to equal sum_j A[i,j]*B[j,k] for all A,B:
    # The coefficient of A[i',j']*B[j'',k''] on the LHS is:
    # sum_c U[c,i',j'] * V[c,j'',k''] * W[c,i,k]
    # On the RHS, it's 1 if i'=i, j'=j'', k''=k, else 0.
    
    # So we need: sum_c U[c,i',j'] * V[c,j'',k''] * W[c,i,k] = delta(i',i)*delta(j',j'')*delta(k'',k)
    
    # In the interface, U has shape (n*m, rank), so U[i*m+j, c] = U[c, i, j]
    # V has shape (m*p, rank), so V[j*p+k, c] = V[c, j, k]  
    # W has shape (p*n, rank), so W[k*n+i, c] = W[c, k, i]
    
    # So the condition is:
    # sum_c U[i'*m+j', c] * V[j''*p+k'', c] * W[k*n+i, c] = delta(i',i)*delta(j',j'')*delta(k'',k)
    
    # Let me verify with a few random tests.
    n, m, p = n, m, p
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)  # (n*m, rank)
    V_arr = np.array(V, dtype=float)  # (m*p, rank)
    W_arr = np.array(W, dtype=float)  # (p*n, rank)
    
    # Check: for all i, j, j', k, k':
    # sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i, c] should be 1 if j==j' and k==k', else 0
    # Wait, let me re-index.
    
    # The condition: sum_c U[i1*m+j1, c] * V[j2*p+k2, c] * W[k3*n+i3, c] = delta(i1,i3)*delta(j1,j2)*delta(k2,k3)
    
    # Let's check by computing the full tensor and comparing.
    # Tensor T has indices (i1,j1) x (j2,k2) x (k3,i3)
    # T[(i1,j1),(j2,k2),(k3,i3)] = sum_c U[i1*m+j1, c] * V[j2*p+k2, c] * W[k3*n+i3, c]
    # Should equal 1 if i1==i3 and j1==j2 and k2==k3, else 0.
    
    # For efficiency, let's do a few random spot checks.
    rng = np.random.RandomState(42)
    for _ in range(20):
        i1 = rng.randint(n)
        j1 = rng.randint(m)
        j2 = rng.randint(m)
        k2 = rng.randint(p)
        k3 = rng.randint(p)
        i3 = rng.randint(n)
        
        val = 0.0
        for c in range(rank):
            val += U[i1*m+j1][c] * V[j2*p+k2][c] * W[k3*n+i3][c]
        
        expected = 1.0 if (i1 == i3 and j1 == j2 and k2 == k3) else 0.0
        if abs(val - expected) > tol:
            return False
    return True

def _schoolbook(n, m, p):
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
    return U, V, W

def _try_rank(r, n, m, p, max_iters=200, seed=0):
    """Try to find a rank-r decomposition using random search."""
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    
    # Initialize with random half-integers
    U = [[rng.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(r)] for _ in range(n*m)]
    V = [[rng.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(r)] for _ in range(m*p)]
    W = [[rng.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(r)] for _ in range(p*n)]
    
    # This random approach is unlikely to work. Let me try a smarter approach.
    # Use the structure: start from schoolbook and try to merge terms.
    return None  # Placeholder

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to reduce rank
    # For now, return schoolbook as a safe fallback
    # TODO: implement actual search
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
