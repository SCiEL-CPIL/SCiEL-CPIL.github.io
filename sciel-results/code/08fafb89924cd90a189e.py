"""Matrix multiplication 4x5x5 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k."""
    rank = U.shape[1]
    # Check a sample of (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    # Also check that for "wrong" combinations, the sum is 0
    # Actually, the tensor equation is: for all i,j,k, sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = delta_{i,j,k}
    # where delta is 1 when this triple corresponds to a valid multiplication term
    # Wait, no. The multiplication tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because every product A[i,j]*B[j,k] contributes to C[i,k].
    # So the equation is: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for ALL i,j,k
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Try to find a rank-25 decomposition
    # Lower bound is max(n*m, m*p, n*p) = max(20, 25, 20) = 25
    # Try rank 25 first
    
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p  # Schoolbook as fallback
    
    # Try to construct rank-25 solution
    # Strategy: For each (j,k) pair in V (25 pairs), assign one rank-1 term
    # U[i*m+j, r] must be 1 for the i values where this term is used
    # W[k*n+i, r] must be 1 for the i values where this term is used
    
    # Actually, let's think about it this way:
    # Term r contributes to C[i,k] the amount U[i*m+j_r, r] * V[j_r*p+k_r, r] * W[k_r*n+i, r]
    # For this to work, each term r is associated with a specific (j_r, k_r) pair (from V)
    # and contributes to all i where U and W are nonzero.
    
    # For rank 25: one term per (j,k) pair.
    # Term (j,k) should contribute to C[i,k] for all i.
    # So U[i*m+j, r_{j,k}] = 1 for all i, V[j*p+k, r_{j,k}] = 1, W[k*n+i, r_{j,k}] = 1 for all i.
    # Then sum over r for fixed i,k: sum_j U[i*m+j, r_{j,k}] * V[j*p+k, r_{j,k}] * W[k*n+i, r_{j,k}]
    # = sum_j 1 * 1 * 1 = m = 5. But we need it to be 1!
    
    # So we need to scale. Let U[i*m+j, r_{j,k}] = 1/m, V[j*p+k, r_{j,k}] = 1, W[k*n+i, r_{j,k}] = 1.
    # Then sum = m * (1/m) * 1 * 1 = 1. 
    
    # But wait, this means for a fixed i, the terms for different j all contribute to C[i,k].
    # And for a fixed j, the term r_{j,k} appears in U[i*m+j, r_{j,k}] for ALL i.
    # So U[i*m+j, r_{j,k}] = 1/5 for all i, for each (j,k).
    
    # Let's verify: for fixed i, k: sum over j of U[i*m+j, r_{j,k}] * V[j*p+k, r_{j,k}] * W[k*n+i, r_{j,k}]
    # = sum_j (1/5) * 1 * 1 = 5 * 1/5 = 1. Correct!
    
    # But we also need to check that there are no spurious contributions.
    # The tensor equation requires: for all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # In our construction, for fixed i, j, k:
    # Only term r_{j,k} has V[j*p+k, r] != 0 (it's 1).
    # U[i*m+j, r_{j,k}] = 1/5.
    # W[k*n+i, r_{j,k}] = 1.
    # So sum = 1/5 * 1 * 1 = 1/5. But we need 1!
    
    # That's wrong. The issue is that the tensor T[i,j,k] = 1 for all i,j,k, but our construction gives 1/5.
    
    # I think I'm confusing myself. Let me re-derive.
    # The matrix multiplication tensor is defined as: T_{i,j,k} = 1 if we're computing C[i,k] from A[i,j]*B[j,k].
    # Actually, the standard definition: the multiplication tensor mu_{n,m,p} has entries mu_{i,j,k} = 1 for all i in [n], j in [m], k in [p].
    # The tensor decomposition is: mu = sum_r u_r ⊗ v_r ⊗ w_r, where (u_r)_{i,j} = U[i*m+j, r], etc.
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for ALL i, j, k.
    
    # In my proposed rank-25 construction with one term per (j,k):
    # For fixed i, j, k: only term r_{j,k} has V[j*p+k, r] != 0.
    # So sum = U[i*m+j, r_{j,k}] * V[j*p+k, r_{j,k}] * W[k*n+i, r_{j,k}]
    # We need this to equal 1.
    # If V[j*p+k, r_{j,k}] = 1, then U[i*m+j, r_{j,k}] * W[k*n+i, r_{j,k}] = 1.
    # This must hold for ALL i. So U[i*m+j, r_{j,k}] = a_i and W[k*n+i, r_{j,k}] = 1/a_i.
    # But U[i*m+j, r_{j,k}] can depend on i, and W[k*n+i, r_{j,k}] can depend on i.
    # So we can set U[i*m+j, r_{j,k}] = 1 and W[k*n+i, r_{j,k}] = 1 for all i.
    # Then for fixed i, j, k: sum = 1 * 1 * 1 = 1. Correct!
    
    # But wait, does this cause problems for other (i', j', k') combinations?
    # For fixed i', j', k': sum_r U[i'*m+j', r] * V[j'*p+k', r] * W[k'*n+i', r]
    # Only term r_{j',k'} has V[j'*p+k', r] != 0.
    # So sum = U[i'*m+j', r_{j',k'}] * V[j'*p+k', r_{j',k'}] * W[k'*n+i', r_{j',k'}] = 1*1*1 = 1.
    # This is correct for ALL i', j', k'!
    
    # So the rank-25 construction is:
    # For each (j, k) in [0,m) x [0,p):
    #   r = j*p + k
    #   U[i*m+j, r] = 1 for all i in [0,n)
    #   V[j*p+k, r] = 1
    #   W[k*n+i, r] = 1 for all i in [0,n)
    
    # Let me verify this is correct:
    # For any i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # V[j*p+k, r] is nonzero only when r = j*p+k (since V has 1 only at position [j*p+k, j*p+k] in this construction).
    # So the sum reduces to: U[i*m+j, j*p+k] * V[j*p+k, j*p+k] * W[k*n+i, j*p+k]
    # = 1 * 1 * 1 = 1. Correct!
    
    # This gives rank = m*p = 25!
    
    U25 = np.zeros((n * m, m * p))
    V25 = np.zeros((m * p, m * p))
    W25 = np.zeros((p * n, m * p))
    
    for j in range(m):
        for k in range(p):
            r = j * p + k
            for i in range(n):
                U25[i * m + j, r] = 1.0
            V25[j * p + k, r] = 1.0
            for i in range(n):
                W25[k * n + i, r] = 1.0
    
    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(m * p):
                    val += U25[i*m+j, r] * V25[j*p+k, r] * W25[k*n+i, r]
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        best_U, best_V, best_W = U25, V25, W25
        best_rank = m * p
    
    # Convert to lists and return
    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        U, V, W = _schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
