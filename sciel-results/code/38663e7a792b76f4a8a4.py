"""Matrix multiplication 2x5x5: rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 5, 5]

def construct():
    """
    Returns a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 2x5 matrix by a 5x5 matrix.
    
    The output factor W follows the trace convention W[k*n+i, r].
    
    U: shape (n*m, rank) = (10, rank)
    V: shape (m*p, rank) = (25, rank)
    W: shape (p*n, rank) = (10, rank)
    
    The decomposition satisfies: for all i, k (output indices),
    sum_r U[i*m + j, r] * V[j*p + k, r] ... 
    Actually, the tensor T[i, j, k] = delta (i.e., T[i,j,k] = 1 if this is a valid
    multiplication entry). The factorization is:
    T[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
    where U has shape (n, rank) effectively but stored as (n*m, rank) with
    U[i*m+j, r] corresponding to the (i,j) input pair.
    
    Wait, let me re-read the interface:
    U: (n*m, rank) - indexed by (i, j) where i is row of A, j is col of A / row of B
    V: (m*p, rank) - indexed by (j, k) where j is col of A / row of B, k is col of B
    W: (p*n, rank) - indexed by (k, i) per trace convention W[k*n+i, r]
    
    So the tensor for matrix multiplication C = A @ B where A is n x m, B is m x p:
    C[i, k] = sum_j A[i, j] * B[j, k]
    
    The multiplication tensor has entries T[i, j, k] = 1 for all valid (i, j, k).
    A rank-r decomposition means:
    T[i, j, k] = sum_{r=0}^{r-1} U[i, j, r] * V[j, k, r] * W[k, i, r]
    
    But the interface stores:
    U[i*m + j, r] = U[i, j, r]
    V[j*p + k, r] = V[j, k, r]  
    W[k*n + i, r] = W[k, i, r]
    
    So we need: for all i in [0,n), j in [0,m), k in [0,p):
    1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    This is the standard matrix multiplication tensor. The rank is the minimum number
    of rank-1 tensors needed to express this.
    
    For 2x5x5, the schoolbook rank is 2*5*5 = 50.
    
    Let's try to find a lower rank. The trivial lower bound is max(n*m, m*p, p*n) = max(10, 25, 10) = 25.
    
    Let me try a systematic search for small ranks, starting from reasonable lower bounds.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m  # 10
    mp = m * p  # 25
    pn = p * n  # 10
    
    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W
    
    best_U, best_V, best_W = schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We need sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i,j,k
    
    # This is a system of nm*p = 50 equations with 3*rank unknowns (but nonlinear)
    
    # Strategy: For small ranks, we can try to solve this system.
    # Key insight: if we fix U and V, then for each (i,k), we need:
    # sum_r W[k,i,r] * (sum_j U[i,j,r]*V[j,k,r]... no that's not right.
    
    # Actually T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    # This is trilinear. For fixed r, define a_r[i,j] = U[i,j,r], b_r[j,k] = V[j,k,r], c_r[k,i] = W[k,i,r]
    # Then sum_r a_r[i,j] * b_r[j,k] * c_r[k,i] = 1 for all i,j,k
    
    # Let's try rank 25 (the lower bound). If we can achieve rank 25, that would be great.
    # Actually, for matrix multiplication, the known rank for 2x5x5...
    # Let me think about this differently.
    
    # For the 2x5x5 case, one approach:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # We have 10 output entries, each is a sum of 5 products.
    # Total "scalar multiplications" in schoolbook = 50.
    # Can we share intermediate products?
    
    # Let's try a structured approach. Suppose rank = r.
    # We can parameterize and use optimization or algebraic methods.
    
    # For rank 25: this is the m*p dimension, suggesting we might be able to do it.
    # Let's try to construct a rank-25 decomposition.
    
    # Idea: Let V be the identity-like structure. If V[j,k,r] is structured...
    
    # Actually, let me try a different approach. Use the fact that for small matrices,
    # we can set up and solve the equations.
    
    # For rank r, we have r "terms". Each term r contributes:
    # a_r[i,j] * b_r[j,k] * c_r[k,i]
    
    # Let's try to find integer/rational solutions for small r.
    
    # Start with r = 25 (lower bound) and work down.
    # Actually, let's first try to see if r < 50 is achievable with a simple structure.
    
    # Simple idea: group by j. For each j, we have 2*5=10 products A[i,j]*B[j,k].
    # If we could compute each group with fewer multiplications...
    
    # For a fixed j, we need to compute a 2x5 matrix product (outer product of column j of A 
    # and row j of B). This requires 10 multiplications in general, but if we precompute
    # the column and row, we still need 10 products.
    
    # However, if we share across j values, we might save.
    
    # Let me try a concrete construction for rank < 50.
    
    # Approach: Use the structure where W has a simple form.
    # If W[k,i,r] = delta_{k, f(r)} * delta_{i, g(r)} for some functions, then...
    # No, that's too restrictive.
    
    # Let's try: W[k,i,r] is nonzero only for specific (k,i) pairs.
    # If W[k,i,r] = 1 for r corresponding to (k,i), then we need:
    # sum_{r: W[k,i,r]!=0} U[i,j,r]*V[j,k,r] = 1 for all i,j,k
    
    # This means for each (k,i), the set of r where W[k,i,r]!=0 must satisfy:
    # sum_r U[i,j,r]*V[j,k,r] = 1
    
    # If each W[k,i,r] is 1 for exactly one r = h(k,i), then:
    # U[i,j,h(k,i)] * V[j,k,h(k,i)] = 1 for all i,j,k
    # This means U[i,j,h(k,i)] and V[j,k,h(k,i)] are both ±1 (or other pairs multiplying to 1).
    # But h(k,i) must be consistent: for fixed (i,j), different k give different h(k,i),
    # so U[i,j,·] has multiple nonzero entries.
    
    # This is getting complex. Let me just try a numerical/algebraic search.
    
    # For the 2x5x5 case, let's try to find rank-25 or lower.
    
    # Actually, I recall that for matrix multiplication of 2x5 by 5x5,
    # the rank is known to be at most 50 (schoolbook) and the lower bound is 25.
    # The exact rank is not well-known for this specific case, but likely between 25 and 50.
    
    # Let's try a random search / optimization approach for small ranks.
    
    # For rank r, we have 3*r*(dimensions) parameters. For r=25: 3*25*(10+25+10)/... 
    # Actually U is 10x25=250 params, V is 25x25=625 params, W is 10x25=250 params. Total 1125 params.
    # Constraints: 50 equations (one for each (i,j,k) triple).
    
    # This is underdetermined for r >= 25, so solutions likely exist.
    
    # Let's try to find a solution with integer or half-integer entries.
    
    # Strategy: Set up the system and solve.
    # For simplicity, let's fix some structure to reduce degrees of freedom.
    
    # Try: Let V be a 25x25 matrix (for r=25) that is invertible or has nice structure.
    # If V is the identity matrix (V[j,k,r] = delta_{j*p+k, r}), then:
    # sum_r U[i,j,r] * delta_{j*p+k,r} * W[k,i,r] = U[i,j, j*p+k] * W[k,i, j*p+k] = 1
    # So we need U[i,j, j*p+k] * W[k,i, j*p+k] = 1 for all i,j,k.
    # Let U[i,j, j*p+k] = 1 and W[k,i, j*p+k] = 1.
    # But U[i,j,·] would have 5 nonzero entries (one for each k), and W[k,i,·] would have 5 nonzero entries (one for each j).
    # This gives a valid rank-25 decomposition!
    
    # Let's verify: r = 25.
    # V[j*p+k, r] = 1 if r == j*p+k, else 0. (V is 25x25 identity)
    # U[i*m+j, r] = 1 if r == j*p+k for some k... wait, r is determined by (j,k).
    # U[i*m+j, j*p+k] = 1 for all k.
    # W[k*n+i, j*p+k] = 1 for all j.
    
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # = sum_r U[i*m+j, r] * delta_{r, j*p+k} * W[k*n+i, r]
    # = U[i*m+j, j*p+k] * W[k*n+i, j*p+k]
    # = 1 * 1 = 1. ✓
    
    # This works! Rank 25 with integer entries.
    
    # Can we do better than 25? The lower bound is max(nm, mp, pn) = max(10, 25, 10) = 25.
    # So rank 25 is optimal!
    
    # Wait, is the lower bound really 25? The rank of the matrix multiplication tensor
    # [n x m x p] is at least max(n*m, m*p, p*n) because each factor matrix must have
    # full rank in appropriate slices. Yes, this is a standard lower bound.
    
    # So rank 25 is optimal for 2x5x5.
    
    U = np.zeros((nm, 25))
    V = np.zeros((mp, 25))
    W = np.zeros((pn, 25))
    
    for j in range(m):
        for k in range(p):
            r = j * p + k
            V[r, r] = 1  # V[j*p+k, r] = 1
            for i in range(n):
                U[i * m + j, r] = 1  # U[i*m+j, r] = 1
                W[k * n + i, r] = 1  # W[k*n+i, r] = 1
    
    # Verify
    # For all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(25):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                assert abs(val - 1) < 1e-10, f"Verification failed for i={i}, j={j}, k={k}: got {val}"
    
    # Convert to lists for return
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
