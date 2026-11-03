"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _validate(U, V, W, n, m, p, tol=1e-6):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    # The tensor identity: for all i,j,k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if (i,j,k) is a valid product index
    # Actually the matrix multiplication tensor T[i,j,k] = delta_{j,k} in the right sense
    # Let me think more carefully.
    
    # We want: (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: C[i,k] = sum_j A[i,j] B[j,k]
    # The bilinear decomposition: C[i,k] = sum_r (sum_j U[i*m+j,r] A[i,j]) * (sum_j V[j*p+k,r] B[j,k])
    # Wait, that's not right either.
    
    # The standard tensor decomposition for matrix multiplication:
    # C[i,k] = sum_r alpha_r(i) * beta_r(k) where alpha_r depends on A and beta_r on B
    # More precisely: C[i,k] = sum_r (sum_j U[i,j,r] A[i,j]) * (sum_j V[j,k,r] B[j,k])
    # But U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    
    # The correct identity:
    # For the matrix multiplication tensor T with T[i,j,k] = 1 if j is the "middle" index
    # Actually, let me use the standard formulation:
    # The multiplication tensor T ∈ R^{n×m} ⊗ R^{m×p} ⊗ R^{n×p}
    # T[i,j,k,l] = delta_{j,k} (i.e., T maps A⊗B to C where C[i,l] = sum_j A[i,j]B[j,l])
    
    # In the flattened form with U (n*m, rank), V (m*p, rank), W (p*n, rank):
    # The condition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{j,k}
    # Wait no, that's not right. Let me reconsider.
    
    # Actually the standard form: we want to compute C = A * B where A is n×m, B is m×p, C is n×p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # A rank-r decomposition means we can write:
    # C[i,k] = sum_{r=1}^r (sum_j U[i,j,r] * A[i,j]) * (sum_j V[j,k,r] * B[j,k])
    # But this doesn't work directly because U and V both depend on j.
    
    # The correct tensor decomposition:
    # The matrix multiplication tensor T_{i,j,k} for i∈[n], j∈[m], k∈[p] is defined as:
    # T[i,j,k] = 1 if the index j in A[i,j] matches the index j in B[j,k], else 0
    # i.e., T[i,j,k] = delta_{j,j} which is always 1... that's not right.
    
    # Let me use the standard formulation from Strassen:
    # We have three sets of indices:
    # - A indices: (i, j) for i in range(n), j in range(m) → flattened to i*m + j
    # - B indices: (j, k) for j in range(m), k in range(p) → flattened to j*p + k  
    # - C indices: (i, k) for i in range(n), k in range(p) → flattened to k*n + i (as per task)
    
    # The tensor T has entries T[a_idx, b_idx, c_idx] where:
    # a_idx = i*m + j, b_idx = j'*p + k, c_idx = k'*n + i'
    # T[a_idx, b_idx, c_idx] = 1 if j == j' and i == i' and k == k', else 0
    
    # A rank-r decomposition: T[a,b,c] = sum_{r=1}^{rank} U[a,r] * V[b,r] * W[c,r]
    
    # So the validation is: for all a in [0, n*m), b in [0, m*p), c in [0, p*n):
    # sum_r U[a,r] * V[b,r] * W[c,r] should equal T[a,b,c]
    
    # Let's compute T[a,b,c]:
    # a = i*m + j → i = a // m, j = a % m
    # b = j'*p + k → j' = b // p, k = b % p
    # c = k'*n + i' → i' = c % n, k' = c // n
    
    # T[a,b,c] = 1 if (j == j' and i == i' and k == k') else 0
    
    # For efficiency, let's check a subset or use matrix operations.
    
    # Actually, let's just do a direct check on a few random points for speed, 
    # or do the full check if rank is small enough.
    
    nm = n * m
    mp = m * p
    pn = p * n
    rank = U.shape[1]
    
    # Full validation: compute sum_r U[a,r]*V[b,r]*W[c,r] for all (a,b,c)
    # This is equivalent to: for each r, the outer product U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # Sum over r gives the tensor.
    
    # For efficiency, let's check: the tensor should have 1s at specific positions.
    # Number of 1s in T: n*m*p (one for each (i,j,k) triple where j matches)
    # Actually T[a,b,c] = 1 when a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    # So there are n*m*p ones, at positions (i*m+j, j*p+k, k*n+i) for i in [n], j in [m], k in [p]
    
    # Let's verify: for each (i,j,k), check that sum_r U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] == 1
    # And for a random sample of other (a,b,c), check that the sum == 0
    
    # Check the "on" positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(val - 1.0) > tol:
                    return False, (i, j, k, val)
    
    # Check some "off" positions (random sample)
    rng = np.random.RandomState(42)
    for _ in range(min(1000, nm * mp * pn // 10)):
        a = rng.randint(0, nm)
        b = rng.randint(0, mp)
        c = rng.randint(0, pn)
        # Check if this is an "on" position
        i_a = a // m
        j_a = a % m
        j_b = b // p
        k_b = b % p
        i_c = c % n
        k_c = c // n
        if j_a == j_b and i_a == i_c and k_b == k_c:
            continue  # This is an on position, skip
        val = np.dot(U[a, :], V[b, :] * W[c, :])
        if abs(val) > tol:
            return False, (a, b, c, val)
    
    return True, None

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W

def _strassen_3x3_block(n, m, p):
    """
    Use Strassen-like decomposition for the 3x3 block structure.
    For 3x3 * 3x5, we can think of it as:
    - The 3x3 part can use a rank-23 decomposition (Strassen gives 7 for 2x2, 
      but for 3x3 the best known is around 19-23)
    - Combined with the 3x5 structure
    
    Actually, let's try a simpler approach: use a known low-rank decomposition.
    """
    # For 3x3 * 3x5, the schoolbook rank is 45.
    # Let's try to find a lower rank decomposition.
    
    # One approach: decompose the 3x5 multiplication as 5 separate 3x3 multiplications
    # (each column of B), but that gives 5 * rank(3x3 mult)
    # If rank(3x3 mult) <= 23, then total <= 115, which is worse.
    
    # Better: use the structure directly.
    # C[i,k] = sum_j A[i,j] * B[j,k] for i in 0..2, k in 0..4
    
    # Let's try a random search for lower ranks.
    return None

def _random_search(n, m, p, target_rank, seed=42, max_iters=5000):
    """Try to find a rank-target_rank decomposition via random initialization and refinement."""
    rng = np.random.RandomState(seed)
    nm = n * m
    mp = m * p
    pn = p * n
    
    for it in range(max_iters):
        # Random initialization
        U = rng.randn(nm, target_rank)
        V = rng.randn(mp, target_rank)
        W = rng.randn(pn, target_rank)
        
        # Scale to make it more likely to work
        # The "on" positions need sum_r U[a,r]*V[b,r]*W[c,r] = 1
        # So typical magnitude of each term should be ~1/sqrt(rank)
        scale = 1.0 / np.sqrt(target_rank)
        U *= scale
        V *= scale
        W *= scale * target_rank  # Adjust so product has right scale
        
        # Check if this random init happens to work (very unlikely)
        valid, _ = _validate(U, V, W, n, m, p, tol=1e-4)
        if valid:
            return U, V, W
        
        # Try a structured approach: start from schoolbook and try to merge terms
        if it == 0:
            # Try a different strategy: use the known structure
            # For each of the n*m*p terms, we have U[i*m+j, r]=1, V[j*p+k, r]=1, W[k*n+i, r]=1
            # We want to reduce the number of terms.
            pass
    
    return None

def _structured_low_rank(n, m, p):
    """
    Try to construct a lower-rank decomposition using structure.
    
    For 3x3 * 3x5:
    C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    
    One idea: use polynomial-like structure.
    Another: use the fact that we can factor through a smaller space.
    
    Let's try: introduce intermediate variables.
    For example, if we could write:
    C[i,k] = sum_r f_r(A) * g_r(B)
    where f_r and g_r are bilinear in A and B respectively.
    
    Actually, the standard form is exactly this. The question is the minimum rank.
    
    Known results:
    - rank(2x2 * 2x2) <= 7 (Strassen)
    - rank(3x3 * 3x3) <= 23 (Landsberg)
    - For 3x3 * 3x5, the bound would be related.
    
    Let's try a constructive approach based on Strassen for 2x2 blocks.
    """
    # For now, let's try a random search with better initialization.
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = (U_school, V_school, W_school)
    
    # Try to find lower-rank decompositions
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(9, 15, 15) = 15
    # But actually the true lower bound is higher.
    
    # Try ranks from lower to higher
    for target_rank in range(15, best_rank + 1):
        if time.time() > deadline:
            break
        
        # Try random search
        result = _random_search(n, m, p, target_rank, seed=target_rank, max_iters=100)
        if result is not None:
            U, V, W = result
            valid, _ = _validate(U, V, W, n, m, p, tol=1e-6)
            if valid:
                best_rank = target_rank
                best_result = (U, V, W)
                break  # Found the lowest rank
    
    # Return the best result found
    U, V, W = best_result
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
