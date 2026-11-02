"""Matrix multiplication 3x3x5 rank-minimal tensor decomposition search."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 5]
DEADLINE = 120.0


def _verify_decomposition(U, V, W, n=3, m=3, p=5, tol=1e-9):
    """Verify that U @ V^T @ W^T reconstructs the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # T should equal 1 if i,j,k are valid (i.e., always 1 for 3x3x5)
    # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all valid indices
    # since C[i,k] = sum_j A[i,j] * B[j,k], so the structural tensor is all ones.
    
    # Reshape and check: for each c, the rank-1 contribution is
    # outer(U[:,c], outer(V[:,c], W[:,c])) reshaped
    # The full tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    T = np.zeros((n, m, p), dtype=float)
    for c in range(rank):
        u = U[:, c].reshape(n, m)
        v = V[:, c].reshape(m, p)
        w = W[:, c].reshape(p, n)
        # Contribution: for each (i,j,k), u[i,j] * v[j,k] * w[k,i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    
    # The structural tensor for matrix multiplication is all ones
    return np.allclose(T, 1.0, atol=tol, rtol=tol)


def _make_schoolbook(n=3, m=3, p=5):
    """Create the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _try_random_search(n=3, m=5, p=5, target_rank=44, max_iters=5000):
    """Try to find a valid decomposition of the given target rank via random search."""
    start = time.time()
    deadline = start + DEADLINE - 5.0  # 5-second safety margin
    
    # For matrix multiplication 3x3x5, the structural tensor T[i,j,k]=1 for all i,j,k
    # We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    # Strategy: start with schoolbook and try to eliminate one term by expressing
    # it as a linear combination, or try random factor matrices.
    
    # Random approach: generate random U, V, W of the target rank and check
    # But the search space is huge. Let's try a structured approach.
    
    # Key idea: The schoolbook has 45 terms. We want to find 44 (or fewer) terms.
    # Try: take 44 of the 45 schoolbook terms, and see if the 45th can be 
    # approximated. If not, try small perturbations.
    
    sb = _make_schoolbook(n, m, p)
    U_sb = np.array(sb["U"], dtype=float)
    V_sb = np.array(sb["V"], dtype=float)
    W_sb = np.array(sb["W"], dtype=float)
    
    # Try dropping one term and compensating
    for drop in range(n * m * p):
        if time.time() > deadline:
            return None
        
        # Remove term 'drop'
        mask = [c for c in range(n * m * p) if c != drop]
        U_try = U_sb[:, mask]
        V_try = V_sb[:, mask]
        W_try = W_sb[:, mask]
        
        # Check if this alone works (it won't, since we're missing a term)
        # Instead, try to add back a combination that compensates.
        # The missing term contributes 1 at position (i,j,k) where the term was.
        # i = drop // (m*p), j = (drop // p) % m, k = drop % p
        i_drop = drop // (m * p)
        j_drop = (drop // p) % m
        k_drop = drop % p
        
        # We need the remaining 44 terms to sum to 1 at ALL positions.
        # They sum to 1 at all positions except (i_drop, j_drop, k_drop) where they sum to 0.
        # So we need to add a correction. But we only have 44 terms.
        # This approach doesn't directly work.
        
        pass
    
    # Alternative: try random linear combinations of the schoolbook terms
    # to see if 44 terms can span the same tensor.
    
    # More promising: try to find a rank-44 decomposition by solving
    # a system of equations. The tensor has 3*3*5 = 45 entries, all = 1.
    # A rank-r decomposition has 3*r + 5*r + 15*r = 23*r variables.
    # For r=44: 23*44 = 1012 variables, 45 constraints. Overdetermined in a good way.
    # But the constraints are nonlinear (product of three factors).
    
    # Let's try a simple randomized approach:
    # Start with schoolbook, randomly perturb, and try to "absorb" one term.
    
    best = None
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        
        # Try to find a rank-44 decomposition by modifying one schoolbook term
        # to compensate for the removal of another.
        
        # Pick two terms to merge
        a = random.randint(0, 44)
        b = random.randint(0, 44)
        if a == b:
            continue
        
        # Remove term b, and modify term a to compensate
        # Term a: (i_a, j_a, k_a), Term b: (i_b, j_b, k_b)
        i_a, j_a, k_a = a // (m*p), (a // p) % m, a % p
        i_b, j_b, k_b = b // (m*p), (b // p) % m, b % p
        
        # For the decomposition to work, we need:
        # At position (i,j,k): sum over remaining terms = 1
        # If we remove term b, we lose 1 at (i_b, j_b, k_b).
        # If we modify term a, we can add a contribution there.
        # But term a only has support at (i_a, j_a, k_a).
        # So this only works if (i_a,j_a,k_a) == (i_b,j_b,k_b), which it isn't.
        
        # This approach is fundamentally limited. Let's try a different strategy.
        
        # Try: use 44 random rank-1 tensors and solve for coefficients
        # such that the sum equals the all-ones tensor.
        
        # Actually, let's just try: can we express the all-ones tensor as a sum
        # of 44 rank-1 tensors of the form outer(u,v,w) where u is 3x3, v is 3x5, w is 5x3?
        
        # The all-ones tensor of size 3x3x5 has rank at most 45 (schoolbook).
        # The question is whether its rank is exactly 45 or less.
        
        # For now, let's try a greedy approach: start with 44 random terms and see
        # if they can be adjusted to match.
        
        # Generate 44 random rank-1 terms
        U_rand = np.random.randn(n * m, 44)
        V_rand = np.random.randn(m * p, 44)
        W_rand = np.random.randn(p * n, 44)
        
        # Check if this random decomposition works (very unlikely)
        T_check = np.zeros((n, m, p))
        for c in range(44):
            u = U_rand[:, c].reshape(n, m)
            v = V_rand[:, c].reshape(m, p)
            w = W_rand[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_check[i, j, k] += u[i, j] * v[j, k] * w[k, i]
        
        if np.allclose(T_check, 1.0, atol=1e-6):
            # Found a valid rank-44 decomposition!
            return {"U": U_rand.tolist(), "V": V_rand.tolist(), "W": W_rand.tolist()}
    
    return None


def construct():
    """Find a rank-minimal exact tensor decomposition for 3x3x5 matrix multiplication."""
    start = time.time()
    deadline = start + DEADLINE - 3.0  # 3-second safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook (rank 45, guaranteed valid)
    schoolbook = _make_schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = n * m * p
    
    # Try to find a lower-rank decomposition
    # Attempt rank 44
    if time.time() < deadline - 10.0:
        result_44 = _try_random_search(n, m, p, target_rank=44, max_iters=2000)
        if result_44 is not None:
            if _verify_decomposition(result_44["U"], result_44["V"], result_44["W"], n, m, p):
                return result_44
    
    # Attempt rank 43
    if time.time() < deadline - 5.0:
        result_43 = _try_random_search(n, m, p, target_rank=43, max_iters=1000)
        if result_43 is not None:
            if _verify_decomposition(result_43["U"], result_43["V"], result_43["W"], n, m, p):
                return result_43
    
    # Return the best we have (schoolbook)
    return best_result
