"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix by a 5x6 matrix.
    Uses the schoolbook decomposition (rank 90) as the baseline.
    Attempts limited structured search for lower ranks if time permits.
    """
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    schoolbook_rank = n * m * p
    
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill

    # Build the multiplication tensor T[i, j, k] = 1
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Condition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1

    def schoolbook():
        rank = schoolbook_rank
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
        return {"U": U, "V": V, "W": W}
    
    best = schoolbook()

    def verify_fast(U, V, W, rank):
        """
        Verify the decomposition using NumPy for speed.
        Returns True if the decomposition is exact.
        """
        # Convert to numpy arrays
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        
        # We need to check if sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
        # This is equivalent to checking the tensor product.
        # Let's construct the target tensor T of shape (nm, mp, pn)
        T = np.zeros((nm, mp, pn), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        
        # Compute the decomposed tensor
        # Res[I, J, K] = sum_c U[I, c] V[J, c] W[K, c]
        # This can be computed as:
        # Res = np.einsum('ic,jc,kc->ijk', U_np, V_np, W_np)
        # However, einsum might be slow for large ranks, but rank 90 is small.
        # Let's try einsum.
        try:
            Res = np.einsum('ic,jc,kc->ijk', U_np, V_np, W_np)
        except Exception:
            # Fallback to manual loop if einsum fails (unlikely)
            Res = np.zeros((nm, mp, pn), dtype=np.float64)
            for c in range(rank):
                Res += U_np[:, c][:, None, None] * V_np[:, c][None, :, None] * W_np[:, c][None, None, :]
        
        # Check if Res matches T
        # Since entries are integers/half-integers, we can use a tight tolerance
        return np.allclose(Res, T, atol=1e-9)

    # Verify the schoolbook solution
    if not verify_fast(best["U"], best["V"], best["W"], schoolbook_rank):
        # This should never happen, but if it does, we have a bug.
        # We still return it as it's the best known.
        pass

    # Attempt to find a lower rank
    # Strategy: Try rank 89, 88 with a few random restarts.
    # Given the discrete nature and high dimension, this is unlikely to succeed quickly.
    # We will limit the search to avoid timeout.
    
    def try_random_decomposition(rank, max_iters=200):
        """Try to find a valid decomposition of given rank using random search."""
        # Use a restricted set of values to increase chance of exact integer/half-integer match
        # Values: -1, -0.5, 0, 0.5, 1
        choices = [-1, -0.5, 0, 0.5, 1]
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Initialize with small random values
            U = [[random.choice(choices) for _ in range(rank)] for _ in range(nm)]
            V = [[random.choice(choices) for _ in range(rank)] for _ in range(mp)]
            W = [[random.choice(choices) for _ in range(rank)] for _ in range(pn)]
            
            # Quick check: sum of absolute values should be at least 1 for each (i,j,k)
            # This is a necessary condition but not sufficient.
            
            if verify_fast(U, V, W, rank):
                return {"U": U, "V": V, "W": W}
        return None

    # Try rank 89 and 88 with very limited iterations
    # If we find one, we update best.
    for r in [schoolbook_rank - 1, schoolbook_rank - 2]:
        if time.time() > deadline - 5:
            break
        result = try_random_decomposition(r, max_iters=100)
        if result is not None:
            best = result
            break # Found a lower rank, stop searching for lower ones immediately? 
            # Actually, we could try to go lower, but rank 88 is very unlikely if 89 failed.
            # We'll just return the first improvement found.

    # Final verification
    if not verify_fast(best["U"], best["V"], best["W"], len(best["U"][0])):
        # Fallback to schoolbook if something went wrong
        best = schoolbook()

    return best
# EVOLVE-BLOCK-END
