import numpy as np
import time
from itertools import combinations

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 118  # safety margin before 150s limit

    # Build the target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify_decomp(U, V, W, atol=1e-8):
        """Check if sum_r U[:,r] V[:,r] W[:,r] == T"""
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(recon, T, atol=atol)

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def get_rank50():
        """Construct a known valid rank-50 decomposition using Strassen blocks."""
        rank = 50
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0

        def add_strassen(i0, j0, k0):
            nonlocal col
            # M1: (a+d)(e+h)
            U[i0 * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M2: (c+d)e
            U[(i0 + 1) * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] -= 1
            col += 1

            # M3: a(f-h)
            U[i0 * m + j0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M4: d(g-e)
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + k0, col] += 1
            V[j0 * p + k0, col] -= 1
            W[k0 * n + i0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            col += 1

            # M5: (a+b)h
            U[i0 * m + j0, col] += 1
            U[i0 * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            col += 1

            # M6: (c-a)(e+f)
            U[(i0 + 1) * m + j0, col] += 1
            U[i0 * m + j0, col] -= 1
            V[j0 * p + k0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M7: (b-d)(g+h)
            U[i0 * m + (j0 + 1), col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] -= 1
            V[(j0 + 1) * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            col += 1

        # Tile 2x4 x 4x7:
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                add_strassen(0, j0, k0)

        # Handle column 6 of B (and C) separately with rank-1 terms
        for j in range(m):
            U[0 * m + j, col] = 1
            V[j * p + 6, col] = 1
            W[6 * n + 0, col] = 1
            col += 1
            U[1 * m + j, col] = 1
            V[j * p + 6, col] = 1
            W[6 * n + 1, col] = 1
            col += 1

        assert col == rank, f"Expected rank 50, got {col}"
        return U, V, W

    # Get the rank-50 decomposition as a starting point
    U50, V50, W50 = get_rank50()

    # Verify rank 50
    if not verify_decomp(U50, V50, W50):
        U, V, W = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_rank = 50
    best_U, best_V, best_W = U50, V50, W50

    # NEW STRUCTURAL APPROACH: Direct combinatorial search for structured decompositions
    # The key insight: matrix multiplication tensor has a very specific algebraic structure.
    # Instead of generic ALS, we'll search for decompositions that respect this structure.
    
    def try_structured_decomposition(max_rank=45):
        """
        Search for structured decompositions by exploring combinations of
        elementary operations that respect the matrix multiplication structure.
        
        Strategy: The 2x4x7 tensor can be viewed as:
        - 2 rows of A, each with 4 elements
        - 4 rows of B, each with 7 elements
        - 7 columns of C, each with 2 elements
        
        We'll try to find minimal decompositions by:
        1. Starting from the rank-50 base
        2. Systematically trying to merge/replace Strassen blocks
        3. Using the fact that entries must be integers or half-integers
        """
        U_base = U50.copy()
        V_base = V50.copy()
        W_base = W50.copy()
        base_rank = 50
        
        # Precompute the target for quick verification
        T_flat = T.flatten()
        
        def quick_verify(U, V, W):
            """Fast verification using matrix multiplication"""
            # For each (i,j,k) in T, check if sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == T[i,j,k]
            # This is equivalent to checking the full tensor, but we can shortcut
            recon = np.einsum('ar,br,cr->abc', U, V, W)
            return np.allclose(recon, T, atol=1e-8)
        
        # Strategy: Try to find lower-rank decompositions by:
        # 1. Removing columns that are linear combinations of others
        # 2. Merging similar Strassen patterns
        
        # First, try to find if any columns in the rank-50 are redundant
        # by checking if removing one column still gives a valid decomposition
        U_curr = U_base.copy()
        V_curr = V_base.copy()
        W_curr = W_base.copy()
        curr_rank = base_rank
        
        # Try greedy column removal
        removed = []
        for attempt in range(curr_rank):
            if time.time() > deadline - 30:
                break
            
            found_removal = False
            for idx in range(curr_rank):
                if idx in removed:
                    continue
                if time.time() > deadline - 30:
                    break
                
                # Try removing column idx
                mask = [i for i in range(curr_rank) if i != idx]
                U_test = U_curr[:, mask]
                V_test = V_curr[:, mask]
                W_test = W_curr[:, mask]
                
                if quick_verify(U_test, V_test, W_test):
                    # Column idx is redundant
                    removed.append(idx)
                    U_curr = U_test
                    V_curr = V_test
                    W_curr = W_test
                    curr_rank -= 1
                    found_removal = True
                    break
            
            if not found_removal:
                break
        
        if curr_rank < best_rank:
            best_rank = curr_rank
            best_U = U_curr.copy()
            best_V = V_curr.copy()
            best_W = W_curr.copy()
        
        # Strategy 2: Try to construct decompositions from scratch using
        # a more systematic approach based on the tensor's structure
        
        # The tensor T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k)
        # This means: C[i,k] = sum_j A[i,j] * B[j,k]
        
        # For a 2x4x7 multiplication, we have:
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0] + A[0,3]*B[3,0]
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1] + A[0,3]*B[3,1]
        # ... etc.
        
        # A rank-r decomposition means we can write this as:
        # C[i,k] = sum_{r=1}^r U[i*m+j_r, r] * V[j_r*p+k_r, r] * W[k_r*n+i_r, r]
        # where each term contributes to specific (i,j,k) positions
        
        # Let's try a more targeted search: for each rank from 40 to best_rank-1,
        # try to find a valid decomposition by solving a system of equations
        
        for target_rank in range(min(42, best_rank - 1), 35, -1):
            if time.time() > deadline - 20:
                break
            
            # For each target rank, try to find U, V, W with half-integer entries
            # such that the tensor equation is satisfied
            
            # This is a hard problem, so we'll use a heuristic approach:
            # Start from the current best and try to "compress" it
            
            U_init = best_U[:, :target_rank].copy() if target_rank <= best_rank else best_U.copy()
            V_init = best_V[:, :target_rank].copy() if target_rank <= best_rank else best_V.copy()
            W_init = best_W[:, :target_rank].copy() if target_rank <= best_rank else best_W.copy()
            
            if U_init.shape[1] < target_rank:
                # Pad with zeros
                pad_size = target_rank - U_init.shape[1]
                U_init = np.hstack([U_init, np.zeros((n*m, pad_size))])
                V_init = np.hstack([V_init, np.zeros((m*p, pad_size))])
                W_init = np.hstack([W_init, np.zeros((p*n, pad_size))])
            
            # Try to optimize using a simple gradient-free approach
            # Since we need exact half-integer solutions, we'll try random perturbations
            # and rounding
            
            rng = np.random.default_rng(42 + target_rank)
            
            for restart in range(10):
                if time.time() > deadline - 15:
                    break
                
                # Perturb the initialization
                U_try = U_init.copy()
                V_try = V_init.copy()
                W_try = W_init.copy()
                
                # Add small random perturbations
                U_try += rng.normal(0, 0.1, U_try.shape)
                V_try += rng.normal(0, 0.1, V_try.shape)
                W_try += rng.normal(0, 0.1, W_try.shape)
                
                # Round to nearest half-integer
                U_rounded = np.round(U_try * 2) / 2
                V_rounded = np.round(V_try * 2) / 2
                W_rounded = np.round(W_try * 2) / 2
                
                if quick_verify(U_rounded, V_rounded, W_rounded):
                    best_rank = target_rank
                    best_U = U_rounded
                    best_V = V_rounded
                    best_W = W_rounded
                    break
            
            if best_rank == target_rank:
                break
        
        return best_rank, best_U, best_V, best_W
    
    # Run the structured search
    if time.time() < deadline - 10:
        new_rank, new_U, new_V, new_W = try_structured_decomposition()
        if new_rank < best_rank:
            best_rank = new_rank
            best_U, best_V, best_W = new_U, new_V, new_W
    
    # Final verification
    if not verify_decomp(best_U, best_V, best_W):
        # Fallback to schoolbook
        U, V, W = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
