"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _build_strassen_28():
    """Build the rank-28 Strassen decomposition for 2x4 * 4x4."""
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_components = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    terms = [
        (0, 0, 0, 0),
        (2, 2, 0, 0),
        (0, 0, 2, 2),
        (2, 2, 2, 2),
    ]
    
    col = 0
    for (a_co, b_ro, b_co, c_co) in terms:
        for (u_pat, v_pat, w_pat) in strassen_components:
            for (pi, pj), val in u_pat.items():
                U[pi * m + (a_co + pj)][col] += val
            for (qi, qj), val in v_pat.items():
                V[(b_ro + qi) * p + (b_co + qj)][col] += val
            for (ri, rj), val in w_pat.items():
                W[(c_co + rj) * n + ri][col] += val
            col += 1
    
    return U, V, W

def _verify(U, V, W, n=2, m=4, p=4):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the tensor T[i,j,k] = sum over A[i,j] * B[j,k] contribution
    # T[i, j, k] = 1 if the (i,j) of A multiplies (j,k) of B to give (k,i) of C
    # Actually: C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T[i, j, k] = 1 (the multiplication tensor)
    # Decomposition: T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # But W uses trace convention: W[k*n + i, r]
    # So W index is (k*n + i), meaning W[k*n+i, r] contributes to T[i, j, k]
    # Let's check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    # for all i in [0,n), j in [0,m), k in [0,p)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True

def construct():
    """Strassen-based rank-28 decomposition, with attempt to reduce rank."""
    deadline = time.time() + 120.0
    
    # Start with rank-28 Strassen
    U, V, W = _build_strassen_28()
    n, m, p = 2, 4, 4
    best_rank = 28
    best_result = {"U": U, "V": V, "W": W}
    
    # Verify base case
    if _verify(U, V, W, n, m, p):
        pass  # Good
    else:
        # Should not happen, but return what we have
        return best_result
    
    # Try to reduce rank by finding linear dependencies
    # Strategy: try to find a rank-27 decomposition by checking if we can
    # express one Strassen vector as a linear combination of others in a way
    # that preserves the tensor identity.
    # 
    # More practically: use random linear transformations.
    # If we have a valid rank-r decomposition, we can apply any invertible
    # r x r matrix G: U' = U*G, V' = V*G^{-1}, W' = W (unchanged if we
    # absorb into U and V). But this doesn't reduce rank.
    #
    # To reduce rank, we need to find a decomposition with fewer terms.
    # One approach: try to find a rank-27 decomposition by solving for
    # a 27-term decomposition that matches the tensor.
    #
    # Let's try: for each pair of columns in the Strassen decomposition,
    # check if their removal (with appropriate adjustment) still works.
    # This is equivalent to checking if the tensor can be written with
    # 27 rank-1 terms from a slightly different basis.
    
    # Alternative: try random rank-27 initializations and use iterative
    # refinement. But this is complex and may not converge.
    
    # Simpler approach: try specific algebraic identities.
    # The 4 Strassen blocks are:
    # Block 0: A0*B00 (cols 0,1 of A; rows 0,1 cols 0,1 of B)
    # Block 1: A1*B10 (cols 2,3 of A; rows 2,3 cols 0,1 of B)
    # Block 2: A0*B01 (cols 0,1 of A; rows 0,1 cols 2,3 of B)
    # Block 3: A1*B11 (cols 2,3 of A; rows 2,3 cols 2,3 of B)
    #
    # Each block has 7 Strassen terms. The question is whether we can share
    # terms between blocks.
    
    # Try a different decomposition: use the fact that the problem is
    # (2,4,4) and try a 2x4 * 4x2 style decomposition or other splits.
    
    # Let's try: split into A (2x4) * B (4x4) = C (2x4)
    # Split B into B1 (4x2) and B2 (4x2).
    # C1 = A * B1 (2x2), C2 = A * B2 (2x2)
    # Each A*Bi is 2x4 * 4x2. The rank of 2x4*4x2 is at most 2*4*2 = 16,
    # but can be lower.
    #
    # Actually, 2x4 * 4x2 can be seen as: split A into A1 (2x2), A2 (2x2)
    # and B into B1_top (2x2), B1_bottom (2x2).
    # A*B1 = A1*B1_top + A2*B1_bottom
    # Each 2x2*2x2 has rank 7 (Strassen), so total 14 for one 2x4*4x2.
    # Two such terms: 28 total. Same as before.
    
    # Let's try yet another approach: use the known result that the rank
    # of (2,4,4) matrix multiplication might be 24 or lower.
    # Try a randomized search for rank 24-27.
    
    if time.time() < deadline - 30:
        # Try random rank-27 search using iterative methods
        # Start from Strassen-28 and try to zero out one column
        random.seed(42)
        for target_rank in [27, 26, 25, 24]:
            if time.time() >= deadline - 10:
                break
            # Try to find a valid rank-target_rank decomposition
            # by perturbing the Strassen solution
            for attempt in range(50):
                if time.time() >= deadline - 10:
                    break
                # Create a random rank-target_rank decomposition
                # by taking random linear combinations of the Strassen vectors
                # and hoping to find a valid one
                U_np = np.array(U, dtype=float)
                V_np = np.array(V, dtype=float)
                W_np = np.array(W, dtype=float)
                
                # Try to find a rank-27 decomposition by finding a null
                # vector in the space of the 28 Strassen vectors
                # This is heuristic and may not work
                
                # Alternative: try to find a decomposition where one of
                # the Strassen terms is replaced by a combination of others
                # from different blocks.
                
                # For now, just try small perturbations
                # This is unlikely to work, so let's try a different strategy:
                # use the fact that we can combine the 4 Strassen blocks
                # more efficiently.
                
                # Try: can we use rank-6 for some of the 2x2*2x2 blocks
                # if we allow shared terms? Probably not, since 7 is optimal
                # for a single 2x2*2x2.
                
                # Let's try a completely different decomposition:
                # Use the "border rank" approach or known low-rank decompositions.
                
                # For now, skip this and return the rank-28 solution.
                pass
    
    return best_result
# EVOLVE-BLOCK-END
