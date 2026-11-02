"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 6
    dim_u = n * m  # 4
    dim_v = m * p  # 12
    dim_w = p * n  # 12
    
    # Non-zero entries of T:
    nonzero = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nonzero.add((a, b, c))
    
    # Cohn-Winograd polynomial embedding approach:
    # Map the 2x6 matrix B to a polynomial using Kronecker substitution.
    # For B[j,k], assign to monomial t^(j*p + k) = t^(6j+k).
    # Then C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # 
    # We want to compute C[i,:] for each i as a linear combination of B[0,:] and B[1,:].
    # 
    # Key idea: Use the fact that for each output position (i,k), we need:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    #
    # We can write this using 20 rank-1 terms with the following structure:
    # Use a "Strassen-like" approach on the 2x2 A matrix, combined with 
    # the 6 independent k positions.
    #
    # Let's use the following 20-term construction:
    # Terms 0-7: Handle the "main" products
    # Terms 8-19: Handle corrections
    
    # Actually, let's use a direct sparse solve approach.
    # Set up the system where each rank-1 term is assigned to specific nonzero entries.
    
    # Strategy: Use 20 terms. Each term r has support on a subset of nonzero entries.
    # For term r, let S_r be the set of (a,b,c) in nonzero that term r contributes to.
    # We need: for each (a,b,c) in nonzero, exactly one term covers it (or multiple terms sum to 1)
    # And for each (a,b,c) NOT in nonzero, all terms give 0.
    
    # A term r with U[a,r] != 0 for a in A_r, V[b,r] != 0 for b in B_r, W[c,r] != 0 for c in C_r
    # will have nonzero product only for (a,b,c) with a in A_r, b in B_r, c in C_r.
    # For the product to be zero for non-nonzero entries in this box, we need careful structure.
    
    # Let's try a structured assignment:
    # Group the 24 nonzero entries into 20 groups (4 groups of 2, 16 groups of 1)
    
    # The 24 nonzero entries are:
    # For k=0: (0,0,0), (1,0,1), (2,6,0), (3,6,1)
    # For k=1: (0,1,2), (1,1,3), (2,7,2), (3,7,3)
    # For k=2: (0,2,4), (1,2,5), (2,8,4), (3,8,5)
    # For k=3: (0,3,6), (1,3,7), (2,9,6), (3,9,7)
    # For k=4: (0,4,8), (1,4,9), (2,10,8), (3,10,9)
    # For k=5: (0,5,10), (1,5,11), (2,11,10), (3,11,11)
    
    # Notice the pattern: for fixed k, the 4 entries share:
    # a in {0,1} or {2,3}, b in {k} or {6+k}, c in {2k} or {2k+1}
    
    # Let's try to pair entries that can share a term:
    # Pair (0,k,2k) with (2,6+k,2k): same c=2k, a differs, b differs
    #   Term: U[0,r]=u0, U[2,r]=u2, V[k,r]=v0, V[6+k,r]=v1, W[2k,r]=w
    #   Need: u0*v0*w = 1, u2*v1*w = 1 => u0*v0 = u2*v1
    #   Zeros: For a in {0,2}, b in {k,6+k}, c=2k: 
    #     (0,6+k,2k) should be 0: u0*v1*w = 0 => need u0*v1=0 or w=0
    #     (2,k,2k) should be 0: u2*v0*w = 0 => need u2*v0=0 or w=0
    #   Since w != 0 (from u0*v0*w=1), we need u0*v1=0 and u2*v0=0
    #   From u0*v0=1/w and u2*v1=1/w, if v0 != 0 then u0 != 0, so v1 must be 0
    #   But if v1=0, then u2*v1=0 != 1/w. Contradiction!
    #   So this pairing doesn't work.
    
    # Try pairing (0,k,2k) with (1,k,2k+1): same b=k, different a, different c
    #   Term: U[0,r]=u0, U[1,r]=u1, V[k,r]=v, W[2k,r]=w0, W[2k+1,r]=w1
    #   Need: u0*v*w0 = 1, u1*v*w1 = 1
    #   Zeros: (0,k,2k+1) and (1,k,2k) should be 0
    #     (0,k,2k+1): u0*v*w1 = 0 => u0*w1 = 0 (since v != 0)
    #     (1,k,2k): u1*v*w0 = 0 => u1*w0 = 0 (since v != 0)
    #   From u0*v*w0=1, u0 != 0, w0 != 0, so u1*w0=0 => u1=0
    #   But u1*v*w1=1 requires u1 != 0. Contradiction!
    
    # Try pairing (0,k,2k) with (3,6+k,2k+1): all different
    #   This is too spread out, will create many zeros to satisfy.
    
    # It seems hard to pair entries. Let's try a different approach:
    # Use 24 terms but with some terms having zero in all three matrices (wasted),
    # or use a different algebraic structure.
    
    # Alternative: Use the fact that this is a "rectangular" matrix multiplication.
    # The rank of 2x2 * 2x6 matrix multiplication tensor.
    # 
    # Lower bound: The border rank or tensor rank.
    # For M_{2,2,6}, the rank is at least max(4, 12, 12) = 12? No, that's not a valid bound.
    # 
    # Actually, the rank of M_{n,m,p} is at least n*m (by fixing b and looking at the 
    # resulting matrix in a,c). Here n*m = 4. Also at least m*p = 12? No.
    # 
    # A better bound: The rank is at least the maximum of the "slice" ranks.
    # 
    # Let me just try to solve for rank 20 using numerical optimization.
    
    import random
    random.seed(42)
    
    def verify_decomp(U, V, W, target_rank):
        for a in range(dim_u):
            for b in range(dim_v):
                for c in range(dim_w):
                    expected = 1 if (a, b, c) in nonzero else 0
                    actual = 0.0
                    for r in range(target_rank):
                        actual += U[a, r] * V[b, r] * W[c, r]
                    if abs(actual - expected) > 1e-8:
                        return False
        return True
    
    best_rank = None
    best_decomp = None
    
    # Try rank 20 with a structured initialization
    for target_rank in [20, 22, 24]:
        # Initialize: assign each nonzero entry to a unique term
        U = np.zeros((dim_u, target_rank))
        V = np.zeros((dim_v, target_rank))
        W = np.zeros((dim_w, target_rank))
        
        nonzero_list = sorted(nonzero)
        
        if target_rank >= len(nonzero_list):
            # One term per nonzero entry
            for col, (a, b, c) in enumerate(nonzero_list[:target_rank]):
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
            
            if verify_decomp(U, V, W, target_rank):
                best_rank = target_rank
                best_decomp = (U, V, W)
                break
        else:
            # Need to share terms. Use random initialization and gradient descent.
            U = np.random.randn(dim_u, target_rank) * 0.1
            V = np.random.randn(dim_v, target_rank) * 0.1
            W = np.random.randn(dim_w, target_rank) * 0.1
            
            # Simple iterative refinement
            for iteration in range(1000):
                # Compute error
                error = 0.0
                for a in range(dim_u):
                    for b in range(dim_v):
                        for c in range(dim_w):
                            expected = 1 if (a, b, c) in nonzero else 0
                            actual = 0.0
                            for r in range(target_rank):
                                actual += U[a, r] * V[b, r] * W[c, r]
                            error += (actual - expected) ** 2
                if error < 1e-10:
                    break
                
                # Update using simple gradient step
                lr = 0.01
                for a in range(dim_u):
                    for b in range(dim_v):
                        for c in range(dim_w):
                            expected = 1 if (a, b, c) in nonzero else 0
                            actual = 0.0
                            for r in range(target_rank):
                                actual += U[a, r] * V[b, r] * W[c, r]
                            err = actual - expected
                            for r in range(target_rank):
                                U[a, r] -= lr * err * V[b, r] * W[c, r]
                                V[b, r] -= lr * err * U[a, r] * W[c, r]
                                W[c, r] -= lr * err * U[a, r] * V[b, r]
            
            if verify_decomp(U, V, W, target_rank):
                best_rank = target_rank
                best_decomp = (U, V, W)
                break
    
    if best_decomp is None:
        # Fallback: schoolbook rank 24
        rank = 24
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        best_decomp = (U, V, W)
    
    U, V, W = best_decomp
    
    # Convert to lists of integers/half-integers
    def to_list(mat):
        result = []
        for row in mat:
            row_list = []
            for val in row:
                if abs(val - round(val)) < 1e-10:
                    row_list.append(int(round(val)))
                elif abs(val * 2 - round(val * 2)) < 1e-10:
                    row_list.append(round(val * 2) / 2)
                else:
                    row_list.append(float(val))
            result.append(row_list)
        return result
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
