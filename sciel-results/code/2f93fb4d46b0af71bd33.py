"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that U,V,W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build expected tensor Texp
    Texp = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                Texp[a, b, c] = 1.0
    
    # Compute actual tensor T = sum_r U[:,r] V[:,r] W[:,r]
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # This can be computed efficiently by iterating over r
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    
    err = np.max(np.abs(T - Texp))
    return err < tol

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # For 2x4 x 4x7, the tensor is T[i,j,k] = 1 if (i,j,k) consistent, else 0.
    # We analyze the structure to find a lower rank decomposition.
    #
    # Let A be 2x4, B be 4x7, C be 2x7.
    # C = A B.
    #
    # We can write A as [A1 A2] where A1, A2 are 2x2.
    # B as [B1; B2] where B1, B2 are 2x7.
    # C = A1 B1 + A2 B2.
    #
    # The multiplication 2x2 x 2x7 has a rank decomposition of rank 7?
    # No, that was flawed in the seed due to consistency issues.
    #
    # However, consider the tensor T for <2, 4, 7>.
    # It is the sum of two <2, 2, 7> tensors if we split the middle dimension?
    # No, the middle dimension is 4.
    # T = T_left + T_right where T_left corresponds to j in {0,1} and T_right to j in {2,3}.
    # T_left is a tensor on indices (i, j_left, k) with i in {0,1}, j_left in {0,1}, k in {0,6}.
    # This is exactly the <2, 2, 7> multiplication tensor.
    # T_right is similarly <2, 2, 7>.
    #
    # So R(<2, 4, 7>) <= R(<2, 2, 7>) + R(<2, 2, 7>) = 2 * R(<2, 2, 7>).
    #
    # What is R(<2, 2, 7>)?
    # T[i,j,k] = 1 for i in {0,1}, j in {0,1}, k in {0..6}.
    # We derived in the seed that rank 7 works:
    # U[i, r] = 1
    # V[j, r] = 1
    # W[k, r] = delta_{k,r}
    # Let's re-verify this carefully.
    #
    # Check T[i,j,k] = sum_{r=0}^6 U[i,r] V[j,r] W[k,r].
    # = sum_{r=0}^6 1 * 1 * delta_{k,r} = 1 (since k is in 0..6).
    # This works for ALL i,j,k.
    #
    # But wait. In the global context, does this cause errors for inconsistent indices?
    # The global tensor T[a,b,c] is 1 if consistent, 0 if inconsistent.
    # a = i*4 + j_global.
    # If we split j_global into j_left (0,1) and j_right (2,3).
    #
    # Let's use the rank 7 decomposition for the left block (j=0,1).
    # Terms r = 0..6.
    # U_global[i*4 + j, r] = 1 if j in {0,1}.
    # V_global[j*7 + k, r] = 1 if j in {0,1}.
    # W_global[k*2 + i, r] = 1 if k == r.
    #
    # Let's check consistency for this block.
    # Consider a triple (a,b,c) where j_a (from a) != j_b (from b).
    # Example: j_a = 0, j_b = 1.
    # U[a,r] = 1 (since j_a=0 in {0,1}).
    # V[b,r] = 1 (since j_b=1 in {0,1}).
    # W[c,r] = 1 if k_c = r.
    # Sum_r 1 * 1 * delta_{k_c, r} = 1.
    # But T[a,b,c] should be 0 because j_a != j_b.
    # So this decomposition is INVALID for the global tensor.
    #
    # Why? Because the local tensor <2,2,7> assumes that the index j is the SAME for A and B.
    # In the global tensor, the index j is split. The local decomposition doesn't "know" that
    # j_a and j_b must match. It just sees them as independent indices in the local space.
    #
    # To fix this, we need a decomposition of <2,2,7> that is "sparse" enough or structured
    # such that when embedded, the cross-terms vanish.
    #
    # Is there a decomposition of <2,2,p> with rank < 2p that works?
    # The schoolbook rank for <2,2,p> is 4p.
    # The rank 7 decomposition for <2,2,2> (Laderman) works because it uses specific values.
    #
    # Let's look at <2,2,2>. Rank 7.
    # Does the rank 7 Laderman decomposition work when embedded in a larger matrix?
    # If we have <2, 4, 4>, can we use two Laderman blocks?
    # <2,4,4> = <2,2,4> + <2,2,4>.
    # If R(<2,2,4>) is known?
    #
    # Actually, there is a known result: R(<2,2,p>) = 4p - ?
    # For p=2, R=7. (Schoolbook 8).
    # For p=1, R=4? No, T[i,j,0]=1. Rank 4?
    # T[i,j,0] = 1. This is a 2x2 matrix of ones. Rank 4?
    # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r].
    # If p=1, W is 1xR.
    # T[i,j,0] = sum_r U[i,r] V[j,r] w_r.
    # This is a matrix product. The rank of the tensor is the rank of the matrix T[:,:,0].
    # T[:,:,0] is 2x2 all ones. Rank 2.
    # So R(<2,2,1>) = 4? No, the rank of the tensor is at least the matrix rank of any unfolding.
    # Unfolding T as (2x2) x 1. The matrix is 4x1? No.
    # T is 2x2x1.
    # Unfold 1: 2 x (2*1) = 2x2. Rank 2.
    # Unfold 2: (2*1) x 2 = 2x2. Rank 2.
    # Unfold 3: (2*2) x 1 = 4x1. Rank 1.
    # So lower bound is 2.
    # Can we achieve rank 4? Yes, schoolbook.
    # Can we achieve rank 2?
    # T[i,j,0] = U[i,0]V[j,0]W[0,0] + U[i,1]V[j,1]W[0,1].
    # We want sum = 1 for all i,j.
    # This requires the matrix sum U V^T * W to be J_2.
    # If W = [1,1], we need U V^T to have rank 2 and sum to J_2?
    # If U = [[1,0],[0,1]], V = [[1,1],[1,1]], W = [0.5, 0.5].
    # Then T[i,j] = 0.5 (U[i,0]V[j,0] + U[i,1]V[j,1])? No.
    # T[i,j] = sum_r U[i,r] V[j,r] W[0,r].
    # If W = [1, 0], then T[i,j] = U[i,0] V[j,0]. This is rank 1 matrix. Cannot be J_2.
    # So we need at least rank 2 for the "matrix" part, but the tensor rank might be higher.
    #
    # Let's stick to the safe schoolbook for 2x4x4x7 which is 56.
    # But we can try to find a lower rank for <2,4,7> specifically.
    #
    # Note: R(<2,4,7>) <= R(<2,4,4>) + R(<2,4,3>).
    # If we can find good decompositions for these...
    #
    # Actually, let's just use the schoolbook. It is valid and simple.
    # The previous attempts to optimize failed due to consistency.
    # Unless we find a specific known low-rank algorithm for this shape, 56 is the best we can guarantee.
    #
    # Is there a way to use the fact that n=2 is small?
    # For 2 x m x p, the rank is at most 2mp - something?
    #
    # Let's return schoolbook.
    
    result = _schoolbook(n, m, p)
    
    # Verification is optional but good for debugging.
    # Given the 150s limit, and schoolbook is O(1) construction, it's fast.
    
    return result

# EVOLVE-BLOCK-END
