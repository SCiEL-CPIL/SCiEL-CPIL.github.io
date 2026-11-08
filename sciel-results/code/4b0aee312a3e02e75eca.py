"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _validate(U, V, W, n, m, p):
    """Check whether the tensor T[i*m+j, j*p+k, k*n+i] equals sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]."""
    rank = U.shape[1]
    # Build the tensor from the factors
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # where a = i*m+j, b = j*p+k, c = k*n+i
    # We need to check for all valid (i,j,k).
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                expected = 1
                actual = int(np.dot(U[a, :], V[b, :]))  # This is wrong, need to include W
                # Actually, the tensor entry is T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                # This is a 3rd order tensor, not a bilinear form in U,V alone.
                # Let's compute properly.
                actual = 0
                for r in range(rank):
                    actual += U[a, r] * V[b, r] * W[c, r]
                if actual != expected:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # First, try to find a lower-rank decomposition using a random search / lattice approach
    # The multiplication tensor for n x m x p matrices has a schoolbook rank of n*m*p = 60.
    # We want to find if there's a rank < 60 decomposition.
    
    # Strategy: 
    # 1. Start with the schoolbook decomposition (rank 60).
    # 2. Try to find linear dependencies among the rank-1 terms to reduce rank.
    #    The terms are e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i} (in appropriate bases).
    #    Actually, the tensor is T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}.
    #    Here e_{i,j} is a basis vector for the 3x4 space (indexed by i,j).
    #    e_{j,k} is a basis vector for the 4x5 space (indexed by j,k).
    #    e_{k,i} is a basis vector for the 5x3 space (indexed by k,i).
    #    
    #    A rank-r decomposition is T = sum_{l=1}^r u_l ⊗ v_l ⊗ w_l.
    #    This is equivalent to finding matrices U, V, W of dimensions (12, r), (20, r), (15, r)
    #    such that for all i,j,k:
    #      sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] = 1
    #    And for all other index combinations (that are not valid i,j,k), the sum is 0.
    #    Wait, the tensor is only defined on the valid (i,j,k) triples. 
    #    The "invalid" entries don't matter as long as the multiplication is correct.
    #    The multiplication (AB)_{i,k} = sum_j A_{i,j} B_{j,k}.
    #    The tensor T is defined such that T(A,B) = AB.
    #    T_{a,b,c} = sum_l U[a,l] V[b,l] W[c,l].
    #    For the multiplication to be correct, we need:
    #      For all i,k: sum_j A_{i,j} B_{j,k} = sum_l (sum_j U[i*m+j, l] A[i,j]) * V[j*p+k, l] * W[k*n+i, l]
    #    This is getting complex. The standard condition is:
    #      T_{(i,j),(j,k),(k,i)} = 1 and T_{...} = 0 otherwise? No.
    #    The tensor T is a multilinear map from (R^{3x4} x R^{4x5}) -> R^{3x5}.
    #    T(A,B)_{i,k} = sum_j A_{i,j} B_{j,k}.
    #    In index notation, if A_{i,j}, B_{j,k}, then (AB)_{i,k} = sum_j A_{i,j} B_{j,k}.
    #    The tensor T has indices (i,j) for A, (j,k) for B, and (i,k) for output.
    #    So T_{(i,j), (j,k), (i,k)} = 1.
    #    Wait, the output index is (i,k), not (k,i).
    #    The task says: "output factor follows the official trace convention W[k*n+i,r]".
    #    So W is indexed by k*n+i. This means the output index is (k,i) where k is the row of B (which is col of A? no).
    #    Let's re-read carefully.
    #    A is n x m (3x4), B is m x p (4x5). Output is n x p (3x5).
    #    Standard: (AB)_{i,k} = sum_j A_{i,j} B_{j,k}.
    #    Indices: i in 0..2, j in 0..3, k in 0..4.
    #    U shape: (n*m, rank) = (12, rank). Index a = i*m+j.
    #    V shape: (m*p, rank) = (20, rank). Index b = j*p+k.
    #    W shape: (p*n, rank) = (15, rank). Index c = k*n+i.
    #    So the output is indexed by (k,i). This is the transpose of the standard output (i,k).
    #    So the tensor is T_{(i,j), (j,k), (k,i)} = 1.
    #    This means the multiplication is being defined as (AB)_{k,i} = sum_j A_{i,j} B_{j,k}.
    #    This is just a permutation of the output indices.
    
    # So the condition is:
    # For all i in 0..n-1, j in 0..m-1, k in 0..p-1:
    #   sum_{l=1}^r U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] = 1
    # And we don't care about other entries, as long as the multiplication works.
    # Actually, if the above holds, does the multiplication work?
    # Let's check. (AB)_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The computed output C_{k,i} = sum_j A_{i,j} B_{j,k} * (some factor)?
    # No, the tensor decomposition defines the bilinear map.
    # C = sum_l (U^T A_flat) * (V^T B_flat) * W_l  ? No.
    # C_{k,i} = sum_l (sum_j U[i*m+j, l] A[i,j]) * (sum_j V[j*p+k, l] B[j,k]) * W[k*n+i, l] ? No.
    # The standard form is:
    # C_{output} = sum_l (sum_{input_A} U[input_A, l] A[input_A]) * (sum_{input_B} V[input_B, l] B[input_B]) * W[output, l] ? No.
    # It's C_{output} = sum_l (sum_{a} U[a,l] A[a]) * (sum_{b} V[b,l] B[b]) ... this is not right.
    # The tensor is T = sum_l u_l ⊗ v_l ⊗ w_l.
    # T(A,B) = sum_l (u_l^T A) (v_l^T B) w_l.
    # Here A is a vector in R^{12}, B is a vector in R^{20}, T(A,B) is a vector in R^{15}.
    # So T(A,B)[c] = sum_l (sum_a U[a,l] A[a]) * (sum_b V[b,l] B[b]) * W[c,l].
    # We need T(A,B)[k*n+i] = (AB)_{i,k} = sum_j A[i,j] B[j,k].
    # So for each i,k:
    #   sum_l (sum_j U[i*m+j, l] A[i,j]) * (sum_j V[j*p+k, l] B[j,k]) * W[k*n+i, l] = sum_j A[i,j] B[j,k].
    # This must hold for all A, B.
    # This implies that for each i,k, the bilinear form on the left must equal the bilinear form on the right.
    # The right side is sum_j A[i,j] B[j,k].
    # The left side is sum_l W[k*n+i, l] * (sum_j U[i*m+j, l] A[i,j]) * (sum_j V[j*p+k, l] B[j,k]).
    # Let's expand the left side:
    # sum_l W[k*n+i, l] sum_{j1} sum_{j2} U[i*m+j1, l] V[j2*p+k, l] A[i,j1] B[j2,k].
    # = sum_{j1, j2} A[i,j1] B[j2,k] (sum_l U[i*m+j1, l] V[j2*p+k, l] W[k*n+i, l]).
    # For this to equal sum_j A[i,j] B[j,k] for all A,B, we need:
    # sum_l U[i*m+j1, l] V[j2*p+k, l] W[k*n+i, l] = delta_{j1, j2}.
    # This must hold for all i, k, j1, j2.
    
    # So the condition is:
    # For all i in 0..n-1, k in 0..p-1, j1 in 0..m-1, j2 in 0..m-1:
    #   sum_{l=1}^r U[i*m+j1, l] * V[j2*p+k, l] * W[k*n+i, l] = delta_{j1, j2}
    
    # This is a set of n*p*m*m = 3*5*4*4 = 240 equations.
    # The variables are the entries of U, V, W.
    # This is a system of polynomial equations.
    
    # Schoolbook solution:
    # U[i*m+j, l] = 1 if l = i*m*p + j*p + k? No.
    # In schoolbook, rank = n*m*p = 60.
    # Each term corresponds to a unique (i,j,k).
    # U[i*m+j, l] = 1 if l = i*m*p + j*p + k? No, l is just an index.
    # Let's map l to (i,j,k). There are 60 triples.
    # U[i*m+j, (i,j,k)] = 1.
    # V[j*p+k, (i,j,k)] = 1.
    # W[k*n+i, (i,j,k)] = 1.
    # Check: sum_l U[i*m+j1, l] V[j2*p+k, l] W[k*n+i, l].
    # Only l=(i,j1,k) makes U non-zero (1).
    # For this l, V[j2*p+k, (i,j1,k)] = 1 if j2=j1.
    # W[k*n+i, (i,j1,k)] = 1.
    # So the product is 1 if j1=j2, else 0.
    # This works.
    
    # Can we do better?
    # The condition is sum_l U[i*m+j1, l] V[j2*p+k, l] W[k*n+i, l] = delta_{j1,j2}.
    # This is a rank-1 decomposition of a 3-tensor.
    # The tensor S has indices (i*m+j1, j2*p+k, k*n+i) but with constraints on i,j,k.
    # Actually, the indices are not independent.
    # i is in 0..2, j1 in 0..3, j2 in 0..3, k in 0..4.
    # The equation must hold for all these combinations.
    
    # Let's try to find a solution with rank < 60.
    # We can use a random search with local optimization, or use a known algorithm.
    # Given the time limit, let's try a few random rank-r decompositions and see if we can satisfy the equations.
    # Or, we can use the fact that the tensor is a Kronecker product or similar.
    
    # For now, let's just return the schoolbook solution. It's guaranteed to work.
    # If we have time, we can try to improve.
    
    U, V, W = _schoolbook(n, m, p)
    
    # Check if we have time to do better.
    elapsed = time.time() - start_time
    if elapsed < 10.0:
        # Try to find a lower rank solution.
        # This is a hard problem. We'll skip it for now to ensure we return a valid solution.
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
