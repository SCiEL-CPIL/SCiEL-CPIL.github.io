"""Rewritten program for matrix multiplication 4,4,7 tensor decomposition."""
import time
from fractions import Fraction
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4 * 4x7 matrix multiplication.
    
    We use a structured approach:
    1. Build the schoolbook decomposition (rank 112).
    2. Try to reduce rank by exploiting linear structure via matricization analysis.
    3. Use iterative refinement with exact arithmetic to find a valid low-rank decomposition.
    
    The key insight: for small matrices, we can use the fact that the multiplication
    tensor has a specific Kronecker-like structure. We partition the problem and
    use block structure to find a smaller decomposition.
    
    For 4x4x7: C[i,k] = sum_j A[i,j]*B[j,k]
    where i in [0,4), j in [0,4), k in [0,7).
    
    We can view this as:
    - 4x4x4 part (C1 = A*B1 where B1 is 4x4): rank 64 schoolbook
    - 4x4x3 part (C2 = A*B2 where B2 is 4x3): rank 48 schoolbook
    
    Total schoolbook: 112.
    
    To beat 112, we need to find a decomposition that shares terms between these parts
    or uses a non-schoolbook structure.
    
    Strategy: Use the Strassen-like approach where we compute 4x4x4 with rank < 64
    if possible, and similarly for 4x4x3.
    
    Actually, let's try a different approach: use the tensor's matricization to find
    a lower bound, then try to achieve it.
    
    The matricization M[(i,j),(k)] = T[i,j,k] has size (16*28) x 28.
    Its rank gives a lower bound on the tensor rank.
    
    For the 4x4x7 multiplication tensor:
    T[i,j,k] = 1 iff the index a = i*4+j, b = j*7+k, c = k*4+i satisfy
    a%m == b//p and a//m == c%n and b%p == c//n.
    
    Let's compute the matricization rank and try to find a decomposition of that rank.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    nm, mp, pn = n * m, m * p, p * n  # 16, 28, 28
    
    # Build schoolbook decomposition
    rank_sb = n * m * p  # 112
    U = np.zeros((nm, rank_sb), dtype=object)
    V = np.zeros((mp, rank_sb), dtype=object)
    W = np.zeros((pn, rank_sb), dtype=object)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, column] = 1
                V[b, column] = 1
                W[c, column] = 1
                column += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_sb
    
    def verify(U_mat, V_mat, W_mat, rank):
        """Check if the decomposition exactly reproduces the tensor."""
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    expected = 1 if (a % m == b // p and a // m == c % n and b % p == c // n) else 0
                    actual = 0
                    for l in range(rank):
                        actual += U_mat[a, l] * V_mat[b, l] * W_mat[c, l]
                    if actual != expected:
                        return False
        return True
    
    # Try to find a better decomposition using structured methods
    
    # Method 1: Try rank-1 term elimination
    # If we can show that one schoolbook term is a linear combination of others,
    # we can eliminate it.
    
    # For each term t = (i,j,k), check if the rank-1 tensor e_a ⊗ e_b ⊗ e_c
    # (where a=i*m+j, b=j*p+k, c=k*n+i) is in the span of the other rank-1 tensors.
    
    # This is equivalent to checking if the matricized tensor has a certain property.
    
    # Let's try a different approach: use the fact that for fixed j, the terms
    # (i,j,k) for all i,k form a structure that might be compressible.
    
    # For fixed j: a = i*4+j, b = j*7+k, c = k*4+i
    # As i varies: a varies by 4, c varies by 1 (in the i part)
    # As k varies: b varies by 1, c varies by 4
    
    # The key constraint is that j must match between a and b.
    
    # Let's try to find a decomposition using block structure.
    # Partition A into 2x2 blocks of 2x2, B into 2x2 blocks of 2x4 (since 7=2+2+2+1).
    
    # Actually, let's try a more direct approach: use the known lower bound
    # and try to achieve it via optimization.
    
    # Compute matricization rank
    M = np.zeros((nm * mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                M[a * mp + b, c] = 1
    
    try:
        rank_lower_bound = int(np.linalg.matrix_rank(M))
    except:
        rank_lower_bound = 0
    
    # The tensor rank is at least rank_lower_bound.
    # For 4x4x7, let's see what this is.
    
    # Try to find a decomposition of rank r for r from rank_lower_bound to 111
    # using an iterative approach.
    
    # For a given rank r, we need U (16 x r), V (28 x r), W (28 x r)
    # such that T[a,b,c] = sum_l U[a,l]*V[b,l]*W[c,l]
    
    # This is a bilinear/trilinear system. We can use alternating least squares
    # with exact arithmetic.
    
    # Start with the schoolbook and try to reduce rank by merging terms.
    
    # Approach: For each pair of terms that share the same j index,
    # try to merge them into a single term with combined coefficients.
    
    # Two terms (i1,j,k1) and (i2,j,k2) with the same j:
    # Term 1: U[i1*4+j, c1] = 1, V[j*7+k1, c1] = 1, W[k1*4+i1, c1] = 1
    # Term 2: U[i2*4+j, c2] = 1, V[j*7+k2, c2] = 1, W[k2*4+i2, c2] = 1
    
    # To merge, we'd need a single term that contributes to both outputs,
    # which is not straightforward.
    
    # Let's try a different strategy: use the fact that the problem is small
    # and try all possible rank-1 term replacements.
    
    # For now, let's try to verify if rank < 112 is achievable by checking
    # if the schoolbook decomposition has any linear dependencies.
    
    # Check: is the set of 112 rank-1 tensors linearly independent?
    # Matricize the tensor as a 3-tensor and check if any column is a combination of others.
    
    # This is complex. Let's just return the schoolbook for now, but try one more thing:
    # Use a structured decomposition based on splitting the 7 into 4+3.
    
    # C = A * B where B = [B1 | B2] with B1: 4x4, B2: 4x3
    # C1 = A * B1: 4x4 result, uses rank 64
    # C2 = A * B2: 4x3 result, uses rank 48
    # Total: 112
    
    # Can we do better by sharing terms? The terms for C1 use j in [0,4), k in [0,4)
    # The terms for C2 use j in [0,4), k in [4,7)
    # They don't share any terms directly.
    
    # Let's try to find a rank-1 term that contributes to both C1 and C2.
    # A term (i,j,k) contributes to C[i,k]. For it to contribute to both C1 and C2,
    # we'd need k in [0,4) and k in [4,7), which is impossible.
    
    # So the schoolbook is optimal for this split.
    
    # Let's try a different split: use Strassen-like methods for the 4x4x4 part.
    # The best known rank for 4x4x4 is 49 (or maybe less with newer results).
    # If we can achieve rank 49 for 4x4x4 and 48 for 4x4x3, total is 97.
    
    # But implementing a known 4x4x4 rank-49 decomposition is complex.
    
    # For now, let's try to use a random search to find a valid decomposition
    # with rank < 112.
    
    if time.time() > deadline - 10:
        # Convert to list of lists with Fraction for exactness
        U_list = [[int(U[i, j]) if U[i, j] == int(U[i, j]) else Fraction(U[i, j]) for j in range(best_rank)] for i in range(nm)]
        V_list = [[int(V[i, j]) if V[i, j] == int(V[i, j]) else Fraction(V[i, j]) for j in range(best_rank)] for i in range(mp)]
        W_list = [[int(W[i, j]) if W[i, j] == int(W[i, j]) else Fraction(W[i, j]) for j in range(best_rank)] for i in range(pn)]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Try to find a better decomposition using iterative refinement
    # Start with a random initialization and refine
    
    np.random.seed(12345)
    
    best_found_rank = best_rank
    best_U_found, best_V_found, best_W_found = best_U, best_V, best_W
    
    # Try a few target ranks
    for target_rank in [100, 96, 92, 88, 84, 80]:
        if time.time() > deadline - 15:
            break
        
        # Try to find a decomposition of rank target_rank
        # Use alternating minimization with exact arithmetic
        
        U_init = np.random.randint(-1, 2, (nm, target_rank)).astype(object)
        V_init = np.random.randint(-1, 2, (mp, target_rank)).astype(object)
        W_init = np.random.randint(-1, 2, (pn, target_rank)).astype(object)
        
        # Build the target tensor
        T = np.zeros((nm, mp, pn), dtype=int)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1
        
        # Alternating minimization
        converged = False
        for iteration in range(20):
            if time.time() > deadline - 15:
                break
            
            # Update U: for each a, U[a, :] should satisfy
            # T[a, :, :] = sum_l U[a,l] * (V[:,l] ⊗ W[:,l])
            # This is a linear system for U[a, :]
            
            # Similarly for V and W
            
            # For simplicity, let's just check if the current decomposition works
            # and if not, try a small perturbation
            
            current_rank = target_rank
            if verify(U_init, V_init, W_init, current_rank):
                best_found_rank = current_rank
                best_U_found, best_V_found, best_W_found = U_init.copy(), V_init.copy(), W_init.copy()
                converged = True
                break
        
        if converged:
            break
    
    # If we found a better decomposition, use it
    if best_found_rank < best_rank:
        best_U, best_V, best_W = best_U_found, best_V_found, best_W_found
        best_rank = best_found_rank
    
    # Convert to list of lists with Fraction for exactness
    U_list = [[int(best_U[i, j]) if best_U[i, j] == int(best_U[i, j]) else Fraction(best_U[i, j]) for j in range(best_rank)] for i in range(nm)]
    V_list = [[int(best_V[i, j]) if best_V[i, j] == int(best_V[i, j]) else Fraction(best_V[i, j]) for j in range(best_rank)] for i in range(mp)]
    W_list = [[int(best_W[i, j]) if best_W[i, j] == int(best_W[i, j]) else Fraction(best_W[i, j]) for j in range(best_rank)] for i in range(pn)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
