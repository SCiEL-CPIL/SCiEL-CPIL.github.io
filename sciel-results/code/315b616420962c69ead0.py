import time
import random
import numpy as np

DIMENSIONS = [2, 2, 3]
TIME_LIMIT = 140.0  # seconds, with 10s safety margin

# EVOLVE-BLOCK-START

def _build_rank11():
    """Build the known rank-11 decomposition."""
    n, m, p = DIMENSIONS
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def set_u(i, j, r, val):
        U[i * m + j][r] = val
    def set_v(j, k, r, val):
        V[j * p + k][r] = val
    def set_w(k, i, r, val):
        W[k * n + i][r] = val

    # Term 1: (A00 + A11) * (B00 + B11)
    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1); set_w(1, 1, 0, 1)

    # Term 2: (A10 + A11) * B00
    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(0, 1, 1, 1); set_w(1, 1, 1, -1)

    # Term 3: A00 * (B01 - B11)
    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
    set_w(1, 0, 2, 1); set_w(1, 1, 2, 1)

    # Term 4: A11 * (B10 - B00)
    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
    set_w(0, 0, 3, 1); set_w(0, 1, 3, 1)

    # Term 5: (A00 + A01) * B11
    set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(0, 0, 4, -1); set_w(1, 0, 4, 1)

    # Term 6: (A10 - A00) * (B00 + B01)
    set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
    set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
    set_w(1, 1, 5, 1)

    # Term 7: (A01 - A11) * (B10 + B11)
    set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
    set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
    set_w(0, 0, 6, 1)

    # Term 8: A00 * B02
    set_u(0, 0, 7, 1)
    set_v(0, 2, 7, 1)
    set_w(2, 0, 7, 1)

    # Term 9: A01 * B12
    set_u(0, 1, 8, 1)
    set_v(1, 2, 8, 1)
    set_w(2, 0, 8, 1)

    # Term 10: A10 * B02
    set_u(1, 0, 9, 1)
    set_v(0, 2, 9, 1)
    set_w(2, 1, 9, 1)

    # Term 11: A11 * B12
    set_u(1, 1, 10, 1)
    set_v(1, 2, 10, 1)
    set_w(2, 1, 10, 1)

    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n=2, m=2, p=3):
    """Verify that the decomposition correctly computes C = A*B for random inputs."""
    rank = len(U[0])
    for _ in range(5):
        A = np.random.randint(-5, 6, (n, m))
        B = np.random.randint(-5, 6, (m, p))
        C_expected = A @ B
        
        # Compute C from decomposition
        # C[i][k] = sum_r U[i*m+j... wait
        # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # C[i,k] = sum_{r} W[k*n + i, r] * (sum_j U[i*m+j, r] * V[j*p+k, r])
        # Actually the tensor decomposition is:
        # C[i,k] = sum_r sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        C_computed = np.zeros((n, p), dtype=int)
        for i in range(n):
            for k in range(p):
                total = 0
                for r in range(rank):
                    for j in range(m):
                        total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                C_computed[i, k] = total
        if not np.array_equal(C_expected, C_computed):
            return False
    return True


def _search_rank10_gf13(deadline):
    """
    Search for a rank-10 decomposition over GF(13) by modifying the rank-11 solution.
    Strategy: try to find a linear dependency among the 11 terms that allows
    eliminating one term while preserving the decomposition.
    """
    n, m, p = DIMENSIONS
    MOD = 13
    
    # Build rank-11 solution as numpy arrays over GF(13)
    sol11 = _build_rank11()
    U11 = np.array(sol11["U"], dtype=int) % MOD
    V11 = np.array(sol11["V"], dtype=int) % MOD
    W11 = np.array(sol11["W"], dtype=int) % MOD
    
    # The tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # We want to find rank-10: a subset of 10 terms (or linear combos) that still works.
    
    # Approach: For each pair of terms to merge/eliminate, try random linear combinations.
    # If we can express term r as a combination of other terms in a way that the
    # tensor equation still holds, we can eliminate it.
    
    # Alternative approach: Try all ways to remove one term and check if the remaining
    # 10 terms can be adjusted (via linear combinations of columns) to still work.
    
    # Most direct: check if any single term is redundant given the others.
    # For each term r to remove, check if the tensor contribution of term r can be
    # absorbed by adjusting the other 10 terms.
    
    # Let's try: for each term r, set it to zero and see if we can find adjustments
    # to the remaining terms that compensate.
    
    # Actually, a cleaner approach: 
    # The condition for rank r decomposition is that T = sum_{s=1}^{r} U_s ⊗ V_s ⊗ W_s
    # where U_s is a vector in R^{n*m}, etc.
    # 
    # For rank 10 from rank 11: we need to find 10 vectors (u_s, v_s, w_s) such that
    # sum_{s=1}^{10} u_s ⊗ v_s ⊗ w_s = T
    #
    # Strategy: Start with the 11-term solution. Try to find a linear relation
    # sum_{s=1}^{11} c_s * (u_s ⊗ v_s ⊗ w_s) = 0 where not all c_s are zero.
    # If such a relation exists, we can use it to eliminate one term.
    
    # Let's build the full tensor T and check linear dependencies.
    # T has dimensions n*m*p = 4*6 = 24 entries (flattened: C[i,k] for i in 0..1, k in 0..2)
    # Wait, T is a 3-tensor of shape (n, m, p) = (2,2,3), so 12 entries.
    # But the decomposition acts on the bilinear map, so we need to check all
    # (i, j, k) triples where the tensor coefficient should be 1 if i,j,k form
    # a valid multiplication term, i.e., T[i,j,k] = 1 if the output index matches.
    #
    # Actually for matrix multiplication: C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor is T[i,j,k] = 1 (it's the identity in the "j" dimension sense)
    # More precisely, the bilinear form has T[i,j,k] = 1 for all valid (i,j,k).
    # The tensor has entries T[i,j,k] for i in [0,n), j in [0,m), k in [0,p).
    # For matrix mult, T[i,j,k] = 1 for all i,j,k (it's the standard multiplication tensor).
    # Wait no. The tensor for matrix multiplication M_{n,m,p} has:
    # T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k].
    # So T is indexed by (i, j, k) and T[i,j,k] = 1 for all valid indices.
    # The rank is the minimum number of rank-1 tensors summing to T.
    
    # Let me verify: the tensor T has shape (n, m, p) = (2, 2, 3) with all entries = 1?
    # No! For matrix multiplication, the tensor is:
    # T[i, j, k] = 1 (it's always 1 because each (A[i,j], B[j,k]) pair contributes to C[i,k])
    # But the decomposition is: C[i,k] = sum_j A[i,j] B[j,k]
    # This is a bilinear map from (A, B) to C. The tensor represents this map.
    # T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # Wait, that's not right either. Let me think more carefully.
    
    # The matrix multiplication tensor T ∈ R^{n×m × m×p × n×p} (in flattened form)
    # or T ∈ R^{n × m × p} in the sense that T[i,j,k] gives the coefficient of A[i,j]*B[j,k] in C[i,k].
    # For standard matrix mult: C[i,k] = sum_j A[i,j] B[j,k], so T[i,j,k] = 1 for all valid (i,j,k).
    # The tensor has n*m*p = 2*2*3 = 12 entries, all equal to 1.
    # A rank-1 tensor is u ⊗ v ⊗ w where u ∈ R^n, v ∈ R^m, w ∈ R^p.
    # (u ⊗ v ⊗ w)[i,j,k] = u[i] * v[j] * w[k]
    # We need: sum_{s=1}^{rank} u_s[i] * v_s[j] * w_s[k] = 1 for all i,j,k.
    
    # So the tensor is just the all-ones tensor of shape (2,2,3)!
    # That makes the problem: find the minimum rank of the all-ones tensor of shape 2×2×3.
    
    # Let me verify with the given decomposition:
    # The decomposition uses U of shape (n*m, rank) = (4, rank), V of shape (m*p, rank) = (6, rank), W of shape (p*n, rank) = (6, rank).
    # The verification is: for all i,k: C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * V[j*p+k, r])
    # This should equal sum_j A[i,j]*B[j,k] for all A, B.
    # This means: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # So the tensor equation is: for all (i,j,k) in [0,2)×[0,2)×[0,3):
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is the all-ones tensor of shape (2,2,3) decomposed into rank-1 tensors
    # where the rank-1 tensor is indexed as (i*m+j, j*p+k, k*n+i) -> this is a
    # "permutation" of the standard (i,j,k) indexing.
    
    # Let me just verify the rank-11 solution works, then search for rank-10.
    
    # Verify rank-11
    if not _verify_decomposition(sol11["U"], sol11["V"], sol11["W"]):
        return None
    
    # Now search for rank-10 over GF(13)
    # Strategy: try to find 10 rank-1 tensors whose sum equals the all-ones tensor.
    # 
    # Approach 1: Start with rank-11, try to eliminate one term by finding a
    # linear dependency.
    #
    # Approach 2: Random search for rank-10 directly.
    #
    # Let's try a structured approach:
    # The all-ones tensor of shape (2,2,3) can be written as:
    # 1 = 1*1*1 for all (i,j,k)
    # A rank-1 tensor is u[i]*v[j]*w[k].
    # We need sum_s u_s[i]*v_s[j]*w_s[k] = 1 for all i,j,k.
    #
    # This is equivalent to: the matrix M_{ij,k} = sum_s u_s[i]*v_s[j]*w_s[k] should be all-ones.
    # For fixed k, M_{ij,k} is a 2x2 matrix that should be all-ones.
    # M_{ij,k} = sum_s (u_s[i]*v_s[j]) * w_s[k]
    # So for each k, the 2x2 matrix M_k = sum_s w_s[k] * (u_s ⊗ v_s) should be all-ones.
    #
    # The all-ones 2x2 matrix has rank 1. So each M_k is a rank-1 matrix (all-ones).
    # M_k = w_k * (some rank-1 combination of u_s ⊗ v_s)
    #
    # Actually, let's think about it differently.
    # sum_s u_s[i]*v_s[j]*w_s[k] = 1
    # For i=0: sum_s u_s[0]*v_s[j]*w_s[k] = 1 for all j,k
    # For i=1: sum_s u_s[1]*v_s[j]*w_s[k] = 1 for all j,k
    #
    # Let a_s = u_s[0], b_s = u_s[1].
    # sum_s a_s * v_s[j] * w_s[k] = 1 for all j,k  ... (1)
    # sum_s b_s * v_s[j] * w_s[k] = 1 for all j,k  ... (2)
    #
    # From (1): the matrix V*W^T weighted by a_s sums to all-ones 2x3 matrix.
    # From (2): the matrix V*W^T weighted by b_s sums to all-ones 2x3 matrix.
    #
    # Let S_a = sum_s a_s * (v_s ⊗ w_s) and S_b = sum_s b_s * (v_s ⊗ w_s)
    # We need S_a = S_b = J (all-ones 2x3 matrix).
    #
    # This means S_a - S_b = 0, i.e., sum_s (a_s - b_s) * (v_s ⊗ w_s) = 0.
    #
    # Also, S_a = J. The all-ones 2x3 matrix J can be written as 1 ⊗ 1 (outer product of
    # all-ones vectors). So we need sum_s a_s * v_s * w_s^T = 1 * 1^T.
    #
    # This is a matrix factorization problem. J = sum_s a_s * v_s * w_s^T where v_s ∈ R^2, w_s ∈ R^3.
    # The rank of J is 1, so we need at least 1 term. But we also need the same decomposition
    # to work for the b_s coefficients.
    
    # Let me just do a direct random search for rank-10 over GF(13).
    # We need to find u_s ∈ GF(13)^2, v_s ∈ GF(13)^2, w_s ∈ GF(13)^3 for s=1..10
    # such that for all i,j,k: sum_s u_s[i]*v_s[j]*w_s[k] = 1 (mod 13).
    
    # This is 12 equations in 10*(2+2+3) = 50 unknowns. Should be feasible.
    
    # Use randomized approach: fix some structure and solve for the rest.
    
    # Strategy: Use the fact that the all-ones tensor has special structure.
    # Try: u_s[i] = 1 for all s,i (fix U to all-ones).
    # Then we need: sum_s v_s[j]*w_s[k] = 1 for all j,k.
    # This means sum_s (v_s ⊗ w_s) = J (all-ones 2x3 matrix).
    # J has rank 1, so we can write J = v * w^T where v = [1,1]^T, w = [1,1,1]^T.
    # But we need it as a sum of 10 rank-1 matrices.
    # We can use: J = sum_{s=1}^{10} v_s * w_s^T where most terms cancel.
    # E.g., v_1 = [1,1], w_1 = [1,1,1], and v_s = 0 for s>1.
    # But then the b_s constraint: sum_s b_s * v_s[j] * w_s[k] = 1 also needs to hold.
    # If v_s = 0 for s>1, then sum_s b_s * v_s[j] * w_s[k] = b_1 * v_1[j] * w_1[k] = b_1 * 1 * 1 = b_1.
    # We need b_1 = 1. And a_1 = 1. So a_1 = b_1 = 1, which means u_1 = [1,1].
    # But then all other u_s can be anything since v_s = 0.
    # Wait, but we need 10 terms. If 9 terms have v_s = 0, they don't contribute.
    # So effectively we have rank 1? No, the rank is the number of nonzero terms.
    # If 9 terms are zero, the effective rank is 1. But the problem asks for rank = number of terms.
    # We need exactly 10 nonzero terms.
    
    # Let me reconsider. The rank is the number of terms in the decomposition, and all terms
    # should be nonzero (otherwise we could remove them and get a lower rank).
    
    # OK let me just do a brute-force random search over GF(13) for rank 10.
    
    def _check_rank10_gf13(U, V, W, MOD):
        """Check if rank-10 decomposition works over GF(MOD)."""
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(10):
                        total = (total + U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]) % MOD
                    if total != 1 % MOD:
                        return False
        return True
    
    # Random search
    rng = random.Random(42)
    attempts = 0
    max_attempts = 500000
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        if attempts % 10000 == 0:
            # Check time
            pass
        
        # Generate random rank-10 decomposition
        U = [[rng.randint(0, MOD-1) for _ in range(10)] for _ in range(n*m)]
        V = [[rng.randint(0, MOD-1) for _ in range(10)] for _ in range(m*p)]
        W = [[rng.randint(0, MOD-1) for _ in range(10)] for _ in range(p*n)]
        
        if _check_rank10_gf13(U, V, W, MOD):
            # Found a valid rank-10 decomposition over GF(13)!
            # Convert to integers (they're already small integers)
            return {"U": U, "V": V, "W": W}
    
    return None


def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 by 2x3 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # First, get the known rank-11 solution as fallback
    sol11 = _build_rank11()
    
    # Verify it works
    if not _verify_decomposition(sol11["U"], sol11["V"], sol11["W"]):
        # Shouldn't happen, but return anyway
        return sol11
    
    # Try to find rank-10 over GF(13)
    remaining = deadline - time.time()
    if remaining > 5:  # Need at least 5 seconds for search
        sol10 = _search_rank10_gf13(deadline - 5)
        if sol10 is not None:
            # Verify the GF(13) solution over integers (it should work since entries are small)
            if _verify_decomposition(sol10["U"], sol10["V"], sol10["W"]):
                return sol10
            # If integer verification fails, try to fix by adjusting entries
            # The GF(13) solution might need lifting
            # For now, fall back to rank-11
    
    return sol11

# EVOLVE-BLOCK-END
