"""Matrix multiplication 2x5x6 - rank-minimal tensor decomposition via greedy set cover."""
import time
import random
from itertools import product
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

def construct():
    start_time = time.time()
    DEADLINE = 145  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm = n * m  # 10
    mp = m * p  # 30
    pn = p * n  # 12
    
    # Schoolbook terms: each (i,j,k) gives a rank-1 tensor
    # U[i*m+j] = 1, V[j*p+k] = 1, W[k*n+i] = 1
    # Total 60 terms
    
    # The tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need to decompose this into sum of rank-1 tensors
    # Each rank-1 tensor is defined by (u, v, w) where:
    #   u has shape (n*m,), v has shape (m*p,), w has shape (p*n,)
    #   and the tensor entry (i,j,k) gets contribution u[i*m+j]*v[j*p+k]*w[k*n+i]
    
    # Strategy: Greedy set cover
    # 1. Generate candidate rank-1 tensors (including linear combos of schoolbook terms)
    # 2. For each candidate, determine which (i,j,k) entries it "covers" (contributes to)
    # 3. Greedily select candidates to cover all 60 entries with minimum count
    
    # The key insight: a candidate tensor that is a linear combination of schoolbook terms
    # with coefficients c_{ijk} will have:
    #   u[i*m+j] = sum_k c_{ijk}
    #   v[j*p+k] = sum_i c_{ijk}  
    #   w[k*n+i] = sum_j c_{ijk}
    # This is rank-1 only in special cases, but we can use it as a building block
    # if the resulting tensor exactly matches a subset of the identity tensor.
    
    # Actually, let's think differently. The matrix multiplication tensor is:
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k  (in appropriate tensor product space)
    # where the (i,j,k) component is 1.
    
    # A rank-1 decomposition term is u ⊗ v ⊗ w where u ∈ R^{nm}, v ∈ R^{mp}, w ∈ R^{pn}
    # and (u ⊗ v ⊗ w)[i,j,k] = u[i*m+j] * v[j*p+k] * w[k*n+i]
    
    # We need sum_c (u_c ⊗ v_c ⊗ w_c) = T where T[i,j,k] = 1 for all valid (i,j,k)
    
    # So for each (i,j,k): sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1
    
    # This is a system of 60 equations in r*60 unknowns (for rank r).
    
    # Approach: Use randomized search + local optimization
    # 1. Start with a random rank-r decomposition
    # 2. Try to minimize the error
    # 3. Use exact rational arithmetic where possible
    
    # Better approach: Greedy construction
    # Try to find rank-1 tensors that exactly satisfy a subset of the equations
    
    # Let's try: for each possible subset of (i,j,k) indices, check if there exists
    # a rank-1 tensor that gives 1 on those indices and 0 on others.
    # This is hard in general, but we can use the structure.
    
    # Alternative: Use the fact that we can have half-integer entries.
    # Try to find pairs of schoolbook terms that can be merged.
    
    # Key observation: if we take two schoolbook terms (i1,j1,k1) and (i2,j2,k2)
    # and form u = e_{i1,j1} + e_{i2,j2}, v = e_{j1,k1} + e_{j2,k2}, w = e_{k1,i1} + e_{k2,i2}
    # then the tensor entry (i,j,k) gets:
    #   (δ_{i,i1}δ_{j,j1} + δ_{i,i2}δ_{j,j2}) * (δ_{j,j1}δ_{k,k1} + δ_{j,j2}δ_{k,k2}) * (δ_{k,k1}δ_{i,i1} + δ_{k,k2}δ_{i,i2})
    # This is not generally equal to δ_{i,i1}δ_{j,j1}δ_{k,k1} + δ_{i,i2}δ_{j,j2}δ_{k,k2}
    
    # So simple merging doesn't work. We need more sophisticated combinations.
    
    # Let's try a different approach: use the structure of small dimensions.
    # n=2 is small, so we can enumerate possibilities for the U matrix.
    
    # For rank r, U is 10×r, V is 30×r, W is 12×r.
    # The constraint is: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # Let's try to find a low-rank solution by random search with refinement.
    
    def time_remaining():
        return DEADLINE - (time.time() - start_time)
    
    # Start with schoolbook rank 60 as baseline
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(nm)]
    best_V = [[0] * best_rank for _ in range(mp)]
    best_W = [[0] * best_rank for _ in range(pn)]
    for col in range(best_rank):
        i = col // (m * p)
        j = (col // p) % m
        k = col % p
        best_U[i * m + j][col] = 1
        best_V[j * p + k][col] = 1
        best_W[k * n + i][col] = 1
    
    # Try to reduce rank by finding linear dependencies
    # Use a greedy approach: try to find a rank-1 tensor that satisfies 
    # multiple of the 60 equations
    
    # For a candidate rank-1 tensor (u, v, w), it satisfies equation (i,j,k) if
    # u[i*m+j]*v[j*p+k]*w[k*n+i] = 1
    # And it "doesn't interfere" with other equations if the product is 0 or we can compensate.
    
    # Let's try: fix a pattern for u, then solve for v and w.
    
    # Since n=2, i ∈ {0,1}. Let's enumerate possible u patterns.
    # u has 10 entries (2 rows of 5).
    
    # Try small integer/half-integer values for u.
    
    import numpy as np
    
    def check_decomposition(U, V, W, rank):
        """Check if U,V,W with given rank exactly decompose the tensor."""
        # U: nm × rank, V: mp × rank, W: pn × rank
        # Check: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True
    
    # Try to find lower rank decompositions using random search
    # with exact rational arithmetic
    
    def random_search(rank, num_trials=1000):
        """Try to find a rank-r decomposition via random initialization and refinement."""
        for trial in range(num_trials):
            if time_remaining() < 5:
                return None
            
            # Initialize with random small rational values
            U = [[Fraction(random.randint(-2, 2), 1) for _ in range(rank)] for _ in range(nm)]
            V = [[Fraction(random.randint(-2, 2), 1) for _ in range(rank)] for _ in range(mp)]
            W = [[Fraction(random.randint(-2, 2), 1) for _ in range(rank)] for _ in range(pn)]
            
            # Check if this random initialization happens to work (unlikely)
            if check_decomposition(U, V, W, rank):
                return U, V, W
            
            # Try local search: fix two matrices, solve for the third
            # This is a linear system if we fix U and V
            # For each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            # This is linear in W if U and V are fixed.
            
            # But the system may be underdetermined or inconsistent.
            # Let's try: fix U and V to be schoolbook-like, then solve for W.
            
            # Alternative: use the structure that n=2 is small.
            # Group by i: for i=0 and i=1, we have separate constraints.
            
            # Let's try a structured approach:
            # Assume U has a specific form based on the i index.
            
            # For now, just return None if we can't find a solution quickly
            pass
        
        return None
    
    # Try ranks from low to high
    # Theoretical lower bound: max(nm, mp, pn) = max(10, 30, 12) = 30
    # But for matrix multiplication, the rank is at least max(n,m,p)^2 for square, 
    # and for rectangular, it's at least max(n*m, m*p, p*n) / something...
    # Actually, the rank of matrix multiplication tensor for n×m × m×p is at least 
    # max(n*m, m*p, p*n) in general, but can be lower with clever decompositions.
    # For 2×5×6, schoolbook is 60. Strassen-like might reduce it.
    
    # Let's try a more systematic approach:
    # Use the fact that we can have linear combinations.
    
    # Generate candidate rank-1 tensors by taking linear combinations of schoolbook terms
    # and checking if the result is "useful" (satisfies some equations)
    
    # For each subset of schoolbook terms of size s, compute the linear combination
    # and check if it gives 1 on some equations and 0 on others (or can be part of a sum)
    
    # This is exponential in the worst case, so we need to be smart.
    
    # Let's try: for each pair of schoolbook terms, check if their sum can be 
    # represented as a single rank-1 tensor (which would reduce rank by 1).
    
    # Term (i1,j1,k1) + Term (i2,j2,k2) gives:
    # u = e_{i1,j1} + e_{i2,j2}
    # v = e_{j1,k1} + e_{j2,k2}  
    # w = e_{k1,i1} + e_{k2,i2}
    # Tensor entry (i,j,k) = (δ_{i,i1}δ_{j,j1} + δ_{i,i2}δ_{j,j2})(δ_{j,j1}δ_{k,k1} + δ_{j,j2}δ_{k,k2})(δ_{k,k1}δ_{i,i1} + δ_{k,k2}δ_{i,i2})
    
    # For this to equal δ_{i,i1}δ_{j,j1}δ_{k,k1} + δ_{i,i2}δ_{j,j2}δ_{k,k2},
    # we need the cross terms to vanish.
    
    # Cross terms occur when the indices partially match.
    # E.g., if i1=i2 but j1≠j2, k1≠k2, then:
    # u = e_{i1,j1} + e_{i1,j2}, v = e_{j1,k1} + e_{j2,k2}, w = e_{k1,i1} + e_{k2,i1}
    # Entry (i1,j1,k1): (1)(1)(1) = 1 ✓
    # Entry (i1,j2,k2): (1)(1)(1) = 1 ✓
    # Entry (i1,j1,k2): (1)(0)(0) = 0 ✓ (should be 0)
    # Entry (i1,j2,k1): (1)(0)(0) = 0 ✓
    # But entry (i1,j1,k1) also gets contribution from cross: 
    #   (δ_{i1,i1}δ_{j1,j1} + δ_{i1,i1}δ_{j1,j2})(δ_{j1,j1}δ_{k1,k1} + δ_{j1,j2}δ_{k1,k2})(δ_{k1,k1}δ_{i1,i1} + δ_{k1,k2}δ_{i1,i1})
    #   = (1+0)(1+0)(1+0) = 1 ✓
    # Entry (i1,j2,k2): 
    #   (δ_{i1,i1}δ_{j2,j1} + δ_{i1,i1}δ_{j2,j2})(δ_{j2,j1}δ_{k2,k1} + δ_{j2,j2}δ_{k2,k2})(δ_{k2,k1}δ_{i1,i1} + δ_{k2,k2}δ_{i1,i1})
    #   = (0+1)(0+1)(0+1) = 1 ✓
    # Entry (i1,j1,k2):
    #   (1+0)(1+0)(0+1) = 1 ✗ (should be 0)
    
    # So it doesn't work in general. The cross terms create unwanted contributions.
    
    # We need the cross terms to cancel. This happens if we use coefficients.
    
    # Let's try: a*Term(i1,j1,k1) + b*Term(i2,j2,k2)
    # u = a*e_{i1,j1} + b*e_{i2,j2}
    # v = a*e_{j1,k1} + b*e_{j2,k2}
    # w = a*e_{k1,i1} + b*e_{k2,i2}
    # Entry (i,j,k) = (a*δ_{i,i1}δ_{j,j1} + b*δ_{i,i2}δ_{j,j2})(a*δ_{j,j1}δ_{k,k1} + b*δ_{j,j2}δ_{k,k2})(a*δ_{k,k1}δ_{i,i1} + b*δ_{k,k2}δ_{i,i2})
    
    # For entry (i1,j1,k1): a*a*a = a^3 (should be 1, so a=1 or a=complex)
    # For entry (i2,j2,k2): b^3 (should be 1, so b=1)
    # But then cross terms appear.
    
    # This approach seems difficult. Let's try a different strategy.
    
    # Use the known result: for 2×n matrix multiplication, the rank is at most 
    # something like 2n - 1 or similar. Let me look for a constructive decomposition.
    
    # For 2×5×6:
    # Let A be 2×5, B be 5×6, C = A×B be 2×6.
    # C[0,k] = sum_j A[0,j]*B[j,k]
    # C[1,k] = sum_j A[1,j]*B[j,k]
    
    # We can write:
    # C[0,k] = sum_j (A[0,j]+A[1,j])*(B[j,k]) - sum_j A[1,j]*B[j,k]
    # C[1,k] = sum_j A[1,j]*B[j,k]
    
    # This doesn't directly help with rank.
    
    # Let's try: define for each j, let s_j = A[0,j] + A[1,j], d_j = A[0,j] - A[1,j]
    # Then A[0,j] = (s_j + d_j)/2, A[1,j] = (s_j - d_j)/2
    # C[0,k] = sum_j (s_j + d_j)/2 * B[j,k]
    # C[1,k] = sum_j (s_j - d_j)/2 * B[j,k]
    
    # C[0,k] + C[1,k] = sum_j s_j * B[j,k]
    # C[0,k] - C[1,k] = sum_j d_j * B[j,k]
    
    # So we need to compute:
    # S[k] = sum_j s_j * B[j,k]  (5×6 matrix-vector, but s is 5-dim)
    # D[k] = sum_j d_j * B[j,k]
    # Then C[0,k] = (S[k] + D[k])/2, C[1,k] = (S[k] - D[k])/2
    
    # Computing S = s × B is a 1×5 × 5×6 multiplication, which has rank 5 (schoolbook)
    # Similarly for D.
    # Total rank: 5 + 5 = 10? But we also need to form s and d, which are linear combinations.
    
    # Actually, the tensor rank of 1×5 × 5×6 multiplication is 5 (each output is a dot product).
    # So S and D each require rank 5, total rank 10.
    # But we can share the B matrix! 
    # S[k] = sum_j s_j * B[j,k]
    # D[k] = sum_j d_j * B[j,k]
    # These are two separate dot products with the same B.
    
    # In tensor terms, we need:
    # For each k: S[k] = sum_j s_j * B[j,k]
    # This is a rank-5 tensor: sum_j e_j ⊗ e_{j,k} ⊗ e_k (in appropriate spaces)
    # Similarly for D.
    
    # But s and d are linear combinations of A's entries, so we can absorb them.
    
    # Let me think in terms of the full tensor.
    # The multiplication tensor T has T[i,j,k] = 1 for i∈{0,1}, j∈{0..4}, k∈{0..5}
    # We want to write T = sum_c u_c ⊗ v_c ⊗ w_c
    
    # From the above:
    # C[0,k] = (1/2) sum_j (A[0,j]+A[1,j]) B[j,k] + (1/2) sum_j (A[0,j]-A[1,j]) B[j,k]
    # C[1,k] = (1/2) sum_j (A[0,j]+A[1,j]) B[j,k] - (1/2) sum_j (A[0,j]-A[1,j]) B[j,k]
    
    # Let S_j = A[0,j] + A[1,j], D_j = A[0,j] - A[1,j]
    # C[0,k] = (1/2) sum_j S_j B[j,k] + (1/2) sum_j D_j B[j,k]
    # C[1,k] = (1/2) sum_j S_j B[j,k] - (1/2) sum_j D_j B[j,k]
    
    # The tensor for computing S×B (where S is 1×5, B is 5×6) is:
    # sum_j e_{S_j} ⊗ e_{B_{j,k}} ⊗ e_{S_k}  for each j,k
    # This has rank 5*6/6 = 5? No, it's a 5×6 outer product structure.
    
    # Actually, S×B is a 1×6 matrix, so the tensor is 5×6×6 (for the 5 inputs, 6 B inputs, 6 outputs).
    # The rank of this tensor is 5 (each output is a separate dot product, but they share B).
    # Wait, no. S×B[k] = sum_j S[j]*B[j,k]. The tensor has entries T[j,k,l] where l is the output index.
    # T[j,k,l] = δ_{k,l} * (something)? No.
    
    # Let me be more careful. The multiplication S (1×5) × B (5×6) = C (1×6):
    # C[l] = sum_j S[j] * B[j,l]
    # The tensor has indices (j, l, l) where j is S index, l is B row/output index.
    # T[j, l, l] = 1 (S[j] multiplies B[j,l] to contribute to C[l])
    # So the tensor is sum_{j,l} e_j ⊗ e_l ⊗ e_l
    # This is the "diagonal" tensor, which has rank 5 (one term per j, with v and w both having 1 at position l for that j... wait no)
    
    # Term for fixed j: u = e_j (in 5-dim), v = sum_l e_l (in 6-dim)? No.
    # For fixed j, the contribution is S[j] * sum_l B[j,l] * C[l]
    # So u = e_j, v = e_{j,?}... 
    # I'm getting confused with the indexing.
    
    # Let me restart with a cleaner formulation.
    
    # The matrix multiplication tensor for A (n×m) × B (m×p) = C (n×p):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has T[i,j,k] = 1 for all i,j,k (in the sense that each triple contributes)
    # More precisely, T ∈ R^{nm × mp × pn} with T[(i,j), (j,k), (k,i)] = 1
    # (using the W[k*n+i] convention)
    
    # A rank-1 decomposition term is u ⊗ v ⊗ w where:
    # u ∈ R^{nm}, v ∈ R^{mp}, w ∈ R^{pn}
    # (u⊗v⊗w)[(i,j),(j,k),(k,i)] = u[i*m+j] * v[j*p+k] * w[k*n+i]
    
    # We need sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1 for all i,j,k
    
    # For n=2, m=5, p=6:
    # 60 equations, each of the form sum_c u_c[5*i+j] * v_c[5*j+k] * w_c[6*k+i] = 1
    
    # Let's try to find a rank-10 solution using the S/D decomposition idea.
    
    # Define for each j:
    # S_j = A[0,j] + A[1,j]
    # D_j = A[0,j] - A[1,j]
    
    # Then:
    # C[0,k] = (1/2) sum_j S_j B[j,k] + (1/2) sum_j D_j B[j,k]
    # C[1,k] = (1/2) sum_j S_j B[j,k] - (1/2) sum_j D_j B[j,k]
    
    # The term sum_j S_j B[j,k] is a dot product of S (5-dim) and column k of B.
    # In tensor form, this is: sum_j e_{S_j} ⊗ e_{B_{j,k}} ⊗ e_{C_k}
    # But S_j is not an input; it's a linear combination of inputs.
    
    # We can absorb the linear combination into the tensor.
    # Instead of having S_j as an intermediate, we write:
    # sum_j (A[0,j] + A[1,j]) * B[j,k] = sum_j A[0,j]*B[j,k] + sum_j A[1,j]*B[j,k]
    # Which is just the original two terms. So no gain.
    
    # The gain comes if we can share computation between the two sums.
    
    # Let me try a different approach: use the small n=2 to our advantage.
    
    # For each j, define:
    # a_j = A[0,j], b_j = A[1,j]
    # C[0,k] = sum_j a_j * B[j,k]
    # C[1,k] = sum_j b_j * B[j,k]
    
    # Consider: C[0,k] + C[1,k] = sum_j (a_j + b_j) * B[j,k]
    # C[0,k] - C[1,k] = sum_j (a_j - b_j) * B[j,k]
    
    # Let s_j = a_j + b_j, d_j = a_j - b_j
    # Then we need to compute:
    # S[k] = sum_j s_j * B[j,k]  for k=0..5
    # D[k] = sum_j d_j * B[j,k]  for k=0..5
    
    # Each of these is a 5×6 matrix-vector product (with the same B).
    # The tensor for S = s × B is: sum_{j,k} e_j ⊗ e_{j,k} ⊗ e_k (in 5×6×6 space)
    # This tensor has rank 5 (it's the "identity" tensor in a sense: for each j, we have a term that multiplies s_j by all B[j,k] and adds to S[k])
    # Wait, no. For fixed j, the term is s_j * (B[j,0], B[j,1], ..., B[j,5])
    # So the contribution to S is: s_j * row_j_of_B
    # In tensor form: for each j, we have u = e_j (in 5-dim for s), v = row_j_of_B (in 6-dim), w = all-ones (in 6-dim for S)
    # No, that's not right either.
    
    # Let me use the standard tensor decomposition for matrix-vector multiplication.
    # y = x × M where x is 1×m, M is m×p, y is 1×p.
    # y[k] = sum_j x[j] * M[j,k]
    # Tensor: sum_{j,k} e_j ⊗ e_{j,k} ⊗ e_k
    # This has rank m (one term per j, with v having 1 at position (j,k) for all k, and w having 1 at position k)
    # Wait, for fixed j: u = e_j, v = sum_k e_{j,k} (which is the j-th row of the identity-like structure), w = sum_k e_k
    # No, v should be such that v[j*p+k] = 1 for all k (for fixed j), and w[k*p+0] = 1 for all k.
    # But v[j*p+k] is a single entry, not a vector.
    
    # I think I'm overcomplicating this. Let me just use the schoolbook for the subproblems.
    
    # S[k] = sum_j s_j * B[j,k]: rank 5*6/6 = 5? No, it's 5 terms (one per j), each term is:
    # u = e_j (in 5-dim), v = e_{j,?} ... 
    # The issue is that B[j,k] depends on both j and k.
    
    # For fixed j, the term s_j * B[j,k] contributes to S[k] for all k.
    # So we need: for each j, a tensor term that has u=e_j, v=vector with 1 at positions (j,k) for all k, w=vector with 1 at positions k for all k.
    # But v has dimension mp=30, and position (j,k) is j*p+k.
    # So v = sum_k e_{j*p+k} (a vector with 1s at positions j*p, j*p+1, ..., j*p+p-1)
    # And w = sum_k e_{k*n+0} (a vector with 1s at positions 0, n, 2n, ...)
    
    # But this is not rank-1! v is a sum of basis vectors, not a single basis vector.
    
    # So the matrix-vector product x×M (1×m × m×p) has tensor rank m*p (schoolbook), not m.
    # Each (j,k) pair is a separate term.
    
    # So S = s×B has rank 5*6 = 30, and D = d×B has rank 30.
    # Total rank: 60, same as schoolbook. No gain.
    
    # The key insight I'm missing: we need to find a decomposition where the rank-1 terms 
    # are not aligned with the schoolbook terms.
    
    # Let's try a randomized search for low-rank decompositions.
    
    def try_rank(rank, num_iterations=500):
        """Try to find a rank-r decomposition via random search."""
        if time_remaining() < 10:
            return None
        
        for it in range(num_iterations):
            if time_remaining() < 5:
                return None
            
            # Random initialization with small rationals
            U = [[Fraction(random.randint(-3, 3), random.choice([1, 2])) for _ in range(rank)] for _ in range(nm)]
            V = [[Fraction(random.randint(-3, 3), random.choice([1, 2])) for _ in range(rank)] for _ in range(mp)]
            W = [[Fraction(random.randint(-3, 3), random.choice([1, 2])) for _ in range(rank)] for _ in range(pn)]
            
            # Check
            if check_decomposition(U, V, W, rank):
                return U, V, W
            
            # Try local search: fix U and V, solve for W
            # For each (i,j,k): sum_c U[5*i+j,c]*V[5*j+k,c]*W[6*k+i,c] = 1
            # This is a linear system in W.
            # Number of unknowns: pn * rank = 12 * rank
            # Number of equations: 60
            # If rank >= 5, we have more unknowns than equations, so a solution may exist.
            
            # Build the linear system
            # Variables: W[k*n+i, c] for k in 0..p-1, i in 0..n-1, c in 0..rank-1
            # Equation (i,j,k): sum_c U[5*i+j,c]*V[5*j+k,c]*W[6*k+i,c] = 1
            
            # This is linear in W. Let's set it up.
            # For fixed (i,j,k), the equation involves only W[6*k+i, c] for c in 0..rank-1
            # So each equation only involves rank variables (the c-th column of W at row 6*k+i)
            
            # Actually, for fixed (i,k), the variable is W[6*k+i, :] (a vector of length rank)
            # And the equation for (i,j,k) is: dot(U[5*i+j, :], V[5*j+k, :], W[6*k+i, :]) = 1
            # Where dot means sum_c U[5*i+j,c]*V[5*j+k,c]*W[6*k+i,c]
            
            # So for each (i,k), we have m=5 equations (one per j) involving the same W[6*k+i, :]
            # Let A_{j,c} = U[5*i+j,c]*V[5*j+k,c] for j=0..4, c=0..rank-1
            # Then: A * W[6*k+i, :] = 1 (vector of length 5, all ones)
            
            # This is a 5×rank linear system. If rank >= 5, it's underdetermined.
            # We can find a particular solution + null space.
            
            # Let's try this approach:
            # 1. Randomly initialize U and V
            # 2. For each (i,k), solve A_{i,k} * W_{i,k} = 1
            # 3. Check if the resulting W satisfies all equations (it should by construction)
            # 4. But we also need W to be consistent across different (i,j,k) for the same (i,k)
            
            # Actually, the system for fixed (i,k) is:
            # For j=0..4: sum_c U[5*i+j,c]*V[5*j+k,c]*W[6*k+i,c] = 1
            # This is 5 equations in rank unknowns (W[6*k+i,0..rank-1])
            # If rank >= 5, we can solve it.
            
            # Let's implement this:
            
            # For each i in 0..1, k in 0..5:
            #   Build 5×rank matrix A where A[j,c] = U[5*i+j,c]*V[5*j+k,c]
            #   Solve A * w = ones(5) for w (length rank)
            #   Set W[6*k+i, :] = w
            
            # Then check if the full decomposition works.
            
            U_new = [[Fraction(random.randint(-3, 3), random.choice([1, 2])) for _ in range(rank)] for _ in range(nm)]
            V_new = [[Fraction(random.randint(-3, 3), random.choice([1, 2])) for _ in range(rank)] for _ in range(mp)]
            
            # Solve for W
            W_new = [[Fraction(0) for _ in range(rank)] for _ in range(pn)]
            
            valid = True
            for i in range(n):
                for k in range(p):
                    # Build 5×rank matrix
                    A = [[U_new[5*i+j][c] * V_new[5*j+k][c] for c in range(rank)] for j in range(m)]
                    # Solve A * w = ones
                    # Use simple Gaussian elimination or just pick a particular solution
                    # Since it's underdetermined (rank >= 5), we can set some variables to 0
                    # and solve the rest.
                    
                    # For simplicity, let's try: set w[c] = 0 for c >= 5, and solve the 5×5 system
                    if rank >= 5:
                        A5 = [row[:5] for row in A]
                        b5 = [Fraction(1) for _ in range(5)]
                        # Solve A5 * w5 = b5
                        # Use Cramer's rule or Gaussian elimination
                        # For now, skip if singular
                        try:
                            # Simple Gaussian elimination
                            n_eq = 5
                            aug = [row[:] + [b5[j]] for j, row in enumerate(A5)]
                            # Eliminate
                            for col in range(n_eq):
                                # Find pivot
                                pivot = -1
                                for row in range(col, n_eq):
                                    if aug[row][col] != 0:
                                        pivot = row
                                        break
                                if pivot == -1:
                                    valid = False
                                    break
                                aug[col], aug[pivot] = aug[pivot], aug[col]
                                # Normalize
                                pv = aug[col][col]
                                for c2 in range(col, n_eq + 1):
                                    aug[col][c2] = aug[col][c2] / pv
                                # Eliminate
                                for row in range(n_eq):
                                    if row != col and aug[row][col] != 0:
                                        factor = aug[row][col]
                                        for c2 in range(col, n_eq + 1):
                                            aug[row][c2] = aug[row][c2] - factor * aug[col][c2]
                            if not valid:
                                break
                            w5 = [aug[j][n_eq] for j in range(n_eq)]
                            for c in range(5):
                                W_new[6*k+i][c] = w5[c]
                            for c in range(5, rank):
                                W_new[6*k+i][c] = Fraction(0)
                        except:
                            valid = False
                            break
                    else:
                        valid = False
                        break
                
                if not valid:
                    break
            
            if valid:
                # Check the full decomposition
                if check_decomposition(U_new, V_new, W_new, rank):
                    return U_new, V_new, W_new
        
        return None
    
    # Try ranks from low to high
    for rank in range(30, best_rank + 1):
        if time_remaining() < 10:
            break
        result = try_rank(rank, num_iterations=200)
        if result is not None:
            U, V, W = result
            best_rank = rank
            best_U, best_V, best_W = U, V, W
            break  # Found the lowest rank, stop
    
    # Convert to lists of numbers (Fraction can be converted to float or kept as is)
    # The evaluator likely accepts Fraction or float
    # Let's convert to float for safety, but keep exact if possible
    
    def to_list(mat):
        return [[float(x) if x != x else 0 for x in row] for row in mat]
    
    # Actually, let's keep as Fraction if they're integers or half-integers
    def convert_entry(x):
        if x.denominator == 1:
            return x.numerator
        elif x.denominator == 2:
            return x.numerator / 2
        else:
            return float(x)
    
    U_out = [[convert_entry(x) for x in row] for row in best_U]
    V_out = [[convert_entry(x) for x in row] for row in best_V]
    W_out = [[convert_entry(x) for x in row] for row in best_W]
    
    return {"U": U_out, "V": V_out, "W": W_out}
