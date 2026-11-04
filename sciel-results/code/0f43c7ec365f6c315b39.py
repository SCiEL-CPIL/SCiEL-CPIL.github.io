"""Tensor network contraction for 4x5x6 matrix multiplication."""
from fractions import Fraction
import time
from collections import defaultdict

DIMENSIONS = [4, 5, 6]

def construct():
    deadline = time.time() + 130  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Strategy: Use Strassen decomposition on 2x2 blocks where possible
    # Split i into {0,1} and {2,3}, split k into {0,1} and {2,3}
    # For j=5, split into {0,1,2} and {3,4}
    # 
    # For each 2x2x2 sub-block (i in {a,a+1}, j in {b,b+1}, k in {c,c+1})
    # we can use Strassen to reduce 8 products to 7
    #
    # Let's build a term list: each term is (U_indices, V_indices, W_indices, coeff)
    # where U_indices maps (i,j) -> value, etc.
    
    # Actually, let me use a cleaner approach.
    # The tensor M[i,j,k] = 1 for all valid (i,j,k) in the multiplication.
    # We need: sum_c U_c[i,j] * V_c[j,k] * W_c[k,i] = 1 for all (i,j,k)
    
    # Let's enumerate all rank-1 terms and find a minimal set.
    # Start with schoolbook (120 terms) and try to reduce using Strassen identities.
    
    # Strassen for 2x2 matrices:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For our 4x5x6 case, let's decompose into overlapping 2x2x2 blocks.
    # i splits: {0,1}, {2,3}
    # k splits: {0,1}, {2,3}, {4,5} - but 4,5 are only 2 columns, so {4,5} works
    # j: 5 columns, split as {0,1}, {2,3}, {4} - but we need pairs for Strassen
    
    # Let me try a different approach: use the known result that for 4x4x4
    # the rank is at most 23 (vs 64 schoolbook). But we have 4x5x6.
    
    # Let me just implement a greedy reduction using Strassen-like identities
    # on the schoolbook decomposition.
    
    # Schoolbook terms: for each (i,j,k), one term with U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
    # Total: 4*5*6 = 120 terms
    
    # Try to find Strassen reductions on 2x2x2 sub-tensors.
    # A 2x2x2 sub-tensor corresponds to fixed (i1,i2) for i, (j1,j2) for j, (k1,k2) for k
    # The 8 schoolbook terms for this sub-tensor can be replaced by 7 Strassen terms.
    
    # Let's identify all possible 2x2x2 sub-blocks and apply Strassen.
    # We need to be careful about overlaps.
    
    # For simplicity, let's partition the problem:
    # i: {0,1}, {2,3}
    # j: {0,1}, {2,3}, {4}
    # k: {0,1}, {2,3}, {4,5}
    
    # 2x2x2 blocks where we can apply Strassen:
    # i_pair x j_pair x k_pair
    # i_pairs: [(0,1), (2,3)]
    # j_pairs: [(0,1), (2,3)]  (j=4 handled separately)
    # k_pairs: [(0,1), (2,3), (4,5)]
    
    # For each combination, we have a 2x2x2 sub-problem.
    # But j=4 is a singleton, so terms with j=4 use schoolbook.
    
    # Let me count:
    # For j in {0,1,2,3}: 2 i_pairs * 2 j_pairs * 3 k_pairs = 12 Strassen blocks
    # Each uses 7 terms instead of 8, saving 1 term per block = 12 savings
    # For j=4: 2 i_pairs * 3 k_pairs * 1 (singleton j) = 6 schoolbook terms (no savings)
    
    # Total: 12*7 + 6*1 = 84 + 6 = 90 terms (vs 120 schoolbook)
    
    # Let's implement this.
    
    terms = []  # list of (U_dict, V_dict, W_dict, coeff)
    
    # Strassen coefficients for a 2x2x2 block
    # Given i in {i1,i2}, j in {j1,j2}, k in {k1,k2}
    # The 7 Strassen products are:
    # M1 = (A[i1,j1]+A[i2,j2]) * (B[j1,k1]+B[j2,k2])
    # M2 = (A[i2,j1]+A[i2,j2]) * B[j1,k1]
    # M3 = A[i1,j1] * (B[j2,k1]-B[j2,k2])
    # M4 = A[i2,j2] * (B[j1,k2]-B[j1,k1])
    # M5 = (A[i1,j1]+A[i1,j2]) * B[j2,k2]
    # M6 = (A[i2,j1]-A[i1,j1]) * (B[j1,k1]+B[j1,k2])
    # M7 = (A[i1,j2]-A[i2,j2]) * (B[j2,k1]+B[j2,k2])
    
    # C[i1,k1] += M1 + M4 - M5 + M7
    # C[i1,k2] += M3 + M5
    # C[i2,k1] += M2 + M4
    # C[i2,k2] += M1 - M2 + M3 + M6
    
    # Each Strassen term M_s corresponds to a rank-1 tensor:
    # M1: U has +1 at (i1,j1) and (i2,j2); V has +1 at (j1,k1) and (j2,k2); W has +1 at (k1,i1), (k1,i2), (k2,i1), (k2,i2) with appropriate signs
    # Actually, let me think more carefully.
    
    # The rank-1 term for M1 contributes to all (i,k) where the product is non-zero.
    # M1 = (A[i1,j1]+A[i2,j2]) * (B[j1,k1]+B[j2,k2])
    # This is a scalar that gets added to C[i1,k1] and C[i2,k2] (with signs from the Strassen formula)
    
    # Wait, I need to be more careful. The Strassen algorithm computes:
    # C[i1,k1] = M1 + M4 - M5 + M7
    # C[i1,k2] = M3 + M5
    # C[i2,k1] = M2 + M4
    # C[i2,k2] = M1 - M2 + M3 + M6
    
    # Each M_s is a product of a linear combination of A entries and a linear combination of B entries.
    # So M_s = (sum_{(i,j) in S_A} a_s(i,j) * A[i,j]) * (sum_{(j',k) in S_B} b_s(j',k) * B[j',k])
    
    # The contribution to C[i,k] from M_s is:
    # sum over the Strassen formula: for each (i,k), C[i,k] += coeff_s(i,k) * M_s
    
    # So the rank-1 tensor for M_s has:
    # U_s[i,j] = a_s(i,j)
    # V_s[j,k] = b_s(j',k)  -- wait, this doesn't work directly because M_s is a single scalar
    
    # Actually, M_s is a scalar (product of two linear combinations), and it contributes to multiple (i,k) entries.
    # So the rank-1 term is:
    # For each (i,k) where M_s contributes:
    #   U_s[i,j] * V_s[j,k] * W_s[k,i] should equal the coefficient of A[i,j]*B[j,k] in the final C[i,k]
    
    # This is getting complicated. Let me use a different formulation.
    
    # The tensor M has entries M[i,j,k] = 1 for all (i,j,k).
    # A rank-1 decomposition is: sum_c U_c[i,j] * V_c[j,k] * W_c[k,i] = 1 for all (i,j,k)
    
    # For Strassen on a 2x2x2 block with i in {i1,i2}, j in {j1,j2}, k in {k1,k2}:
    # The 8 schoolbook terms are:
    # (i1,j1,k1), (i1,j1,k2), (i1,j2,k1), (i1,j2,k2)
    # (i2,j1,k1), (i2,j1,k2), (i2,j2,k1), (i2,j2,k2)
    
    # Strassen replaces these with 7 terms. Each Strassen term is a rank-1 tensor
    # that is a linear combination of the 8 schoolbook terms.
    
    # Let me define the 7 Strassen rank-1 tensors explicitly.
    
    # M1 = (A[i1,j1]+A[i2,j2]) * (B[j1,k1]+B[j2,k2])
    # This contributes to:
    # C[i1,k1]: +M1
    # C[i2,k2]: +M1
    # So the rank-1 tensor for M1 has:
    # U_M1[i1,j1] = 1, U_M1[i2,j2] = 1 (others 0)
    # V_M1[j1,k1] = 1, V_M1[j2,k2] = 1 (others 0)
    # W_M1[k1,i1] = 1, W_M1[k2,i2] = 1 (others 0)
    # But wait, this gives U[i1,j1]*V[j1,k1]*W[k1,i1] = 1 for (i1,j1,k1)
    # and U[i1,j1]*V[j2,k2]*W[k2,i1] = 0 (since V[j2,k2] is not multiplied by W[k2,i1] in this term)
    
    # I think I'm overcomplicating this. Let me just directly construct the rank-1 terms.
    
    # For a Strassen term M_s, it's a product:
    # M_s = (sum_{(i,j) in I_s} alpha_s(i,j) * A[i,j]) * (sum_{(j,k) in J_s} beta_s(j,k) * B[j,k])
    
    # The contribution to C[i,k] is:
    # C[i,k] += gamma_s(i,k) * M_s
    
    # where gamma_s is from the Strassen formula.
    
    # So the rank-1 tensor for this term has:
    # For each (i,j,k):
    #   U[i,j] * V[j,k] * W[k,i] = gamma_s(i,k) * alpha_s(i,j) * beta_s(j,k)
    
    # But this is not a rank-1 tensor in general, because the RHS factors as a product of three separate functions only if gamma_s(i,k) can be written as W_s[k,i] and the other factors separate.
    
    # Actually, for Strassen, each M_s is a single product of two linear combinations, so:
    # M_s = (sum_{(i,j)} alpha_s(i,j) A[i,j]) * (sum_{(j,k)} beta_s(j,k) B[j,k])
    
    # The contribution to C[i,k] is gamma_s(i,k) * M_s.
    
    # So we need: sum_c U_c[i,j] * V_c[j,k] * W_c[k,i] = M[i,j,k] = 1
    
    # For the Strassen term, the contribution to M[i,j,k] is:
    # gamma_s(i,k) * alpha_s(i,j) * beta_s(j,k)
    
    # This must be a rank-1 tensor, so we need:
    # U_s[i,j] * V_s[j,k] * W_s[k,i] = gamma_s(i,k) * alpha_s(i,j) * beta_s(j,k)
    
    # This works if we can factor gamma_s(i,k) * alpha_s(i,j) * beta_s(j,k) into U[i,j] * V[j,k] * W[k,i].
    
    # For the standard Strassen case, let's check M1:
    # M1 = (A[i1,j1]+A[i2,j2]) * (B[j1,k1]+B[j2,k2])
    # alpha_M1(i1,j1)=1, alpha_M1(i2,j2)=1, others 0
    # beta_M1(j1,k1)=1, beta_M1(j2,k2)=1, others 0
    # gamma_M1(i1,k1)=1, gamma_M1(i2,k2)=1, others 0
    
    # So U_M1[i1,j1]*V_M1[j1,k1]*W_M1[k1,i1] = 1
    # U_M1[i1,j1]*V_M1[j1,k2]*W_M1[k2,i1] = 0 (gamma=0)
    # U_M1[i1,j1]*V_M1[j2,k1]*W_M1[k1,i1] = 0 (beta=0)
    # U_M1[i1,j1]*V_M1[j2,k2]*W_M1[k2,i1] = 0 (gamma=0)
    # U_M1[i2,j1]*... = 0 (alpha=0)
    # U_M1[i2,j2]*V_M1[j1,k1]*W_M1[k1,i2] = 0 (gamma=0)
    # U_M1[i2,j2]*V_M1[j1,k2]*W_M1[k2,i2] = 0 (beta=0)
    # U_M1[i2,j2]*V_M1[j2,k1]*W_M1[k1,i2] = 0 (gamma=0)
    # U_M1[i2,j2]*V_M1[j2,k2]*W_M1[k2,i2] = 1
    
    # So U_M1[i1,j1]=1, U_M1[i2,j2]=1
    # V_M1[j1,k1]=1, V_M1[j2,k2]=1
    # W_M1[k1,i1]=1, W_M1[k2,i2]=1
    
    # This is a valid rank-1 tensor!
    
    # Let me define all 7 Strassen terms similarly.
    
    def add_strassen_block(i1, i2, j1, j2, k1, k2):
        """Add 7 Strassen rank-1 terms for a 2x2x2 block."""
        # M1 = (A[i1,j1]+A[i2,j2]) * (B[j1,k1]+B[j2,k2])
        # Contributes: C[i1,k1] += M1, C[i2,k2] += M1
        terms.append({
            'U': {(i1,j1): 1, (i2,j2): 1},
            'V': {(j1,k1): 1, (j2,k2): 1},
            'W': {(k1,i1): 1, (k2,i2): 1},
            'coeff': 1
        })
        
        # M2 = (A[i2,j1]+A[i2,j2]) * B[j1,k1]
        # Contributes: C[i2,k1] += M2, C[i2,k2] -= M2
        terms.append({
            'U': {(i2,j1): 1, (i2,j2): 1},
            'V': {(j1,k1): 1},
            'W': {(k1,i2): 1, (k2,i2): -1},
            'coeff': 1
        })
        
        # M3 = A[i1,j1] * (B[j2,k1]-B[j2,k2])
        # Contributes: C[i1,k2] += M3, C[i2,k2] += M3
        terms.append({
            'U': {(i1,j1): 1},
            'V': {(j2,k1): 1, (j2,k2): -1},
            'W': {(k2,i1): 1, (k2,i2): 1},
            'coeff': 1
        })
        
        # M4 = A[i2,j2] * (B[j1,k2]-B[j1,k1])
        # Contributes: C[i1,k1] += M4, C[i2,k1] += M4
        terms.append({
            'U': {(i2,j2): 1},
            'V': {(j1,k2): 1, (j1,k1): -1},
            'W': {(k1,i1): 1, (k1,i2): 1},
            'coeff': 1
        })
        
        # M5 = (A[i1,j1]+A[i1,j2]) * B[j2,k2]
        # Contributes: C[i1,k1] -= M5, C[i1,k2] += M5
        terms.append({
            'U': {(i1,j1): 1, (i1,j2): 1},
            'V': {(j2,k2): 1},
            'W': {(k1,i1): -1, (k2,i1): 1},
            'coeff': 1
        })
        
        # M6 = (A[i2,j1]-A[i1,j1]) * (B[j1,k1]+B[j1,k2])
        # Contributes: C[i2,k2] += M6
        terms.append({
            'U': {(i2,j1): 1, (i1,j1): -1},
            'V': {(j1,k1): 1, (j1,k2): 1},
            'W': {(k2,i2): 1},
            'coeff': 1
        })
        
        # M7 = (A[i1,j2]-A[i2,j2]) * (B[j2,k1]+B[j2,k2])
        # Contributes: C[i1,k1] += M7
        terms.append({
            'U': {(i1,j2): 1, (i2,j2): -1},
            'V': {(j2,k1): 1, (j2,k2): 1},
            'W': {(k1,i1): 1},
            'coeff': 1
        })
    
    def add_schoolbook_term(i, j, k):
        """Add a single schoolbook rank-1 term."""
        terms.append({
            'U': {(i,j): 1},
            'V': {(j,k): 1},
            'W': {(k,i): 1},
            'coeff': 1
        })
    
    # Apply Strassen to 2x2x2 blocks
    i_pairs = [(0, 1), (2, 3)]
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for i1, i2 in i_pairs:
        for j1, j2 in j_pairs:
            for k1, k2 in k_pairs:
                add_strassen_block(i1, i2, j1, j2, k1, k2)
    
    # Add schoolbook terms for j=4
    for i in range(n):
        for k in range(p):
            add_schoolbook_term(i, 4, k)
    
    # Now convert terms to matrix form
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, term in enumerate(terms):
        coeff = term['coeff']
        for (i, j), val in term['U'].items():
            U[i * m + j][c] = val * coeff
        for (j, k), val in term['V'].items():
            V[j * p + k][c] = val
        for (k, i), val in term['W'].items():
            W[k * n + i][c] = val
    
    return {"U": U, "V": V, "W": W}
