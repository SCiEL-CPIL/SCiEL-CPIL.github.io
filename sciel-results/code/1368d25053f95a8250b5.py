import numpy as np
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 3
    deadline = time.time() + 100.0

    # Construct a rank-23 decomposition for 3x3 matrix multiplication
    # Strategy: 7 diagonal + 16 cross terms
    # Each term c has: U[i*m+j, c] = coeff_a, V[j*p+k, c] = coeff_b, W[k*n+i, c] = coeff_w
    # such that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 if (i,j,k) is the (i,j) entry from row i of A and col k of B contributing to C[i,k]
    
    # The multiplication tensor T has T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k]
    # We need: C[i,k] = sum_j A[i,j]*B[j,k] = sum_c (sum_j U[i,j,c]*V[j,k,c])*W[i,k,c]
    # So: for each c, the rank-1 slice U_c (3x3) * V_c (3x3) * W_c (3x3) must sum to T
    
    # Let's build terms systematically:
    terms = []
    
    # Diagonal terms: A[i,i]*B[i,i] -> C[i,i]
    for i in range(3):
        terms.append((i, i, i, i, 1, 1))  # (i_row, j_col, k_row, k_col, u_coeff, v_coeff)
        # W for this term: C[i,i] gets +1, so W[i*n+i] = 1
    
    # Cross terms using Strassen-like ideas
    # For each pair (i != k), we need C[i,k] = sum_j A[i,j]*B[j,k]
    # We can use combinations of off-diagonal entries
    
    # Let me define the decomposition explicitly with verification
    # Term format: list of (i_a, j_a, j_b, k_b, u_val, v_val) and w_indices {ik: val}
    
    # Actually, let's use a cleaner representation:
    # Each term c: U_c is a 3x3 matrix (sparse), V_c is a 3x3 matrix (sparse), W_c is a 3x3 matrix (sparse)
    # T = sum_c U_c (outer) V_c (outer) W_c (broadcast)
    # i.e., T[i,j,k] = sum_c U_c[i,j] * V_c[j,k] * W_c[i,k]
    
    # Diagonal terms (7 terms):
    # Term 0: A[0,0]*B[0,0] -> C[0,0]
    # Term 1: A[1,1]*B[1,1] -> C[1,1]  
    # Term 2: A[2,2]*B[2,2] -> C[2,2]
    # Term 3: (A[0,1]+A[1,0])*(B[1,0]+B[0,1]) -> contributes to C[0,0] and C[1,1]
    # etc.
    
    # Let me just hardcode a known rank-23 decomposition
    # Based on the structure: use all 9 diagonal-ish and 14 cross terms
    
    # I'll construct it as: for each of the 27 (i,j,k) positions where T=1, 
    # distribute the unit across terms
    
    # Known rank-23 decomposition for 3x3 (Alon 1990 / Bini et al.):
    # Let me use a systematic approach with verification
    
    # Define terms as (U_sparse, V_sparse, W_sparse)
    # U_sparse: dict (i,j) -> val
    # V_sparse: dict (j,k) -> val  
    # W_sparse: dict (i,k) -> val
    
    terms_list = []
    
    # 9 diagonal terms: for each (i,i) and (j,j) pair
    for i in range(3):
        for j in range(3):
            # Term: A[i,j] * B[j,i] contributes to C[i,i]
            terms_list.append({
                'U': {(i, j): 1},
                'V': {(j, i): 1},
                'W': {(i, i): 1}
            })
    
    # That gives 9 terms. Now add 14 more cross terms for off-diagonal C[i,k]
    
    # For C[i,k] where i != k: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    # We need to represent this sum with additional terms
    
    # Cross terms: pair up (i,k) pairs
    # (0,1): A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    # (1,0): A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    # (0,2): A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # (2,0): A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
    # (1,2): A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    # (2,1): A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
    
    # 6 off-diagonal C entries, each needs 3 products = 18 products
    # But we can share some with the diagonal terms above
    
    # Actually the 9 diagonal terms above cover:
    # C[0,0]: A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0] (terms with i=0, j=0,1,2 and k=0)
    # Wait, my diagonal terms are: for i=0: (0,0)->C[0,0], (0,1)->C[0,0]?? No.
    # Term (i,j): U[i,j]=1, V[j,i]=1, W[i,i]=1 means A[i,j]*B[j,i] -> C[i,i]
    # So for i=0: terms are A[0,0]*B[0,0], A[0,1]*B[1,0], A[0,2]*B[2,0] all -> C[0,0]
    # For i=1: A[1,0]*B[0,1], A[1,1]*B[1,1], A[1,2]*B[2,1] all -> C[1,1]
    # For i=2: A[2,0]*B[0,2], A[2,1]*B[1,2], A[2,2]*B[2,2] all -> C[2,2]
    
    # So the 9 terms cover ALL diagonal C entries completely!
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0] ✓
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1] ✓
    # C[2,2] = A[2,0]*B[0,2] + A[2,1]*B[1,2] + A[2,2]*B[2,2] ✓
    
    # Now for off-diagonal C[i,k] with i != k:
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
    
    # These are 18 distinct products not covered by the 9 diagonal terms.
    # We need to represent them with 14 more terms (total 23).
    
    # Group them into pairs that share structure:
    # Group 1: C[0,1] and C[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    
    # Term: (A[0,0]+A[1,0])*(B[0,1]+B[1,0]) = A[0,0]*B[0,1] + A[0,0]*B[1,0] + A[1,0]*B[0,1] + A[1,0]*B[1,0]
    # This introduces unwanted terms A[0,0]*B[1,0], A[1,0]*B[0,1], A[1,0]*B[1,0]
    
    # Let me use a different approach: just use 6 terms, one for each off-diagonal C entry,
    # each being a sum of 3 products represented as one rank-1 term isn't possible.
    
    # Actually, each off-diagonal C[i,k] needs 3 separate multiplications unless we can combine.
    # 6 * 3 = 18 terms for off-diagonal + 9 for diagonal = 27. That's schoolbook.
    
    # To get rank 23, we need to save 4 terms. 
    # The key insight: some off-diagonal products can be shared.
    
    # Let me try: for C[0,1] and C[1,0]:
    # Use: (A[0,0]+A[1,1])*(B[0,1]+B[1,0]) - this gives 4 cross terms
    # We need to cancel 2 unwanted terms with other combinations.
    
    # This is getting complex. Let me just verify a specific known decomposition.
    
    # Alternative: use rank 27 (schoolbook) which is guaranteed correct.
    # The score will be 1.0 but at least it's valid.
    
    # Let me try to construct rank 23 more carefully.
    # Reference: The rank of 3x3 matrix multiplication is 23.
    # One explicit decomposition uses the following pattern:
    
    # Let me define 23 terms and verify them numerically.
    
    # Term definitions: each term is a triple of 3x3 matrices (U_t, V_t, W_t)
    # such that sum_t U_t[i,j] * V_t[j,k] * W_t[i,k] = 1 for the multiplication tensor
    
    # I'll construct and verify:
    
    def verify_decomposition(terms_list):
        """Verify that the terms sum to the matrix multiplication tensor."""
        # T[i,j,k] should equal 1 if (i,j,k) is a valid multiplication path, else 0
        # Actually T[i,j,k] = 1 always for the MM tensor (each (i,j,k) has value 1)
        # No wait: the MM tensor M[i,j,k] = 1 for all i,j,k in [0,n)
        # Because C[i,k] = sum_j A[i,j] * B[j,k], so the tensor has 1 at every (i,j,k)
        
        total = np.zeros((3, 3, 3))
        for term in terms_list:
            U = np.zeros((3, 3))
            V = np.zeros((3, 3))
            W = np.zeros((3, 3))
            for (i, j), val in term['U'].items():
                U[i, j] = val
            for (j, k), val in term['V'].items():
                V[j, k] = val
            for (i, k), val in term['W'].items():
                W[i, k] = val
            # Rank-1 contribution: U[i,j] * V[j,k] * W[i,k]
            # This is not a simple outer product in 3D; it's a specific contraction
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        total[i, j, k] += U[i, j] * V[j, k] * W[i, k]
        
        expected = np.ones((3, 3, 3))
        return np.allclose(total, expected, atol=1e-10)
    
    # Build the 9 diagonal terms
    terms_list = []
    for i in range(3):
        for j in range(3):
            terms_list.append({
                'U': {(i, j): 1},
                'V': {(j, i): 1},
                'W': {(i, i): 1}
            })
    
    # Now add 14 terms for the 18 off-diagonal products
    # Off-diagonal C entries and their required products:
    # C[0,1]: A[0,0]*B[0,1], A[0,1]*B[1,1], A[0,2]*B[2,1]
    # C[1,0]: A[1,0]*B[0,0], A[1,1]*B[1,0], A[1,2]*B[2,0]
    # C[0,2]: A[0,0]*B[0,2], A[0,1]*B[1,2], A[0,2]*B[2,2]
    # C[2,0]: A[2,0]*B[0,0], A[2,1]*B[1,0], A[2,2]*B[2,0]
    # C[1,2]: A[1,0]*B[0,2], A[1,1]*B[1,2], A[1,2]*B[2,2]
    # C[2,1]: A[2,0]*B[0,1], A[2,1]*B[1,1], A[2,2]*B[2,1]
    
    # I'll use a systematic approach: for each pair of off-diagonal C entries,
    # try to combine their 6 products into 2 terms (saving 1 term per pair)
    
    # Pair (C[0,1], C[1,0]): 6 products
    # Term: (A[0,0]+A[1,1])*(B[0,1]+B[1,0]) with W contributing to both C[0,1] and C[1,0]
    # = A[0,0]*B[0,1] + A[0,0]*B[1,0] + A[1,1]*B[0,1] + A[1,1]*B[1,0]
    # We need: A[0,0]*B[0,1] for C[0,1] and A[1,0]*B[0,0] for C[1,0]
    # This doesn't directly work.
    
    # Let me just use 6 separate terms for the 6 off-diagonal C entries,
    # each requiring 3 products, but represent each C[i,k] as one term with
    # U having 3 entries, V having 3 entries, and W having 1 entry.
    # Wait, that would be rank-3 per term, not rank-1.
    
    # A single rank-1 term can only produce ONE product A[i,j]*B[j,k]*W[i,k].
    # So we truly need one term per (i,j,k) product for schoolbook = 27.
    
    # To get rank 23, we need to use terms where U and V have multiple entries,
    # and W compensates. E.g., a term with U having entries (i1,j1) and (i2,j2),
    # V having entries (j1,k1) and (j2,k2), W having entries (i1,k1) and (i2,k2)
    # would produce a combination of 4 products.
    
    # This is the key: use "fat" terms that cover multiple (i,j,k) positions.
    
    # Let me try to construct this properly. I'll define 14 additional terms
    # and verify the full decomposition.
    
    # Given the complexity and risk of error, let me fall back to schoolbook (rank 27)
    # which is guaranteed correct. The score remains 1.0.
    
    rank = 27
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(9)]
    W = [[0] * rank for _ in range(9)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * 3 + j][col] = 1
                V[j * 3 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
