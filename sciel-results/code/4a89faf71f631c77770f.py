import time
import numpy as np
from itertools import product, combinations_with_replacement

def construct():
    n, m, p = 2, 4, 8
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Dimensions
    dim_A = n * m  # 8
    dim_B = m * p  # 32
    dim_C = p * n  # 16
    
    # The tensor T is defined by T[r, c, o] = 1 if (r, c, o) corresponds to (i, j, k)
    # r = i*m + j
    # c = j*p + k
    # o = k*n + i
    # We want to find U, V, W such that T = sum_{t} U[:, t] V[:, t] W[:, t]
    # This is equivalent to finding rank-1 decompositions of the "slices" or using bilinear complexity logic.
    
    # Strategy: Algebraic Elimination / Bilinear Complexity Search
    # We search for a set of "multiplications" m_t = u_t^T A v_t.
    # Each multiplication is a scalar.
    # The output C is a linear combination of these scalars: C_{i,k} = sum_t w_{i,k,t} m_t.
    
    # We will try to find a decomposition of rank R.
    # We start with a low rank bound (32) and go up.
    # To make this tractable, we restrict the search space of u_t and v_t to "structured" vectors.
    # Specifically, we look for u_t and v_t that are sparse or have small integer entries.
    
    # A more robust approach for this specific size (2x4x8):
    # We can view the multiplication as a sum of products of sub-matrices.
    # However, a direct algebraic search for general u, v is NP-hard.
    
    # Let's use a "Randomized Bilinear Elimination" approach.
    # 1. Generate a pool of candidate rank-1 terms (u, v).
    #    u is in R^8, v is in R^32.
    #    We restrict u and v to have entries in {-1, 0, 1} for simplicity and exactness.
    #    This creates a finite (though large) set of candidates.
    # 2. Select a subset of these candidates of size R.
    # 3. Solve for W (coefficients) such that the tensor equation holds.
    #    If a solution W exists with integer/half-integer entries (or rational), we have a valid decomposition.
    
    # Optimization:
    # Instead of brute force over all subsets, we can use a greedy selection or random sampling.
    # Given the 150s limit, we cannot exhaustively search.
    
    # Let's try a specific structured decomposition first:
    # The "Block Strassen" approach yielded 56.
    # Can we do better?
    # The lower bound is 32.
    
    # Let's try to construct a rank 32 decomposition.
    # If we can find one, score is 64/32 = 2.0.
    
    # Construction Idea for Rank 32:
    # Note that 2x4x8 is equivalent to multiplying (2x4) by (4x8).
    # This can be seen as 4 independent (2x1x2) multiplications? No.
    # It is 2 rows of A.
    # Let A = [A1; A2] where A1, A2 are 1x4.
    # B is 4x8.
    # C1 = A1 * B (1x8)
    # C2 = A2 * B (1x8)
    # This is two independent (1x4x8) multiplications.
    # Rank of (1x4x8) tensor?
    # Lower bound: max(4, 32, 8) = 32.
    # So rank(1x4x8) >= 32.
    # Total rank for 2x4x8 >= 2 * 32? No, the A parts are distinct, so they are independent.
    # Wait, the lower bound for the whole tensor is max(8, 32, 16) = 32.
    # If rank(1x4x8) is 32, then two of them would be 64?
    # No, the tensor for 2x4x8 is NOT the direct sum of two 1x4x8 tensors in a way that allows independent rank reduction.
    # Actually, T_{2x4x8} is isomorphic to T_{1x4x8} \otimes I_2? No.
    # Let's check the matricization ranks again.
    # M1 (A x BC): 8 x 512. Rank <= 8.
    # M2 (B x AC): 32 x 128. Rank <= 32.
    # M3 (C x AB): 16 x 256. Rank <= 16.
    # So Rank(T) >= 32.
    
    # If we can achieve rank 32, it is optimal.
    
    # Let's try to find a rank 32 decomposition.
    # We will use a randomized search for a set of 32 rank-1 terms.
    # We will restrict u, v to have small integer entries.
    
    # To speed this up, we will use a "SVD-like" heuristic on the tensor slices.
    # But we need EXACT integer/half-integer solutions.
    
    # Alternative: Use the known result for 2x2x2 (Strassen, rank 7).
    # 2x4x8 can be decomposed using Strassen blocks.
    # We already have a rank 56 decomposition.
    # Can we reduce it to 32?
    
    # Let's try a different structural decomposition:
    # Split B into 4 columns of 4x2?
    # No, let's stick to the algebraic elimination search for a better rank than 56.
    # We will try ranks 32 to 55.
    
    # Since exhaustive search is too slow, we will implement a "Greedy Linear Combination" search.
    # 1. Start with the schoolbook decomposition (rank 64).
    # 2. Try to replace pairs of terms with a single term (if possible).
    # 3. This is essentially searching for a lower rank CP decomposition.
    
    # Given the time constraints and the difficulty of finding a rank < 56 exactly,
    # and knowing that 56 is a strong result (score 1.14),
    # I will implement a robust version of the Strassen-based construction (rank 56)
    # but add a quick check for a rank 32 solution using a very limited random search.
    # If the random search fails (likely), it returns the rank 56 solution.
    
    # The Rank 56 construction:
    # Split B into B_left (4x4) and B_right (4x4).
    # C_left = A * B_left.
    # C_right = A * B_right.
    # Each 2x4x4 multiplication is done using 28 scalar multiplications (14 for each half of A? No, 14 total for 2x4x4 as derived? Wait, 14 was impossible. 28 is valid).
    # Let's re-verify the 2x4x4 rank.
    # Schoolbook: 32.
    # Strassen based:
    # A = [A1 A2], B = [B11 B12; B21 B22].
    # C1 = A1 B11 + A2 B21.
    # C2 = A1 B12 + A2 B22.
    # We computed A1 B11 (rank 7), A2 B21 (rank 7).
    # Total 14 terms for C1?
    # But C1 is 2x2. The 7 terms for A1 B11 produce a 2x2 result.
    # The 7 terms for A2 B21 produce a 2x2 result.
    # We sum them.
    # So we have 14 scalar multiplications.
    # But the rank of the tensor for 2x4x4 is at least 16.
    # So we cannot just sum 14 rank-1 tensors to get the full 2x4x4 tensor.
    # The issue is that the "terms" in the Strassen decomposition are not rank-1 in the 2x4x4 tensor space, they are rank-1 in the 2x2x2 space.
    # When embedded in 2x4x4, each 2x2x2 term becomes a tensor supported on a subspace.
    # The rank of the sum is not simply the sum of the ranks if the supports overlap in the output space.
    # Here, the output space is shared.
    # So the rank is likely higher than 14.
    # The previous program found rank 56 for 2x4x8.
    # This implies rank 28 for 2x4x4?
    # If 2x4x4 rank is 28, then 2x4x8 rank is 56.
    # Is 28 the true rank of 2x4x4?
    # Lower bound 16.
    # It is plausible.
    
    # I will return the Rank 56 solution as it is a proven improvement over 64.
    # I will not risk a failed search for 32 given the time limit, as returning an invalid decomposition scores 0.
    
    # Constructing the Rank 56 Decomposition (Strassen-based)
    
    # Strassen 2x2x2 factors
    # Indices for A (2x2): 0=A00, 1=A01, 2=A10, 3=A11
    # Indices for B (2x2): 0=B00, 1=B01, 2=B10, 3=B11
    # Indices for C (2x2): 0=C00, 1=C01, 2=C10, 3=C11
    
    # Terms:
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00 -> C10 += M2
    # M3 = A00(B01-B11) -> C01 += M3
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    # M5 = (A00+A01)B11 -> C01 += M5, C11 -= M5
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    # M7 = (A01-A11)(B10+B11) -> C00 += M7, C01 -= M7
    
    # We define the (u, v, w) vectors for these 7 terms in the 2x2x2 space.
    # u (len 4), v (len 4), w (len 4)
    
    S_terms = []
    # M1
    S_terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    # M2
    S_terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))
    # M3
    S_terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,0]))
    # M4
    S_terms.append(([0,0,0,1], [0,0,1,-1], [1,0,1,0]))
    # M5
    S_terms.append(([1,1,0,0], [0,0,0,1], [0,1,0,-1]))
    # M6
    S_terms.append(([0,-1,1,0], [1,1,0,0], [0,0,0,1]))
    # M7
    S_terms.append(([0,1,0,-1], [0,0,1,1], [1,-1,0,0]))
    
    # Now, we map these to the 2x4x4 problem.
    # A (2x4) indices: 0..7.
    # A1 = A[:, 0:2] -> indices 0,1,2,3.
    # A2 = A[:, 2:4] -> indices 4,5,6,7.
    
    # B (4x4) indices: 0..15.
    # B11 = B[0:2, 0:2] -> indices 0,1,4,5.
    # B12 = B[0:2, 2:4] -> indices 2,3,6,7.
    # B21 = B[2:4, 0:2] -> indices 8,9,12,13.
    # B22 = B[2:4, 2:4] -> indices 10,11,14,15.
    
    # C (2x2) indices: 0..3.
    # C11 = C[0,0] = 0.
    # C12 = C[0,1] = 1.
    # C21 = C[1,0] = 2.
    # C22 = C[1,1] = 3.
    
    # We compute C1 = A1*B11 + A2*B21.
    # We use 7 terms for A1*B11 and 7 terms for A2*B21.
    # For each term, we extend u, v to length 8 and 16.
    # w remains length 4 (for the 2x2 output block).
    
    # Left Block (B_left, indices 0..15 of B, C indices 0..3)
    # Wait, C is 2x4 for the 2x4x4 case?
    # No, in 2x4x4, C is 2x4.
    # My previous analysis assumed C was 2x2.
    # If C is 2x4, then:
    # C1 = A * B_left (2x4).
    # C1[:, 0:2] = A1*B11 + A2*B21.
    # C1[:, 2:4] = A1*B12 + A2*B22.
    
    # So we have 4 sub-problems of 2x2x2:
    # 1. A1*B11 -> contributes to C[:, 0:2]
    # 2. A2*B21 -> contributes to C[:, 0:2]
    # 3. A1*B12 -> contributes to C[:, 2:4]
    # 4. A2*B22 -> contributes to C[:, 2:4]
    
    # Total 4 * 7 = 28 terms for 2x4x4.
    # For 2x4x8, we have two such blocks (B_left and B_right).
    # Total 2 * 28 = 56 terms.
    
    # Let's construct the U, V, W matrices for rank 56.
    
    rank = 56
    U = np.zeros((dim_A, rank), dtype=float)
    V = np.zeros((dim_B, rank), dtype=float)
    W = np.zeros((dim_C, rank), dtype=float)
    
    # We will fill the columns of U, V, W.
    # We iterate over the two halves of B (left and right).
    # And for each half, we iterate over the 4 sub-problems.
    
    # Helper to extend a 2x2 vector to 2x4 or 4x4 context
    def extend_u(u2, offset_A):
        # u2 is length 4.
        # A1 is at offset 0, A2 is at offset 4.
        # If offset_A is 0, u is in first half. If 4, in second half.
        u_ext = np.zeros(8)
        u_ext[offset_A:offset_A+4] = u2
        return u_ext

    def extend_v(v2, offset_B):
        # v2 is length 4.
        # B11 is at 0, B12 at 2, B21 at 8, B22 at 10?
        # B indices:
        # 0:0, 1:1, 2:2, 3:3
        # 4:0, 5:1, 6:2, 7:3
        # 8:0, 9:1, 10:2, 11:3
        # 12:0, 13:1, 14:2, 15:3
        # B11: 0,1,4,5 -> indices 0,1,4,5
        # B12: 2,3,6,7 -> indices 2,3,6,7
        # B21: 8,9,12,13 -> indices 8,9,12,13
        # B22: 10,11,14,15 -> indices 10,11,14,15
        
        # We need to map the 4 entries of v2 to the correct positions in the 16-length vector.
        # The positions depend on which block we are in.
        # We pass the base offset and the block type.
        pass # Will implement directly below

    # Let's just explicitly list the mappings for the 4 blocks in one 4x4 B.
    # Block 1: B11. V indices: 0,1,4,5.
    # Block 2: B12. V indices: 2,3,6,7.
    # Block 3: B21. V indices: 8,9,12,13.
    # Block 4: B22. V indices: 10,11,14,15.
    
    # C indices for the 2x4 output:
    # C[:, 0:2] -> indices 0,1,4,5 (Row 0: 0,1; Row 1: 4,5)
    # C[:, 2:4] -> indices 2,3,6,7 (Row 0: 2,3; Row 1: 6,7)
    
    # Wait, C is 2x4.
    # C[i, k].
    # i=0, k=0 -> 0
    # i=0, k=1 -> 1
    # i=0, k=2 -> 2
    # i=0, k=3 -> 3
    # i=1, k=0 -> 4
    # i=1, k=1 -> 5
    # i=1, k=2 -> 6
    # i=1, k=3 -> 7
    
    # For the first 2x2 block of C (cols 0,1):
    # C00=0, C01=1, C10=4, C11=5.
    # For the second 2x2 block of C (cols 2,3):
    # C02=2, C03=3, C12=6, C13=7.
    
    # We will build the 56 terms.
    # Term index t from 0 to 55.
    
    # Structure:
    # For half_h in [0, 1] (Left/Right B)
    #   For sub_block in [0, 1, 2, 3] (B11, B12, B21, B22)
    #     For s_term in [0..6] (Strassen terms)
    #       Construct u, v, w.
    
    # B half offset:
    # Left half: B indices 0..15.
    # Right half: B indices 16..31.
    
    # C offset:
    # Left half: C indices 0..7.
    # Right half: C indices 8..15.
    
    # Let's define the specific index mappings for V and W.
    
    # V mappings for the 4 blocks within a 4x4 B:
    # B11: [0, 1, 4, 5]
    # B12: [2, 3, 6, 7]
    # B21: [8, 9, 12, 13]
    # B22: [10, 11, 14, 15]
    
    # W mappings for the 2x2 C blocks:
    # C_left (cols 0,1): [0, 1, 4, 5]
    # C_right (cols 2,3): [2, 3, 6, 7]
    
    # We iterate.
    t_idx = 0
    for half in range(2):
        B_offset = half * 16
        C_offset = half * 8
        
        # Sub-block 0: B11 (A1*B11 -> C_left)
        # A1 offset: 0
        # V indices: 0,1,4,5 + B_offset
        # W indices: 0,1,4,5 + C_offset
        for s_u, s_v, s_w in S_terms:
            u = np.zeros(8)
            u[0:4] = s_u # A1
            v = np.zeros(32)
            v_indices = [0, 1, 4, 5]
            for i, val in enumerate(s_v):
                v[v_indices[i] + B_offset] = val
            w = np.zeros(16)
            w_indices = [0, 1, 4, 5]
            for i, val in enumerate(s_w):
                w[w_indices[i] + C_offset] = val
            U[:, t_idx] = u
            V[:, t_idx] = v
            W[:, t_idx] = w
            t_idx += 1
            
        # Sub-block 1: B12 (A1*B12 -> C_right)
        # A1 offset: 0
        # V indices: 2,3,6,7 + B_offset
        # W indices: 2,3,6,7 + C_offset
        for s_u, s_v, s_w in S_terms:
            u = np.zeros(8)
            u[0:4] = s_u # A1
            v = np.zeros(32)
            v_indices = [2, 3, 6, 7]
            for i, val in enumerate(s_v):
                v[v_indices[i] + B_offset] = val
            w = np.zeros(16)
            w_indices = [2, 3, 6, 7]
            for i, val in enumerate(s_w):
                w[w_indices[i] + C_offset] = val
            U[:, t_idx] = u
            V[:, t_idx] = v
            W[:, t_idx] = w
            t_idx += 1
            
        # Sub-block 2: B21 (A2*B21 -> C_left)
        # A2 offset: 4
        # V indices: 8,9,12,13 + B_offset
        # W indices: 0,1,4,5 + C_offset
        for s_u, s_v, s_w in S_terms:
            u = np.zeros(8)
            u[4:8] = s_u # A2
            v = np.zeros(32)
            v_indices = [8, 9, 12, 13]
            for i, val in enumerate(s_v):
                v[v_indices[i] + B_offset] = val
            w = np.zeros(16)
            w_indices = [0, 1, 4, 5]
            for i, val in enumerate(s_w):
                w[w_indices[i] + C_offset] = val
            U[:, t_idx] = u
            V[:, t_idx] = v
            W[:, t_idx] = w
            t_idx += 1
            
        # Sub-block 3: B22 (A2*B22 -> C_right)
        # A2 offset: 4
        # V indices: 10,11,14,15 + B_offset
        # W indices: 2,3,6,7 + C_offset
        for s_u, s_v, s_w in S_terms:
            u = np.zeros(8)
            u[4:8] = s_u # A2
            v = np.zeros(32)
            v_indices = [10, 11, 14, 15]
            for i, val in enumerate(s_v):
                v[v_indices[i] + B_offset] = val
            w = np.zeros(16)
            w_indices = [2, 3, 6, 7]
            for i, val in enumerate(s_w):
                w[w_indices[i] + C_offset] = val
            U[:, t_idx] = u
            V[:, t_idx] = v
            W[:, t_idx] = w
            t_idx += 1
            
    # Verify the decomposition
    # T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    # T_true = ...
    # If np.allclose(T_recon, T_true), return.
    # Else fallback to schoolbook.
    
    T_true = np.zeros((8, 32, 16), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m + j
                c = j * p + k
                o = k * n + i
                T_true[r, c, o] = 1.0
                
    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    if not np.allclose(T_recon, T_true, atol=1e-9):
        # Fallback to schoolbook
        rank = 64
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i * m * p + j * p + k
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
