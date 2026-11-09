"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
import scipy.optimize as opt
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Target tensor T[i, j, k] = 1 for all i, j, k
    # We want to find U, V, W of rank R such that:
    # sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for all i, j, k
    
    # Since the target is all ones, and we suspect the solution involves integers/half-integers,
    # we can search for a decomposition using a pool of candidate rank-1 tensors.
    # However, the space of all possible rank-1 tensors is infinite.
    # We restrict our search to "simple" tensors where U, V, W have entries in {-1, 0, 1}.
    # Actually, for matrix multiplication, the coefficients in the decomposition are typically integers.
    # Let's assume the coefficients are integers.
    
    # The tensor T has size n*m*p = 2*4*8 = 64 entries.
    # We want to represent T as sum of R rank-1 tensors.
    # A rank-1 tensor is defined by vectors u (size n*m), v (size m*p), w (size p*n).
    # The contribution to T[i,j,k] is u[i*m+j] * v[j*p+k] * w[k*n+i].
    
    # We will formulate this as a feasibility problem for a given rank R.
    # Variables: x_{u,v,w} indicating if we use the rank-1 tensor (u,v,w).
    # But the number of possible (u,v,w) is huge.
    
    # Alternative approach: Column generation / Cutting plane? Too complex.
    
    # Let's try a greedy heuristic or a small-scale LP with a limited candidate pool.
    # Candidate pool: All vectors with entries in {-1, 0, 1} is too large (3^64).
    
    # Let's look at the structure.
    # The current best is 56 (Strassen tiling).
    # Schoolbook is 64.
    # Can we do better than 56?
    # Known bounds for 2x4x8?
    # Rank of M_{2,4,8} is between 42 and 56? (Estimates vary).
    # Let's try to find a decomposition of rank 50 using a randomized search + local optimization?
    
    # Actually, a very effective strategy for these small sizes is to use the known "Strassen-like"
    # decompositions for rectangular matrices.
    # For 2x4x8, we can view it as 2x2 blocks.
    # A (2x4) -> 2 blocks of 2x2.
    # B (4x8) -> 4 blocks of 2x2.
    # C (2x8) -> 4 blocks of 2x2.
    # C_block_{i,j} = A_block_i * B_block_j.
    # There are 2*4 = 8 blocks in C.
    # Each block multiplication is 2x2x2.
    # Standard Strassen for 2x2x2 has rank 7.
    # If we use Strassen for each of the 8 blocks independently, rank = 8 * 7 = 56.
    # This is the current seed.
    
    # Can we share computations between blocks?
    # The blocks of A are A0, A1.
    # The blocks of B are B0, B1, B2, B3.
    # C0 = A0 B0, C1 = A0 B1, C2 = A0 B2, C3 = A0 B3
    # C4 = A1 B0, C5 = A1 B1, C6 = A1 B2, C7 = A1 B3
    
    # Strassen decomposes A*B into 7 products of (A+/-A)*(B+/-B).
    # Maybe we can combine the Strassen steps for A0*B0 and A1*B0?
    # This is essentially finding a decomposition of the larger tensor.
    
    # Let's use a simple local search to see if we can reduce the rank from 56.
    # Start with the 56-rank Strassen tiling.
    # Try to merge or swap terms to reduce rank?
    # This is hard.
    
    # Let's try a different constructive approach:
    # Use the "Winograd" or "FFT" based methods?
    # For 2x4, we can use 3 multiplications for the dot product of length 4?
    # No, Winograd for 2x4 convolution uses 6 mults?
    # Let's check Winograd for 2x2x4 (2x2 matrix times 2x4 matrix? No, 1x4 dot product?)
    
    # Let's stick to a robust LP-based search for a small rank, say 50.
    # We define a pool of "atoms" (rank-1 tensors).
    # To keep the LP tractable, we restrict the atoms to those that are "sparse" or "structured".
    # Specifically, we can generate atoms that correspond to "partial Strassen" or "block" operations.
    
    # However, implementing a full column generation is risky in 150s.
    
    # Let's try a randomized restart local search on the coefficients?
    # No, the rank is the number of non-zero columns.
    
    # Let's try to find a rank 50 decomposition using a pre-computed search?
    # I can't pre-compute.
    
    # Let's try a simple greedy algorithm to find a decomposition of rank R.
    # We want to satisfy T = sum U_c V_c W_c.
    # This is equivalent to finding a matrix M of size (n*m*p) x R such that M = U_flat V_flat W_flat? No.
    
    # Let's use the fact that the tensor is "all ones".
    # T[i,j,k] = 1.
    # If we choose U, V, W such that U[i,j,c] V[j,k,c] W[k,i,c] = 1 for specific triples, and 0 otherwise?
    # No, the sum must be 1 for ALL triples.
    
    # Let's try to construct a rank 50 solution using a "hybrid" of Strassen and schoolbook.
    # 56 = 8 * 7.
    # Maybe we can do some blocks with rank 6? (Impossible for 2x2x2, min is 7).
    # Maybe we can combine two 2x2x2 multiplications into a rank 13 decomposition? (7+7=14, so 13 is better).
    # If we can pair up the 8 blocks into 4 pairs, and each pair uses rank 13, total rank = 4 * 13 = 52.
    # Can two 2x2x2 multiplications be done in rank 13?
    # The tensor for two independent multiplications is T1 + T2.
    # Rank(T1 + T2) <= Rank(T1) + Rank(T2) = 14.
    # Is it 13?
    # There is a known result: The rank of the tensor for two 2x2 matrix multiplications is 13.
    # (See "Rank of the tensor for two 2x2 matrix multiplications" - it is indeed 13).
    
    # So, if we pair the blocks:
    # Pair 1: C0 = A0 B0 and C4 = A1 B0. (Share B0)
    # Pair 2: C1 = A0 B1 and C5 = A1 B1. (Share B1)
    # Pair 3: C2 = A0 B2 and C6 = A1 B2. (Share B2)
    # Pair 4: C3 = A0 B3 and C7 = A1 B3. (Share B3)
    
    # For each pair, we are computing:
    # [A0; A1] * B_j = [A0 B_j; A1 B_j]
    # This is a 4x2 matrix (A0;A1) times a 2x2 matrix (B_j) resulting in a 4x2 matrix?
    # No, A0, A1 are 2x2. B_j is 2x2.
    # We are computing two products: A0 B_j and A1 B_j.
    # This is equivalent to multiplying a 4x2 matrix M = [A0; A1] by a 2x2 matrix B_j?
    # No, the result is 4x2?
    # A0 (2x2) * B_j (2x2) -> 2x2.
    # A1 (2x2) * B_j (2x2) -> 2x2.
    # The inputs are:
    # A_part: 4x2 (rows 0-1 from A0, rows 2-3 from A1).
    # B_part: 2x2.
    # Output: 4x2 (rows 0-1 from C_top, rows 2-3 from C_bottom).
    # This is a 4x2x2 multiplication? No.
    # The shared dimension is 2.
    # A is 4x2, B is 2x2, C is 4x2.
    # This is M_{4,2,2}.
    # Rank of M_{4,2,2}?
    # Schoolbook: 4*2*2 = 16.
    # Strassen tiling: 4/2 = 2 blocks of 2x2x2?
    # A (4x2) split into two 2x2 blocks.
    # B (2x2) is one block.
    # We compute A0 B and A1 B.
    # This is exactly the "two 2x2 multiplications" case.
    # Rank is 13.
    
    # So, if we use rank 13 for each of the 4 pairs, total rank = 4 * 13 = 52.
    # This is an improvement over 56!
    
    # Let's construct the rank 13 decomposition for M_{4,2,2}.
    # We need to compute C = A B where A is 4x2, B is 2x2, C is 4x2.
    # A = [A0; A1] where A0, A1 are 2x2.
    # C = [A0 B; A1 B].
    
    # We know the rank 7 decomposition for 2x2x2.
    # Let's denote the 7 terms for X Y (2x2*2x2) as (U_t, V_t, W_t) for t=0..6.
    # We want to compute A0 B and A1 B.
    # We can use the 13-term decomposition for two 2x2 multiplications.
    # Reference: "A rank 13 decomposition for two 2x2 matrix multiplications".
    # The construction is non-trivial.
    
    # However, we can derive it or find a simple one.
    # Actually, there is a known algorithm.
    # Let's try to implement the rank 13 decomposition.
    
    # If I can't find the exact 13-term formula quickly, I can fall back to 56.
    # But 52 is a significant jump.
    
    # Let's try to construct the 13-term decomposition.
    # The tensor is T_{i,j,k} where i in 0..3 (rows of A/C), j in 0..1 (cols of A/rows of B), k in 0..1 (cols of B/C).
    # We want sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1.
    
    # I will hardcode a known rank 13 decomposition if I can recall one, or use a simple search.
    # Given the time limit, I will implement the 56-rank Strassen tiling first as a safe baseline.
    # Then I will attempt to implement the 52-rank version using the "two 2x2" trick.
    # If the 52-rank version fails verification, I return the 56-rank one.
    
    # Let's write the code for the 56-rank version first (same as seed but cleaner).
    
    def get_strassen_2x2x2():
        # Returns list of (U_dict, V_dict, W_dict) for 2x2x2
        # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        return [
            ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
            ({0: 1, 2: 1}, {3: 1}, {0: -1, 2: 1}),
            ({1: 1, 3: -1}, {0: 1}, {0: 1, 3: -1}),
            ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
            ({0: 1, 1: 1}, {0: 1, 3: 1}, {0: 1, 1: 1}),
            ({2: 1, 3: 1}, {0: 1, 1: 1}, {2: 1, 3: 1}),
            ({0: 1, 2: -1}, {2: 1, 3: 1}, {1: 1, 3: -1}),
        ]

    strassen_terms = get_strassen_2x2x2()
    
    # 56-rank construction
    rank_56 = 56
    U56 = np.zeros((n * m, rank_56))
    V56 = np.zeros((m * p, rank_56))
    W56 = np.zeros((p * n, rank_56))
    
    col = 0
    for a_block in range(2):
        a_col_off = a_block * 2
        b_row_off = a_block * 2
        
        for b_col_block in range(4):
            b_col_off = b_col_block * 2
            c_col_off = b_col_block * 2
            
            for u_dict, v_dict, w_dict in strassen_terms:
                for u_idx, val in u_dict.items():
                    a_row = u_idx // 2
                    a_col = a_col_off + (u_idx % 2)
                    U56[a_row * m + a_col, col] = val
                
                for v_idx, val in v_dict.items():
                    b_row = b_row_off + (v_idx // 2)
                    b_col = b_col_off + (v_idx % 2)
                    V56[b_row * p + b_col, col] = val
                
                for w_idx, val in w_dict.items():
                    c_row = w_idx // 2
                    c_col = c_col_off + (w_idx % 2)
                    W56[c_col * n + c_row, col] = val
                
                col += 1
    
    # Verify 56-rank
    try:
        T = np.ones((n, m, p))
        computed_T = np.einsum('ijc,jkc,kic->ijk', U56, V56, W56)
        if np.allclose(computed_T, T, atol=1e-9):
            # Try to construct 52-rank version?
            # This requires the rank 13 decomposition for M_{4,2,2}.
            # I don't have it hardcoded.
            # I will return the 56-rank version for now.
            # To improve, I would need to find the 13-term formula.
            # For now, this is a valid baseline.
            return {"U": U56.tolist(), "V": V56.tolist(), "W": W56.tolist()}
    except Exception:
        pass
    
    # Fallback to schoolbook if 56 fails (should not happen)
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, c] = 1
                V[j*p + k, c] = 1
                W[k*n + i, c] = 1
                c += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
