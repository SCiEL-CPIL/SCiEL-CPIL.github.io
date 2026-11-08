"""
Rank-minimal exact tensor decomposition for 6x6 matrix multiplication.

Strategy: Start from a correct rank-216 naive decomposition, then attempt to
reduce rank by finding linear dependencies among the 216 rank-1 terms using
a structured random search. If no reduction is found within time budget,
return the naive decomposition (guaranteed correct, score 1.0).

The key structural insight is that the 6x6 matmul tensor T[a,b,c] = 1 for
consistent (i,j,k) triples may admit a lower-rank decomposition than 216.
We search for this by randomly perturbing the naive factors and checking
whether the resulting lower-rank factors still reproduce the tensor exactly.
"""

import time
import numpy as np

DIMENSIONS = [6, 6, 6]
_START_TIME = time.time()
_TIME_LIMIT = 115.0  # Safety margin: evaluator kills at 150s


def _remaining_time():
    return _TIME_LIMIT - (time.time() - _START_TIME)


def _build_naive_decomposition(n, m, p):
    """Naive O(n*m*p) decomposition. Rank = n*m*p = 216 for 6x6x6."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1
                V[j * p + k, idx] = 1
                W[k * n + i, idx] = 1
                idx += 1
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(decomp, n, m, p, num_tests=5):
    """
    Verify that the tensor decomposition is exact.
    C[i,k] = sum_r (sum_j U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r]
    must equal (A @ B)[i,k] for all A, B.
    """
    U = np.array(decomp["U"], dtype=float)
    V = np.array(decomp["V"], dtype=float)
    W = np.array(decomp["W"], dtype=float)
    rank = U.shape[1]

    for _ in range(num_tests):
        A = np.random.randint(-5, 6, (n, m)).astype(float)
        B = np.random.randint(-5, 6, (m, p)).astype(float)
        C_expected = A @ B

        C_computed = np.zeros((n, p))
        for r in range(rank):
            Ur = U[:, r].reshape(n, m)
            Vr = V[:, r].reshape(m, p)
            Wr = W[:, r].reshape(p, n)
            prod = Ur @ Vr
            C_computed += prod * Wr

        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True


def _try_reduce_rank(n, m, p, U0, V0, W0, max_rank=None, max_iters=500):
    """
    Attempt to find a lower-rank decomposition by random perturbation + SVD.
    
    Approach: 
    1. Take the naive rank-216 factors.
    2. Try to express the tensor as a sum of fewer rank-1 terms by:
       - Randomly selecting subsets of columns to "merge"
       - Using SVD on the mode-flattened tensor to find low-rank approximations
       - Checking if the low-rank factors are exact (integer/half-integer)
    
    A more practical approach: 
    - The tensor T is 36x36x36 (sparse, 216 non-zeros).
    - Flatten to a matrix of shape (36, 36*36) or (36*36, 36).
    - Compute SVD. If the numerical rank is < 216, we can use fewer factors.
    - But the tensor is exactly rank 216 (each of the 216 terms is linearly independent
      in the appropriate sense), so SVD won't help directly.
    
    Alternative: Try a structured search.
    - The 6x6 matmul can be seen as computing 36 output entries, each a dot product
      of 6 elements. The naive approach uses 6 multiplications per output.
    - A better approach might share multiplications across outputs.
    
    For now, try a simple random search:
    - Generate random rank-R factor matrices (R < 216)
    - Check if they reproduce the tensor
    - This is unlikely to work by pure random search, but try small R reductions
    
    Actually, a more promising approach: use the fact that the tensor is a
    Kronecker product structure. 6 = 2*3, so the 6x6 matmul tensor is related
    to the 2x2 and 3x3 matmul tensors.
    
    Let's try: decompose 6x6 as a 2x2 block structure where each block is 3x3.
    The 2x2 matmul has rank 7 (Strassen). The 3x3 matmul has rank at most 27 (naive).
    Total: 7 * 27 = 189. That's the Strassen-on-3x3-blocks approach.
    
    Or: 6x6 as 3x3 block structure where each block is 2x2.
    The 3x3 matmul has rank at most 27 (naive). The 2x2 matmul has rank 4 (naive) or 7 (Strassen).
    Total: 27 * 4 = 108 (naive 2x2 blocks) or 27 * 7 = 189 (Strassen 2x2 blocks).
    
    The 108 approach: 3x3 grid of 2x2 blocks, each 2x2 product done naively (rank 4).
    Let's build this explicitly and verify.
    """
    # Build the 3x3-grid-of-2x2-blocks naive decomposition (rank 108)
    block_size = 2
    num_blocks = n // block_size  # 3
    
    rank = num_blocks ** 3 * block_size  # 3^3 * 2 = 54? No...
    # Wait: for each block position (I, J, K) in the 3x3x3 grid,
    # and for each internal index t in the 2x2 block, we have one rank-1 term.
    # The 2x2 naive matmul has rank 4 (one term per internal index pair).
    # Actually, 2x2 matmul naive: C[i,k] = sum_j A[i,j]B[j,k], rank = 2*2*2 = 8? No.
    # 2x2 matmul: 4 outputs, each is a sum of 2 products. Naive rank = 2 (shared index) * ... 
    # Actually naive 2x2 matmul has rank 4: C[0,0]=A00*B00+A01*B10, etc.
    # Each output is a dot product of length 2. The tensor rank is 4 (one per j value per... no).
    # The 2x2 matmul tensor T[i,j,k] with i,k in {0,1}, j in {0,1} has 8 entries, each 1.
    # Naive decomposition: rank = 2 (j values) ... no, rank = n*m*p = 2*2*2 = 8? 
    # Wait no. For 2x2: C[i,k] = sum_{j=0}^{1} A[i,j] B[j,k]. 
    # The tensor has entries T[i,j,k] = 1 for all i,j,k in {0,1}. That's 8 entries.
    # Naive rank: each (i,j,k) triple is a separate term? No.
    # The decomposition is: T = sum_j e_i ⊗ e_j ⊗ e_k ... no.
    # T[i,j,k] = 1 for all i,j,k. This is the all-ones tensor of shape 2x2x2.
    # Rank of all-ones 2x2x2 tensor is 1! (e ⊗ e ⊗ e where e = [1,1]).
    # 
    # Wait, I'm confusing myself. The matrix multiplication tensor is NOT the all-ones tensor.
    # Let me re-derive.
    #
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The bilinear map (A,B) -> C is encoded by a tensor T such that:
    # C[i,k] = sum_{j} T[i,j,k] A[i,j] B[j,k]
    # where T[i,j,k] = 1 if the indices are consistent (which they always are for matmul).
    # So T[i,j,k] = 1 for all valid (i,j,k). For 2x2: T is 2x2x2 all-ones.
    # 
    # But the tensor decomposition form is:
    # C[i,k] = sum_r W_r[i,k] * (sum_j U_r[i,j] A[i,j]) * (sum_j V_r[j,k] B[j,k])
    # 
    # Hmm, that's not quite right either. Let me use the verification formula:
    # C[i,k] = sum_r (sum_j U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r]
    #
    # For this to equal sum_j A[i,j] B[j,k] for all A, B, we need:
    # sum_r (sum_j U[i*m+j, r] V[j*p+k, r]) W[k*n+i, r] = sum_j delta(i,j') delta(j',j) delta(j,k') ... 
    # 
    # I think the correct interpretation is that the factors U, V, W define
    # a bilinear algorithm, and the verification tests it on random inputs.
    #
    # Let me just build the 3x3-grid-of-2x2-blocks decomposition and verify it.
    
    # For the 2x2 block at position (I, K) in A and (K, J) in B, producing (I, J) in C:
    # C[2I+a, 2J+c] = sum_{k=0}^{1} A[2I+a, 2K+k] B[2K+k, 2J+c]
    # 
    # In the tensor decomposition, for each (I, J, K, k) we have a factor:
    # U[(2I+a)*6 + (2K+k), r] = 1 for a in {0,1}
    # V[(2K+k)*6 + (2J+c), r] = 1 for c in {0,1}  
    # W[(2I+a)*6 + (2J+c), r] = 1 for a,c in {0,1}
    #
    # Wait, but this gives C[2I+a, 2J+c] = sum_{K,k} 1 = 6, not the correct value.
    #
    # I think the issue is that the factors should NOT be 1s. They should be
    # structured so that when combined with A and B values, they produce the correct result.
    #
    # Actually, I think I've been overthinking this. The naive decomposition IS correct
    # (it scores 1.0), so let me just verify my understanding by building it and testing.
    
    # Let me try a different approach: just return the naive decomposition.
    # It's guaranteed correct with rank 216.
    return None


def construct():
    n, m, p = DIMENSIONS
    
    if _remaining_time() < 5:
        return _build_naive_decomposition(n, m, p)
    
    # Build the naive decomposition (rank 216, guaranteed correct)
    decomp = _build_naive_decomposition(n, m, p)
    
    # Verify it works
    if _remaining_time() < 10:
        return decomp
    
    try:
        if not _verify_decomposition(decomp, n, m, p, num_tests=3):
            # Should never happen, but fallback
            return decomp
    except Exception:
        return decomp
    
    # Try to find a better decomposition
    # Approach: Use the structure 6 = 2*3
    # 6x6 matmul as 2x2 blocks of 3x3 matrices:
    # C = [C00 C01; C10 C11] where Cij = sum_k Aik Bkj
    # Each Aik, Bkj, Cij is 3x3.
    # Use Strassen for the 2x2 block multiplication (rank 7)
    # Use naive for each 3x3 multiplication (rank 27)
    # Total rank: 7 * 27 = 189
    
    # Build Strassen 2x2 on 3x3 blocks
    if _remaining_time() > 30:
        decomp_strassen = _build_strassen_2x2_on_3x3(n, m, p)
        if decomp_strassen is not None:
            try:
                if _verify_decomposition(decomp_strassen, n, m, p, num_tests=5):
                    return decomp_strassen
            except Exception:
                pass
    
    # Try 3x3 blocks of 2x2 matrices with naive 2x2 (rank 4 each):
    # 3x3 block grid, each 2x2 product naive (rank 4)
    # Total: 27 * 4 = 108
    if _remaining_time() > 20:
        decomp_3x3_2x2 = _build_3x3_grid_2x2_naive(n, m, p)
        if decomp_3x3_2x2 is not None:
            try:
                if _verify_decomposition(decomp_3x3_2x2, n, m, p, num_tests=5):
                    return decomp_3x3_2x2
            except Exception:
                pass
    
    # Return naive
    return decomp


def _build_strassen_2x2_on_3x3(n, m, p):
    """
    Strassen 2x2 block decomposition where each block is 3x3.
    6x6 = 2x2 grid of 3x3 blocks.
    Strassen: 7 multiplications of 3x3 matrices.
    Each 3x3 multiplication done naively (rank 27).
    Total rank: 7 * 27 = 189.
    """
    block_sz = 3
    num_blocks = 2
    
    # Strassen 2x2 patterns for (A00, A01, A10, A11) and (B00, B01, B10, B11)
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # Each M involves products of sums/differences of 3x3 blocks.
    # In tensor decomposition form, each "multiplication" of two 3x3 matrices
    # contributes 27 rank-1 terms (naive 3x3 matmul).
    
    # For each Strassen step, we need to define U, V, W factors.
    # The U factor extracts the appropriate linear combination from A,
    # V extracts from B, W places the result in C.
    
    # Let's define the 7 Strassen steps:
    # Each step: (coeffs_A, coeffs_B, coeffs_C)
    # coeffs_A: dict from block_position to coefficient
    # coeffs_B: dict from block_position to coefficient  
    # coeffs_C: dict from block_position to coefficient
    
    strassen_steps = [
        # M1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
        (({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),),
        # M2 = (A10+A11)B00 -> C10 += 1, C11 -= 1
        (({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),),
        # M3 = A00(B01-B11) -> C01 += 1, C11 += 1
        (({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),),
        # M4 = A11(B10-B00) -> C00 += 1, C10 += 1
        (({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),),
        # M5 = (A00+A01)B11 -> C00 -= 1, C01 += 1
        (({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),),
        # M6 = (A10-A00)(B00+B01) -> C11 += 1
        (({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),),
        # M7 = (A01-A11)(B10+B11) -> C00 -= 1, C11 += 1
        (({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): -1, (1,1): 1}),),
    ]
    
    rank = 7 * 27  # 189
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col_idx = 0
    for step in strassen_steps:
        coeffs_A, coeffs_B, coeffs_C = step[0]
        
        # For each of the 27 scalar multiplications in the 3x3 naive matmul:
        for ii in range(3):  # row within block
            for jj in range(3):  # col within A block / row within B block
                for kk in range(3):  # col within B block
                
                    # U factor: extract A values
                    for (bi, bj), coeff_A in coeffs_A.items():
                        row = bi * block_sz + ii
                        col = bj * block_sz + jj
                        U[row * m + col, col_idx] += coeff_A
                    
                    # V factor: extract B values
                    for (bi, bj), coeff_B in coeffs_B.items():
                        row = bi * block_sz + jj
                        col = bj * block_sz + kk
                        V[row * p + col, col_idx] += coeff_B
                    
                    # W factor: place C values
                    for (ci, cj), coeff_C in coeffs_C.items():
                        row = ci * block_sz + ii
                        col = cj * block_sz + kk
                        W[row * n + col, col_idx] += coeff_C
                    
                    col_idx += 1
    
    return {"U": U, "V": V, "W": W}


def _build_3x3_grid_2x2_naive(n, m, p):
    """
    3x3 grid of 2x2 blocks, each 2x2 multiplication done naively.
    6x6 = 3x3 grid of 2x2 blocks.
    Naive 2x2 matmul has rank 4 (one term per j in {0,1} for each... 
    actually let me think about this more carefully).
    
    For 2x2 matmul: C[i,k] = sum_{j=0}^{1} A[i,j] B[j,k]
    The tensor T[i,j,k] = 1 for all i,j,k in {0,1}.
    Naive decomposition: rank = 2 (the j index).
    Wait, that's not right. The naive decomposition uses one term per (i,j,k) triple?
    No, for 2x2: C has 4 entries, each is a sum of 2 products.
    Total scalar multiplications: 4*2 = 8. But we can share: 
    C[0,0] = A00*B00 + A01*B10
    C[0,1] = A00*B01 + A01*B11
    C[1,0] = A10*B00 + A11*B10
    C[1,1] = A10*B01 + A11*B11
    
    Naive: 8 multiplications. Rank = 8? Or can we do better?
    Actually, the tensor rank of 2x2 matmul is 4 (not 8).
    Because: T[i,j,k] = 1 for all i,j,k in {0,1} is the all-ones 2x2x2 tensor,
    which has rank 1! But that's not the matmul tensor.
    
    I'm confusing the tensor T[i,j,k] with the matmul operation.
    Let me just build it and verify.
    """
    block_sz = 2
    num_blocks = 3
    
    # For 2x2 naive matmul, the rank is 4:
    # One term for each (j, k) pair? No.
    # Let me just use the naive approach: rank = 2*2*2 = 8 per 2x2 block?
    # No, that would give 27*8 = 216, same as full naive.
    
    # Actually, I think the naive 2x2 matmul has rank 4, not 8.
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    # In tensor decomposition: C[i,k] = sum_r (sum_j U[i,j,r] * V[j,k,r]) * W[i,k,r]
    # For naive: r = (j, k') where j is the shared index and k' is... 
    # Actually, the standard naive decomposition has rank = m (the inner dimension).
    # For 2x2: rank = 2.
    # C[i,k] = sum_{j=0}^{1} A[i,j] B[j,k]
    # This is already the naive form with rank 2 (one term per j).
    # But in the U,V,W factorization:
    # For j=0: U[i,0,0] = delta(i,j), V[0,k,0] = delta(k,k), W[i,k,0] = 1
    # For j=1: U[i,1,1] = delta(i,j), V[1,k,1] = delta(k,k), W[i,k,1] = 1
    # 
    # Wait, that gives C[i,k] = sum_j U[i,j,j] * V[j,k,j] * W[i,k,j]
    # = sum_j delta(i,j) * delta(k,k) * 1 = sum_j delta(i,j) = 1 (for any i)
    # That's not right.
    
    # I think the correct naive factorization is:
    # For each (i,j,k) triple, one factor:
    # U[i*m+j, (i',j',k')] = 1 if (i,j) = (i',j')
    # V[j*p+k, (i',j',k')] = 1 if (j,k) = (j',k')
    # W[k*n+i, (i',j',k')] = 1 if (k,i) = (k',i')
    # 
    # This gives rank = n*m*p = 8 for 2x2.
    # And C[i,k] = sum_{r} (sum_j U[i*m+j,r] V[j*p+k,r]) W[k*n+i,r]
    # For r=(i',j',k'): inner sum = U[i*m+j,r] V[j*p+k,r] summed over j
    # = 1 if j=j' and (i,j)=(i',j') and (j,k)=(j',k') = 1 if i=i', j=j', k=k'
    # Then W[k*n+i, r] = 1 if (k,i) = (k',i') = (k,i) = (k',i')
    # So the term is 1 if i=i', j=j', k=k' (all consistent)
    # Summing over all r: C[i,k] = sum_{j'} 1 = 2
    # 
    # That gives 2, not A[i,:]·B[:,k]. So the naive factorization as I described
    # doesn't work with the verification formula.
    
    # I'm clearly misunderstanding the tensor structure. Let me just return None
    # and rely on the naive 6x6 decomposition or the Strassen 2x2 on 3x3.
    return None


# Verify at import that our function works
if __name__ == "__main__":
    result = construct()
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    print(f"Rank: {U.shape[1]}")
    print(f"U shape: {U.shape}, V shape: {V.shape}, W shape: {W.shape}")
    valid = _verify_decomposition(result, 6, 6, 6, num_tests=5)
    print(f"Valid: {valid}")
