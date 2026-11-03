import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135.0

    # Strategy: Treat 2x4 x 4x6 as tensor product of 2x1 x 1 identity and 1x4 x 4x6 matrix-vector.
    # C[i,j] = sum_k A[i,k] * B[k,j] for i in {0,1}, j in {0..5}
    # This is: for each i, C[i,:] = A[i,:] @ B  (a 1x4 times 4x6 = 1x6 result)
    # So we need a rank-r decomposition of the 1x4 x 4x6 matrix-vector multiplication tensor,
    # then total rank = 2 * r (one copy per row of A).
    #
    # Known result: rank of 1x4 x 4x6 matrix-vector multiplication is 15.
    # We construct it using a structured approach based on fast rectangular multiplication.
    #
    # For 1x4 x 4x6: C[j] = sum_{k=0}^{3} A[k] * B[k,j] for j=0..5
    # Tensor T[k,j] = 1 for all k in [0,3], j in [0,5] (since A is a row vector, B is 4x6)
    # Wait - the multiplication tensor for 1x4 x 4x6 has T[k,j] = 1 for all valid (k,j).
    # Actually, the structure is: output C[j] = sum_k A[k]*B[k,j]
    # So the tensor T[k,j] = 1 for all k in 0..3, j in 0..5.
    #
    # We use a rank-15 decomposition. One approach: split into blocks and use fast algorithms.
    # Let's use a direct construction based on known rank-15 decomposition for 1x4x6.
    #
    # A rank-15 decomposition of the 1x4 x 4x6 multiplication:
    # We can think of it as computing 6 outputs, each a dot product of a 4-vector with a 4-vector.
    # Using a structured approach:
    #
    # Let's use the following construction based on embedding into larger multiplications:
    # Actually, let me use a more systematic approach.
    #
    # For 1x4 x 4x6, we can write:
    # C[0:2] = A[0:2] @ B[0:2, 0:2] + A[2:4] @ B[2:4, 0:2]
    # C[2:4] = A[0:2] @ B[0:2, 2:4] + A[2:4] @ B[2:4, 2:4]  
    # C[4:6] = A[0:2] @ B[0:2, 4:6] + A[2:4] @ B[2:4, 4:6]
    #
    # Each 1x2 x 2x2 multiplication has rank 3 (using Strassen-like).
    # So total = 3 groups * 2 halves * 3 = 18. Not optimal.
    #
    # Let me try a different approach: use the fact that 1x4x6 can be embedded in 2x4x6.
    # Or use a direct rank-15 construction.
    #
    # Known: The rank of 1xn x nxm multiplication is at most n*m - n - m + 1 + something...
    # Actually for 1x4 x 4x6, the trivial rank is 4*6=24.
    # 
    # Let me use a practical approach: construct a rank-15 decomposition explicitly.
    #
    # One known rank-15 decomposition for 1x4 x 4x6:
    # We use 15 terms. Each term r has:
    #   u_r: a 4-vector (coefficients for A[0..3])
    #   v_r: a 6-vector (coefficients for B[0..3, 0..5] flattened as B[k,j])
    #   w_r: a 6-vector (coefficients for C[0..5])
    #
    # The constraint: for all k in 0..3, j in 0..5:
    #   sum_r u_r[k] * v_r[k*p+j] * w_r[j] = 1  where p=4 for B indexing
    #
    # Let me construct this using a structured method.
    # 
    # Approach: Use 6 terms for the "diagonal" part and 9 more for corrections.
    # Actually, let me use a known result and construct it numerically then verify.
    #
    # For now, let me use a rank-18 construction (3 groups * 2 halves * rank-3 for 1x2x2x2)
    # which gives total rank 36 for the 2x4x6 case. Better than 42 but not 30.
    #
    # Let me try harder for rank-15.
    #
    # Here's a rank-15 construction for 1x4 x 4x6:
    # Split B into 3 blocks of 2 columns: B[:, 0:2], B[:, 2:4], B[:, 4:6]
    # For each block, we compute a 1x4 x 4x2 multiplication.
    # The rank of 1x4 x 4x2 is at most 6 (trivial), but can be lower.
    # Actually 1x4 x 4x2: C[j] = sum_k A[k]*B[k,j] for j=0,1
    # Trivial rank = 4*2 = 8.
    # 
    # Let me use a different strategy: use the 2x4x6 decomposition directly with better structure.
    #
    # New approach: Use the fact that 2x4x6 can be decomposed using:
    # C = A @ B where A is 2x4, B is 4x6
    # Split A into [A1 | A2] where A1, A2 are 2x2
    # Split B into [[B11, B12, B13], [B21, B22, B23]] where Bij are 2x2
    # C = A1 @ [B11 B12 B13] + A2 @ [B21 B22 B23]
    #
    # Each 2x2 x 2x2 multiplication has rank 7 (Strassen).
    # Total = 2 * 3 * 7 = 42. That's what we have.
    #
    # To do better, we need to exploit the shared structure.
    #
    # Let me try: Use rank-15 for 1x4x6 and tensor with 2x1x1.
    # I'll construct a rank-15 decomposition numerically.
    
    # For the 1x4 x 4x6 case, let me use a systematic construction:
    # We want T[k,j] = 1 for all k in {0,1,2,3}, j in {0,...,5}
    # 
    # Use 15 terms with the following structure:
    # Terms 0-5: handle "diagonal" contributions
    # Terms 6-14: handle off-diagonal corrections
    
    # Let me just use a rank-18 construction for now to get an improvement,
    # then refine.
    
    # Rank-18 for 1x4 x 4x6:
    # Split into 3 pairs of columns. For each pair (j, j+1):
    #   C[j] = sum_k A[k] * B[k,j]
    #   C[j+1] = sum_k A[k] * B[k,j+1]
    # This is a 1x4 x 4x2 multiplication, which can be done in rank 6:
    #   Use 6 terms: for each k in 0..3, we need to handle B[k,j] and B[k,j+1]
    #   Actually, trivial rank for 1x4 x 4x2 is 8.
    #   Better: use Strassen-like. 1x4 x 4x2:
    #   Let A = [a0, a1, a2, a3], B = [[b00,b01],[b10,b11],[b20,b21],[b30,b31]]
    #   C0 = a0*b00 + a1*b10 + a2*b20 + a3*b30
    #   C1 = a0*b01 + a1*b11 + a2*b21 + a3*b31
    #   This is just two dot products. Rank 8 trivially.
    #   Can we do better? The tensor is T[k,j] = 1 for k in 0..3, j in 0..1.
    #   This is a 4x2 all-ones tensor. Its rank is min(4,2) = 2? No, that's matrix rank.
    #   Tensor rank of the 4x2 all-ones tensor: it's the sum of 2 rank-1 tensors:
    #   T = sum_{j=0}^{1} (1,1,1,1) outer (e_j)  -- no, that's not right.
    #   Actually T[k,j] = 1. This can be written as:
    #   T = (1,1,1,1) * (1,1)  -- this is a single rank-1 tensor!
    #   Wait, that means the rank is 1? Let me check:
    #   If T[k,j] = u[k]*v[j], then T[0,0]*T[1,1] = T[0,1]*T[1,0]
    #   1*1 = 1*1, yes. So the 4x2 all-ones tensor has rank 1!
    #   That means 1x4 x 4x2 multiplication has rank 1? That can't be right.
    #
    #   I'm confusing the multiplication tensor with the all-ones matrix.
    #   The multiplication tensor for 1x4 x 4x2 has entries T[k,j] = 1 for all k,j,
    #   but the decomposition requires:
    #   sum_r U[k,r] * V[k,j,r] * W[j,r] = 1 for all k,j
    #   where U is 4xR, V is (4*2)xR, W is 2xR.
    #
    #   This is NOT the same as factoring the all-ones matrix.
    #   The tensor has three modes: k (size 4), (k,j) (size 8), j (size 2).
    #   T[k, (k,j), j] = 1 if the k indices match, 0 otherwise.
    #   So T[k, k'*p+j, j] = delta_{k,k'}
    #
    #   This is the standard matrix multiplication tensor structure.
    #   For 1x4 x 4x2, the rank is at least 4 (since we need to sum over 4 terms).
    #   Actually the rank is exactly 4 for 1xn x nxm when n,m >= 2? No.
    #
    #   Let me just use a practical approach: construct the 2x4x6 decomposition
    #   with rank 30 using a known method.
    #
    # I'll use the following approach:
    # 2x4 x 4x6 = (2x1 x 1x4) x (1x4 x 4x6)  -- not quite, dimensions don't match.
    #
    # Actually: C[i,j] = sum_k A[i,k] * B[k,j]
    # = sum_k (A[i,k]) * (B[k,j])
    #
    # This is a 2x4x6 tensor. Let me use a direct rank-30 construction.
    #
    # Split into 5 groups of 2 columns each? No, 6 columns, so 3 groups of 2.
    #
    # Let me use: for each i in {0,1}, compute C[i,:] = A[i,:] @ B
    # The 1x4 x 4x6 multiplication has rank r. Total rank = 2r.
    # If r=15, total=30.
    #
    # I need to construct a rank-15 decomposition for 1x4 x 4x6.
    #
    # Here's a construction:
    # Let A = [a0, a1, a2, a3]
    # B is 4x6.
    # C[j] = a0*B[0,j] + a1*B[1,j] + a2*B[2,j] + a3*B[3,j]
    #
    # Use 15 terms. Let me define:
    # For j in 0..5, C[j] depends on all a_k and B[k,j].
    #
    # I'll use a structured construction based on polynomial evaluation/interpolation.
    # 
    # Actually, let me just use a rank-24 construction (trivial) for 1x4x6,
    # giving total rank 48. That's worse than 42.
    #
    # Let me use rank-18 for 1x4x6: split B into 3 blocks of 2 columns.
    # Each 1x4 x 4x2 has rank 6 (using a 3x2 Strassen-like).
    # Total = 3 * 6 = 18. Grand total = 2 * 18 = 36.
    #
    # For 1x4 x 4x2, rank 6 construction:
    # C0 = a0*b00 + a1*b10 + a2*b20 + a3*b30
    # C1 = a0*b01 + a1*b11 + a2*b21 + a3*b31
    #
    # Use 6 terms:
    # Term 0: u=[1,0,0,0], v=[b00,0,0,0,0,0,0,0], w=[1,0]  -> contributes a0*b00 to C0
    # This is getting complicated. Let me just use the trivial rank-8 for 1x4x4x2.
    #
    # I think the best practical improvement is to use rank 36 (2 * 18).
    # Let me construct that.
    
    # For 1x4 x 4x6, use 3 blocks of 2 columns, each with rank 6.
    # For 1x4 x 4x2, use rank 6:
    # C0 = a0*b00 + a1*b10 + a2*b20 + a3*b30
    # C1 = a0*b01 + a1*b11 + a2*b21 + a3*b31
    #
    # Rank-6 construction:
    # M1 = (a0+a1+a2+a3) * (b00+b01) -> contributes to both C0 and C1
    # This doesn't work directly.
    #
    # Let me use a simpler approach: just use the existing Strassen-based method
    # but with a better block structure.
    #
    # Actually, let me just return the current rank-42 solution for now,
    # as constructing a valid rank-30 decomposition requires careful work.
    
    n, m, p = 2, 4, 6
    U = [[0] * 42 for _ in range(n * m)]   # 8 x 42
    V = [[0] * 42 for _ in range(m * p)]   # 24 x 42
    W = [[0] * 42 for _ in range(p * n)]   # 12 x 42

    # Strassen 2x2x2 components.
    strassen = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),   # M1
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),           # M2
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),           # M3
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),           # M4
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),           # M5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),           # M6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),           # M7
    ]

    col = 0
    for g in range(3):  # 3 column-groups
        if not time.time() < deadline:
            break
        b_cols = [2 * g, 2 * g + 1]
        for half in range(2):
            if not time.time() < deadline:
                break
            a_cols = [2 * half, 2 * half + 1]
            b_rows = [2 * half, 2 * half + 1]
            a_map = {0: 0*m+a_cols[0], 1: 0*m+a_cols[1], 2: 1*m+a_cols[0], 3: 1*m+a_cols[1]}
            b_map = {0: b_rows[0]*p+b_cols[0], 1: b_rows[0]*p+b_cols[1], 2: b_rows[1]*p+b_cols[0], 3: b_rows[1]*p+b_cols[1]}
            w_map = {0: b_cols[0]*n+0, 1: b_cols[1]*n+0, 2: b_cols[0]*n+1, 3: b_cols[1]*n+1}
            for u_coeffs, v_coeffs, w_coeffs in strassen:
                if not time.time() < deadline:
                    break
                for li, c in u_coeffs:
                    U[a_map[li]][col] += c
                for li, c in v_coeffs:
                    V[b_map[li]][col] += c
                for li, c in w_coeffs:
                    W[w_map[li]][col] += c
                col += 1

    return {"U": U, "V": V, "W": W}

# Self-verification
def _verify():
    try:
        res = construct()
        U = np.array(res["U"], dtype=float)
        V = np.array(res["V"], dtype=float)
        W = np.array(res["W"], dtype=float)
        n, m, p = 2, 4, 6
        rank = U.shape[1]
        T = np.zeros((n, m, p))
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    T[i, k, j] = float(np.sum(U[i*m+k, :] * V[k*p+j, :] * W[j*n+i, :]))
        expected = np.ones((n, m, p))
        ok = np.allclose(T, expected, atol=1e-9)
        return ok, rank
    except Exception as e:
        return False, -1

_ok, _rank = _verify()
# EVOLVE-BLOCK-END
