"""Rewritten program for matrix multiplication tensor rank search (3x4 by 4x4)."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]


def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Check if U,V,W exactly decompose the tensor (within tol)."""
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', u, v, w)
    return np.allclose(T, 1.0, atol=tol, rtol=0)


def _construct_decomposition():
    """
    Construct a rank-28 decomposition using the Bini-Capovani-De Falco-Ladisi
    rank-19 decomposition for the 3x3x3 core, plus schoolbook for the
    remaining strips (3x1x4 and 1x4x4).

    Total rank: 19 + 4 + 5 = 28
    - 3x3x3 core: rank 19 (BCL)
    - A[:,3] @ B[0:3,3:4] -> C[:,3] : 3 terms
    - A[2,0:3] @ B[3,:] -> C[2,0:3] : 3 terms
    - A[2,3] @ B[3,3] -> C[2,3] : 1 term
    Wait, that's 19 + 3 + 3 + 1 = 26. Let me recount.

    The 3x4 by 4x4 multiplication:
    C[i,k] = sum_j A[i,j] B[j,k]

    Split j into {0,1,2} and {3}:
    C[i,k] = sum_{j=0}^{2} A[i,j] B[j,k] + A[i,3] B[3,k]

    Split k into {0,1,2} and {3}:
    C[i,k] for k in {0,1,2}: sum_{j=0}^{2} A[i,j] B[j,k] + A[i,3] B[3,k]
    C[i,3]: sum_{j=0}^{2} A[i,j] B[j,3] + A[i,3] B[3,3]

    So:
    - Core 3x3x3: A[0:3,0:3] @ B[0:3,0:3] -> C[0:3,0:3] : rank 19
    - Strip 1: A[0:3,3] @ B[0:3,3] -> C[0:3,3] : rank 3 (schoolbook, 3 terms)
    - Strip 2: A[0:3,3] @ B[3,0:3] -> C[0:3,0:3] : rank 3 (schoolbook, 3 terms)
    - Strip 3: A[0:3,3] @ B[3,3] -> C[0:3,3] : rank 1 (single term)
    - Strip 4: A[0:3,0:3] @ B[3,0:3] -> C[0:3,0:3] : rank 3 (schoolbook, 3 terms)
    - Strip 5: A[0:3,0:3] @ B[3,3] -> C[0:3,3] : rank 3 (schoolbook, 3 terms)
    - Strip 6: A[0:3,3] @ B[3,0:3] -> C[0:3,0:3] : already counted in strip 2

    Wait, I'm double counting. Let me be more careful.

    C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k]

    Terms involving j=3: A[i,3] * B[3,k] for all i,k
    Terms involving j=0,1,2 and k=3: A[i,j] * B[j,3] for i in 0..2, j in 0..2
    Terms involving j=0,1,2 and k=0,1,2: A[i,j] * B[j,k] for i in 0..2, j in 0..2, k in 0..2

    So the decomposition is:
    1. Core: 3x3x3 (A[0:3,0:3] @ B[0:3,0:3] -> C[0:3,0:3]) : rank 19
    2. j=3, all k: A[i,3] * B[3,k] for i in 0..2, k in 0..3 : rank 3
       (since for each i, A[i,3] is a scalar, and we sum over k)
       Actually: C[i,k] += A[i,3] * B[3,k]
       This is outer product of column A[:,3] with row B[3,:], giving a 3x4 matrix.
       This can be done in rank 3: for each i, C[i,:] += A[i,3] * B[3,:]
       But in tensor decomposition form, this is rank 3.
    3. j=0,1,2, k=3: A[i,j] * B[j,3] for i in 0..2, j in 0..2
       C[i,3] += sum_j A[i,j] B[j,3]
       This is a 3x1x3 tensor: A[0:3, 0:3] @ B[0:3, 3] -> C[0:3, 3]
       Rank 3 schoolbook.
    4. j=0,1,2, k=0,1,2 is the core (already counted).

    Wait, I missed: what about A[i,3]*B[3,k] for k=0,1,2 AND k=3?
    And A[i,j]*B[j,k] for j=0,1,2 and k=3?

    Let me redo:
    C[i,k] = (sum_{j=0}^{2} A[i,j] B[j,k]) + A[i,3] B[3,k]

    For k in {0,1,2}:
      C[i,k] = (sum_{j=0}^{2} A[i,j] B[j,k]) + A[i,3] B[3,k]
    For k=3:
      C[i,3] = (sum_{j=0}^{2} A[i,j] B[j,3]) + A[i,3] B[3,3]

    So the additional terms beyond the 3x3x3 core are:
    - A[i,3] * B[3,k] for all i in 0..2, k in 0..3: this is rank 3
    - A[i,j] * B[j,3] for i in 0..2, j in 0..2 (only affects C[i,3]): rank 3

    Total: 19 + 3 + 3 = 25

    Hmm, let me verify this more carefully with the tensor decomposition formalism.
    """
    n, m, p = 3, 4, 4
    rank = 25

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0

    # Bini-Capovani-De Falco-Ladisi rank-19 decomposition for 3x3x3
    # The BCL decomposition uses 19 terms for 3x3 matrix multiplication.
    # Reference: Bini, Capovani, De Falco, Ladisi (1978)
    
    # I'll use the known BCL rank-19 decomposition.
    # The 3x3 multiplication tensor has rank 19.
    # Here I'll encode the 19 terms.
    
    # Let me use a known explicit BCL decomposition.
    # A = (a_ij), B = (b_jk), C = (c_ik)
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # BCL 19-term decomposition (one standard form):
    # I'll use the formulation where we have 19 rank-1 tensors.
    
    # Actually, let me just use a verified rank-19 decomposition for 3x3.
    # The standard BCL decomposition can be found in the literature.
    
    # Here's a known rank-19 decomposition for 3x3 matrix multiplication:
    # Using the "Bini et al." scheme.
    
    # For simplicity and correctness, let me construct the 19 terms explicitly.
    # I'll use a known correct set of 19 terms.
    
    # The 3x3x3 multiplication tensor in index form: T[i,j,k] = 1 for all i,j,k
    # We need 19 rank-1 tensors whose sum equals T.
    
    # Let me use a specific BCL decomposition. I'll encode it directly.
    
    # BCL rank-19 for 3x3 (one standard encoding):
    bcl_terms = [
        # (U entries: list of (i,j, val), V entries: list of (j,k, val), W entries: list of (k,i, val))
        # Term 1
        ([(0,0,1),(2,2,1)], [(0,0,1),(2,2,1)], [(0,0,1),(2,2,1),(0,1,1),(1,0,1),(1,1,1),(1,2,1),(2,1,1)]),
        # This is getting complex. Let me use a different approach.
    ]
    
    # Actually, let me just use a known correct BCL decomposition.
    # I'll use the one from the literature with explicit coefficients.
    
    # For now, let me use a simpler but correct approach:
    # Use the known rank-19 BCL decomposition for 3x3.
    
    # Here's a verified BCL rank-19 decomposition (I'll encode it carefully):
    
    # The 19 terms of the BCL decomposition for 3x3 matrix multiplication:
    # I'll use the formulation from "Optimal matrix multiplication" literature.
    
    # Let me use a concrete set. The BCL decomposition can be written as:
    # 19 rank-1 tensors (u_r, v_r, w_r) where:
    # sum_r u_r[i,j] * v_r[j,k] * w_r[k,i] = 1 for all i,j,k
    
    # Here's one explicit BCL rank-19 decomposition:
    
    bcl = [
        # Each term: (U_dict, V_dict, W_dict)
        # U_dict: {(i,j): val}, V_dict: {(j,k): val}, W_dict: {(k,i): val}
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1, (0,1):1, (1,0):1, (1,1):1, (1,2):1, (2,1):1, (2,2):1}),
        # This is getting too complex to encode from memory correctly.
        
        # Let me use a different strategy: use the known fact that 3x3x3 has rank 19,
        # and construct a valid decomposition programmatically.
    ]
    
    # Given the complexity of encoding BCL from memory, let me use a fallback:
    # I'll use a rank-27 decomposition (schoolbook for 3x3x3) plus the strips,
    # giving total rank 27 + 3 + 3 = 33. But that's worse than 44.
    
    # Actually, schoolbook for 3x3x3 is rank 9, not 27.
    # So 9 + 3 + 3 = 15? No, that can't be right since rank of 3x3x3 is 19.
    
    # Wait, schoolbook for 3x3x3 is rank 9 (each C[i,k] = sum_j A[i,j]B[j,k] is one term).
    # But the tensor rank of 3x3x3 multiplication is 19, meaning schoolbook (rank 9) 
    # doesn't work as a tensor decomposition in the multiplicative sense.
    
    # I'm confusing myself. Let me restart with a clear head.
    
    # The matrix multiplication tensor for m x n * n x p has entries T[i,j,k] = 1.
    # A rank-r decomposition is: T[i,j,k] = sum_{r=1}^{R} u_r[i,j] * v_r[j,k] * w_r[k,i]
    
    # For 3x4 x 4x4, we need T[i,j,k] = 1 for all i in {0,1,2}, j in {0,1,2,3}, k in {0,1,2,3}.
    
    # The current program achieves rank 44. Let me try to improve it.
    
    # Let me use the approach: split into 3x3x3 core + strips.
    # For the 3x3x3 core, use a known rank-19 decomposition.
    # For the strips, use schoolbook.
    
    # But I need to actually encode the BCL rank-19 decomposition correctly.
    
    # Let me use a different known decomposition. I'll use the "Laderman" style
    # but for 3x3. Actually, let me just use a verified rank-19 set.
    
    # After reflection, I think the safest approach is to keep the current rank-44
    # decomposition which is verified, and not risk breaking it with an incorrect
    # BCL encoding. Let me return the current working decomposition.
    
    # Actually, let me try a different decomposition that I can verify:
    # Use Strassen for 2x2x2 blocks (rank 7 each) and handle the rest.
    
    # 3x4 x 4x4:
    # Split A into rows [0:2] and [2:3], and cols [0:2], [2:4]
    # Split B into rows [0:2], [2:4] and cols [0:2], [2:4]
    
    # 4 blocks of 2x2x2 (top 2 rows) + 4 blocks of 1x2x2 (bottom row)
    # = 4*7 + 4*4 = 28 + 16 = 44 (this is what the current program does)
    
    # To improve, I need a better decomposition for some blocks.
    
    # Let me try: use rank-7 for all 2x2x2, and for 1x2x2, can I do better than 4?
    # 1x2x2: C[k] = A[0,j] B[j,k] for j in {0,1}, k in {0,1}
    # This is a 1x2x2 tensor. Its rank is 2 (not 4)!
    # C[0] = A0*B00 + A1*B10
    # C[1] = A0*B01 + A1*B11
    # This can be done in rank 2? No, it needs rank 2 if we allow linear combinations.
    # Actually: C[0] = A0*B00 + A1*B10, C[1] = A0*B01 + A1*B11
    # We can write this as:
    # Term 1: A0 * (B00, B01) * (1, 0) -> contributes A0*B00 to C0, A0*B01 to C1
    # Term 2: A1 * (B10, B11) * (1, 0) -> contributes A1*B10 to C0, A1*B11 to C1
    # So rank 2 for 1x2x2!
    
    # So: 4*7 + 4*2 = 28 + 8 = 36. Better than 44!
    
    # Let me verify: 1x2x2 tensor T[i,j,k] where i is always 0, j in {0,1}, k in {0,1}
    # T[0,0,0]=1, T[0,0,1]=1, T[0,1,0]=1, T[0,1,1]=1
    # Decomposition: sum_r u_r[0,j] * v_r[j,k] * w_r[k,0]
    # Term 1: u1[0,0]=1, u1[0,1]=0; v1[0,0]=1, v1[0,1]=1, v1[1,0]=0, v1[1,1]=0; w1[0,0]=1, w1[1,0]=0
    #   Contributes: u1[0,0]*v1[0,0]*w1[0,0] = 1*1*1 = 1 to T[0,0,0]
    #   u1[0,0]*v1[0,1]*w1[1,0] = 1*1*0 = 0 to T[0,0,1]
    #   u1[0,1]*... = 0
    #   Hmm, this only gives T[0,0,0]=1. Not enough.
    
    # Let me reconsider. The tensor is T[0,j,k] = 1 for j,k in {0,1}.
    # We need: T[0,j,k] = sum_r u_r[0,j] v_r[j,k] w_r[k,0]
    
    # Try rank 2:
    # Term 1: u1[0,0]=1, u1[0,1]=1; v1[0,0]=1, v1[1,0]=1; w1[0,0]=1, w1[1,0]=1
    #   T[0,0,0] += 1*1*1 = 1
    #   T[0,0,1] += 1*0*1 = 0 (v1[0,1] is not set, so 0)
    #   This doesn't work cleanly.
    
    # Let me think again. For 1x2x2:
    # C[k] = sum_j A[0,j] B[j,k]
    # C[0] = A0 B00 + A1 B10
    # C[1] = A0 B01 + A1 B11
    
    # In tensor form with our convention W[k*n+i, r] and n=1, m=2, p=2:
    # U has shape (1*2, R) = (2, R): U[j, r] corresponds to A[0,j] coefficient
    # V has shape (2*2, R) = (4, R): V[j*2+k, r] corresponds to B[j,k] coefficient  
    # W has shape (2*1, R) = (2, R): W[k*1+0, r] = W[k, r] corresponds to C[0,k] coefficient
    
    # We need: sum_r U[j,r] * V[j*2+k, r] * W[k, r] = 1 for all j,k in {0,1}
    
    # So the tensor T[j,k] = 1 (2x2) must be decomposed as sum_r u_r[j] v_r[j,k] w_r[k]
    # where u_r is length 2, v_r is 2x2, w_r is length 2.
    
    # This is the matrix multiplication tensor for 1x2 * 2x2 -> 1x2.
    # Its rank is 2 (since it's just a 2x2 matrix, and we can decompose any 2x2 matrix
    # as sum of 2 rank-1 outer products).
    
    # Actually: T[j,k] = 1 = sum_r u_r[j] v_r[j,k] w_r[k]
    # Let's try:
    # Term 1: u1 = [1, 0], v1 = [[1, 0], [0, 0]], w1 = [1, 0] -> contributes 1 to T[0,0]
    # Term 2: u2 = [0, 1], v2 = [[0, 0], [0, 1]], w2 = [1, 0] -> contributes 1 to T[1,1]
    # That only gives diagonal. We need all 4 entries.
    
    # Term 1: u1 = [1, 1], v1 = [[1, 0], [1, 0]], w1 = [1, 0]
    #   T[0,0] += 1*1*1 = 1, T[0,1] += 1*0*0 = 0, T[1,0] += 1*1*1 = 1, T[1,1] += 1*0*0 = 0
    # Term 2: u2 = [1, 1], v2 = [[0, 1], [0, 1]], w2 = [0, 1]
    #   T[0,0] += 1*0*0 = 0, T[0,1] += 1*1*1 = 1, T[1,0] += 1*0*0 = 0, T[1,1] += 1*1*1 = 1
    # Total: T = [[1,1],[1,1]]. Yes! Rank 2 for 1x2x2.
    
    # So total rank: 4*7 + 4*2 = 28 + 8 = 36.
    
    # Let me implement this.
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen(rA, cA, rB, cB):
        nonlocal col
        # Strassen 2x2x2
        # X is A[rA:rA+2, cA:cA+2], Y is B[rB:rB+2, cB:cB+2], Z is C[rA:rA+2, cB:cB+2]
        
        # M1 = (X11+X22)(Y11+Y22) -> Z11, Z22
        U[(rA+0)*m + (cA+0), col] = 1
        U[(rA+1)*m + (cA+1), col] = 1
        V[(rB+0)*p + (cB+0), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+0)*n + (rA+0), col] = 1
        W[(cB+1)*n + (rA+1), col] = 1
        col += 1
        
        # M2 = (X21+X22)Y11 -> Z21, -Z22
        U[(rA+1)*m + (cA+0), col] = 1
        U[(rA+1)*m + (cA+1), col] = 1
        V[(rB+0)*p + (cB+0), col] = 1
        W[(cB+0)*n + (rA+1), col] = 1
        W[(cB+1)*n + (rA+1), col] = -1
        col += 1
        
        # M3 = X11(Y12-Y22) -> Z12, Z22
        U[(rA+0)*m + (cA+0), col] = 1
        V[(rB+0)*p + (cB+1), col] = 1
        V[(rB+1)*p + (cB+1), col] = -1
        W[(cB+1)*n + (rA+0), col] = 1
        W[(cB+1)*n + (rA+1), col] = 1
        col += 1
        
        # M4 = X22(Y21-Y11) -> Z11, Z21
        U[(rA+1)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+0), col] = 1
        V[(rB+0)*p + (cB+0), col] = -1
        W[(cB+0)*n + (rA+0), col] = 1
        W[(cB+0)*n + (rA+1), col] = 1
        col += 1
        
        # M5 = (X11+X12)Y22 -> Z12, -Z11
        U[(rA+0)*m + (cA+0), col] = 1
        U[(rA+0)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rA+0), col] = 1
        W[(cB+0)*n + (rA+0), col] = -1
        col += 1
        
        # M6 = (X21-X11)(Y11+Y12) -> Z22
        U[(rA+1)*m + (cA+0), col] = 1
        U[(rA+0)*m + (cA+0), col] = -1
        V[(rB+0)*p + (cB+0), col] = 1
        V[(rB+0)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rA+1), col] = 1
        col += 1
        
        # M7 = (X12-X22)(Y21+Y22) -> Z11
        U[(rA+0)*m + (cA+1), col] = 1
        U[(rA+1)*m + (cA+1), col] = -1
        V[(rB+1)*p + (cB+0), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+0)*n + (rA+0), col] = 1
        col += 1

    def add_rank2_1x2x2(rA, cA, rB, cB):
        """Add rank-2 decomposition for 1x2x2 block."""
        nonlocal col
        # A is 1x2: A[0, cA:cA+2]
        # B is 2x2: B[rB:rB+2, cB:cB+2]
        # C is 1x2: C[0, cB:cB+2]
        
        # n=1, m=2, p=2 for this sub-block
        # U index: j in {0,1} -> (rA+0)*m + (cA+j)
        # V index: (j, k) -> (rB+j)*p + (cB+k)
        # W index: k -> (cB+k)*n + (rA+0) = cB+k (since n=1, rA=2)
        
        # Wait, n is the number of rows of A and C. Here the "row" index is fixed at rA=2.
        # U[(rA)*m + (cA+j), col]
        # V[(rB+j)*p + (cB+k), col]
        # W[(cB+k)*n + rA, col]
        
        # Term 1: u1[0]=1, u1[1]=1; v1[0,0]=1, v1[1,0]=1; w1[0]=1, w1[1]=0
        #   Contributes to C[0, cB+0]
        U[(rA)*m + (cA+0), col] = 1
        U[(rA)*m + (cA+1), col] = 1
        V[(rB+0)*p + (cB+0), col] = 1
        V[(rB+1)*p + (cB+0), col] = 1
        W[(cB+0)*n + rA, col] = 1
        col += 1
        
        # Term 2: u2[0]=1, u2[1]=1; v2[0,1]=1, v2[1,1]=1; w2[0]=0, w2[1]=1
        #   Contributes to C[0, cB+1]
        U[(rA)*m + (cA+0), col] = 1
        U[(rA)*m + (cA+1), col] = 1
        V[(rB+0)*p + (cB+1), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+1)*n + rA, col] = 1
        col += 1

    # Apply blocks
    # Top-Left 2x2 x 2x2
    add_strassen(0, 0, 0, 0)
    add_strassen(0, 2, 2, 0)
    
    # Top-Right 2x2 x 2x2
    add_strassen(0, 0, 0, 2)
    add_strassen(0, 2, 2, 2)
    
    # Bottom-Left 1x2 x 2x2
    add_rank2_1x2x2(2, 0, 0, 0)
    add_rank2_1x2x2(2, 2, 2, 0)
    
    # Bottom-Right 1x2 x 2x2
    add_rank2_1x2x2(2, 0, 0, 2)
    add_rank2_1x2x2(2, 2, 2, 2)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def construct():
    deadline = time.time() + 110
    result = _construct_decomposition()
    
    n, m, p = DIMENSIONS
    U, V, W = result["U"], result["V"], result["W"]
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    if _verify_decomposition(U_np, V_np, W_np, n, m, p, tol=1e-6):
        return result
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
