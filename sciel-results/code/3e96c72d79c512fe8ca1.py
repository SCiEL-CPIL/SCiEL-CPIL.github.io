"""Matrix multiplication 2,4,6 — Strassen-based block decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]
_START_TIME = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    """
    Build a rank-42 decomposition of the 2x4 * 4x6 matrix multiplication tensor
    using Strassen's algorithm on 2x2 blocks.

    Strategy:
      - Split the 4x4 inner product space into two 2x2 blocks.
      - Each 2x2 block multiplication (A[i:][:2] * B[j:j+2]) uses Strassen's 7 multiplications.
      - Two blocks -> 14 multiplications per (i, k) pair... wait, let me think more carefully.

    Actually, let's think about the structure:
      A is 2x4, B is 4x6.
      Split A into [A1 A2] where A1, A2 are 2x2.
      Split B into [B1; B2] where B1, B2 are 2x6.
      C = A1*B1 + A2*B2.

    Each A1*B1 is a 2x2 * 2x6 multiplication.
    Each A2*B2 is a 2x2 * 2x6 multiplication.

    For a 2x2 * 2x6 multiplication, we can use Strassen on the 2x2 left factor:
      Split B into two 2x3 blocks: B = [B_left, B_right].
      A*B = A * [B_left, B_right] = [A*B_left, A*B_right].
      Each A*B_left is 2x2 * 2x3.

    For 2x2 * 2x3: split the 2x3 into two 2x1.5... that's not clean.

    Alternative approach: Use the known result that 2x2 * 2x2 has rank 7 (Strassen).
    
    Let me reconsider. The cleanest approach:
      A (2x4) = [A1, A2] with A1, A2 in R^{2x2}
      B (4x6) = [B1; B2] with B1, B2 in R^{2x6}
      C = A1*B1 + A2*B2

    Now A1*B1 is 2x2 * 2x6. We can view this as:
      Split B1 into three 2x2 blocks: B1 = [B1a, B1b, B1c]
      A1*B1 = [A1*B1a, A1*B1b, A1*B1c]
      Each A1*B1a is 2x2*2x2, which has rank 7 by Strassen.
      So A1*B1 has rank 3*7 = 21? No, that's not right either because the multiplications are independent.

    Wait, actually each 2x2*2x2 multiplication is independent, so yes, 3 blocks of 2x2*2x2 each need 7 mults = 21 mults for A1*B1.
    Similarly A2*B2 needs 21 mults.
    Total: 42 mults. That gives rank 42.

    But can we do better? The sum of two independent tensors has rank at most the sum of their ranks, and sometimes less. But 42 is a solid improvement over 48.

    Let me implement the 42-rank decomposition:
      For each of the two 2x2 left blocks (A1, A2) and each of the three 2x2 right sub-blocks of the corresponding 2x6 block, we apply Strassen.

    Strassen for 2x2*2x2:
      Given M (2x2) and N (2x2):
      M = [[m00, m01], [m10, m11]]
      N = [[n00, n01], [n10, n11]]
      
      p1 = (m00 + m11) * (n00 + n11)
      p2 = (m10 + m11) * n00
      p3 = m00 * (n01 - n11)
      p4 = m11 * (n10 - n00)
      p5 = (m00 + m01) * n11
      p6 = (m10 - m00) * (n00 + n01)
      p7 = (m01 - m11) * (n10 + n11)

      Result:
      [[p1 + p4 - p5 + p7, p3 + p5],
       [p2 + p4,            p1 - p2 + p3 + p6]]

    Now I need to build U, V, W matrices.

    The tensor for matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p:
    T[i, k, j] = 1 if the entry C[i,k] = sum_j A[i,j]*B[j,k]

    The decomposition says: T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank) [using W[k*n+i, r] convention]

    For each Strassen multiplication p_r = (linear combo of A entries) * (linear combo of B entries),
    we add a rank-1 term to U, V, W.

    Let me index carefully:
    A is 2x4: indices (i, j) where i in {0,1}, j in {0,1,2,3}
    B is 4x6: indices (j, k) where j in {0,1,2,3}, k in {0,1,2,3,4,5}
    C is 2x6: indices (i, k)

    Split: A1 = A[:, 0:2], A2 = A[:, 2:4]
           B1 = B[0:2, :], B2 = B[2:4, :]

    For A1*B1:
      A1 is 2x2 with entries A1[i, j] = A[i, j] for j in {0,1}
      B1 is 2x6 with entries B1[j, k] = B[j, k] for j in {0,1}
      Split B1 into B1a = B[0:2, 0:2], B1b = B[0:2, 2:4], B1c = B[0:2, 4:6]

    For A2*B2:
      A2 is 2x2 with entries A2[i, j] = A[i, j+2] for j in {0,1}
      B2 is 2x6 with entries B2[j, k] = B[j+2, k] for j in {0,1}
      Split B2 into B2a = B[2:4, 0:2], B2b = B[2:4, 2:4], B2c = B[2:4, 4:6]

    For each 2x2 * 2x2 multiplication, Strassen gives 7 scalar multiplications.
    Total: 2 (left blocks) * 3 (right sub-blocks) * 7 = 42.

    Let me build the factor matrices.
    """
    n, m, p = 2, 4, 6
    rank = 42
    U = [[0.0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0.0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0.0] * rank for _ in range(p * n)]  # 12 x 42

    def strassen_coeffs():
        """
        Return the 7 Strassen linear combinations for a 2x2 * 2x2 multiplication.
        For M (2x2) and N (2x2):
        M indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        N indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        
        p1 = (m00 + m11) * (n00 + n11)
        p2 = (m10 + m11) * n00
        p3 = m00 * (n01 - n11)
        p4 = m11 * (n10 - n00)
        p5 = (m00 + m01) * n11
        p6 = (m10 - m00) * (n00 + n01)
        p7 = (m01 - m11) * (n10 + n11)

        Each returns (U_coeffs, V_coeffs) where U_coeffs is a dict {A_local_idx: coeff}
        and V_coeffs is a dict {B_local_idx: coeff}.
        """
        # A local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        # B local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        
        p1_U = {0: 1, 3: 1}
        p1_V = {0: 1, 3: 1}
        
        p2_U = {2: 1, 3: 1}
        p2_V = {0: 1}
        
        p3_U = {0: 1}
        p3_V = {1: 1, 3: -1}
        
        p4_U = {3: 1}
        p4_V = {2: 1, 0: -1}
        
        p5_U = {0: 1, 1: 1}
        p5_V = {3: 1}
        
        p6_U = {2: 1, 0: -1}
        p6_V = {0: 1, 1: 1}
        
        p7_U = {1: 1, 3: -1}
        p7_V = {2: 1, 3: 1}
        
        return [(p1_U, p1_V), (p2_U, p2_V), (p3_U, p3_V), (p4_U, p4_V),
                (p5_U, p5_V), (p6_U, p6_V), (p7_U, p7_V)]

    def strassen_result_coeffs():
        """
        For each of the 7 products, return the coefficients for the 4 output entries.
        Output indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        
        C[0,0] = p1 + p4 - p5 + p7
        C[0,1] = p3 + p5
        C[1,0] = p2 + p4
        C[1,1] = p1 - p2 + p3 + p6
        """
        # For each product r (0-indexed), and each output entry s, the coefficient
        res = [[0]*7 for _ in range(4)]  # res[output_entry][product_idx] = coeff
        # C[0,0] = p1 + p4 - p5 + p7
        res[0][0] = 1   # p1
        res[0][3] = 1   # p4
        res[0][4] = -1  # p5
        res[0][6] = 1   # p7
        # C[0,1] = p3 + p5
        res[1][2] = 1   # p3
        res[1][4] = 1   # p5
        # C[1,0] = p2 + p4
        res[2][1] = 1   # p2
        res[2][3] = 1   # p4
        # C[1,1] = p1 - p2 + p3 + p6
        res[3][0] = 1   # p1
        res[3][1] = -1  # p2
        res[3][2] = 1   # p3
        res[3][5] = 1   # p6
        return res

    coeffs = strassen_coeffs()
    result_coeffs = strassen_result_coeffs()

    col = 0
    # For each left block (A1 or A2) and each right sub-block
    for left_block in range(2):  # 0: A1 (cols 0-1 of A), 1: A2 (cols 2-3 of A)
        for right_sub in range(3):  # 0: cols 0-1, 1: cols 2-3, 2: cols 4-5 of B
            for s in range(7):  # Strassen product index
                U_coeffs, V_coeffs = coeffs[s]
                # Add to U
                for a_idx, coeff in U_coeffs.items():
                    # a_idx is local index in 2x2: 0=(0,0),1=(0,1),2=(1,0),3=(1,1)
                    a_i, a_j_local = (a_idx // 2, a_idx % 2)
                    a_j_global = left_block * 2 + a_j_local
                    U[a_i * m + a_j_global][col] += coeff
                # Add to V
                for b_idx, coeff in V_coeffs.items():
                    # b_idx is local index in 2x2: 0=(0,0),1=(0,1),2=(1,0),3=(1,1)
                    b_j_local, b_k_local = (b_idx // 2, b_idx % 2)
                    b_j_global = left_block * 2 + b_j_local
                    b_k_global = right_sub * 2 + b_k_local
                    V[b_j_global * p + b_k_global][col] += coeff
                # Add to W: for each output entry
                for out_idx in range(4):
                    coeff = result_coeffs[out_idx][s]
                    if coeff != 0:
                        out_i, out_k_local = (out_idx // 2, out_idx % 2)
                        out_k_global = right_sub * 2 + out_k_local
                        # W convention: W[k*n + i, r]
                        W[out_k_global * n + out_i][col] += coeff
                col += 1

    # Verify correctness (quick check)
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    n, m, p = 2, 4, 6
    # Check: for all (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    # Sample check on a few entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                if abs(s - 1.0) > 1e-9:
                    # Fallback: return schoolbook rank-48
                    return _schoolbook()
    return {"U": U, "V": V, "W": W}

def _schoolbook():
    n, m, p = 2, 4, 6
    rank = n * m * p  # 48
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][r] = 1.0
                V[j*p + k][r] = 1.0
                W[k*n + i][r] = 1.0
                r += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
