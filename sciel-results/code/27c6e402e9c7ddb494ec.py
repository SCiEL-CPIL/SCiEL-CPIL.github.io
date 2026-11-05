import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # Use Strassen's algorithm over F[t]/(t^3 - t) ≅ F ⊕ F ⊕ F
    # The 6 columns are encoded as 3 pairs (coefficients of 1, t, t^2)
    # Strassen for 3x3 uses rank 7, so total rank = 7 * 3 = 21
    rank = 21
    
    # Strassen's 7 products for 3x3 matrix multiplication:
    # For A (3x3) and B (3x3), partition into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each block is (n/2)x(m/2)
    # B = [[B11, B12], [B21, B22]] where each block is (m/2)x(p/2)
    # 
    # For 3x3: we can't cleanly partition, so use the known rank-7 scheme
    # for 3x3 matrices (a variant of Strassen).
    #
    # Actually, for 3x3, the known minimal rank is 19 or 20, but let's use
    # a simpler approach: treat 3x3 as block matrices with 1x1 and 2x2 blocks.
    # 
    # Alternative: Use the fact that we're over a field extension of degree 3.
    # The 3x6 matrix B is viewed as 3x3 over R = F[t]/(t^3-t).
    # Multiplication C = A*B becomes 3x3 matrix multiplication over R.
    # Over R, each entry is a triple (a0, a1, a2) representing a0 + a1*t + a2*t^2.
    # Since t^3 = t, we have t*t = t^2, t^2*t = t^3 = t.
    # The algebra R ≅ F × F × F via the idempotents:
    #   e0 = (2t^2 - 1)/3, e1 = (t^2 + t)/3, e2 = (t^2 - t)/3
    # But over integers, let's just use the direct tensor product structure.
    #
    # For each Strassen product over R, we get 3 scalar products (one per component).
    # So if Strassen uses 7 products for 3x3, total rank = 7*3 = 21.
    #
    # Let me define the Strassen scheme for 3x3 matrices explicitly.
    # A 3x3 matrix can be partitioned as:
    #   A = [[a11, a12, a13],
    #        [a21, a22, a23],
    #        [a31, a32, a33]]
    # 
    # One known rank-7 scheme for 3x3 (from the literature):
    # Define 7 bilinear forms. For simplicity, let's use a constructive approach.
    #
    # Actually, let me just use the standard Strassen for 2x2 blocks embedded in 3x3.
    # Partition 3x3 into:
    #   Top-left 2x2, top-right 2x1, bottom-left 1x2, bottom-right 1x1
    # This is messy. Let me use a different approach.
    #
    # Simplest correct approach: The tensor for 3x3x6 matrix mult has rank at most 21
    # by the field extension trick. Let me construct it explicitly.
    #
    # The multiplication tensor T[i,j,k] = 1 if output (i,k) depends on A[i,j] and B[j,k].
    # Over R = F[t]/(t^3-t), the 3x3x3 multiplication tensor decomposes as:
    # sum_{r=1}^{7} u_r ⊗ v_r ⊗ w_r (Strassen rank 7 for 3x3)
    # Then over F, each u_r, v_r, w_r has 3 components, giving 21 terms.
    #
    # Let me define the Strassen products for 3x3 explicitly.
    # Using the scheme from: "Lower bound for the rank of the 3x3 matrix multiplication tensor"
    # 
    # A simpler approach: use the fact that 3x3 multiplication can be done with
    # 19 multiplications (Winograd), but let's just use 21 = 7*3 for simplicity
    # and correctness.
    #
    # Define 7 Strassen-like bilinear forms for 3x3:
    # M1 = (A11 + A22 + A33) * (B11 + B22 + B33)  -- not quite right
    #
    # Let me use a concrete rank-7 scheme for 3x3 matrix multiplication:
    # Partition into 1x1, 1x2, 2x1, 2x2 blocks isn't standard.
    #
    # Alternative: Just verify that rank 21 works by explicit construction.
    # The key insight: if we have a rank-r decomposition for 3x3x3 over F,
    # then we get rank-3r for 3x3x6 by the field extension.
    #
    # Known: 3x3 matrix multiplication has rank 19 (Winograd 1971) or possibly less.
    # But for simplicity and guaranteed correctness, let me use a rank-21 scheme
    # derived from 3 copies of a rank-7 scheme.
    #
    # Here's a concrete rank-7 scheme for 3x3 (adapted from Strassen):
    # Let A = (a_ij), B = (b_ij) for i,j in {0,1,2}
    #
    # M0 = (a00+a01+a02+a10+a11+a12+a20+a21+a22) * (b00+b01+b02+b10+b11+b12+b20+b21+b22)
    # This is too crude. Let me use a known scheme.
    #
    # I'll use the following approach: embed 3x3 multiplication into 4x4,
    # use Strassen for 4x4 (rank 7), and project back. But that gives rank 7*4=28.
    #
    # Let me just directly construct a valid rank-21 decomposition.
    # The field extension approach: 
    #   R = F[t]/(t^3 - t) = F[t]/(t(t-1)(t+1)) ≅ F × F × F
    # The isomorphism: f(t) -> (f(0), f(1), f(-1))
    # So a polynomial a0 + a1*t + a2*t^2 maps to (a0, a0+a1+a2, a0-a1+a2)
    #
    # For 3x3x3 multiplication over R, we use Strassen rank 7.
    # Each Strassen product gives 3 scalar products via the three components.
    # Total: 7 * 3 = 21 scalar products.
    #
    # Let me define the 7 Strassen products for 3x3 matrices.
    # Using the scheme from B. Strassen (1969) for 3x3:
    #
    # Actually, the simplest correct rank for 3x3 is known to be 19, but 
    # constructing it is complex. Let me use a rank-21 scheme that's easier.
    #
    # Here's a rank-21 scheme for 3x3x3:
    # Use 3 separate 2x2x2 Strassen decompositions plus corrections.
    #
    # Let me just hardcode a known valid rank-21 decomposition for 3x3x3.
    # 
    # The 3x3x3 tensor: T[a,b,c] = 1 if C[a,c] = sum_b A[a,b]*B[b,c]
    # 
    # I'll construct it using the field extension method explicitly.
    
    # Strassen's 7 products for 3x3 (one valid scheme):
    # Define row sums and column sums
    # M1 = (A00 + A11 + A22) * (B00 + B11 + B22)  -- trace-like
    # This isn't quite right either.
    
    # Let me use a different, verified approach:
    # The rank of 3x3x3 matrix multiplication is known to be 19.
    # But for this problem, we need 3x3x6, which via field extension has rank 3*rank(3x3x3).
    # If rank(3x3x3) = 19, then rank(3x3x6) <= 57. Not helpful.
    #
    # Wait, I need to reconsider. The field extension trick:
    # If we have a rank-r decomposition for n×m×p over F, then we get a rank-3r 
    # decomposition for n×m×(3p) over F by the field extension F[t]/(t^3-t).
    # So for 3×3×3 with rank r, we get 3×3×6 with rank 3r.
    #
    # The schoolbook rank for 3×3×3 is 27. Strassen gives 7 for 2×2×2, but for 3×3×3,
    # the best known is 19 (Winograd). So 3×3×6 would have rank <= 3*19 = 57.
    # But schoolbook for 3×3×6 is 54. So this doesn't help!
    #
    # I made an error. Let me reconsider.
    #
    # The field extension F[t]/(t^3-t) has dimension 3 over F.
    # A 3×6 matrix B over F can be viewed as a 3×3 matrix over R = F[t]/(t^3-t).
    # Then C = A*B (3×6) corresponds to C_R = A_R * B_R (3×3 over R).
    # If we have a rank-r decomposition for 3×3×3 over R, then over F it has rank 3r
    # (since each R-entry is 3 F-entries).
    #
    # The minimum rank for 3×3×3 over any field is 19 (Winograd).
    # So 3×3×6 has rank <= 3*19 = 57. But schoolbook is 54.
    # So this approach gives 57 > 54. Not an improvement!
    #
    # I need a different strategy. Let me think about this more carefully.
    #
    # Actually, maybe I can do better by exploiting the specific structure.
    # The 3×3×6 tensor might have rank less than 54.
    #
    # Let me try a direct search for a lower rank decomposition.
    # But that's computationally expensive.
    #
    # Alternative: Use the fact that 6 = 2*3, and try to exploit block structure.
    # Write B as [B1 | B2] where B1, B2 are 3×3.
    # Then C = [A*B1 | A*B2].
    # This is two 3×3×3 multiplications, each of rank 19, total 38.
    # But we can share some products!
    #
    # If we compute A*B1 and A*B2 simultaneously, we might save multiplications.
    # The combined tensor is 3×3×6, which is the direct sum of two 3×3×3 tensors.
    # The rank of a direct sum is at most the sum of ranks, but can be less.
    #
    # For two 3×3×3 multiplications, the rank is at most 2*19 = 38.
    # But it could be less if there's sharing.
    #
    # Actually, the rank of the direct sum of two 3×3×3 tensors is exactly 38
    # (by a result of Strassen on ranks of direct sums). So no improvement.
    #
    # Hmm. Let me try a different approach.
    #
    # What if I use a rank-18 scheme for 3×3×3? No, 19 is the lower bound.
    #
    # Let me just try to find a rank < 54 decomposition by optimization.
    # But that's very hard.
    #
    # Given the difficulty, let me stick with schoolbook rank 54 for now,
    # but try to see if I can do slightly better with a simple block decomposition.
    
    # For now, return schoolbook
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
# EVOLVE-BLOCK-END
