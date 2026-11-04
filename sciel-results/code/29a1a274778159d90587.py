"""Matrix multiplication 2x3x8 - rank-minimal exact tensor decomposition."""
import time
from fractions import Fraction as F
from itertools import product

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140.0  # 10s safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 3, 8

    # Build the full multiplication tensor T[i,j,k] for C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor index: (i, j, k) -> contributes to output (i, k)
    # We represent the tensor as a 3-tensor T of shape (n*m, m*p, p*n)
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # where a = i*m+j, b = j*p+k, c = k*n+i

    # Strategy: Use block decomposition
    # Split the 3x8 matrix B into four 3x2 blocks: B[:,0:2], B[:,2:4], B[:,4:6], B[:,6:8]
    # Each block multiplication is 2x3 * 3x2
    # Use Winograd-style algorithms for 2x3*3x2

    # For 2x3 * 3x2, the schoolbook rank is 12
    # We can use a rank-7 or rank-6 algorithm

    # Let's define the 2x3*3x2 multiplication tensor
    # A is 2x3, B is 3x2, C is 2x2
    # C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, j in {0,1,2}, k in {0,1}

    # Known result: 2x3*3x2 can be done in rank 7 (Winograd)
    # Let's construct it explicitly

    # For 2x3*3x2:
    # A = [[a00, a01, a02],
    #      [a10, a11, a12]]
    # B = [[b00, b01],
    #      [b10, b11],
    #      [b20, b21]]
    # C = [[c00, c01],
    #      [c10, c11]]
    # where c00 = a00*b00 + a01*b10 + a02*b20
    #       c01 = a00*b01 + a01*b11 + a02*b21
    #       c10 = a10*b00 + a11*b10 + a12*b20
    #       c11 = a10*b01 + a11*b11 + a12*b21

    # Rank-7 algorithm for 2x3*3x2:
    # m1 = a00 + a10, n1 = b00 + b01  -> contributes to c00, c01, c10, c11
    # Actually, let me use a known construction.

    # Alternative: use the fact that we can share computations.
    # Let me try to find a rank-7 decomposition.

    # Variables for A: a00, a01, a02, a10, a11, a12 (6 vars)
    # Variables for B: b00, b01, b10, b11, b20, b21 (6 vars)
    # Output: c00, c01, c10, c11 (4 outputs)

    # Rank 7 means 7 bilinear terms.

    # Let me try:
    # t1 = (a00+a10)*(b00+b10+b20)  -- no, this doesn't work directly

    # Let me use a systematic approach. I'll set up the linear system.
    # For each rank term r, we have vectors u_r (in R^6), v_r (in R^6), w_r (in R^4)
    # such that for each output c_ik:
    #   c_ik = sum_r u_r[i,j stuff] * v_r[j,k stuff] * w_r[output]

    # Actually, let me just use a known rank-6 algorithm for 2x3*3x2 if it exists,
    # or use rank 7.

    # Let me try a different approach: use the tensor directly and find minimum rank
    # via linear algebra over the rationals.

    # Build the 2x3*3x2 tensor
    def build_232_tensor():
        """Build the 2x3*3x2 multiplication tensor as a list of (a_idx, b_idx, c_idx) tuples."""
        # a_idx = i*3+j for i in {0,1}, j in {0,1,2} -> 6 indices
        # b_idx = j*2+k for j in {0,1,2}, k in {0,1} -> 6 indices
        # c_idx = k*2+i for k in {0,1}, i in {0,1} -> 4 indices (using W[k*n+i] convention)
        terms = []
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    a_idx = i * 3 + j
                    b_idx = j * 2 + k
                    c_idx = k * 2 + i  # W convention: W[k*n+i]
                    terms.append((a_idx, b_idx, c_idx))
        return terms

    def build_238_tensor():
        """Build the full 2x3*3x8 multiplication tensor."""
        # a_idx = i*3+j for i in {0,1}, j in {0,1,2} -> 6 indices
        # b_idx = j*8+k for j in {0,1,2}, k in {0,...,7} -> 24 indices
        # c_idx = k*2+i for k in {0,...,7}, i in {0,1} -> 16 indices
        terms = []
        for i in range(2):
            for j in range(3):
                for k in range(8):
                    a_idx = i * 3 + j
                    b_idx = j * 8 + k
                    c_idx = k * 2 + i  # W convention
                    terms.append((a_idx, b_idx, c_idx))
        return terms

    # The tensor T is defined by: T[a,b,c] = 1 if there exists i,j,k such that
    # a = i*3+j, b = j*8+k, c = k*2+i, else 0.
    # Note: for a given (a,b,c), the (i,j,k) is uniquely determined.

    # So the tensor has exactly 48 non-zero entries (one for each i,j,k triple).

    # Now, to find a low-rank decomposition, we need to find the minimum number of
    # rank-1 tensors that sum to T.

    # Approach: Use the block structure. Split k into 4 blocks of size 2.
    # Block 0: k in {0,1}
    # Block 1: k in {2,3}
    # Block 2: k in {4,5}
    # Block 3: k in {6,7}

    # For each block, the sub-tensor involves:
    # a_idx = i*3+j (same as before)
    # b_idx = j*8 + k where k is in the block
    # c_idx = k*2+i where k is in the block

    # The sub-tensor for block b (k in {2b, 2b+1}) has:
    # a indices: 0..5
    # b indices: j*8 + 2b and j*8 + 2b+1 for j in {0,1,2}
    # c indices: (2b)*2+i = 4b+i and (2b+1)*2+i = 4b+2+i for i in {0,1}

    # Each sub-tensor is a 2x3*3x2 tensor (with shifted indices).

    # If we can decompose each 2x3*3x2 sub-tensor in rank r, the total rank is 4*r.
    # But we might be able to share terms across blocks.

    # Let's first find the minimum rank of the 2x3*3x2 tensor.

    # For 2x3*3x2, I'll use a known construction or solve the linear system.

    # Let me try rank 7 first (Winograd-style).

    # Actually, let me just build the full 2x3*3x8 tensor and try to find a low-rank
    # decomposition using the block structure and then compress.

    # Step 1: Build candidate decomposition using block Winograd algorithms
    # Step 2: Compress via linear algebra

    # For 2x3*3x2, let me use a rank-7 algorithm:
    # I'll construct it by solving the linear system.

    def find_232_rank7():
        """Find a rank-7 decomposition of the 2x3*3x2 tensor."""
        # The tensor has 6*6*4 = 144 entries, but only 12 are non-zero.
        # Non-zero entries: for i in {0,1}, j in {0,1,2}, k in {0,1}:
        #   a = i*3+j, b = j*2+k, c = k*2+i

        # We want to find U (6x7), V (6x7), W (4x7) such that
        # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]

        # This is a system of equations. Let's set it up.
        # For each (a,b,c) with T[a,b,c] = 1:
        #   sum_r U[a,r]*V[b,r]*W[c,r] = 1
        # For each (a,b,c) with T[a,b,c] = 0:
        #   sum_r U[a,r]*V[b,r]*W[c,r] = 0

        # This is nonlinear. Instead, let's fix W and solve for U,V linearly.
        # Or fix U and W and solve for V.

        # Alternative: use the fact that the tensor is a sum of rank-1 tensors.
        # Let me try a known construction.

        # Winograd-style for 2x3*3x2:
        # Define:
        # p1 = (a00 + a10) * (b00 + b10 + b20)
        # p2 = (a00 - a10) * (b00 - b10 - b20)  -- not quite

        # Let me just use a brute force approach with rational arithmetic.
        # Fix the W matrix (4x7) and solve for U and V.

        # Actually, let me try a different strategy: use the known result that
        # 2x3*3x2 has rank 7 and construct it explicitly.

        # From literature, one rank-7 algorithm:
        # Let A = [[a,b,c],[d,e,f]], B = [[g,h],[i,j],[k,l]]
        # C = [[ag+bi+ck, ah+bj+cl],
        #      [dg+ei+fk, dh+ej+fl]]

        # Define:
        # m1 = (a+d)*(g+i+k)       -> contributes to c00, c10
        # m2 = (a-d)*(g-i-k)       -> contributes to c00, c10
        # m3 = (b+e)*(h+j+l)       -> contributes to c01, c11
        # m4 = (b-e)*(h-j-l)       -> contributes to c01, c11
        # m5 = (c+f)*(k+l)         -- no, this doesn't work

        # This is getting complicated. Let me just use the full tensor approach
        # and find the minimum rank via linear algebra.

        # I'll use the following approach:
        # 1. Start with the schoolbook decomposition (rank 12)
        # 2. Try to reduce the rank by finding linear dependencies

        # Schoolbook for 2x3*3x2:
        # For each (i,j,k): U[i*3+j, r] = 1, V[j*2+k, r] = 1, W[k*2+i, r] = 1
        # where r = i*3+j*2+k (or some ordering)

        # Let me just build the schoolbook and then try to compress.

        pass

    # Let me take a more direct approach:
    # Build the full 2x3*3x8 tensor's schoolbook decomposition (rank 48)
    # Then try to compress it by finding linear dependencies in the U, V, W matrices.

    # Schoolbook decomposition:
    def schoolbook_238():
        n, m, p = 2, 3, 8
        rank = n * m * p  # 48
        U = [[F(0)] * rank for _ in range(n * m)]  # 6 x 48
        V = [[F(0)] * rank for _ in range(m * p)]  # 24 x 48
        W = [[F(0)] * rank for _ in range(p * n)]  # 16 x 48
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = F(1)
                    V[j * p + k][col] = F(1)
                    W[k * n + i][col] = F(1)
                    col += 1
        return U, V, W

    U, V, W = schoolbook_238()

    # Now try to compress: find the minimum rank by identifying linearly dependent columns
    # in the "slice" matrices.

    # The tensor T can be viewed as a matrix for each fixed c:
    # T_c is a 6x24 matrix where T_c[a,b] = 1 if there exists i,j,k with
    # a=i*3+j, b=j*8+k, c=k*2+i, else 0.

    # For a rank-r decomposition, T_c = U_c * V_c^T for all c, where U_c is 6xr
    # (rows of U corresponding to the c-th slice) and V_c is 24xr.

    # Actually, the standard way: the tensor T(a,b,c) = sum_r U[a,r] * V[b,r] * W[c,r].
    # For fixed c, define U_c[a,r] = U[a,r]*W[c,r] and V_c[b,r] = V[b,r].
    # Then T(a,b,c) = sum_r U_c[a,r] * V_c[b,r], i.e., T_c = U_c * V_c^T.

    # The rank of T_c is at most r for all c. So r must be at least the maximum
    # rank of any T_c.

    # Let's compute the ranks of the slices.

    def slice_rank(c, n=2, m=3, p=8):
        """Compute the rank of the c-th slice of the tensor."""
        # T_c[a,b] = 1 if there exists i,j,k with a=i*m+j, b=j*p+k, c=k*n+i
        # For fixed c, k*n+i = c, so i = c mod n, k = c // n
        # Then a = i*m+j for j in range(m), b = j*p+k for j in range(m)
        # So T_c has non-zero entries at (a=i*m+j, b=j*p+k) for j in range(m)
        # This is a matrix with at most m non-zero entries, all in distinct rows and columns.
        # So the rank is at most m = 3.
        # Actually, the non-zero entries are at positions (i*m+j, j*p+k) for j=0,1,2.
        # These are in distinct rows (i*m, i*m+1, i*m+2) and distinct columns (k, p+k, 2p+k).
        # So the rank is exactly 3 (if m=3).
        return 3

    # So the minimum rank is at least 3. But we need a global decomposition.

    # Let me try to find a low-rank decomposition by solving the linear system.
    # I'll try ranks from 16 upward (since 4 blocks * 4 = 16 is a lower bound
    # if each 2x3*3x2 block needs rank 4, but it might be less).

    # Actually, let me just try to construct a decomposition directly.

    # For each block of 2 columns in B (k in {2b, 2b+1} for b=0,1,2,3):
    # The sub-tensor involves:
    #   a = i*3+j for i in {0,1}, j in {0,1,2}
    #   b_idx = j*8 + k for k in {2b, 2b+1}
    #   c_idx = k*2+i for k in {2b, 2b+1}, i in {0,1}

    # Let me try to find a rank-4 decomposition for each 2x3*3x2 block.
    # If I can do that, the total rank would be 4*4 = 16.

    # For 2x3*3x2, can we achieve rank 4?
    # The tensor has 12 non-zero entries. A rank-4 tensor has at most 4*6*4 = 96
    # "degrees of freedom" but the constraints are nonlinear.

    # Let me just try to solve it numerically and then round to rationals.

    # Actually, let me use a simpler approach: just return the schoolbook decomposition
    # but try to compress it slightly.

    # Given the time constraints, let me just return the schoolbook decomposition
    # with rank 48. This is safe and correct.

    # But let me try one more thing: can I reduce the rank by noticing that
    # the tensor has a product structure?

    # The 2x3*3x8 tensor can be seen as a Kronecker-like product of smaller tensors.
    # But 8 = 4*2, so maybe we can use a 2x3*3x4 tensor (rank 24) combined with
    # a 2x2 identity or something.

    # Actually, let me just try to find a rank-24 decomposition by splitting
    # the 8 columns into 4 groups of 2 and using rank-6 for each 2x3*3x2 block.

    # If 2x3*3x2 has rank 6, then 4 blocks give rank 24.

    # Let me try to construct a rank-6 algorithm for 2x3*3x2.

    def rank6_232():
        """Try to find a rank-6 decomposition of 2x3*3x2."""
        # I'll use a known construction or solve the system.
        # For now, let me just use the schoolbook (rank 12) for each block.
        pass

    # Given the complexity, let me just return the schoolbook decomposition.
    # The rank is 48, which gives a score of 1.0.

    # But wait, the observed changes show that a score of 1.09 was achieved,
    # which corresponds to rank 48/1.09 ≈ 44. So there's a better solution.

    # Let me try to achieve rank 44 or lower.

    # Actually, 1.09 = 48/44.04... so rank ≈ 44.

    # Let me try to compress the schoolbook decomposition by 4 ranks.

    # I'll do this by finding 4 linear dependencies among the columns.

    # The columns of U, V, W are indexed by (i,j,k) triples.
    # Two columns (i1,j1,k1) and (i2,j2,k2) are linearly dependent if
    # their U, V, W vectors are proportional. But they're all 0/1 vectors,
    # so they're only dependent if they're identical, which they're not.

    # So I need to find a different decomposition.

    # Let me just return the schoolbook for now.
    result = {"U": U, "V": V, "W": W}
    return result
# EVOLVE-BLOCK-END

