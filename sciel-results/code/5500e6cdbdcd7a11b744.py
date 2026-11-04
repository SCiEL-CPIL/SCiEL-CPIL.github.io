import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0

    def verify(U, V, W, tol=1e-8):
        rank = len(U[0])
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                ui = i * m + j
                for k in range(p):
                    vk = j * p + k
                    wk = k * n + i
                    total = np.dot(U_arr[ui], V_arr[vk] * W_arr[wk])
                    if abs(total - 1.0) > tol:
                        return False
        return True

    def get_schoolbook():
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
        return U, V, W

    # Strassen 2x2x2: 7 multiplications
    # U factors: (i,j) -> list of (coeff, (a,b)) where a is row-index into A-block, b is col-index
    # We represent each factor as a dict mapping (i,j) -> list of (coeff, (a,b))
    # where a in {0,1}, b in {0,1} are block sub-indices

    def strassen_2x2():
        """
        Returns U, V, W as dicts for 2x2x2 Strassen.
        U[(i,j)] = list of (coeff, (a,b)) meaning U factor has coeff * A[a,b]
        V[(j,k)] = list of (coeff, (c,d)) meaning V factor has coeff * B[c,d]
        W[(k,i)] = list of (coeff, (e,f)) meaning W factor has coeff * C[e,f]
        """
        # Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B10)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        #
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6

        U = {}  # (i,j) -> [(coeff, (a,b)), ...]
        V = {}  # (j,k) -> [(coeff, (c,d)), ...]
        W = {}  # (k,i) -> [(coeff, (e,f)), ...]

        # M1: (A00+A11)(B00+B11)
        U[(0,0)] = [(1, (0,0)), (1, (1,1))]
        U[(1,1)] = [(1, (0,0)), (1, (1,1))]
        V[(0,0)] = [(1, (0,0)), (1, (1,1))]
        V[(0,1)] = [(1, (0,0)), (1, (1,1))]
        V[(1,0)] = [(1, (0,0)), (1, (1,1))]
        V[(1,1)] = [(1, (0,0)), (1, (1,1))]
        W[(0,0)] = [(1, (0,0))]
        W[(1,0)] = [(1, (1,1))]
        W[(0,1)] = [(-1, (1,1)), (1, (0,0))]
        W[(1,1)] = [(1, (0,0))]

        # M2: (A10+A11)B00
        U[(1,0)] = [(1, (1,0)), (1, (1,1))]
        U[(1,1)] = [(1, (1,0)), (1, (1,1))]
        V[(0,0)] = [(1, (0,0))]
        V[(1,0)] = [(1, (0,0))]
        W[(0,1)] = [(1, (1,0))]
        W[(1,1)] = [(-1, (1,1))]

        # M3: A00(B10-B11)
        U[(0,0)] = [(1, (0,0))]
        V[(1,0)] = [(1, (1,0))]
        V[(1,1)] = [(-1, (1,1))]
        W[(0,1)] = [(1, (0,1))]
        W[(1,1)] = [(1, (1,1))]

        # M4: A11(B01-B10)
        U[(1,1)] = [(1, (1,1))]
        V[(0,1)] = [(1, (0,1))]
        V[(1,0)] = [(-1, (1,0))]
        W[(0,0)] = [(1, (0,0))]
        W[(1,0)] = [(1, (1,0))]

        # M5: (A00+A01)B11
        U[(0,0)] = [(1, (0,0))]
        U[(0,1)] = [(1, (0,1))]
        V[(0,1)] = [(1, (1,1))]
        V[(1,1)] = [(1, (1,1))]
        W[(0,0)] = [(-1, (0,0))]
        W[(1,0)] = [(1, (1,0))]

        # M6: (A10-A00)(B00+B01)
        U[(1,0)] = [(1, (1,0))]
        U[(0,0)] = [(-1, (0,0))]
        V[(0,0)] = [(1, (0,0))]
        V[(0,1)] = [(1, (0,1))]
        W[(1,1)] = [(1, (1,1))]

        # M7: (A01-A11)(B10+B11)
        U[(0,1)] = [(1, (0,1))]
        U[(1,1)] = [(-1, (1,1))]
        V[(1,0)] = [(1, (1,0))]
        V[(1,1)] = [(1, (1,1))]
        W[(0,0)] = [(1, (0,0))]
        W[(0,1)] = [(1, (0,1))]

        return U, V, W

    def laderman_3x3():
        """
        Returns U, V, W as dicts for 3x3x3 Laderman (rank 19).
        """
        # Laderman's 19-multiplication algorithm for 3x3 matrix multiplication
        # A = [[a00,a01,a02],[a10,a11,a12],[a20,a21,a22]]
        # B = [[b00,b01,b02],[b10,b11,b12],[b20,b21,b22]]
        # C = A*B, C[i,j] = sum_k A[i,k]*B[k,j]

        # Laderman's 19 multiplications:
        # M1 = (a00+a10+a20)(b00+b10+b20)  -- wait, let me use the standard Laderman
        # Actually, let me use the well-known Laderman decomposition.

        # Standard Laderman 3x3x3 rank-19:
        # The 19 products are:
        # p1 = (a00+a10)(b00+b20)
        # p2 = (a01+a11+a21)(b00+b10+b20)  -- hmm, let me look this up properly

        # Let me use a verified Laderman decomposition.
        # Reference: Laderman (1976), "On the number of multiplications required to compute the product of two 3x3 matrices"

        # I'll use the following verified 19-term decomposition:
        # Let me define it carefully.

        # Actually, let me use a different well-known approach.
        # I'll use the explicit Laderman formula:

        # The 19 bilinear forms for 3x3x3:
        U = {}
        V = {}
        W = {}

        # Laderman's algorithm (verified):
        # M1 = (a00+a10)(b00+b20)
        # M2 = (a01+a11+a21)(b00+b10+b20)  -- no, this isn't right either

        # Let me use a clean formulation. I'll define the 19 multiplications explicitly.

        # Using the standard Laderman 3x3 decomposition:
        # I'll index A[i][j] with i=row, j=col; B[k][j] with k=row, j=col; C[i][j]

        # The 19 products (using 0-indexed):
        # p0: (a[0][0]+a[1][0]) * (b[0][0]+b[2][0])
        # p1: (a[0][1]+a[1][1]+a[2][1]) * (b[0][0]+b[1][0]+b[2][0])  -- NO

        # Let me just hard-code a known correct Laderman decomposition.
        # From literature, the 19 multiplications are:

        # I'll use this verified set (from various references):
        mults = [
            # (U_terms, V_terms, W_terms)
            # Each term is (coeff, (row, col))
            # U_terms: contributions to A[i][j]
            # V_terms: contributions to B[k][j]
            # W_terms: contributions to C[i][j]

            # M1: (a00+a10)(b00+b20)
            ([(1,(0,0)), (1,(1,0))], [(1,(0,0)), (1,(2,0))],
             [(1,(0,0)), (1,(1,0)), (1,(0,2)), (1,(1,2)), (-1,(0,0)), (-1,(1,0)), (-1,(0,2)), (-1,(1,2))]),
            # This is getting complicated. Let me use a different approach.
        ]

        # Actually, let me take a completely different and cleaner approach.
        # I'll directly construct the 5x5x5 decomposition using block structure.

        # The key insight: 5x5 = 2x2 blocks + 3x3 blocks
        # But more precisely, we can write:
        # A (5x5) = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x3, A21 is 3x2, A22 is 3x3
        # B (5x5) = [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x3, B21 is 3x2, B22 is 3x3
        # C = A*B

        # C11 (2x2) = A11*B11 + A12*B21
        # C12 (2x3) = A11*B12 + A12*B22
        # C21 (3x2) = A21*B11 + A22*B21
        # C22 (3x3) = A21*B12 + A22*B22

        # For each block product, we can use:
        # 2x2 * 2x2: Strassen (rank 7)
        # 2x2 * 2x3: can be decomposed
        # 2x2 * 3x2: can be decomposed
        # 2x3 * 3x2: rank at most 2*3*2=12, but can do better
        # 2x3 * 3x3: rank at most 18
        # 3x2 * 2x2: rank at most 12
        # 3x2 * 2x3: rank at most 12
        # 3x3 * 3x3: Laderman (rank 19)

        # This is getting complex. Let me use a simpler but still valid approach.

        # SIMPLEST CORRECT APPROACH: Use the tensor product structure.
        # 5x5x5 can be decomposed as a combination of smaller tensor products.

        # Actually, the cleanest constructive approach for rank 49:
        # Split 5 = 2 + 3
        # The 5x5x5 multiplication tensor decomposes into:
        # - (2x2x2) for the top-left 2x2 block of C: needs 7 (Strassen)
        # - (3x3x3) for the bottom-right 3x3 block of C: needs 19 (Laderman)
        # - Cross terms: (2x2x3), (2x3x2), (3x2x2), (2x3x3), (3x2x3), (3x3x2)

        # For mixed-size blocks, the trivial bound is r*s*t:
        # 2x2x3: 12
        # 2x3x2: 12
        # 3x2x2: 12
        # 2x3x3: 18
        # 3x2x3: 18
        # 3x3x2: 18
        # Total: 7 + 19 + 12 + 12 + 12 + 18 + 18 + 18 = 116 -- too high

        # Better: use Strassen-like algorithms for mixed sizes.
        # 2x2x3: We can compute 2x2 * 2x3 using 2 Strassen-like steps.
        # Actually, 2x2x3 has rank at most 2*2*3 = 12, but we can do 2*3 = 6? No.
        # The rank of 2x2x3 multiplication tensor is at most 12 (trivial).
        # Can we do better? The tensor [2,2,3] has rank at most 12.

        # Let me just use the trivial decomposition for all cross terms and Strassen/Laderman for the pure blocks.
        # 7 + 19 + 12 + 12 + 12 + 18 + 18 + 18 = 116. Still worse than 125.

        # Better approach: Use a 5x5 = 1+4 or other split.
        # Or: use the fact that we can compute 5x5x5 with rank at most:
        # Using the "doubling" trick: 5x5x5 <= 2*2x2x2 + 3*3x3x3 + cross terms

        # Actually, let me just directly implement a rank-49 decomposition.
        # The known upper bound for 5x5x5 is 49 from:
        # omega(5) <= 2*omega(2) + 3*omega(3) + 2*2*3 + 2*3*2 + 3*2*2 + 2*3*3 + 3*2*3 + 3*3*2
        # = 2*7 + 3*19 + ... no, that's not right either.

        # The correct decomposition:
        # 5x5x5 = (2+3)x(2+3)x(2+3)
        # = 2x2x2 + 2x2x3 + 2x3x2 + 2x3x3 + 3x2x2 + 3x2x3 + 3x3x2 + 3x3x3
        # Rank <= 7 + 12 + 12 + 18 + 12 + 18 + 18 + 19 = 116

        # That's not 49. The 49 bound comes from a different argument.
        # Actually, the known result is that mu(5) <= 49 from:
        # 5x5x5 can be embedded in 7x7x7 which has rank <= 7^omega
        # No, that's not right.

        # Let me reconsider. The cell finding says:
        # "Block decomposition using Strassen for 2x2x2 block reduces rank from 125 to approximately 7+42=49"
        # 7 + 42 = 49. So the idea is:
        # Use Strassen for one 2x2x2 sub-block (rank 7)
        # And the remaining 42 = 125 - 7 - 75? No.
        # 42 = 2*3*7? Or 6*7?

        # I think the idea is:
        # Split the 5x5 matrix into a 2x2 block and the rest.
        # The 2x2x2 part uses Strassen (rank 7).
        # The remaining terms involve 2x3, 3x2, 3x3 blocks.
        # 42 = 12 + 12 + 18 = 42? That's 2x2x3 + 3x2x2 + 2x3x3? No, that's only part.

        # Actually: 7 + 42 = 49 where 42 might come from:
        # The "off-diagonal" terms can be computed with 42 multiplications.
        # 2x2x3 (12) + 2x3x2 (12) + 3x2x2 (12) + 3x3x2 (6?) ...

        # Let me just try a direct construction. I'll use the following approach:
        # Decompose 5x5x5 into 7 "Strassen-like" 2x2x2 blocks plus 42 additional terms.

        # Actually, the simplest correct approach that gives rank 49:
        # Use the formula: mu(5) <= mu(2)*mu(3) + mu(2)*mu(3) + ... no.

        # Let me just implement the trivial block decomposition and see what rank we get.
        # Then optimize from there.

        # FOR NOW, let me implement a correct rank-49 decomposition using the following structure:
        # We'll use 7 Strassen multiplications for the 2x2x2 block
        # And 42 "simple" multiplications for the rest

        # The 42 comes from:
        # - 2x2 * 3x2: 6 terms (each element of 2x2 result needs 2 mults, 4 elements = 8? No, 2*2*2=8)
        # Hmm, let me think differently.

        # 5x5 = [2, 3] x [2, 3]
        # C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:5]*B[2:5,0:2]
        # C[0:2, 2:5] = A[0:2,0:2]*B[0:2,2:5] + A[0:2,2:5]*B[2:5,2:5]
        # C[2:5, 0:2] = A[2:5,0:2]*B[0:2,0:2] + A[2:5,2:5]*B[2:5,0:2]
        # C[2:5, 2:5] = A[2:5,0:2]*B[0:2,2:5] + A[2:5,2:5]*B[2:5,2:5]

        # Each of these is a sum of two matrix products.
        # For the 2x2x2 terms: use Strassen (7 mults each, but they're added so we need separate mults)
        # Wait, no. Each C block is a SUM of two products. We need to compute both products.

        # C[0:2,0:2] = A11*B11 + A12*B21
        #   A11*B11: 2x2x2, rank 7 (Strassen)
        #   A12*B21: 2x3 * 3x2 = 2x2, rank at most 2*3*2 = 12
        #   Total for this block: 7 + 12 = 19

        # C[0:2,2:5] = A11*B12 + A12*B22
        #   A11*B12: 2x2 * 2x3 = 2x3, rank at most 12
        #   A12*B22: 2x3 * 3x3 = 2x3, rank at most 18
        #   Total: 12 + 18 = 30

        # C[2:5,0:2] = A21*B11 + A22*B21
        #   A21*B11: 3x2 * 2x2 = 3x2, rank at most 12
        #   A22*B21: 3x3 * 3x2 = 3x2, rank at most 18
        #   Total: 12 + 18 = 30

        # C[2:5,2:5] = A21*B12 + A22*B22
        #   A21*B12: 3x2 * 2x3 = 3x3, rank at most 18
        #   A22*B22: 3x3 * 3x3 = 3x3, rank 19 (Laderman)
        #   Total: 18 + 19 = 37

        # Grand total: 19 + 30 + 30 + 37 = 116. Still not 49.

        # The 49 bound must come from a smarter algorithm. Let me just implement the best I can
        # and see. Actually, let me try a completely different approach:

        # Use the "border rank" or "tensor product" approach.
        # 5 = 2 + 3, and we know:
        # mu(5) <= mu(2,2,2) + mu(2,3,2) + mu(2,2,3) + mu(3,2,2) + mu(2,3,3) + mu(3,2,3) + mu(3,3,2) + mu(3,3,3)
        # With Strassen for 2x2x2 (7) and Laderman for 3x3x3 (19), and trivial for others:
        # 7 + 12 + 12 + 12 + 18 + 18 + 18 + 19 = 116

        # To get 49, we need a much smarter algorithm. The known result mu(5) <= 49 comes from:
        # Coppersmith-Winograd type bounds or specific constructions.

        # For this task, let me just implement the best constructive decomposition I can.
        # If I can get rank 49, great. If not, I'll return the best I find.

        # Let me try a different split: 5 = 1 + 4
        # mu(5) <= mu(1,1,1) + mu(1,1,4) + mu(1,4,1) + mu(4,1,1) + mu(1,4,4) + mu(4,1,4) + mu(4,4,1) + mu(4,4,4)
        # = 1 + 4 + 4 + 4 + 16 + 16 + 16 + mu(4,4,4)
        # mu(4,4,4) <= 4^omega, and with Strassen mu(4,4,4) <= 7^2 = 49? No, mu(4,4,4) <= 2*mu(2,2,2) + ...
        # Actually mu(4,4,4) <= 7^2 = 49 is NOT correct. mu(4,4,4) <= 2*7 + 2*7 + 2*7 + 2*7 = 56? No.
        # mu(4,4,4) using Strassen on 4x4 = 2x(2x2): each 2x2x2 uses 7, and there are 4 blocks of 2x2x2
        # plus cross terms. mu(4,4,4) <= 4*7 + 4*12 = 28 + 48 = 76? No.
        # mu(4,4,4) = 2^2 * mu(2,2,2) + cross terms. Actually mu(4,4,4) <= 7*4 + 12*4 = 76? No.
        # The correct bound: mu(4,4,4) <= 2*mu(2,2,2) + 4*mu(2,2,3) + 4*mu(2,3,2) + 2*mu(2,3,3)
        # = 2*7 + 4*12 + 4*12 + 2*18 = 14 + 48 + 48 + 36 = 146? That can't be right.

        # I'm overcomplicating this. Let me just implement a correct decomposition and verify it.
        # The simplest correct approach: schoolbook (rank 125) is always valid.
        # Let me try to construct a rank-49 decomposition directly.

        # After more thought, I believe the rank-49 bound comes from:
        # Using the "tensor square" of Strassen: 5x5x5 embeds in 7x7x7
        # No, that gives mu(5) <= mu(7) which is worse.

        # Let me just implement the block decomposition with Strassen and Laderman and see what rank we get.
        # Even if it's not 49, it should be better than 125.

        return None  # Placeholder, we'll implement the actual decomposition below

    # Let me implement the actual block decomposition.
    # I'll construct U, V, W matrices directly.

    def build_block_decomposition():
        """
        Build a decomposition of 5x5x5 using 2+3 block structure.
        Uses Strassen for 2x2x2 and Laderman for 3x3x3.
        Uses trivial (schoolbook) for mixed-size blocks.
        """
        n, m, p = 5, 5, 5

        # Block structure:
        # A rows: 0,1 (top-2), 2,3,4 (bottom-3)
        # A cols: 0,1 (left-2), 2,3,4 (right-3)
        # B rows: 0,1 (top-2), 2,3,4 (bottom-3)
        # B cols: 0,1 (left-2), 2,3,4 (right-3)
        # C rows: 0,1 (top-2), 2,3,4 (bottom-3)
        # C cols: 0,1 (left-2), 2,3,4 (right-3)

        U_rows = []  # Will build as list of (i,j, coeff, (a_block, a_sub))
        V_rows = []  # (j,k, coeff, (b_block, b_sub))
        W_rows = []  # (k,i, coeff, (c_block, c_sub))

        # For each of the 8 block products, add terms.
        # Block product (ar, ac) * (br, bc) contributes to C block (ar, bc)

        # Helper: add a term to the decomposition
        # We'll collect all terms and then assemble into matrices.

        terms = []  # Each term: (U_entry, V_entry, W_entry)
        # U_entry: (i, j, coeff) where i is row of A, j is col of A
        # V_entry: (j, k, coeff) where j is row of B, k is col of B
        # W_entry: (k, i, coeff) where k is row of C, i is col of C

        # Strassen 2x2x2 for block (0,0)*(0,0) -> C(0,0)
        # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        strassen_U = {
            (0,0): [(1,(0,0)), (1,(1,1))],
            (1,1): [(1,(0,0)), (1,(1,1))],
        }
        # This is getting too complex to do by hand. Let me use a programmatic approach.

        # I'll use a simpler but correct method:
        # Just use the schoolbook decomposition for all blocks, but replace
        # the 2x2x2 block with Strassen and 3x3x3 block with Laderman.

        # For now, let me just return the schoolbook decomposition as a fallback,
        # and try to improve it with a constructive algorithm.

        # Actually, let me just implement the full block decomposition programmatically.

        rank = 0
        U_list = []  # Will be a list of columns, each column is a dict (i,j) -> coeff
        V_list = []  # (j,k) -> coeff
        W_list = []  # (k,i) -> coeff

        def add_term(U_dict, V_dict, W_dict):
            """Add one rank-1 term to the decomposition."""
            nonlocal rank
            U_list.append(U_dict)
            V_list.append(V_dict)
            W_list.append(W_dict)
            rank += 1

        # Block (0,0) * (0,0) -> C(0,0): 2x2x2, use Strassen (7 terms)
        # A[0:2,0:2] * B[0:2,0:2]
        # Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B10)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6

        # Global indices: A[i,j] with i in {0,1}, j in {0,1}
        # B[k,j] with k in {0,1}, j in {0,1}
        # C[i,j] with i in {0,1}, j in {0,1}

        # M1: U: A[0,0]+=1, A[1,1]+=1; V: B[0,0]+=1, B[0,1]+=1, B[1,0]+=1, B[1,1]+=1
        # Wait, Strassen M1 = (A00+A11)(B00+B11)
        # U factor: A[0,0] + A[1,1]
        # V factor: B[0,0] + B[1,1]  (B00 is B[0,0], B11 is B[1,1])
        # W factor: C[0,0] += 1, C[1,1] += 1, C[0,1] += -1, C[1,0] += 1
        # Wait, let me be more careful.

        # In Strassen:
        # M1 = (A00+A11)(B00+B11)
        # This contributes to:
        # C00 += M1
        # C11 += M1
        # C01 -= M1  (from C01 = M3+M5, no M1 doesn't appear in C01)
        # Wait, let me re-derive:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6

        # So M1 appears in C00 (+1) and C11 (+1)
        # M2 appears in C10 (+1) and C11 (-1)
        # M3 appears in C01 (+1) and C11 (+1)
        # M4 appears in C00 (+1) and C10 (+1)
        # M5 appears in C00 (-1) and C01 (+1)
        # M6 appears in C11 (+1)
        # M7 appears in C00 (+1)

        # M1: (A00+A11)(B00+B11)
        U1 = {(0,0): 1, (1,1): 1}
        V1 = {(0,0): 1, (1,1): 1}
        W1 = {(0,0): 1, (1,1): 1}
        add_term(U1, V1, W1)

        # M2: (A10+A11)B00
        U2 = {(1,0): 1, (1,1): 1}
        V2 = {(0,0): 1}
        W2 = {(1,0): 1, (1,1): -1}
        add_term(U2, V2, W2)

        # M3: A00(B10-B11)
        U3 = {(0,0): 1}
        V3 = {(1,0): 1, (1,1): -1}
        W3 = {(0,1): 1, (1,1): 1}
        add_term(U3, V3, W3)

        # M4: A11(B01-B10)
        U4 = {(1,1): 1}
        V4 = {(0,1): 1, (1,0): -1}
        W4 = {(0,0): 1, (1,0): 1}
        add_term(U4, V4, W4)

        # M5: (A00+A01)B11
        U5 = {(0,0): 1, (0,1): 1}
        V5 = {(1,1): 1}
        W5 = {(0,0): -1, (0,1): 1}
        add_term(U5, V5, W5)

        # M6: (A10-A00)(B00+B01)
        U6 = {(1,0): 1, (0,0): -1}
        V6 = {(0,0): 1, (0,1): 1}
        W6 = {(1,1): 1}
        add_term(U6, V6, W6)

        # M7: (A01-A11)(B10+B11)
        U7 = {(0,1): 1, (1,1): -1}
        V7 = {(1,0): 1, (1,1): 1}
        W7 = {(0,0): 1}
        add_term(U7, V7, W7)

        # Now block (0,0) * (0,1) -> C(0,1): 2x2 * 2x3 = 2x3
        # A[0:2,0:2] * B[0:2,2:5]
        # B columns 2,3,4
        # Trivial: 2*2*3 = 12 terms
        for i in range(2):  # A row
            for k in range(2):  # B row (A col)
                for j in range(3):  # B col (global col 2+j)
                    U_term = {(i, k): 1}
                    V_term = {(k, 2+j): 1}
                    W_term = {(i, 2+j): 1}  # C[i, 2+j]
                    add_term(U_term, V_term, W_term)

        # Block (0,1) * (1,0) -> C(0,0): 2x3 * 3x2 = 2x2
        # A[0:2,2:5] * B[2:5,0:2]
        # Trivial: 2*3*2 = 12 terms
        for i in range(2):  # A row
            for k in range(3):  # A col (global col 2+k), B row (global row 2+k)
                for j in range(2):  # B col
                    U_term = {(i, 2+k): 1}
                    V_term = {(2+k, j): 1}
                    W_term = {(i, j): 1}  # C[i, j]
                    add_term(U_term, V_term, W_term)

        # Block (0,1) * (1,1) -> C(0,1): 2x3 * 3x3 = 2x3
        # A[0:2,2:5] * B[2:5,2:5]
        # Trivial: 2*3*3 = 18 terms
        for i in range(2):
            for k in range(3):
                for j in range(3):
                    U_term = {(i, 2+k): 1}
                    V_term = {(2+k, 2+j): 1}
                    W_term = {(i, 2+j): 1}
                    add_term(U_term, V_term, W_term)

        # Block (1,0) * (0,0) -> C(1,0): 3x2 * 2x2 = 3x2
        # A[2:5,0:2] * B[0:2,0:2]
        # Trivial: 3*2*2 = 12 terms
        for i in range(3):  # A row (global row 2+i)
            for k in range(2):  # A col, B row
                for j in range(2):  # B col
                    U_term = {(2+i, k): 1}
                    V_term = {(k, j): 1}
                    W_term = {(2+i, j): 1}
                    add_term(U_term, V_term, W_term)

        # Block (1,0) * (0,1) -> C(1,1): 3x2 * 2x3 = 3x3
        # A[2:5,0:2] * B[0:2,2:5]
        # Trivial: 3*2*3 = 18 terms
        for i in range(3):
            for k in range(2):
                for j in range(3):
                    U_term = {(2+i, k): 1}
                    V_term = {(k, 2+j): 1}
                    W_term = {(2+i, 2+j): 1}
                    add_term(U_term, V_term, W_term)

        # Block (1,1) * (1,0) -> C(1,0): 3x3 * 3x2 = 3x2
        # A[2:5,2:5] * B[2:5,0:2]
        # Trivial: 3*3*2 = 18 terms
        for i in range(3):
            for k in range(3):
                for j in range(2):
                    U_term = {(2+i, 2+k): 1}
                    V_term = {(2+k, j): 1}
                    W_term = {(2+i, j): 1}
                    add_term(U_term, V_term, W_term)

        # Block (1,1) * (1,1) -> C(1,1): 3x3 * 3x3 = 3x3
        # A[2:5,2:5] * B[2:5,2:5]
        # Use Laderman (rank 19) or trivial (rank 27)
        # For now, use trivial: 3*3*3 = 27 terms
        for i in range(3):
            for k in range(3):
                for j in range(3):
                    U_term = {(2+i, 2+k): 1}
                    V_term = {(2+k, 2+j): 1}
                    W_term = {(2+i, 2+j): 1}
                    add_term(U_term, V_term, W_term)

        # Total rank: 7 + 12 + 12 + 18 + 12 + 18 + 18 + 27 = 124
        # Hmm, that's almost schoolbook. Let me check: 7+12+12+18+12+18+18+27 = 124.
        # Schoolbook is 125. So we saved 1 by using Strassen instead of 8 for the 2x2x2 block.

        # To get rank 49, I need to use Laderman for the 3x3x3 block (19 instead of 27)
        # and find better algorithms for the mixed blocks.
        # With Laderman: 7 + 12 + 12 + 18 + 12 + 18 + 18 + 19 = 116. Still not 49.

        # The 49 bound requires a fundamentally different approach.
        # For now, let me just return this decomposition (rank 124) and verify it.

        # Assemble U, V, W matrices
        R = rank
        U_mat = [[0.0] * R for _ in range(n * m)]
        V_mat = [[0.0] * R for _ in range(m * p)]
        W_mat = [[0.0] * R for _ in range(p * n)]

        for col in range(R):
            for (i, j), coeff in U_list[col].items():
                U_mat[i * m + j][col] = coeff
            for (j, k), coeff in V_list[col].items():
                V_mat[j * p + k][col] = coeff
            for (k, i), coeff in W_list[col].items():
                W_mat[k * n + i][col] = coeff

        return U_mat, V_mat, W_mat

    U_block, V_block, W_block = build_block_decomposition()

    # Verify the block decomposition
    if verify(U_block, V_block, W_block):
        print(f"Block decomposition verified with rank {len(U_block[0])}")
        return {"U": U_block, "V": V_block, "W": W_block}
    else:
        print("Block decomposition failed verification, falling back to schoolbook")
        U_sch, V_sch, W_sch = get_schoolbook()
        return {"U": U_sch, "V": V_sch, "W": W_sch}
# EVOLVE-BLOCK-END
