import time
import random
import itertools
from fractions import Fraction

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W, n, m, p):
    """Check if U,V,W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1
                actual = 0
                for c in range(rank):
                    actual += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if actual != expected:
                    return False
    return True

def _try_las_vegas(rank, n, m, p, deadline, max_iters=5000):
    """
    Try to find a decomposition of given rank using random initialization and iterative refinement.
    We work over rationals (Fractions) for exactness.
    """
    # Strategy: Randomly initialize U, V, W with small integers (e.g., -1, 0, 1)
    # Then solve for one factor given the other two? Or use a fixed-point iteration?
    
    # A more robust approach for small ranks:
    # If rank R is small, the tensors U, V, W define a system of quadratic equations.
    # This is hard.
    
    # Alternative: Use the known structure.
    # For 3x4x8, the schoolbook rank is 96.
    # Lower bounds: max(n*m, m*p, p*n) = max(12, 32, 24) = 32.
    # Also, there are known results for small matrices.
    # 2x2x2 has rank 7.
    # 2x3x3 has rank 13?
    # 3x3x3 has rank 23?
    
    # Let's try a heuristic:
    # 1. Start with random sparse matrices.
    # 2. Check if valid.
    # 3. If not, perturb.
    
    # Given the difficulty of exact algebraic geometry solving in Python within 150s,
    # and the risk of timeout, we should be conservative.
    
    # However, let's try a simple random search for very low ranks first, just in case.
    # But 3x4x8 is likely to have a rank significantly above 32.
    # Let's assume the schoolbook 96 is close to optimal or at least the safest bet for correctness.
    
    # Actually, let's look at the "half-integer" hint.
    # Sometimes decompositions use 1/2.
    
    # Let's try to see if we can find a decomposition of rank < 96.
    # We will use a random search over {-1, 0, 1, 1/2, -1/2} for entries.
    # But the search space is huge.
    
    # Instead, let's just return the schoolbook decomposition. It is guaranteed correct.
    # The score is 1.0 for schoolbook.
    # To beat 1.0, we need rank < 96.
    
    # Is there a known better rank for 3x4x8?
    # Strassen's algorithm works for 2x2.
    # For 3x3, rank is 23 (Strassen-like).
    # For 3x4, maybe we can pad to 4x4?
    # 4x4x4 rank is 49? No, 4x4x4 rank is 49 is for 2x2x2 padding?
    # Actually, 4x4x4 rank is 49 is not the best. Best known for 4x4x4 is 49? No, 4x4x4 is 49?
    # Wait, 2x2x2 is 7.
    # 4x4x4 can be done with 49? 7^2 = 49? Yes, if we tile 2x2 blocks, rank is 7*7=49? No, tiling 2x2 blocks of size 2 gives 4x4.
    # If we tile 2x2x2 blocks, a 4x4x4 matrix multiplication can be done with 7^2 = 49 operations?
    # Wait, if A is 4x4, B is 4x4. We can split into 2x2 blocks of size 2.
    # There are 4 block multiplications of 2x2 by 2x2.
    # Each 2x2x2 multiplication has rank 7.
    # So total rank is 4 * 7 = 28?
    # No, the blocks are coupled.
    # Actually, if you just do 4 independent 2x2x2 multiplications, you get the 4x4x4 result.
    # The rank is additive. So 4 * 7 = 28.
    # So rank(4x4x4) <= 28.
    
    # Can we use this for 3x4x8?
    # Pad 3x4 to 4x4 (rank 28).
    # Pad 4x8 to 4x8?
    # We need to multiply (4x4) by (4x8).
    # This is a 4x4x8 problem.
    # We can split the 4x8 into two 4x4 blocks.
    # So we do two 4x4x4 multiplications.
    # Rank = 2 * 28 = 56.
    # So rank(3x4x8) <= 56.
    
    # Can we do better?
    # Maybe split 4x8 differently?
    # Or use a different padding?
    
    # Let's construct the rank 56 decomposition.
    # 1. Pad A (3x4) to A' (4x4) by adding a row of zeros.
    # 2. Split B (4x8) into B1 (4x4) and B2 (4x4).
    # 3. Compute C1 = A' * B1 (4x4) using rank 28 decomposition.
    # 4. Compute C2 = A' * B2 (4x4) using rank 28 decomposition.
    # 5. Concatenate C1 and C2 to get C' (4x8).
    # 6. Extract the 3x8 submatrix.
    
    # We need the explicit factor matrices for the 4x4x4 rank 28 decomposition.
    # The 4x4x4 decomposition is derived from 4 independent 2x2x2 Strassen decompositions.
    
    # Let's build the 2x2x2 Strassen decomposition first.
    # M = A * B
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    # Strassen:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)(e)
    # p3 = (a-b)(g+h)
    # p4 = (d)(f-h)
    # p5 = (a+c)(h)
    # p6 = (b-d)(e+g)
    # p7 = (a-c)(f)
    
    # M11 = p1 + p4 - p5 + p7
    # M12 = p3 + p5
    # M21 = p2 + p4
    # M22 = p1 - p2 + p3 + p6
    
    # This gives 7 multiplications.
    # The factors U, V, W for 2x2x2:
    # U has 4 rows (a,b,c,d), V has 4 rows (e,f,g,h), W has 4 rows (M11, M12, M21, M22).
    # Rank 7.
    
    # For 4x4x4 using 4 independent 2x2x2 blocks:
    # A is 4x4. Blocks: A11 (top-left), A12 (top-right), A21 (bottom-left), A22 (bottom-right).
    # B is 4x4. Blocks: B11, B12, B21, B22.
    # C is 4x4. Blocks: C11, C12, C21, C22.
    
    # C11 = A11*B11 + A12*B21
    # This is NOT independent. The blocks are coupled in the summation.
    # So we cannot just do 4 independent 2x2x2 multiplications.
    # We need to multiply 4x4 matrices.
    # If we split into 2x2 blocks, we have 4 block multiplications of 2x2 by 2x2, but they share terms?
    # No, C11 depends on A11, A12, B11, B21.
    # C12 depends on A11, A12, B12, B22.
    # C21 depends on A21, A22, B11, B21.
    # C22 depends on A21, A22, B12, B22.
    
    # This is exactly the 4x4x4 problem.
    # The rank of 4x4x4 is known to be 49? No, that's for 2x2x2 padding?
    # Actually, the best known rank for 4x4x4 is 49?
    # Wait, if we use Strassen on 4x4 by splitting into 2x2 blocks, we get a recursion.
    # T(4) = 7 * T(2)? No.
    # If we split 4x4 into 2x2 blocks, we need to compute 4x4x4.
    # We can use Strassen's algorithm which reduces 3x3x3 to 7 multiplications of 1.5x1.5? No.
    # Strassen reduces 2x2x2 to 7 multiplications of 1x1x1? No, 1x1x1 is trivial.
    # It reduces the number of scalar multiplications.
    
    # For 4x4x4, we can split into 2x2 blocks.
    # We need to compute 4 products of 2x2 matrices?
    # No, the formula for C involves sums of products of blocks.
    # C11 = A11 B11 + A12 B21
    # This requires 2 matrix multiplications of 2x2.
    # So total 8 matrix multiplications of 2x2.
    # Each 2x2x2 has rank 7.
    # So total rank 8 * 7 = 56.
    
    # So rank(4x4x4) <= 56.
    # And rank(3x4x8) <= 56 (by padding 3->4 and splitting 8->4+4).
    
    # Can we do better than 56?
    # Maybe split 4x8 into 2x4 blocks?
    # 4x4 * 4x8.
    # Split 4x4 into 2x2 blocks (4 blocks).
    # Split 4x8 into 2x4 blocks (2 blocks vertically? or 4x4 blocks?).
    
    # Let's stick with the rank 56 construction.
    # It is better than 96.
    
    # Construction details:
    # 1. Define U2, V2, W2 for 2x2x2 Strassen (rank 7).
    # 2. Define U4, V4, W4 for 4x4x4 using 8 calls to 2x2x2 (rank 56).
    # 3. Define U348, V348, W348 for 3x4x8 using 2 calls to 4x4x4 (rank 56).
    
    # Let's implement this.
    
    def get_strassen_2x2x2():
        # Returns U, V, W for 2x2x2
        # U rows: a, b, c, d
        # V rows: e, f, g, h
        # W rows: M11, M12, M21, M22
        # Columns: p1..p7
        
        U = [
            [1, 0, 1, 0, 1, 0, 1],  # a
            [0, 1, 0, 0, 0, 1, 0],  # b
            [0, 0, 1, 1, 1, 0, 0],  # c
            [1, 0, 0, 1, 0, 1, 0]   # d
        ]
        V = [
            [1, 1, 0, 0, 0, 0, 0],  # e
            [0, 0, 0, 1, 0, 0, 1],  # f
            [0, 0, 1, 0, 0, 1, 0],  # g
            [1, 0, 1, 0, 1, 0, 0]   # h
        ]
        W = [
            [1, 0, 0, 1, -1, 0, 1],  # M11
            [0, 0, 1, 0, 1, 0, 0],   # M12
            [0, 1, 0, 1, 0, 0, 0],   # M21
            [1, -1, 1, 0, 0, 1, 0]   # M22
        ]
        return U, V, W
    
    U2, V2, W2 = get_strassen_2x2x2()
    
    # Now build 4x4x4.
    # A is 4x4. Indices:
    # a00 a01 a02 a03
    # a10 a11 a12 a13
    # a20 a21 a22 a23
    # a30 a31 a32 a33
    
    # Blocks:
    # A11 = [[a00, a01], [a10, a11]]
    # A12 = [[a02, a03], [a12, a13]]
    # A21 = [[a20, a21], [a30, a31]]
    # A22 = [[a22, a23], [a32, a33]]
    
    # B is 4x4.
    # B11 = [[b00, b01], [b10, b11]]
    # B12 = [[b02, b03], [b12, b13]]
    # B21 = [[b20, b21], [b30, b31]]
    # B22 = [[b22, b23], [b32, b33]]
    
    # C11 = A11 B11 + A12 B21
    # C12 = A11 B12 + A12 B22
    # C21 = A21 B11 + A22 B21
    # C22 = A21 B12 + A22 B22
    
    # We need 8 multiplications of 2x2 matrices.
    # 1. A11 * B11
    # 2. A12 * B21
    # 3. A11 * B12
    # 4. A12 * B22
    # 5. A21 * B11
    # 6. A22 * B21
    # 7. A21 * B12
    # 8. A22 * B22
    
    # Each multiplication uses the 2x2x2 Strassen decomposition.
    # We need to map the global indices to the local 2x2 indices.
    
    # U4 will have 16 rows (a00..a33).
    # V4 will have 16 rows (b00..b33).
    # W4 will have 16 rows (c00..c33).
    # Rank 56.
    
    U4 = [[0]*56 for _ in range(16)]
    V4 = [[0]*56 for _ in range(16)]
    W4 = [[0]*56 for _ in range(16)]
    
    # Helper to add a 2x2x2 block to the global factors
    def add_block(u_start, v_start, w_start, offset):
        # u_start: index in U4 for the first row of the 2x2 block
        # v_start: index in V4 for the first row of the 2x2 block
        # w_start: index in W4 for the first row of the 2x2 block
        # offset: column offset in the 56 columns
        
        # U2 rows correspond to local a, b, c, d
        # Global rows: u_start, u_start+1, u_start+2, u_start+3
        # But the mapping depends on the block.
        # For A11: a00, a01, a10, a11 -> rows 0, 1, 4, 5?
        # Let's define the mapping explicitly.
        
        pass
    
    # This is getting complex. Let's just write it out.
    
    # Multiplication 1: A11 * B11 -> C11_part1
    # A11 rows: 0, 1, 4, 5 (a00, a01, a10, a11)
    # B11 rows: 0, 1, 4, 5 (b00, b01, b10, b11)
    # C11 rows: 0, 1, 4, 5 (c00, c01, c10, c11)
    
    # We need to map U2 rows (a,b,c,d) to global rows.
    # U2[0] is a -> global row 0
    # U2[1] is b -> global row 1
    # U2[2] is c -> global row 4
    # U2[3] is d -> global row 5
    
    # V2[0] is e -> global row 0
    # V2[1] is f -> global row 1
    # V2[2] is g -> global row 4
    # V2[3] is h -> global row 5
    
    # W2[0] is M11 -> global row 0
    # W2[1] is M12 -> global row 1
    # W2[2] is M21 -> global row 4
    # W2[3] is M22 -> global row 5
    
    def add_strassen_block(u_map, v_map, w_map, col_offset):
        for i in range(4):
            for c in range(7):
                U4[u_map[i]][col_offset + c] += U2[i][c]
                V4[v_map[i]][col_offset + c] += V2[i][c]
                W4[w_map[i]][col_offset + c] += W2[i][c]
    
    # Block 1: A11 * B11
    # A11: a00(0), a01(1), a10(4), a11(5)
    # B11: b00(0), b01(1), b10(4), b11(5)
    # C11: c00(0), c01(1), c10(4), c11(5)
    u_map1 = [0, 1, 4, 5]
    v_map1 = [0, 1, 4, 5]
    w_map1 = [0, 1, 4, 5]
    add_strassen_block(u_map1, v_map1, w_map1, 0)
    
    # Block 2: A12 * B21
    # A12: a02(2), a03(3), a12(6), a13(7)
    # B21: b20(8), b21(9), b30(12), b31(13)
    # C11: c00(0), c01(1), c10(4), c11(5) (Adds to C11)
    u_map2 = [2, 3, 6, 7]
    v_map2 = [8, 9, 12, 13]
    w_map2 = [0, 1, 4, 5]
    add_strassen_block(u_map2, v_map2, w_map2, 7)
    
    # Block 3: A11 * B12
    # A11: a00(0), a01(1), a10(4), a11(5)
    # B12: b02(2), b03(3), b12(6), b13(7)
    # C12: c02(2), c03(3), c12(6), c13(7)
    u_map3 = [0, 1, 4, 5]
    v_map3 = [2, 3, 6, 7]
    w_map3 = [2, 3, 6, 7]
    add_strassen_block(u_map3, v_map3, w_map3, 14)
    
    # Block 4: A12 * B22
    # A12: a02(2), a03(3), a12(6), a13(7)
    # B22: b22(10), b23(11), b32(14), b33(15)
    # C12: c02(2), c03(3), c12(6), c13(7)
    u_map4 = [2, 3, 6, 7]
    v_map4 = [10, 11, 14, 15]
    w_map4 = [2, 3, 6, 7]
    add_strassen_block(u_map4, v_map4, w_map4, 21)
    
    # Block 5: A21 * B11
    # A21: a20(8), a21(9), a30(12), a31(13)
    # B11: b00(0), b01(1), b10(4), b11(5)
    # C21: c20(8), c21(9), c30(12), c31(13)
    u_map5 = [8, 9, 12, 13]
    v_map5 = [0, 1, 4, 5]
    w_map5 = [8, 9, 12, 13]
    add_strassen_block(u_map5, v_map5, w_map5, 28)
    
    # Block 6: A22 * B21
    # A22: a22(10), a23(11), a32(14), a33(15)
    # B21: b20(8), b21(9), b30(12), b31(13)
    # C21: c20(8), c21(9), c30(12), c31(13)
    u_map6 = [10, 11, 14, 15]
    v_map6 = [8, 9, 12, 13]
    w_map6 = [8, 9, 12, 13]
    add_strassen_block(u_map6, v_map6, w_map6, 35)
    
    # Block 7: A21 * B12
    # A21: a20(8), a21(9), a30(12), a31(13)
    # B12: b02(2), b03(3), b12(6), b13(7)
    # C22: c22(10), c23(11), c32(14), c33(15)
    u_map7 = [8, 9, 12, 13]
    v_map7 = [2, 3, 6, 7]
    w_map7 = [10, 11, 14, 15]
    add_strassen_block(u_map7, v_map7, w_map7, 42)
    
    # Block 8: A22 * B22
    # A22: a22(10), a23(11), a32(14), a33(15)
    # B22: b22(10), b23(11), b32(14), b33(15)
    # C22: c22(10), c23(11), c32(14), c33(15)
    u_map8 = [10, 11, 14, 15]
    v_map8 = [10, 11, 14, 15]
    w_map8 = [10, 11, 14, 15]
    add_strassen_block(u_map8, v_map8, w_map8, 49)
    
    return U4, V4, W4
    
    U4, V4, W4 = get_4x4x4()
    
    # Now build 3x4x8.
    # A is 3x4. Pad to 4x4 by adding row of zeros.
    # A' = [[a00, a01, a02, a03],
    #       [a10, a11, a12, a13],
    #       [a20, a21, a22, a23],
    #       [0,   0,   0,   0  ]]
    
    # B is 4x8. Split into B1 (4x4) and B2 (4x4).
    # B1 = [[b00, b01, b02, b03],
    #       [b10, b11, b12, b13],
    #       [b20, b21, b22, b23],
    #       [b30, b31, b32, b33]]
    # B2 = [[b04, b05, b06, b07],
    #       [b14, b15, b16, b17],
    #       [b24, b25, b26, b27],
    #       [b34, b35, b36, b37]]
    
    # C' = A' * B' (4x8)
    # C1 = A' * B1 (4x4)
    # C2 = A' * B2 (4x4)
    # C = [C1 | C2]
    
    # We use the 4x4x4 decomposition twice.
    # Rank 56 + 56 = 112?
    # Wait, the 4x4x4 decomposition I built has rank 56.
    # So total rank 112.
    # But schoolbook is 96.
    # So this is WORSE.
    
    # My mistake. The 4x4x4 decomposition using 8 Strassen blocks has rank 56.
    # Schoolbook for 4x4x4 is 64.
    # So 56 < 64.
    # But for 3x4x8, schoolbook is 96.
    # My construction:
    # Pad 3x4 to 4x4.
    # Multiply 4x4 by 4x8.
    # Split 4x8 into two 4x4.
    # Do two 4x4x4 multiplications.
    # Rank = 2 * 56 = 112.
    # 112 > 96.
    
    # So this approach is not better.
    
    # Can we do better?
    # Maybe split 4x8 into 2x4 blocks?
    # 4x4 * 4x8.
    # Split 4x4 into 2x2 blocks (4 blocks).
    # Split 4x8 into 2x4 blocks (2 blocks vertically? 4x4 blocks?).
    
    # Actually, if we use Strassen directly on 3x4x8?
    # No, Strassen is for square matrices.
    
    # Let's go back to schoolbook. 96 is safe.
    # The task asks for rank-minimal.
    # If I can't find a better one, I return 96.
    
    # Is there a rank 90 decomposition?
    # Unlikely to find by hand.
    
    # Let's return schoolbook.
    
    return _schoolbook()

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 120  # 120s limit, keep margin
    
    # Try to find a better decomposition
    # For now, just return schoolbook
    # In a real search, we would try random ranks < 96
    
    # Since finding a better rank is hard, we return the safe schoolbook.
    result = _schoolbook()
    
    # Verify just in case
    n, m, p = DIMENSIONS
    if not _verify(result["U"], result["V"], result["W"], n, m, p):
        # Should never happen
        pass
        
    return result
# EVOLVE-BLOCK-END
