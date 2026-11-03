"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import time
import itertools
from fractions import Fraction

# We will use Strassen for 2x2 blocks to reduce rank.
# The matrix multiplication is (3x4) * (4x4) -> (3x4).
# We can view the 3x4 matrix as 3 rows of 4 elements, or 4 columns of 3.
# Let's view A as 3x4 and B as 4x4.
# We can split the 4 columns of A and 4 rows of B into two groups of 2.
# A = [A1, A2] where A1, A2 are 3x2.
# B = [B1; B2] where B1, B2 are 2x4.
# Then C = A*B = A1*B1 + A2*B2.
# Each term is (3x2) * (2x4) -> (3x4).
# We can further split the 3 rows into 2+1.
# A1 = [A1a; A1b] where A1a is 2x2, A1b is 1x2.
# B1 = [B1a B1b] where B1a is 2x2, B1b is 2x2.
# A1*B1 = A1a*B1a + A1b*B1b.
# A1a*B1a is (2x2)*(2x2) -> (2x2). We can use Strassen (rank 7) here.
# A1b*B1b is (1x2)*(2x2) -> (1x2). This is just 2 scalar multiplications (rank 2).
# Similarly for A2*B2.
#
# Let's be more systematic.
# A is 3x4, B is 4x4.
# Split columns of A into 2+2: A = [A_left, A_right], each 3x2.
# Split rows of B into 2+2: B = [B_top; B_bottom], each 2x4.
# C = A_left * B_top + A_right * B_bottom.
#
# Now consider A_left * B_top. A_left is 3x2, B_top is 2x4.
# Split rows of A_left into 2+1: A_left = [A_lt; A_lb], A_lt 2x2, A_lb 1x2.
# Split columns of B_top into 2+2: B_top = [B_tt B_tb], B_tt 2x2, B_tb 2x2.
# A_left * B_top = A_lt * B_tt + A_lb * B_tb.
# A_lt * B_tt: (2x2)*(2x2) -> (2x2). Use Strassen (rank 7).
# A_lb * B_tb: (1x2)*(2x2) -> (1x2). Rank 2.
#
# Similarly for A_right * B_bottom.
# A_right = [A_rt; A_rb], A_rt 2x2, A_rb 1x2.
# B_bottom = [B_bt B_bb], B_bt 2x2, B_bb 2x2.
# A_right * B_bottom = A_rt * B_bt + A_rb * B_bb.
# A_rt * B_bt: (2x2)*(2x2) -> (2x2). Use Strassen (rank 7).
# A_rb * B_bb: (1x2)*(2x2) -> (1x2). Rank 2.
#
# Total rank = 7 + 2 + 7 + 2 = 18.
# Wait, this seems too low. Let's re-evaluate.
# The output C is 3x4.
# C_top_left (2x2) comes from A_lt * B_tt. (Strassen rank 7)
# C_top_right (2x2) comes from A_lt * B_tb. (Strassen rank 7)
# C_bottom_left (1x2) comes from A_lb * B_tt. (Rank 2)
# C_bottom_right (1x2) comes from A_lb * B_tb. (Rank 2)
#
# Wait, A_left * B_top = [A_lt; A_lb] * [B_tt B_tb] = [A_lt*B_tt  A_lt*B_tb; A_lb*B_tt  A_lb*B_tb].
# So the top-left 2x2 block of C (which is 2x2) is A_lt*B_tt.
# The top-right 2x2 block of C (which is 2x2) is A_lt*B_tb.
# The bottom-left 1x2 block of C (which is 1x2) is A_lb*B_tt.
# The bottom-right 1x2 block of C (which is 1x2) is A_lb*B_tb.
#
# Similarly for A_right * B_bottom.
# A_right * B_bottom = [A_rt; A_rb] * [B_bt B_bb] = [A_rt*B_bt  A_rt*B_bb; A_rb*B_bt  A_rb*B_bb].
# This contributes to the same blocks of C.
#
# So:
# C[0:2, 0:2] = A_lt*B_tt + A_rt*B_bt  (two 2x2 Strassen products)
# C[0:2, 2:4] = A_lt*B_tb + A_rt*B_bb  (two 2x2 Strassen products)
# C[2:3, 0:2] = A_lb*B_tt + A_rb*B_bt  (two 1x2 products, rank 2 each)
# C[2:3, 2:4] = A_lb*B_tb + A_rb*B_bb  (two 1x2 products, rank 2 each)
#
# Total rank = 7 + 7 + 7 + 7 + 2 + 2 + 2 + 2 = 36.
# This is better than 48.
#
# Let's verify the Strassen decomposition for 2x2.
# M1 = (A11 + A22)(B11 + B22)
# M2 = (A21 + A22)B11
# M3 = A11(B21 - B22)
# M4 = A22(B12 - B11)
# M5 = (A11 + A12)B22
# M6 = (A21 - A11)(B11 + B12)
# M7 = (A12 - A22)(B21 + B22)
# C11 = M1 + M4 - M5 + M7
# C12 = M3 + M5
# C21 = M2 + M4
# C22 = M1 - M2 + M3 + M6
#
# We need to construct the U, V, W matrices for the whole 3x4 * 4x4 multiplication.
# The tensor T_{i,j,k} = 1 if we are computing C[i,k] from A[i,j]*B[j,k].
# The rank decomposition is a sum of rank-1 tensors.
# Each rank-1 tensor corresponds to a scalar multiplication u * v * w.
# U has shape (n*m, rank) = (12, rank).
# V has shape (m*p, rank) = (16, rank).
# W has shape (p*n, rank) = (12, rank).
# Note: W[k*n+i, r] is the coefficient for C[i,k] in the r-th rank-1 term.
#
# Let's build the decomposition term by term.
# We will use a list of (u, v, w) tuples, where u, v, w are sparse vectors.
# Then we can assemble U, V, W.

def construct():
    n, m, p = 3, 4, 4
    # We will build the rank-1 terms.
    # Each term is (u, v, w) where:
    # u is a vector of length n*m = 12. u[i*m + j] is the coefficient for A[i,j].
    # v is a vector of length m*p = 16. v[j*p + k] is the coefficient for B[j,k].
    # w is a vector of length p*n = 12. w[k*n + i] is the coefficient for C[i,k].
    
    terms = []
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices, coeff=1):
        # u_indices: list of (index, value)
        # v_indices: list of (index, value)
        # w_indices: list of (index, value)
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        for idx, val in u_indices:
            u[idx] = val
        for idx, val in v_indices:
            v[idx] = val
        for idx, val in w_indices:
            w[idx] = val
        terms.append((u, v, w))
    
    # Strassen 2x2 block for A_lt (rows 0,1; cols 0,1) * B_tt (rows 0,1; cols 0,1)
    # A_lt is A[0:2, 0:2], B_tt is B[0:2, 0:2].
    # Output is C[0:2, 0:2].
    # M1 = (A11 + A22)(B11 + B22) -> C11 += M1, C22 += M1
    # In our indexing:
    # A11 = A[0,0] -> u[0]
    # A22 = A[1,1] -> u[1]
    # B11 = B[0,0] -> v[0]
    # B22 = B[1,1] -> v[1]
    # C11 = C[0,0] -> w[0]
    # C22 = C[1,1] -> w[1*3 + 1] = w[4]
    
    # M1
    add_term([(0,1), (1,1)], [(0,1), (1,1)], [(0,1), (4,1)])
    # M2 = (A21 + A22)B11 -> C21 += M2
    # A21 = A[1,0] -> u[1*4 + 0] = u[4]
    # A22 = A[1,1] -> u[1]
    # B11 = B[0,0] -> v[0]
    # C21 = C[1,0] -> w[0*3 + 1] = w[1]
    add_term([(4,1), (1,1)], [(0,1)], [(1,1)])
    # M3 = A11(B21 - B22) -> C12 += M3
    # A11 = A[0,0] -> u[0]
    # B21 = B[1,0] -> v[1*4 + 0] = v[4]
    # B22 = B[1,1] -> v[1]
    # C12 = C[0,1] -> w[1*3 + 0] = w[3]
    add_term([(0,1)], [(4,1), (1,-1)], [(3,1)])
    # M4 = A22(B12 - B11) -> C11 += M4, C21 += M4
    # A22 = A[1,1] -> u[1]
    # B12 = B[0,1] -> v[0*4 + 1] = v[1]
    # B11 = B[0,0] -> v[0]
    # C11 = C[0,0] -> w[0]
    # C21 = C[1,0] -> w[1]
    add_term([(1,1)], [(1,1), (0,-1)], [(0,1), (1,1)])
    # M5 = (A11 + A12)B22 -> C11 -= M5, C12 += M5
    # A11 = A[0,0] -> u[0]
    # A12 = A[0,1] -> u[0*4 + 1] = u[1]
    # B22 = B[1,1] -> v[1]
    # C11 = C[0,0] -> w[0]
    # C12 = C[0,1] -> w[3]
    add_term([(0,1), (1,1)], [(1,1)], [(0,-1), (3,1)])
    # M6 = (A21 - A11)(B11 + B12) -> C22 += M6
    # A21 = A[1,0] -> u[4]
    # A11 = A[0,0] -> u[0]
    # B11 = B[0,0] -> v[0]
    # B12 = B[0,1] -> v[1]
    # C22 = C[1,1] -> w[4]
    add_term([(4,1), (0,-1)], [(0,1), (1,1)], [(4,1)])
    # M7 = (A12 - A22)(B21 + B22) -> C11 += M7
    # A12 = A[0,1] -> u[1]
    # A22 = A[1,1] -> u[1]  (Wait, A22 is u[1] in this block? No, A22 is A[1,1] which is u[1*4+1]=u[5])
    # Let's re-check indices.
    # A_lt is A[0:2, 0:2].
    # A[0,0] -> u[0]
    # A[0,1] -> u[1]
    # A[1,0] -> u[4]
    # A[1,1] -> u[5]
    # B_tt is B[0:2, 0:2].
    # B[0,0] -> v[0]
    # B[0,1] -> v[1]
    # B[1,0] -> v[4]
    # B[1,1] -> v[5]
    # C[0,0] -> w[0]
    # C[0,1] -> w[3]
    # C[1,0] -> w[1]
    # C[1,1] -> w[4]
    
    # Let's restart the Strassen terms with correct indices.
    terms.clear()
    
    # Block 1: A_lt (rows 0,1; cols 0,1) * B_tt (rows 0,1; cols 0,1) -> C[0:2, 0:2]
    # A indices: 0,1,4,5. B indices: 0,1,4,5. C indices: 0,3,1,4.
    # M1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1]) -> C[0,0]+=1, C[1,1]+=1
    add_term([(0,1), (5,1)], [(0,1), (5,1)], [(0,1), (4,1)])
    # M2 = (A[1,0]+A[1,1])B[0,0] -> C[1,0]+=1
    add_term([(4,1), (5,1)], [(0,1)], [(1,1)])
    # M3 = A[0,0](B[1,0]-B[1,1]) -> C[0,1]+=1
    add_term([(0,1)], [(4,1), (5,-1)], [(3,1)])
    # M4 = A[1,1](B[0,1]-B[0,0]) -> C[0,0]+=1, C[1,0]+=1
    add_term([(5,1)], [(1,1), (0,-1)], [(0,1), (1,1)])
    # M5 = (A[0,0]+A[0,1])B[1,1] -> C[0,0]-=1, C[0,1]+=1
    add_term([(0,1), (1,1)], [(5,1)], [(0,-1), (3,1)])
    # M6 = (A[1,0]-A[0,0])(B[0,0]+B[0,1]) -> C[1,1]+=1
    add_term([(4,1), (0,-1)], [(0,1), (1,1)], [(4,1)])
    # M7 = (A[0,1]-A[1,1])(B[1,0]+B[1,1]) -> C[0,0]+=1
    add_term([(1,1), (5,-1)], [(4,1), (5,1)], [(0,1)])
    
    # Block 2: A_lt (rows 0,1; cols 0,1) * B_tb (rows 0,1; cols 2,3) -> C[0:2, 2:4]
    # A indices: 0,1,4,5. B indices: 2,3,6,7. C indices: 2,5,2+3=5? No.
    # C[0,2] -> w[2*3+0]=w[6]
    # C[0,3] -> w[3*3+0]=w[9]
    # C[1,2] -> w[2*3+1]=w[7]
    # C[1,3] -> w[3*3+1]=w[10]
    
    # M1 = (A[0,0]+A[1,1])(B[0,2]+B[1,3]) -> C[0,2]+=1, C[1,3]+=1
    add_term([(0,1), (5,1)], [(2,1), (7,1)], [(6,1), (10,1)])
    # M2 = (A[1,0]+A[1,1])B[0,2] -> C[1,2]+=1
    add_term([(4,1), (5,1)], [(2,1)], [(7,1)])
    # M3 = A[0,0](B[1,2]-B[1,3]) -> C[0,3]+=1
    add_term([(0,1)], [(6,1), (7,-1)], [(9,1)])
    # M4 = A[1,1](B[0,3]-B[0,2]) -> C[0,2]+=1, C[1,2]+=1
    add_term([(5,1)], [(3,1), (2,-1)], [(6,1), (7,1)])
    # M5 = (A[0,0]+A[0,1])B[1,3] -> C[0,2]-=1, C[0,3]+=1
    add_term([(0,1), (1,1)], [(7,1)], [(6,-1), (9,1)])
    # M6 = (A[1,0]-A[0,0])(B[0,2]+B[0,3]) -> C[1,3]+=1
    add_term([(4,1), (0,-1)], [(2,1), (3,1)], [(10,1)])
    # M7 = (A[0,1]-A[1,1])(B[1,2]+B[1,3]) -> C[0,2]+=1
    add_term([(1,1), (5,-1)], [(6,1), (7,1)], [(6,1)])
    
    # Block 3: A_lb (row 2; cols 0,1) * B_tt (rows 0,1; cols 0,1) -> C[2:3, 0:2]
    # A indices: 8,9. B indices: 0,1,4,5. C indices: 0*3+2=2, 1*3+2=5.
    # C[2,0] -> w[0*3+2]=w[2]
    # C[2,1] -> w[1*3+2]=w[5]
    # This is (1x2)*(2x2). Just 2 multiplications.
    # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0]
    # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1]
    # Term 1: A[2,0]*B[0,0] -> C[2,0]
    add_term([(8,1)], [(0,1)], [(2,1)])
    # Term 2: A[2,1]*B[1,0] -> C[2,0]
    add_term([(9,1)], [(4,1)], [(2,1)])
    # Term 3: A[2,0]*B[0,1] -> C[2,1]
    add_term([(8,1)], [(1,1)], [(5,1)])
    # Term 4: A[2,1]*B[1,1] -> C[2,1]
    add_term([(9,1)], [(5,1)], [(5,1)])
    
    # Block 4: A_lb (row 2; cols 0,1) * B_tb (rows 0,1; cols 2,3) -> C[2:3, 2:4]
    # A indices: 8,9. B indices: 2,3,6,7. C indices: 2*3+2=8, 3*3+2=11.
    # C[2,2] -> w[2*3+2]=w[8]
    # C[2,3] -> w[3*3+2]=w[11]
    # Term 1: A[2,0]*B[0,2] -> C[2,2]
    add_term([(8,1)], [(2,1)], [(8,1)])
    # Term 2: A[2,1]*B[1,2] -> C[2,2]
    add_term([(9,1)], [(6,1)], [(8,1)])
    # Term 3: A[2,0]*B[0,3] -> C[2,3]
    add_term([(8,1)], [(3,1)], [(11,1)])
    # Term 4: A[2,1]*B[1,3] -> C[2,3]
    add_term([(9,1)], [(7,1)], [(11,1)])
    
    # Block 5: A_rt (rows 0,1; cols 2,3) * B_bt (rows 2,3; cols 0,1) -> C[0:2, 0:2]
    # A indices: 2,3,6,7. B indices: 8,9,12,13. C indices: 0,3,1,4.
    # M1 = (A[0,2]+A[1,3])(B[2,0]+B[3,1]) -> C[0,0]+=1, C[1,1]+=1
    add_term([(2,1), (7,1)], [(8,1), (13,1)], [(0,1), (4,1)])
    # M2 = (A[1,2]+A[1,3])B[2,0] -> C[1,0]+=1
    add_term([(6,1), (7,1)], [(8,1)], [(1,1)])
    # M3 = A[0,2](B[3,0]-B[3,1]) -> C[0,1]+=1
    add_term([(2,1)], [(12,1), (13,-1)], [(3,1)])
    # M4 = A[1,3](B[2,1]-B[2,0]) -> C[0,0]+=1, C[1,0]+=1
    add_term([(7,1)], [(9,1), (8,-1)], [(0,1), (1,1)])
    # M5 = (A[0,2]+A[0,3])B[3,1] -> C[0,0]-=1, C[0,1]+=1
    add_term([(2,1), (3,1)], [(13,1)], [(0,-1), (3,1)])
    # M6 = (A[1,2]-A[0,2])(B[2,0]+B[2,1]) -> C[1,1]+=1
    add_term([(6,1), (2,-1)], [(8,1), (9,1)], [(4,1)])
    # M7 = (A[0,3]-A[1,3])(B[3,0]+B[3,1]) -> C[0,0]+=1
    add_term([(3,1), (7,-1)], [(12,1), (13,1)], [(0,1)])
    
    # Block 6: A_rt (rows 0,1; cols 2,3) * B_bb (rows 2,3; cols 2,3) -> C[0:2, 2:4]
    # A indices: 2,3,6,7. B indices: 10,11,14,15. C indices: 6,9,7,10.
    # M1 = (A[0,2]+A[1,3])(B[2,2]+B[3,3]) -> C[0,2]+=1, C[1,3]+=1
    add_term([(2,1), (7,1)], [(10,1), (15,1)], [(6,1), (10,1)])
    # M2 = (A[1,2]+A[1,3])B[2,2] -> C[1,2]+=1
    add_term([(6,1), (7,1)], [(10,1)], [(7,1)])
    # M3 = A[0,2](B[3,2]-B[3,3]) -> C[0,3]+=1
    add_term([(2,1)], [(14,1), (15,-1)], [(9,1)])
    # M4 = A[1,3](B[2,3]-B[2,2]) -> C[0,2]+=1, C[1,2]+=1
    add_term([(7,1)], [(11,1), (10,-1)], [(6,1), (7,1)])
    # M5 = (A[0,2]+A[0,3])B[3,3] -> C[0,2]-=1, C[0,3]+=1
    add_term([(2,1), (3,1)], [(15,1)], [(6,-1), (9,1)])
    # M6 = (A[1,2]-A[0,2])(B[2,2]+B[2,3]) -> C[1,3]+=1
    add_term([(6,1), (2,-1)], [(10,1), (11,1)], [(10,1)])
    # M7 = (A[0,3]-A[1,3])(B[3,2]+B[3,3]) -> C[0,2]+=1
    add_term([(3,1), (7,-1)], [(14,1), (15,1)], [(6,1)])
    
    # Block 7: A_rb (row 2; cols 2,3) * B_bt (rows 2,3; cols 0,1) -> C[2:3, 0:2]
    # A indices: 10,11. B indices: 8,9,12,13. C indices: 2,5.
    # Term 1: A[2,2]*B[2,0] -> C[2,0]
    add_term([(10,1)], [(8,1)], [(2,1)])
    # Term 2: A[2,3]*B[3,0] -> C[2,0]
    add_term([(11,1)], [(12,1)], [(2,1)])
    # Term 3: A[2,2]*B[2,1] -> C[2,1]
    add_term([(10,1)], [(9,1)], [(5,1)])
    # Term 4: A[2,3]*B[3,1] -> C[2,1]
    add_term([(11,1)], [(13,1)], [(5,1)])
    
    # Block 8: A_rb (row 2; cols 2,3) * B_bb (rows 2,3; cols 2,3) -> C[2:3, 2:4]
    # A indices: 10,11. B indices: 10,11,14,15. C indices: 8,11.
    # Term 1: A[2,2]*B[2,2] -> C[2,2]
    add_term([(10,1)], [(10,1)], [(8,1)])
    # Term 2: A[2,3]*B[3,2] -> C[2,2]
    add_term([(11,1)], [(14,1)], [(8,1)])
    # Term 3: A[2,2]*B[2,3] -> C[2,3]
    add_term([(10,1)], [(11,1)], [(11,1)])
    # Term 4: A[2,3]*B[3,3] -> C[2,3]
    add_term([(11,1)], [(15,1)], [(11,1)])
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u, v, w) in enumerate(terms):
        for i in range(len(u)):
            if u[i] != 0:
                U[i][r] = u[i]
        for i in range(len(v)):
            if v[i] != 0:
                V[i][r] = v[i]
        for i in range(len(w)):
            if w[i] != 0:
                W[i][r] = w[i]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
