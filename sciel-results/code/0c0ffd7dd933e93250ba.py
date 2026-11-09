"""Matrix multiplication 2x4 by 4x7 rank-minimal decomposition search."""
import numpy as np
import itertools
from fractions import Fraction
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    for r in range(rank):
        T += np.outer(np.outer(U_arr[:, r], V_arr[:, r]), W_arr[:, r])
    T_true = _build_tensor(n, m, p).astype(np.float64)
    return np.allclose(T, T_true, atol=tol)

def _check_half_integer(mat):
    """Check if all entries are integer or half-integer (i.e., 2*x is integer)."""
    for row in mat:
        for val in row:
            if isinstance(val, (int, np.integer)):
                continue
            v = float(val)
            if abs(2 * v - round(2 * v)) > 1e-9:
                return False
    return True

def _to_rational_list(mat):
    """Convert a matrix of floats (that are half-integers) to list of Fractions."""
    result = []
    for row in mat:
        r = []
        for val in row:
            v = float(val)
            r.append(Fraction(round(2 * v), 2))
        result.append(r)
    return result

def all_half_int_arr(arr, tol=1e-6):
    """Check if all entries in array are integer or half-integer."""
    for val in arr.flat:
        v = float(val)
        if abs(2 * v - round(2 * v)) > tol:
            return False
    return True

def _to_list(arr):
    """Convert numpy array to list of lists with rounded half-integer values."""
    result = []
    for row in arr:
        r = []
        for val in row:
            v = float(val)
            rv = round(2 * v) / 2.0
            r.append(rv)
        result.append(r)
    return result

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W

def _strassen_block(n, m, p):
    """
    Construct a Strassen-based decomposition for n x m by m x p matrix multiplication.
    This works best when n, m, p are even (or can be padded, but here we assume compatibility).
    For n=2, m=4, p=7:
    We split the matrix multiplication into blocks.
    A is 2x4. We can view A as two 2x2 blocks? No, 2x4 is not a square of 2x2 blocks in a way that standard Strassen applies directly to the whole product unless we pad or split differently.
    
    Let's look at the dimensions: 2x4 * 4x7 = 2x7.
    We can split the 4x4 middle dimension into two 2x2 blocks?
    A = [A1 A2] where A1 is 2x2, A2 is 2x2.
    B = [B1; B2] where B1 is 2x7, B2 is 2x7.
    This doesn't immediately allow Strassen on 2x2 blocks because B is not 2x2.
    
    Alternative: Split the output 2x7 into two 2x3 and 2x4? No.
    
    Let's try to split the 4 dimension (middle) into 2+2.
    A = [A_left | A_right] (2x2, 2x2)
    B = [B_top; B_bottom] (2x7, 2x7)
    C = A_left @ B_top + A_right @ B_bottom.
    
    We can apply Strassen to the 2x2 block multiplication if we treat the 7 columns as separate?
    No, Strassen is for 2x2 * 2x2.
    
    However, we can use the fact that we can decompose the 4x4 multiplication part.
    Wait, the multiplication is (2x4) * (4x7).
    We can pad the 2x2 to 4x4? No, that increases rank.
    
    Let's look at the "block-wise Strassen" claim in the knowledge:
    "Replacing the trivial schoolbook algorithm with a block-wise Strassen decomposition reduces the rank from 56 to 50 by exploiting the structure of 2x2 sub-blocks."
    
    How does 56 become 50?
    Schoolbook: 2*4*7 = 56.
    If we use Strassen on 2x2 blocks, the rank of 2x2*2x2 is 7 instead of 8.
    If we can decompose the problem into independent 2x2*2x2 multiplications, we save rank.
    
    Let's try to map the 2x4 * 4x7 to a set of 2x2 * 2x2 multiplications.
    We have 2 rows in A.
    We have 4 cols in A, 4 rows in B.
    We have 7 cols in B.
    
    If we split the 4 columns of A into two 2-column blocks: A1 (2x2), A2 (2x2).
    If we split the 4 rows of B into two 2-row blocks: B1 (2x7), B2 (2x7).
    Then C = A1 B1 + A2 B2.
    This is a sum of two products of 2x2 and 2x7 matrices.
    This is not directly Strassen.
    
    However, we can split the 7 columns of B into 2x2 blocks? No, 7 is not divisible by 2.
    
    Maybe the strategy is to pad the 7 to 8?
    If we pad p=7 to p=8, we can use 2x2 blocks for B as well.
    A (2x4) -> Two 2x2 blocks? No, A is 2x4.
    Let's view A as a 2x2 block of 1x2? No.
    
    Let's reconsider the standard Strassen application.
    Strassen reduces rank of 2x2*2x2 from 8 to 7.
    Rank of 4x4*4x4 via Strassen is 7^2 = 49.
    Rank of 4x4*4x4 schoolbook is 64.
    
    Here we have 2x4 * 4x7.
    If we embed this in 4x4 * 4x4, the rank would be 49.
    But we only need 2x7 output.
    
    Let's try a specific construction for 2x4 * 4x7.
    We can split the 4 (middle dim) into 2+2.
    A = [A1, A2] (2x2, 2x2)
    B = [B1; B2] (2x7, 2x7)
    C = A1 B1 + A2 B2.
    
    We can compute A1 B1 using 7 multiplications?
    A1 is 2x2. B1 is 2x7.
    Product is 2x7.
    We can treat B1 as two 2x2 blocks and one 2x1 block?
    B1 = [B11, B12, B13] where B11, B12 are 2x2, B13 is 2x1.
    A1 B1 = A1 B11 + A1 B12 + A1 B13.
    
    This seems complicated. Let's look for a simpler structural reduction.
    
    Actually, there is a known result: 
    R(2,4,7) <= 50.
    One way to achieve this is to use the following identity:
    Split the 4x4 core into blocks.
    
    Let's try to construct the rank-50 decomposition explicitly.
    
    We can use the following decomposition:
    The tensor T_{2,4,7} can be decomposed using 50 rank-1 tensors.
    
    How to construct it?
    We can use the fact that R(2,4,7) <= R(2,4,8) and R(2,4,8) can be computed by splitting 8 into 4+4?
    Or R(2,4,8) = 2 * R(1,4,8)? No.
    
    Let's try to use the block structure:
    A = [A1 A2] (2x2, 2x2)
    B = [B1; B2] (2x7, 2x7)
    C = A1 B1 + A2 B2.
    
    We need to compute A1 B1 (2x2 * 2x7) and A2 B2 (2x2 * 2x7).
    Let's focus on computing X = M N where M is 2x2 and N is 2x7.
    N = [N1, N2, N3] where N1, N2 are 2x2 and N3 is 2x1.
    X = M N1 + M N2 + M N3.
    
    M N1 (2x2 * 2x2) can be done in 7 mults.
    M N2 (2x2 * 2x2) can be done in 7 mults.
    M N3 (2x2 * 2x1) can be done in 4 mults (schoolbook) or less?
    Rank of 2x2 * 2x1 is 4?
    T_{2,2,1} has rank 4.
    
    So for one 2x2 * 2x7 product: 7 + 7 + 4 = 18 mults.
    We have two such products (A1 B1 and A2 B2).
    Total = 18 + 18 = 36 mults?
    Wait, this assumes we can separate them.
    C = A1 B1 + A2 B2.
    The outputs are summed.
    If we use independent Strassen for A1 B1 and A2 B2, we get 36 rank-1 terms.
    But we can potentially share some terms?
    No, the inputs are different.
    
    Wait, 36 is much better than 50. Why is the knowledge saying 50?
    Maybe my assumption that M N3 is rank 4 is correct, but maybe I missed a constraint or the "block-wise Strassen" refers to a specific known construction that achieves 50, not 36?
    Or maybe R(2,2,1) is not 4?
    T_{2,2,1}: T[i,j,k] = delta_{i,k}.
    Matricization T_{2,2,1} -> 4x1 matrix. Rank is 4.
    So 36 should be possible?
    
    Let's re-read the knowledge: "reduces the rank from 56 to 50".
    Perhaps the 36 construction is not valid due to the way the blocks are summed?
    C = A1 B1 + A2 B2.
    The output C is 2x7.
    The terms from A1 B1 and A2 B2 are added element-wise.
    Since the decomposition is a sum of rank-1 tensors, we can just sum the rank-1 tensors from both parts.
    If Part 1 has 18 rank-1 tensors and Part 2 has 18 rank-1 tensors, the total is 36.
    
    Is there a reason we can't do 36?
    Maybe the "half-integer" constraint prevents the optimal Strassen coefficients?
    Strassen uses coefficients 1, -1. These are integers.
    So 36 should be valid.
    
    Let's try to implement the 36-rank decomposition.
    
    Steps:
    1. Split A into A1 (cols 0-1), A2 (cols 2-3).
    2. Split B into B1 (rows 0-1), B2 (rows 2-3).
    3. Compute C1 = A1 B1.
    4. Compute C2 = A2 B2.
    5. C = C1 + C2.
    
    For C1 = A1 B1:
    A1 is 2x2. B1 is 2x7.
    Split B1 into B1_0 (cols 0-1, 2x2), B1_1 (cols 2-3, 2x2), B1_2 (cols 4-6, 2x3).
    Wait, 7 cols. 2+2+3.
    B1_2 is 2x3.
    
    Let's refine the split of B1 (2x7):
    B1 = [B1_0 (2x2), B1_1 (2x2), B1_2 (2x3)].
    
    A1 B1_0: 2x2 * 2x2. Rank 7 (Strassen).
    A1 B1_1: 2x2 * 2x2. Rank 7 (Strassen).
    A1 B1_2: 2x2 * 2x3.
    
    What is the rank of 2x2 * 2x3?
    T_{2,2,3}.
    Schoolbook: 2*2*3 = 12.
    Can we do better?
    Split 2x3 into 2x2 + 2x1?
    A1 B1_2 = A1 B1_2a + A1 B1_2b.
    A1 B1_2a (2x2 * 2x2): Rank 7.
    A1 B1_2b (2x2 * 2x1): Rank 4.
    Total for A1 B1_2: 11.
    
    So total for A1 B1: 7 + 7 + 11 = 25.
    Total for A2 B2: 25.
    Total Rank: 50.
    
    Ah, so if we split the 7 into 2+2+3, and then 3 into 2+1, we get:
    2x2 * 2x2 (7)
    2x2 * 2x2 (7)
    2x2 * 2x2 (7)
    2x2 * 2x1 (4)
    Sum = 25 per block.
    Two blocks = 50.
    
    This matches the knowledge "50".
    
    So the strategy is:
    1. Split A into two 2x2 blocks (left, right).
    2. Split B into two 2x7 blocks (top, bottom).
    3. For each 2x2 * 2x7 product:
       a. Split the 2x7 matrix into three parts: 2x2, 2x2, 2x3.
       b. Split the 2x3 part into 2x2 and 2x1.
       c. Use Strassen (rank 7) for the three 2x2 * 2x2 parts.
       d. Use Schoolbook (rank 4) for the 2x2 * 2x1 part.
    
    Let's implement this.
    
    Strassen for 2x2 * 2x2:
    M1 = (A11 + A22) * (B11 + B22)
    M2 = (A21 + A22) * B11
    M3 = A11 * (B12 - B22)
    M4 = A22 * (B21 - B11)
    M5 = (A11 + A12) * B22
    M6 = (A21 - A11) * (B11 + B12)
    M7 = (A12 - A22) * (B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    Schoolbook for 2x2 * 2x1:
    A is 2x2, B is 2x1.
    C1 = A11*B1 + A12*B2
    C2 = A21*B1 + A22*B2
    
    We need to construct U, V, W for each of these rank-1 terms.
    
    Let's write a helper to add a rank-1 term to U, V, W.
    """
    def _add_strassen_2x2_2x2(U, V, W, base_u, base_v, base_w, A11_idx, A12_idx, A21_idx, A22_idx, B11_idx, B12_idx, B21_idx, B22_idx, C11_idx, C12_idx, C21_idx, C22_idx):
        # M1 = (A11 + A22) * (B11 + B22)
        # U: A11 + A22
        # V: B11 + B22
        # W: C11 + C22 (with sign +1 for both)
        # Actually, we need to distribute the coefficients.
        
        # It is easier to just generate the 7 terms and add them.
        
        # Term 1: M1
        # U: A11 + A22
        # V: B11 + B22
        # W: C11 + C22
        # Coefficients:
        # U: 1 at A11, 1 at A22
        # V: 1 at B11, 1 at B22
        # W: 1 at C11, 1 at C22
        
        def add_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
            # u_indices, u_vals are lists of (index, value)
            # We need to find the next column in U, V, W
            # Since we are building the matrices, we can just append columns.
            pass
        
        # This function is getting complex. Let's just build the matrices directly in the main function.
        pass

def _construct_strassen_decomposition(n, m, p):
    """
    Construct the rank-50 decomposition for 2x4 * 4x7.
    """
    if n != 2 or m != 4 or p != 7:
        return None
        
    rank = 50
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_map, v_map, w_map):
        nonlocal col
        if col >= rank:
            raise Exception("Rank exceeded")
        for idx, val in u_map.items():
            U[idx, col] += val
        for idx, val in v_map.items():
            V[idx, col] += val
        for idx, val in w_map.items():
            W[idx, col] += val
        col += 1
        
    # Indices for A (2x4)
    # A[i, j] -> U[i*4 + j]
    # A1 (left 2x2): i in 0..1, j in 0..1
    # A2 (right 2x2): i in 0..1, j in 2..3
    
    # Indices for B (4x7)
    # B[i, j] -> V[i*7 + j]
    # B1 (top 2x7): i in 0..1, j in 0..6
    # B2 (bottom 2x7): i in 2..3, j in 0..6
    
    # Indices for C (2x7)
    # C[i, j] -> W[j*2 + i] (Note: W[k*n+i, r] in task description? 
    # "Output factor follows the official trace convention W[k*n+i,r]"
    # Usually C[i,j] = sum A[i,k] B[k,j].
    # Tensor T[i*m+j, j*p+k, k*n+i] = 1.
    # Here i is row of A (0..n-1), j is col of A/row of B (0..m-1), k is col of B (0..p-1).
    # Output index is k*n + i.
    # So W index is k*n + i.
    # C[i, k] is at W[k*n + i].
    
    def a_idx(i, j): return i * m + j
    def b_idx(i, j): return i * p + j
    def c_idx(i, k): return k * n + i
    
    # Strassen 2x2 * 2x2 helper
    # A indices: a00, a01, a10, a11
    # B indices: b00, b01, b10, b11
    # C indices: c00, c01, c10, c11
    def add_strassen_2x2(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11):
        # M1 = (A11 + A22) * (B11 + B22)
        # Note: My previous notation was A11, A12, A21, A22.
        # Let's stick to 00, 01, 10, 11.
        # M1 = (A00 + A11) * (B00 + B11)
        add_term({a00: 1, a11: 1}, {b00: 1, b11: 1}, {c00: 1, c11: 1})
        
        # M2 = (A10 + A11) * B00
        add_term({a10: 1, a11: 1}, {b00: 1}, {c10: 1})
        
        # M3 = A00 * (B01 - B11)
        add_term({a00: 1}, {b01: 1, b11: -1}, {c01: 1})
        
        # M4 = A11 * (B10 - B00)
        add_term({a11: 1}, {b10: 1, b00: -1}, {c10: 1, c11: 1})
        
        # M5 = (A00 + A01) * B11
        add_term({a00: 1, a01: 1}, {b11: 1}, {c00: -1, c01: 1})
        
        # M6 = (A10 - A00) * (B00 + B01)
        add_term({a10: 1, a00: -1}, {b00: 1, b01: 1}, {c11: 1})
        
        # M7 = (A01 - A11) * (B10 + B11)
        add_term({a01: 1, a11: -1}, {b10: 1, b11: 1}, {c00: 1, c11: -1})
        
    # Schoolbook 2x2 * 2x1 helper
    # A indices: a00, a01, a10, a11
    # B indices: b0, b1
    # C indices: c0, c1
    def add_schoolbook_2x2_2x1(a00, a01, a10, a11, b0, b1, c0, c1):
        # C0 = A00*B0 + A01*B1
        # C1 = A10*B0 + A11*B1
        add_term({a00: 1}, {b0: 1}, {c0: 1})
        add_term({a01: 1}, {b1: 1}, {c0: 1})
        add_term({a10: 1}, {b0: 1}, {c1: 1})
        add_term({a11: 1}, {b1: 1}, {c1: 1})
        
    # Process Block 1: A1 (2x2) * B1 (2x7)
    # A1 indices: a00, a01, a10, a11 correspond to A[0,0], A[0,1], A[1,0], A[1,1]
    a1_00 = a_idx(0, 0)
    a1_01 = a_idx(0, 1)
    a1_10 = a_idx(1, 0)
    a1_11 = a_idx(1, 1)
    
    # B1 indices: B[0,0..6], B[1,0..6]
    # Split B1 into B1_0 (cols 0-1), B1_1 (cols 2-3), B1_2 (cols 4-6)
    # B1_0: B[0,0], B[0,1], B[1,0], B[1,1]
    b1_0_00 = b_idx(0, 0)
    b1_0_01 = b_idx(0, 1)
    b1_0_10 = b_idx(1, 0)
    b1_0_11 = b_idx(1, 1)
    
    # B1_1: B[0,2], B[0,3], B[1,2], B[1,3]
    b1_1_00 = b_idx(0, 2)
    b1_1_01 = b_idx(0, 3)
    b1_1_10 = b_idx(1, 2)
    b1_1_11 = b_idx(1, 3)
    
    # B1_2: B[0,4..6], B[1,4..6]
    # Split B1_2 into B1_2a (cols 4-5) and B1_2b (col 6)
    # B1_2a: B[0,4], B[0,5], B[1,4], B[1,5]
    b1_2a_00 = b_idx(0, 4)
    b1_2a_01 = b_idx(0, 5)
    b1_2a_10 = b_idx(1, 4)
    b1_2a_11 = b_idx(1, 5)
    
    # B1_2b: B[0,6], B[1,6]
    b1_2b_0 = b_idx(0, 6)
    b1_2b_1 = b_idx(1, 6)
    
    # C1 output indices for B1_0 (cols 0-1)
    c1_0_00 = c_idx(0, 0)
    c1_0_01 = c_idx(0, 1)
    c1_0_10 = c_idx(1, 0)
    c1_0_11 = c_idx(1, 1)
    
    # C1 output indices for B1_1 (cols 2-3)
    c1_1_00 = c_idx(0, 2)
    c1_1_01 = c_idx(0, 3)
    c1_1_10 = c_idx(1, 2)
    c1_1_11 = c_idx(1, 3)
    
    # C1 output indices for B1_2a (cols 4-5)
    c1_2a_00 = c_idx(0, 4)
    c1_2a_01 = c_idx(0, 5)
    c1_2a_10 = c_idx(1, 4)
    c1_2a_11 = c_idx(1, 5)
    
    # C1 output indices for B1_2b (col 6)
    c1_2b_0 = c_idx(0, 6)
    c1_2b_1 = c_idx(1, 6)
    
    # Add Strassen for A1 * B1_0
    add_strassen_2x2(a1_00, a1_01, a1_10, a1_11, b1_0_00, b1_0_01, b1_0_10, b1_0_11, c1_0_00, c1_0_01, c1_0_10, c1_0_11)
    
    # Add Strassen for A1 * B1_1
    add_strassen_2x2(a1_00, a1_01, a1_10, a1_11, b1_1_00, b1_1_01, b1_1_10, b1_1_11, c1_1_00, c1_1_01, c1_1_10, c1_1_11)
    
    # Add Strassen for A1 * B1_2a
    add_strassen_2x2(a1_00, a1_01, a1_10, a1_11, b1_2a_00, b1_2a_01, b1_2a_10, b1_2a_11, c1_2a_00, c1_2a_01, c1_2a_10, c1_2a_11)
    
    # Add Schoolbook for A1 * B1_2b
    add_schoolbook_2x2_2x1(a1_00, a1_01, a1_10, a1_11, b1_2b_0, b1_2b_1, c1_2b_0, c1_2b_1)
    
    # Process Block 2: A2 (2x2) * B2 (2x7)
    # A2 indices: A[0,2], A[0,3], A[1,2], A[1,3]
    a2_00 = a_idx(0, 2)
    a2_01 = a_idx(0, 3)
    a2_10 = a_idx(1, 2)
    a2_11 = a_idx(1, 3)
    
    # B2 indices: B[2,0..6], B[3,0..6]
    # Split B2 into B2_0 (cols 0-1), B2_1 (cols 2-3), B2_2 (cols 4-6)
    b2_0_00 = b_idx(2, 0)
    b2_0_01 = b_idx(2, 1)
    b2_0_10 = b_idx(3, 0)
    b2_0_11 = b_idx(3, 1)
    
    b2_1_00 = b_idx(2, 2)
    b2_1_01 = b_idx(2, 3)
    b2_1_10 = b_idx(3, 2)
    b2_1_11 = b_idx(3, 3)
    
    b2_2a_00 = b_idx(2, 4)
    b2_2a_01 = b_idx(2, 5)
    b2_2a_10 = b_idx(3, 4)
    b2_2a_11 = b_idx(3, 5)
    
    b2_2b_0 = b_idx(2, 6)
    b2_2b_1 = b_idx(3, 6)
    
    # C2 output indices are the same as C1 because C = C1 + C2
    # So we add to the same W indices.
    
    # Add Strassen for A2 * B2_0
    add_strassen_2x2(a2_00, a2_01, a2_10, a2_11, b2_0_00, b2_0_01, b2_0_10, b2_0_11, c1_0_00, c1_0_01, c1_0_10, c1_0_11)
    
    # Add Strassen for A2 * B2_1
    add_strassen_2x2(a2_00, a2_01, a2_10, a2_11, b2_1_00, b2_1_01, b2_1_10, b2_1_11, c1_1_00, c1_1_01, c1_1_10, c1_1_11)
    
    # Add Strassen for A2 * B2_2a
    add_strassen_2x2(a2_00, a2_01, a2_10, a2_11, b2_2a_00, b2_2a_01, b2_2a_10, b2_2a_11, c1_2a_00, c1_2a_01, c1_2a_10, c1_2a_11)
    
    # Add Schoolbook for A2 * B2_2b
    add_schoolbook_2x2_2x1(a2_00, a2_01, a2_10, a2_11, b2_2b_0, b2_2b_1, c1_2b_0, c1_2b_1)
    
    if col != rank:
        raise Exception(f"Rank mismatch: expected {rank}, got {col}")
        
    return U, V, W

# EVOLVE-BLOCK-END

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try to find a better decomposition
    try:
        result = _construct_strassen_decomposition(n, m, p)
        if result is not None:
            U, V, W = result
            # Verify
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
    except:
        pass
    
    # Fall back to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
