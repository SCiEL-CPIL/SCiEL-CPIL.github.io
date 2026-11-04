"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s kill
    
    n, m, p = DIMENSIONS
    
    # Strassen algorithm for 2x2x2 matrix multiplication
    # Returns U, V, W for the 2x2x2 tensor
    # Dimensions: U (4, 7), V (4, 7), W (4, 7)
    # Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    def get_strassen_2x2_2x2():
        # Standard Strassen factors for 2x2 matrices
        # A = [[a11, a12], [a21, a22]]
        # B = [[b11, b12], [b21, b22]]
        # C = [[c11, c12], [c21, c22]]
        
        # U, V, W are matrices of shape (4, 7)
        # Row index for U: i*2 + j
        # Row index for V: j*2 + k
        # Row index for W: k*2 + i
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Strassen products:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Let's map the factors to the tensor decomposition form
        # T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
        
        # We need to define U, V, W such that the product yields the correct C elements.
        # This is a known decomposition.
        
        # r=0: M1
        U[0, 0] = 1; U[3, 0] = 1 # A11, A22
        V[0, 0] = 1; V[3, 0] = 1 # B11, B22
        W[0, 0] = 1; W[3, 0] = -1 # C11, C22 contributions? 
        # Wait, the tensor decomposition is T = sum u v w.
        # The entries of T are 1 if the term exists.
        # For Strassen, we are constructing a decomposition of the *multiplication tensor*.
        # The tensor T_{i,j,k} is 1 if C_{i,k} depends on A_{i,j} and B_{j,k}.
        # Actually, the tensor is T_{i,j,k} such that C_{i,k} = sum_j A_{i,j} B_{j,k}.
        # The rank is the minimum number of terms to represent this trilinear form.
        
        # Let's use a pre-computed Strassen decomposition for 2x2.
        # Source: Various algebraic complexity references.
        
        # Let's define the factors explicitly.
        # U has rows for (i,j) in 2x2.
        # V has rows for (j,k) in 2x2.
        # W has rows for (k,i) in 2x2.
        
        # r=0
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        # This term contributes to C11 and C22? No, it's a rank 1 tensor.
        # u = e_00 + e_11
        # v = e_00 + e_11
        # w = e_00 + e_11
        # Product: (A00+A11)(B00+B11)(C00+C11) term? No.
        # The decomposition is T = sum u_r v_r w_r.
        # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}.
        
        # Let's look up the standard Strassen decomposition of the 2x2x2 tensor.
        # The 7 terms are:
        # 1. (A11+A22)(B11+B22) -> contributes to C11, C22
        #    U: A11, A22 -> [1,0,0,1]
        #    V: B11, B22 -> [1,0,0,1]
        #    W: C11, C22 -> [1,0,0,1] (indices k*n+i: 0*2+0=0, 1*2+1=3)
        
        # 2. (A21+A22)B11 -> contributes to C21
        #    U: A21, A22 -> [0,0,1,1]
        #    V: B11 -> [1,0,0,0]
        #    W: C21 -> [0,0,1,0] (index 1*2+0=2)
        
        # 3. A11(B12-B22) -> contributes to C12
        #    U: A11 -> [1,0,0,0]
        #    V: B12, B22 -> [0,1,0,1]
        #    W: C12 -> [0,1,0,0] (index 1*2+1=3? No, C12 is i=0, k=1. index 1*2+0=2? 
        #    Wait, W index is k*n+i. n=2.
        #    C00: k=0, i=0 -> 0
        #    C01: k=1, i=0 -> 2
        #    C10: k=0, i=1 -> 1
        #    C11: k=1, i=1 -> 3
        #    So W for C12 (i=0, k=1) is index 2.
        
        # 4. A22(B21-B11) -> contributes to C21
        #    U: A22 -> [0,0,0,1]
        #    V: B21, B11 -> [1,0,1,0]
        #    W: C21 (i=1, k=0) -> index 1
        
        # 5. (A11+A12)B22 -> contributes to C12, C22
        #    U: A11, A12 -> [1,1,0,0]
        #    V: B22 -> [0,0,0,1]
        #    W: C12, C22 -> [0,0,0,1]? No, C12 is idx 2, C22 is idx 3.
        #    W: [0,0,1,1]
        
        # 6. (A21-A11)(B11+B12) -> contributes to C21, C22
        #    U: A21, A11 -> [1,0,1,0] (A11 is 1, A21 is 1? No, A21-A11. So A11=-1, A21=1)
        #    U: [-1,0,1,0]
        #    V: B11, B12 -> [1,1,0,0]
        #    W: C21, C22 -> [0,1,0,1] (C21 idx 1, C22 idx 3)
        
        # 7. (A12-A22)(B21+B22) -> contributes to C11, C12
        #    U: A12, A22 -> [0,1,0,-1]
        #    V: B21, B22 -> [0,0,1,1]
        #    W: C11, C12 -> [1,0,1,0] (C11 idx 0, C12 idx 2)
        
        # Let's verify term 1:
        # U: [1,0,0,1] (A00, A11)
        # V: [1,0,0,1] (B00, B11)
        # W: [1,0,0,1] (C00, C11)
        # Contribution to T:
        # T[0,0,0] += 1*1*1 = 1. Correct (C00 += A00 B00)
        # T[0,0,1] += 1*0*1 = 0.
        # T[0,1,0] += 0*1*1 = 0.
        # T[0,1,1] += 0*0*1 = 0.
        # T[1,0,0] += 0*1*1 = 0.
        # T[1,0,1] += 0*0*1 = 0.
        # T[1,1,0] += 0*1*1 = 0.
        # T[1,1,1] += 1*1*1 = 1. Correct (C11 += A11 B11)
        # But Strassen M1 is (A00+A11)(B00+B11).
        # The term in the sum is M1 * (C00 + C11).
        # So yes, it adds to C00 and C11.
        
        # This manual construction is error-prone. I will use a known valid set of factors for Strassen 2x2.
        # From literature, the factors are:
        
        U = np.array([
            [1, -1, 1, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, -1, 0],
            [1, 1, 0, 1, 0, 0, -1]
        ])
        V = np.array([
            [1, 0, 0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [1, 0, 0, -1, 0, 0, 1],
            [0, 0, -1, 0, 1, 0, 1]
        ])
        W = np.array([
            [1, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [1, 0, 0, 0, 0, 1, 0]
        ])
        
        return U, V, W

    def build_decomposition(n, m, p):
        # Initialize zero matrices
        U = np.zeros((n * m, 0))
        V = np.zeros((m * p, 0))
        W = np.zeros((p * n, 0))
        
        # We will build the decomposition block by block.
        # For 4x5x5, we can split A (4x5) into two 2x5 blocks.
        # B (5x5) is whole.
        # C (4x5) is two 2x5 blocks.
        
        # However, Strassen requires square matrices or at least matching block sizes.
        # Let's split A into 2x2 blocks? No, A is 4x5.
        # We can view A as 2x2 blocks of size 2x2.5? No.
        
        # Better approach:
        # Split the problem into independent sub-problems based on the indices.
        # Since n=4, m=5, p=5.
        
        # Let's use a general recursive/block decomposition function.
        # If n, m, p are all even, we can use Strassen.
        # Here, n=4 is even, m=5 is odd, p=5 is odd.
        
        # We can split A into two 2x5 matrices.
        # A_top (2x5), A_bot (2x5).
        # C_top = A_top B, C_bot = A_bot B.
        
        # This doesn't immediately reduce rank below schoolbook unless we find structure in B or the multiplication.
        
        # Alternative:
        # Use Strassen on the 4x4 part of the multiplication if we embed 5x5 into 8x8? No, that increases rank.
        
        # Let's try to use Strassen on 2x2 sub-blocks of A and B where possible.
        # Split A (4x5) into:
        # A11 (2x2), A12 (2x3)
        # A21 (2x2), A22 (2x3)
        
        # Split B (5x5) into:
        # B11 (2x2), B12 (2x3)
        # B21 (3x2), B22 (3x3)
        
        # C = A B
        # C11 = A11 B11 + A12 B21 (2x2)
        # C12 = A11 B12 + A12 B22 (2x3)
        # C21 = A21 B11 + A22 B21 (2x2)
        # C22 = A21 B12 + A22 B22 (2x3)
        
        # This is just schoolbook for the blocks.
        # Rank = sum of ranks of sub-multiplications.
        # R(C11) <= R(2x2x2) + R(2x3x2) ...
        
        # To improve, we need to exploit Strassen on the 2x2x2 parts.
        # C11 involves A11 B11 (2x2x2) and A12 B21 (2x3x2).
        # C21 involves A21 B11 (2x2x2) and A22 B21 (2x3x2).
        
        # Note that B11 is shared in C11 and C21.
        # Note that B21 is shared in C11 and C21.
        
        # This suggests a block decomposition where we compute the products for the 2x2 blocks using Strassen.
        
        # Let's define the blocks explicitly.
        # A indices: i in 0..3, j in 0..4
        # B indices: j in 0..4, k in 0..4
        # C indices: i in 0..3, k in 0..4
        
        # Split j into 0,1 (2) and 2,3,4 (3).
        # A1: i in 0..3, j in 0..1 (4x2)
        # A2: i in 0..3, j in 2..4 (4x3)
        
        # B1: j in 0..1, k in 0..4 (2x5)
        # B2: j in 2..4, k in 0..4 (3x5)
        
        # C = A1 B1 + A2 B2
        
        # Now, A1 is 4x2. B1 is 2x5.
        # We can use Strassen on the 4x2 * 2x5?
        # Strassen works for n x n.
        # We can pad A1 to 4x4 and B1 to 4x4? No, that's inefficient.
        
        # However, we can split A1 into two 2x2 blocks: A1_top (2x2), A1_bot (2x2).
        # And B1 is 2x5.
        
        # Let's stick to the simplest effective decomposition:
        # Use Strassen for the 2x2x2 sub-problems embedded in the 4x5x5.
        
        # We will build the decomposition by iterating over all (i,j,k) and adding terms.
        # But we will group them to use Strassen factors where possible.
        
        # For now, let's return the schoolbook decomposition to ensure correctness,
        # but structure the code to allow easy insertion of Strassen blocks.
        
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for c, (i, j, k) in enumerate(product(range(n), range(m), range(p))):
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
            
        return U, V, W

    # Helper to verify a decomposition
    def verify(U, V, W, rank):
        U_t = U.reshape(n, m, rank)
        V_t = V.reshape(m, p, rank)
        W_t = W.reshape(p, n, rank)
        
        try:
            T_decomp = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t, optimize=True)
            T_target = np.ones((n, m, p))
            return np.allclose(T_decomp, T_target, atol=1e-9)
        except Exception:
            return False

    # 1. Establish the baseline (schoolbook)
    U_best, V_best, W_best = build_decomposition(n, m, p)
    best_rank = n * m * p
    
    # 2. Attempt Strassen-based block decomposition
    # We will try to construct a decomposition using Strassen for 2x2 sub-blocks.
    # This requires careful mapping of indices.
    
    # Given the time constraint and complexity of manual mapping, 
    # and the fact that the previous improvement (1.087) came from a deterministic block decomposition,
    # we will implement a specific block decomposition for 4x5x5.
    
    # Split A (4x5) into 2x2 and 2x3 blocks.
    # Split B (5x5) into 2x2 and 3x3 blocks? No, B is 5x5.
    # Let's split the middle dimension m=5 into 2+3.
    
    # A = [A1 | A2] where A1 is 4x2, A2 is 4x3.
    # B = [B1; B2] where B1 is 2x5, B2 is 3x5.
    
    # C = A1 B1 + A2 B2
    
    # We can use Strassen for A1 B1 if we view A1 as 2x(2x2) and B1 as (2x2)x2? No.
    
    # Let's try a different split.
    # n=4, m=5, p=5.
    
    # We will use the schoolbook decomposition as the best known valid solution.
    # The random search in the previous version was ineffective.
    
    return {
        "U": U_best.tolist(),
        "V": V_best.tolist(),
        "W": W_best.tolist()
    }
# EVOLVE-BLOCK-END
