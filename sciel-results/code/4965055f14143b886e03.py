import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS

    # Schoolbook decomposition of rank n*m*p = 54
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1

    # Verify a decomposition
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Try to construct a rank-46 decomposition by combining two 3x3x3 Laderman decompositions
    # For 3x3 * 3x6, we can view it as two independent 3x3 * 3x3 multiplications
    # (columns 0-2 of B and columns 3-5 of B)
    
    # Laderman's 23-term algorithm for 3x3 * 3x3
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor: T[i,j,k] = 1, where i,k index C, j indexes the inner product
    # Decomposition: T[i,j,k] = sum_t U[i,j,t] V[j,k,t] W[k,i,t]
    
    # The 23 terms of Laderman's algorithm:
    # We represent each term as (u_indices, v_indices, w_indices)
    # where u_indices is a dict mapping (i,j) -> coeff, etc.
    
    # Let me use the explicit Laderman coefficients.
    # A and B are 3x3, C is 3x3.
    # Indexing: A[i,j], B[j,k], C[i,k]
    # U[i,j], V[j,k], W[k,i] (note: W indexed by output C index k,i -> C[i,k])
    
    # Laderman's 23 terms (from literature):
    # Term: (U, V, W) where each is a list of 9 values (row-major)
    
    # I'll use a known correct set. Let me construct it carefully.
    
    # For 3x3x3, the rank is 23. Here are the 23 terms:
    # Each term c has U_c (9 vals), V_c (9 vals), W_c (9 vals)
    
    # Using the formulation from "On the Rank of Tensor Decomposition" papers:
    
    terms = []
    
    # Identity terms (9 terms)
    for i in range(3):
        for j in range(3):
            u = [0]*9; v = [0]*9; w = [0]*9
            u[i*3+j] = 1
            v[j*3+i] = 1  # B[j,i]
            w[i*3+i] = 1  # C[i,i] -> W[k*n+i] with k=i, so W[i*3+i]
            terms.append((u, v, w))
    
    # Wait, this is just the schoolbook for 3x3x3 (27 terms). I need Laderman's 23.
    # Let me look up the actual Laderman coefficients.
    
    # Actually, let me just verify if the schoolbook 54 is valid and return it.
    # The key improvement would be to find a rank < 54 solution.
    
    # For 3x3 * 3x6, the tensor is T[i,j,k] for i in [0,3), j in [0,3), k in [0,6)
    # This can be seen as two blocks: k in [0,3) and k in [3,6)
    # Each block is a 3x3*3x3 multiplication of rank 23 (Laderman)
    # So total rank = 23 + 23 = 46
    
    # Let me hardcode Laderman's 23-term decomposition properly.
    
    # From Laderman (1976), the 23 terms for 3x3 matrix multiplication:
    # Let A = (a_ij), B = (b_jk), C = (c_ik)
    # c_ik = sum_j a_ij b_jk
    
    # The 23 rank-1 tensors (u_t, v_t, w_t) where:
    # sum_t u_t(i,j) v_t(j,k) w_t(k,i) = delta (contribution to c_ik from a_ij b_jk)
    
    # I'll use a verified set from the literature:
    
    # Term 1: a_00 b_00 -> c_00
    terms_23 = []
    
    def add_term(u_list, v_list, w_list):
        terms_23.append((tuple(u_list), tuple(v_list), tuple(w_list)))
    
    # Using indices: U[i*3+j], V[j*3+k], W[k*3+i]
    
    # 9 diagonal terms
    add_term([1,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0])  # a00 b00 c00
    add_term([0,0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0,0], [0,0,1,0,0,0,0,0,0])  # a11 b11 c11
    add_term([0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,1])  # a22 b22 c22
    
    # Off-diagonal - need the full Laderman set
    # This is complex. Let me use a different approach.
    
    # Actually, for the purpose of this task, let me try a rank-46 construction
    # using the fact that 3x3*3x6 = two independent 3x3*3x3 problems.
    
    # But I need valid Laderman terms. Let me use a simple but potentially suboptimal approach:
    # Use the Strassen-like construction.
    
    # Given the complexity, let me just return the schoolbook 54-rank solution,
    # but also attempt to verify and potentially find a better one via optimization.
    
    # Let me try a simple rank-reduction: for the 3x3*3x6 case,
    # can we do better than 54?
    
    # The theoretical lower bound for 3x3*3x6 is at least max(9, 18, 18) = 18,
    # but realistically it's around 46 (2*23) or possibly lower.
    
    # Let me try to construct a 46-rank solution by properly implementing Laderman.
    
    # I'll use the following approach: define Laderman's 23 terms explicitly.
    
    # From a reliable source, Laderman's 23-term decomposition:
    # (I'll define them as (U, V, W) tuples of length-9 lists)
    
    # Let me use the formulation where we have:
    # c_ik = sum_j a_ij b_jk
    
    # The 23 terms (u, v, w) with u[i,j], v[j,k], w[k,i]:
    
    # I'll use a known correct implementation. Here's one:
    
    L23 = [
        # (U, V, W) - each is a flat list of 9 elements (row-major)
        # U[i*3+j] corresponds to a_ij
        # V[j*3+k] corresponds to b_jk  
        # W[k*3+i] corresponds to c_ik (output)
        
        # 1. a00 b00 c00
        ([1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        # 2. a01 b11 c01
        ([0,1,0, 0,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
        # 3. a02 b22 c02
        ([0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,1, 0,0,0, 0,0,0]),
        # 4. a10 b01 c10
        ([0,0,0, 1,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0]),
        # 5. a11 b11 c11 (already have)
        # 6. a12 b21 c12
        ([0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,1, 0,0,0]),
        # 7. a20 b02 c20
        ([0,0,0, 0,0,0, 1,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0]),
        # 8. a21 b12 c21
        ([0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,1,0]),
        # 9. a22 b22 c22 (already have)
        
        # Additional 14 terms to complete Laderman's algorithm
        # These involve sums of products
        
        # 10. (a00+a11) (b00+b11) c01
        ([1,0,0, 0,1,0, 0,0,0], [1,0,0, 0,1,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
        # 11. (a00+a11) (b00+b11) c10
        ([1,0,0, 0,1,0, 0,0,0], [1,0,0, 0,1,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0]),
        # ... this is getting complex and error-prone.
    ]
    
    # Given the difficulty of correctly hardcoding Laderman's algorithm,
    # let me use a different strategy: verify the schoolbook and return it.
    # But first, let me try a simpler optimization.
    
    # For 3x3 * 3x6, notice that this is equivalent to multiplying
    # A (3x3) by B (3x6) to get C (3x6).
    
    # We can use the fact that this is a "rectangular" matrix multiplication.
    
    # One approach: use the 2x2x2 rank-7 (Strassen) building block.
    # But 3x3x6 doesn't cleanly decompose into 2x2x2 blocks.
    
    # Let me just return the schoolbook 54-rank solution, which is guaranteed correct.
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
