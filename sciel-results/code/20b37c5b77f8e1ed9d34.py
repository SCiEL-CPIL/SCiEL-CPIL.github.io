"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs an exact tensor decomposition for multiplying a 2x3 matrix 
    by a 3x10 matrix.
    
    Strategy: 
    Split the 3x10 matrix B into 5 blocks of size 3x2.
    For each block, we compute C_block = A * B_block (2x3 * 3x2 -> 2x2).
    The matrix multiplication tensor M_{2,3,2} has a known rank decomposition of rank 11.
    We use 5 such blocks, resulting in a total rank of 5 * 11 = 55.
    
    The 11 terms for M_{2,3,2} are derived from Strassen-like identities for the 2x2 blocks
    plus outer product terms for the third column of A and third row of B (which don't exist 
    in a pure 2x2x2, but here A is 2x3 and B is 3x2, so the 'middle' dimension is 3).
    
    Wait, M_{2,3,2} means A is 2x3, B is 3x2, C is 2x2.
    Indices: i in {0,1}, j in {0,1,2}, k in {0,1}.
    C[i,k] = sum_j A[i,j] B[j,k].
    
    The 11 terms for M_{2,3,2} (rank 11) are:
    Let A = [[a00, a01, a02], [a10, a11, a12]]
    Let B = [[b00, b01], [b10, b11], [b20, b21]]
    
    We can decompose this using 7 terms for the 2x2 core (A[:2,:2] * B[:2,:2]) 
    and 4 terms for the remaining parts involving index 2 in the middle dimension.
    
    Core 2x2x2 (A00,A01;A10,A11 * B00,B01;B10,B11) uses Strassen rank 7.
    The terms involving A[i,2] and B[2,k] are simple outer products:
    A[i,2] * B[2,0] contributes to C[i,0]
    A[i,2] * B[2,1] contributes to C[i,1]
    There are 2 values of i and 2 values of k, so 4 terms.
    
    Total rank = 7 + 4 = 11.
    
    We replicate this structure for each of the 5 blocks of the 3x10 matrix.
    """
    n, m, p = DIMENSIONS
    
    # Define the 11 terms for the 2x3x2 block multiplication
    # Each term is a tuple: (u_indices, u_vals, v_indices, v_vals, w_indices, w_vals)
    # u_indices: list of (i, j) pairs
    # v_indices: list of (j, k) pairs
    # w_indices: list of (k, i) pairs
    
    # Strassen-like terms for the 2x2 core (indices 0,1 in j)
    # M1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # Note: The standard Strassen algorithm computes 2x2*2x2.
    # Here we map the 2x2 core to indices j=0,1.
    
    terms = [
        # 1. (A00 + A11) * (B00 + B11)
        # U: A00=1, A11=1. V: B00=1, B11=1. W: C00=1, C11=1.
        ([(0,0), (1,1)], [1, 1], [(0,0), (1,1)], [1, 1], [(0,0), (1,1)], [1, 1]),
        
        # 2. (A10 + A11) * B00
        # U: A10=1, A11=1. V: B00=1. W: C01=1, C11=-1.
        # Check: (A10+A11)B00 = A10B00 + A11B00.
        # C01 = A00B01 + A01B11 + ... 
        # Standard Strassen for C = A*B:
        # M1=(A00+A11)(B00+B11)
        # M2=(A10+A11)B00
        # M3=A00(B01-B11)
        # M4=A11(B10-B00)
        # M5=(A00+A01)B11
        # M6=(A10-A00)(B00+B01)
        # M7=(A01-A11)(B10+B11)
        # C00 = M1 - M3 + M5 - M7
        # C01 = M4 + M5
        # C10 = M2 + M3
        # C11 = M1 - M2 + M4 + M7
        
        # Let's verify the W coefficients for each M against the C indices.
        # Term 1 (M1): Contributes to C00 (+1), C11 (+1).
        # Term 2 (M2): Contributes to C10 (+1), C11 (-1).
        # Term 3 (M3): Contributes to C00 (-1), C10 (+1).
        # Term 4 (M4): Contributes to C01 (+1), C11 (+1).
        # Term 5 (M5): Contributes to C00 (-1), C01 (+1).
        # Term 6 (M6): Contributes to C11 (+1). (Wait, M6 is (A10-A00)(B00+B01). 
        #           C11 = M1 - M2 + M4 + M7. 
        #           Let's re-verify standard Strassen coefficients.
        
        # Actually, it is safer to just use the explicit decomposition found in literature or derive it carefully.
        # Let's use the explicit terms from the current program's fallback which was verified to work (score 1.0909).
        # The current program's fallback terms are:
        # 1. U:(0,0)=1,(1,1)=1; V:(0,0)=1,(1,1)=1; W:(0,0)=1,(1,1)=1
        # 2. U:(1,0)=1,(1,1)=1; V:(0,0)=1; W:(0,1)=1,(1,1)=-1
        # 3. U:(0,0)=1; V:(0,1)=1,(1,1)=-1; W:(1,0)=1,(1,1)=1
        # 4. U:(1,1)=1; V:(1,0)=1,(0,0)=-1; W:(0,0)=1,(0,1)=1
        # 5. U:(0,0)=1,(0,1)=1; V:(1,1)=1; W:(0,0)=-1,(1,0)=1
        # 6. U:(1,0)=1,(0,0)=-1; V:(0,0)=1,(0,1)=1; W:(1,1)=1
        # 7. U:(0,1)=1,(1,1)=-1; V:(1,0)=1,(1,1)=1; W:(0,0)=1
        
        # Let's stick to the proven terms from the current program's fallback to ensure correctness.
    ]
    
    # Re-defining the proven terms clearly
    strassen_terms = [
        # 1
        ([(0,0), (1,1)], [1, 1], [(0,0), (1,1)], [1, 1], [(0,0), (1,1)], [1, 1]),
        # 2
        ([(1,0), (1,1)], [1, 1], [(0,0)], [1], [(0,1), (1,1)], [1, -1]),
        # 3
        ([(0,0)], [1], [(0,1), (1,1)], [1, -1], [(1,0), (1,1)], [1, 1]),
        # 4
        ([(1,1)], [1], [(1,0), (0,0)], [1, -1], [(0,0), (0,1)], [1, 1]),
        # 5
        ([(0,0), (0,1)], [1, 1], [(1,1)], [1], [(0,0), (1,0)], [-1, 1]),
        # 6
        ([(1,0), (0,0)], [1, -1], [(0,0), (0,1)], [1, 1], [(1,1)], [1]),
        # 7
        ([(0,1), (1,1)], [1, -1], [(1,0), (1,1)], [1, 1], [(0,0)], [1]),
        # 8. Outer A02 * B20
        ([(0,2)], [1], [(2,0)], [1], [(0,0)], [1]),
        # 9. Outer A02 * B21
        ([(0,2)], [1], [(2,1)], [1], [(1,0)], [1]),
        # 10. Outer A12 * B20
        ([(1,2)], [1], [(2,0)], [1], [(0,1)], [1]),
        # 11. Outer A12 * B21
        ([(1,2)], [1], [(2,1)], [1], [(1,1)], [1]),
    ]

    num_blocks = p // 2  # 5 blocks
    block_size = 2
    rank_per_block = len(strassen_terms)  # 11
    total_rank = num_blocks * rank_per_block  # 55
    
    U = [[0] * total_rank for _ in range(n * m)]  # 6 x 55
    V = [[0] * total_rank for _ in range(m * p)]  # 30 x 55
    W = [[0] * total_rank for _ in range(p * n)]  # 20 x 55
    
    col_offset = 0
    for b in range(num_blocks):
        for term in strassen_terms:
            u_idx, u_val, v_idx, v_val, w_idx, w_val = term
            for (i, j), val in zip(u_idx, u_val):
                U[i * m + j][col_offset] += val
            for (j, k_local), val in zip(v_idx, v_val):
                k_global = b * block_size + k_local
                V[j * p + k_global][col_offset] += val
            for (k_local, i), val in zip(w_idx, w_val):
                k_global = b * block_size + k_local
                W[k_global * n + i][col_offset] += val
            col_offset += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
