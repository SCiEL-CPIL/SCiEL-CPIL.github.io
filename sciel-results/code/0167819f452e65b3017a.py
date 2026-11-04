"""Matrix multiplication 2x3x10 tensor decomposition search using numerical rank estimation."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 10]

def _schoolbook_decomp(n, m, p):
    """Returns U, V, W for schoolbook multiplication of n x m by m x p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
    return U, V, W

def _strassen_decomp_2x2x2():
    """Returns U, V, W for 2x2x2 Strassen multiplication."""
    # Standard Strassen rank 7
    # U (4x7), V (4x7), W (4x7)
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    U = [
        [0, 0, 0, 0, 0, 0, 0], # A00
        [0, 1, 0, 0, 0, 0, 0], # A01
        [1, 0, 0, 0, 0, 0, 0], # A10
        [0, 0, 1, 1, 0, 0, 0]  # A11
    ]
    V = [
        [1, 0, 0, 0, 0, 0, 0], # B00
        [0, 1, 0, 0, 0, 0, 0], # B01
        [0, 0, 1, 0, 0, 0, 0], # B10
        [0, 0, 0, 1, 1, 0, 0]  # B11
    ]
    # Strassen formulas:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    W = [
        [1, 0, 0, 1, 0, 0, 1], # C00 coeff for M1..M7
        [0, 1, 0, 0, 0, 0, 0], # C01 coeff for M2? No, C01 = M3+M5 -> [0,0,1,0,1,0,0]
        [0, 0, 1, 0, 0, 0, 0], # C10 = M2+M4 -> [0,1,0,1,0,0,0]
        [1, -1, 1, 0, 0, 1, 0] # C11 = M1-M2+M3+M6 -> [1,-1,1,0,0,1,0]
    ]
    # Correction: Let's map carefully.
    # U rows correspond to A00, A01, A10, A11
    # V rows correspond to B00, B01, B10, B11
    # W rows correspond to C00, C01, C10, C11
    
    # M1 = (A00+A11)(B00+B11) -> U: [1,0,0,1], V: [1,0,0,1]
    # M2 = (A10+A11)B00      -> U: [0,0,1,1], V: [1,0,0,0]
    # M3 = A00(B01-B11)       -> U: [1,0,0,0], V: [0,1,0,-1]
    # M4 = A11(B10-B00)       -> U: [0,0,0,1], V: [0,0,1,-1]
    # M5 = (A00+A01)B11       -> U: [1,1,0,0], V: [0,0,0,1]
    # M6 = (A10-A00)(B00+B01) -> U: [0,0,1,-1], V: [1,1,0,0]
    # M7 = (A01-A11)(B10+B11) -> U: [0,1,0,-1], V: [0,0,1,1]
    
    U = [
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V = [
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, -1, 1, 0, 1]
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1], # C00 = M1+M4-M5+M7
        [0, 0, 1, 0, 1, 0, 0],  # C01 = M3+M5
        [0, 1, 0, 1, 0, 0, 0],  # C10 = M2+M4
        [1, -1, 1, 0, 0, 1, 0]  # C11 = M1-M2+M3+M6
    ]
    return U, V, W

def _build_block_decomp(partition_n, partition_m, partition_p, n, m, p):
    """
    Builds a decomposition by partitioning dimensions and using best known sub-decomps.
    This is a placeholder for a more complex search. For now, we just use the known 
    2x2x2 Strassen mix found in the previous improvement, but we will search for better 
    partitions if time permits.
    """
    # For this specific problem, the best known simple block decomp is:
    # Split m=3 into 2+1.
    # Block 1: 2x2x10 (using 5 Strassen 2x2x2 blocks? No, 2x2x10 is not 2x2x2).
    # The previous improvement used:
    # Split m=3 into 2+1.
    # Subproblem 1: 2x2x10. This can be seen as 5 blocks of 2x2x2? No, 10 is not multiple of 2? Yes, 10=5*2.
    # So 2x2x10 = 5 * (2x2x2). Each 2x2x2 uses Strassen rank 7. Total rank 5*7=35.
    # Subproblem 2: 2x1x10. Schoolbook rank 2*1*10=20.
    # Total rank = 35 + 20 = 55.
    
    # Let's verify this structure.
    # n=2, m=3, p=10.
    # Split m into m1=2, m2=1.
    # Part 1: n=2, m1=2, p=10.
    #   Split p into p1=2, p2=2, p3=2, p4=2, p5=2.
    #   5 blocks of 2x2x2.
    #   Each block uses Strassen rank 7.
    # Part 2: n=2, m2=1, p=10.
    #   Schoolbook rank 20.
    
    # Total rank 55.
    # We will construct this explicitly.
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    current_rank = 0
    
    # Part 1: 2x2x10 using 5 Strassen blocks
    for p_block in range(5):
        # Offset in p dimension: p_block * 2
        # Get Strassen decomp
        sU, sV, sW = _strassen_decomp_2x2x2()
        
        for u_i in range(4): # 2x2 A indices
            i = u_i // 2
            j = u_i % 2
            # Map to global U index: i*m + j
            global_u_idx = i * m + j
            
            for v_j in range(4): # 2x2 B indices
                j_s = v_j // 2
                k_s = v_j % 2
                # Map to global V index: (j+m1)*p + (k_s + p_block*2)
                global_v_idx = (j + 2) * p + (k_s + p_block * 2)
                
                for w_k in range(4): # 2x2 C indices
                    k_w = w_k // 2
                    i_w = w_k % 2
                    # Map to global W index: (k_w + p_block*2)*n + i_w
                    global_w_idx = (k_w + p_block * 2) * n + i_w
                    
                    for r in range(7):
                        if sU[u_i][r] != 0 and sV[v_j][r] != 0 and sW[w_k][r] != 0:
                            # This is not how it works. The rank is shared per block.
                            pass
        
        # Correct construction:
        # Each Strassen block contributes 7 ranks.
        for r in range(7):
            global_r = current_rank + r
            
            # Update U
            for u_i in range(4):
                if sU[u_i][r] != 0:
                    i = u_i // 2
                    j = u_i % 2
                    global_u_idx = i * m + j
                    U[global_u_idx][global_r] += sU[u_i][r]
            
            # Update V
            for v_j in range(4):
                if sV[v_j][r] != 0:
                    j_s = v_j // 2
                    k_s = v_j % 2
                    global_v_idx = (j_s + 2) * p + (k_s + p_block * 2)
                    V[global_v_idx][global_r] += sV[v_j][r]
            
            # Update W
            for w_k in range(4):
                if sW[w_k][r] != 0:
                    k_w = w_k // 2
                    i_w = w_k % 2
                    global_w_idx = (k_w + p_block * 2) * n + i_w
                    W[global_w_idx][global_r] += sW[w_k][r]
        
        current_rank += 7
    
    # Part 2: 2x1x10 Schoolbook
    # m2=1, so j=2 (the 3rd row of A, 3rd col of B)
    for i in range(n):
        for k in range(p):
            global_r = current_rank
            current_rank += 1
            global_u_idx = i * m + 2
            global_v_idx = 2 * p + k
            global_w_idx = k * n + i
            
            U[global_u_idx][global_r] = 1
            V[global_v_idx][global_r] = 1
            W[global_w_idx][global_r] = 1
    
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0

    # Default to the known best block decomp (rank 55)
    U, V, W, rank = _build_block_decomp(2, 3, 10, n, m, p)
    
    # Verify the decomposition
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # We want T[i,j,k] = 1 if j==j and k==k (identity in middle) -> Actually T is the multiplication tensor.
    # T_{i,j,k} = delta_{j, j'} delta_{k, k'}? No.
    # The tensor for matrix multiplication C = AB is T_{i,j,k} such that C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The decomposition is C_{i,k} = sum_r (sum_j U_{i,j,r} A_{i,j}) (sum_k V_{j,k,r} B_{j,k}) ... wait.
    # Standard form: C_{i,k} = sum_r U_{i,r} V_{k,r} W_{j,r}? No.
    # The tensor is T_{i,j,k}. Decomposition: T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}.
    # Let's check indices.
    # U shape (n*m, rank) -> U[i*m+j, r]
    # V shape (m*p, rank) -> V[j*p+k, r]
    # W shape (p*n, rank) -> W[k*n+i, r]
    # Product: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # This should equal 1 if the indices correspond to a valid multiplication term?
    # For C_{i,k} = sum_j A_{i,j} B_{j,k}, the tensor T_{i,j,k} is 1 for all i,j,k? No.
    # The tensor is defined such that the bilinear form recovers the product.
    # Actually, the tensor T is the structure tensor. T_{i,j,k} = 1 if the term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # In this case, EVERY (i,j,k) is a valid term. So T_{i,j,k} = 1 for all i,j,k.
    # Wait, no. The tensor is sparse? No, the multiplication tensor has 1s at all (i,j,k) where the indices align?
    # Yes, for C_{i,k} = sum_j A_{i,j} B_{j,k}, the term A_{i,j} B_{j,k} is always present.
    # So T_{i,j,k} = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    # Let's verify our decomp.
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if val != 1:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    
    if not valid:
        # Fallback to schoolbook
        U, V, W = _schoolbook_decomp(n, m, p)
        rank = n * m * p

    return {"U": U, "V": V, "W": W}
