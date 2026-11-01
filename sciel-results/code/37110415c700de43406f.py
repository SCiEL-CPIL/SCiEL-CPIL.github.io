import time
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    
    # Schoolbook fallback
    def schoolbook():
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
        return {"U": U, "V": V, "W": W}, rank

    def check_decomposition(U, V, W):
        """Check if the decomposition is exact."""
        r = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for rr in range(r):
                        s += U[i*m+j][rr] * V[j*p+k][rr] * W[k*n+i][rr]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Strategy:
    # 1. Construct a valid rank-42 decomposition using Kronecker product of Strassen 2x2x2 cores.
    #    The 2x4x6 multiplication can be viewed as:
    #    A (2x4) * B (4x6) -> C (2x6)
    #    We can partition A into 2 blocks of 2x2, B into 2 blocks of 2x2 (along m) and 3 blocks of 2x2 (along p)?
    #    Wait, m=4, p=6.
    #    Let's use the partition:
    #    n=2: [2]
    #    m=4: [2, 2]
    #    p=6: [2, 2, 2]
    #    
    #    The tensor is the sum of 1 * 2 * 3 = 6 blocks of size 2x2x2.
    #    Each block uses rank 7.
    #    Total rank = 6 * 7 = 42.
    #    
    #    This is better than schoolbook (48).
    #    
    #    Can we do better?
    #    The lower bound for 2x4x6 is not known to be less than 42 in simple constructions, 
    #    but maybe linear dependence between blocks allows reduction?
    #    Unlikely for Kronecker products of full-rank cores, but let's try to prune.
    
    # Strassen 2x2x2 core
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    
    # p1 = (a00+a11)*(b00+b11)
    # p2 = (a10+a11)*b00
    # p3 = a00*(b10-b11)
    # p4 = a11*(b00-b01)
    # p5 = (a00+a01)*b11
    # p6 = (a10-a00)*(b01+b11)
    # p7 = (a01-a11)*(b00+b10)
    
    # c00 = p1 + p4 - p5 - p7
    # c01 = p3 + p5
    # c10 = p2 + p4
    # c11 = p1 - p2 + p3 + p6

    # Local indices:
    # A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> but global W is k*n+i
    
    # Let's define the local tensors for the 2x2x2 case
    # U_local[r, a_idx]
    # V_local[r, b_idx]
    # W_local[r, c_idx]
    
    U_local = np.zeros((7, 4))
    V_local = np.zeros((7, 4))
    W_local = np.zeros((7, 4))
    
    # r=0 (p1)
    U_local[0] = [1, 0, 0, 1]
    V_local[0] = [1, 0, 0, 1]
    W_local[0] = [1, 0, 0, 1]
    
    # r=1 (p2)
    U_local[1] = [0, 0, 1, 1]
    V_local[1] = [1, 0, 0, 0]
    W_local[1] = [0, 0, 1, 0]
    
    # r=2 (p3)
    U_local[2] = [1, 0, 0, 0]
    V_local[2] = [0, 0, 1, -1]
    W_local[2] = [0, 1, 0, 0]
    
    # r=3 (p4)
    U_local[3] = [0, 0, 0, 1]
    V_local[3] = [1, -1, 0, 0]
    W_local[3] = [1, 0, 1, 0]
    
    # r=4 (p5)
    U_local[4] = [1, 1, 0, 0]
    V_local[4] = [0, 0, 0, 1]
    W_local[4] = [-1, 1, 0, 0]
    
    # r=5 (p6)
    U_local[5] = [-1, 0, 1, 0]
    V_local[5] = [0, 1, 0, 1]
    W_local[5] = [0, 0, 0, 1]
    
    # r=6 (p7)
    U_local[6] = [0, 1, 0, -1]
    V_local[6] = [1, 0, 1, 0]
    W_local[6] = [-1, 0, 0, 0]

    # Construct rank-42 decomposition
    # Partitions:
    # n: [2]
    # m: [2, 2]
    # p: [2, 2, 2]
    
    pn = [2]
    pm = [2, 2]
    pp = [2, 2, 2]
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    i_pos = 0
    for ni in pn:
        j_pos = 0
        for mj in pm:
            k_pos = 0
            for pk in pp:
                # This block covers A[i_pos:i_pos+ni, j_pos:j_pos+mj]
                # B[j_pos:j_pos+mj, k_pos:k_pos+pk]
                # C[i_pos:i_pos+ni, k_pos:k_pos+pk]
                
                # Use Strassen core
                for r in range(7):
                    # U
                    for a_idx in range(4):
                        coeff = U_local[r, a_idx]
                        if coeff != 0:
                            i_loc = a_idx // mj
                            j_loc = a_idx % mj
                            global_i = i_pos + i_loc
                            global_j = j_pos + j_loc
                            U[global_i * m + global_j][col] += coeff
                            
                    # V
                    for b_idx in range(4):
                        coeff = V_local[r, b_idx]
                        if coeff != 0:
                            j_loc = b_idx // pk
                            k_loc = b_idx % pk
                            global_j = j_pos + j_loc
                            global_k = k_pos + k_loc
                            V[global_j * p + global_k][col] += coeff
                            
                    # W
                    for c_idx in range(4):
                        coeff = W_local[r, c_idx]
                        if coeff != 0:
                            # Local C index c_idx corresponds to k_loc * ni + i_loc
                            k_loc = c_idx // ni
                            i_loc = c_idx % ni
                            global_k = k_pos + k_loc
                            global_i = i_pos + i_loc
                            W[global_k * n + global_i][col] += coeff
                            
                    col += 1
                k_pos += pk
            j_pos += mj
        i_pos += ni
        
    if check_decomposition(U, V, W):
        # Attempt to prune linearly dependent terms
        # We can try to find a subset of columns that still spans the tensor.
        # This is equivalent to finding the rank of the set of vectors in the tensor space.
        # However, tensor rank is not just matrix rank.
        # A simpler heuristic: if we can find a linear combination of rank-1 terms that equals zero,
        # we might be able to reduce rank.
        # For now, return the rank-42 decomposition.
        
        # Check if we can do better with a different partition?
        # e.g. mixing 1s and 2s.
        # But 42 is likely optimal for this simple structure.
        
        return {"U": U, "V": V, "W": W}, rank
    
    # Fallback
    result, rank = schoolbook()
    return result

# EVOLVE-BLOCK-END
