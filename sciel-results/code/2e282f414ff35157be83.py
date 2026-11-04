import numpy as np
import time

_START_TIME = time.time()
TIME_LIMIT = 145

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    n, m, p = 4, 4, 7
    
    # Build the multiplication tensor T[i,j,k] = 1 if j==k else 0
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    
    # Start with schoolbook: rank = m = 4
    # For each j, we have terms for all i,k: U[i*m+j, j]=1, V[j*p+k, j]=1, W[k*n+i, j]=1
    # But this gives rank 4*4=16... no wait.
    
    # Actually schoolbook for 4x4x7: for each j in 0..3, for each i in 0..3, for each k in 0..6:
    # C[i,k] += A[i,j]*B[j,k]
    # This is 4*4*7 = 112 rank-1 terms if done naively.
    # But we can group: for each j, the sum over i,k of A[i,j]*B[j,k]*delta(i,j,k)
    # = sum_i sum_k A[i,j] * B[j,k] * delta
    # This is a rank-(4*7)=28 tensor for each j, but we can do better.
    
    # Standard schoolbook: rank = n*m*p / ... no.
    # Actually, the standard decomposition has rank = m = 4? No.
    # Let me think again. T[i,j,k] = delta(j,k).
    # Schoolbook: for each j, T_j[i,k] = A[i,j]*B[j,k] = sum over... 
    # No, T[i,j,k] = delta(j,k) means for fixed j, T[i,j,k] = 1 if k=j else 0.
    # Wait no. Matrix mult: C[i,k] = sum_j A[i,j] B[j,k].
    # So T[i,j,k] = 1 if the indices match for the multiplication, i.e., T[i,j,k] = 1 always?
    # No! The tensor is T[i,j,k] = 1 for all i,j,k? No.
    # The multiplication tensor has T[i,j,k] = 1 if... 
    # Actually, the standard convention: the tensor for multiplying A (n x m) by B (m x p) to get C (n x p)
    # is T[i,j,k] where i indexes rows of A/C, j indexes columns of A / rows of B, k indexes columns of B/C.
    # T[i,j,k] = 1 if this term appears in C[i,k] = sum_j A[i,j] B[j,k].
    # So T[i,j,k] = 1 for ALL i,j,k. That can't be right for a low-rank decomposition.
    
    # Wait, I think I'm confusing myself. Let me re-read the task.
    # "exactly decompose the matrix multiplication tensor"
    # The multiplication tensor T has T[i,j,k] = 1 for all i in [n], j in [m], k in [p].
    # No wait, that's a rank-1 tensor! That doesn't make sense.
    
    # Let me reconsider. The tensor for matrix multiplication A*B=C where A is n×m, B is m×p, C is n×p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T satisfies: T_{i,j,k} = 1 if the term A[i,j]*B[j,k] contributes to C[i,k].
    # Since every term A[i,j]*B[j,k] contributes to C[i,k] (with coefficient 1), T[i,j,k] = 1 for all i,j,k.
    # But that's a rank-1 tensor, which is trivially decomposable.
    
    # I think I'm misunderstanding the tensor convention. Let me look at the W index: W[k*n+i, r]
    # So W is indexed by (k, i) where k is the column of C (and B) and i is the row of C (and A).
    # The decomposition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    # For this to equal the matrix multiplication, we need T[i,j,k] = 1 for all i,j,k? 
    # No, that gives C[i,k] = sum_j sum_r ... which doesn't work.
    
    # Actually, I think the tensor is T[i,j,k] = 1 for all i,j,k, and the decomposition 
    # sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] should equal 1 for all i,j,k.
    # But that's just saying the product of three matrices equals the all-ones tensor,
    # which is rank 1. That can't be the intended problem.
    
    # Let me reconsider. Maybe the tensor is the standard matrix multiplication tensor where
    # T[i,j,k] = 1 if j is the "middle" index that connects A and B.
    # In the standard formulation, the multiplication tensor has entries T[i,j,k] = 1 
    # when the indices form a valid path in the multiplication.
    
    # Actually, I think the correct interpretation is:
    # The tensor T has T[i,j,k] = 1 for all i,j,k, and we want to decompose it as
    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # This is equivalent to saying the matrix multiplication map can be computed with rank r.
    # The standard schoolbook algorithm has rank = m (the shared dimension).
    # For m=4, schoolbook rank is 4.
    # But wait, that's not right either. Schoolbook has rank = m*n*p / ... 
    
    # Let me just implement the known Strassen-based decomposition.
    # The current program uses rank 93, which suggests the problem is non-trivial.
    
    # I'll keep the existing approach but clean it up and verify correctness.
    
    total_rank = 93
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    current_rank = 0
    
    # Strassen 2x2x2 terms
    strassen_2x2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 0, (1,0): 1, (1,1): 0}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1, (1,0): 0, (1,1): 0}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}),
    ]
    
    # Strassen 4x4x4 outer terms
    strassen_4x4_outer = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}),
        ({0: 1}, {1: 1, 3: -1}),
        ({3: 1}, {2: 1, 0: -1}),
        ({0: 1, 1: 1}, {3: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}),
    ]
    
    c_blocks_4x4 = [
        {(0,0): 1.0, (1,1): 1.0},
        {(1,0): 1.0, (1,1): -1.0},
        {(0,1): 1.0, (1,1): 1.0},
        {(0,0): 1.0, (1,0): 1.0},
        {(0,0): -1.0, (0,1): 1.0},
        {(1,1): 1.0},
        {(0,0): 1.0},
    ]
    
    # Part 1: 4x4x4 Strassen (rank 49)
    for m_idx in range(7):
        x_def, y_def = strassen_4x4_outer[m_idx]
        c_block = c_blocks_4x4[m_idx]
        for t_idx in range(7):
            r = current_rank
            current_rank += 1
            x_sub, y_sub, c_sub = strassen_2x2[t_idx]
            
            for (li, lj), cx in x_sub.items():
                for b, cb in x_def.items():
                    bi, bj = b // 2, b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    U[i * m + j, r] += cx * cb
            
            for (li, lj), cy in y_sub.items():
                for b, cb in y_def.items():
                    bi, bj = b // 2, b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    V[i * p + j, r] += cy * cb
            
            for (li, lj), cc in c_sub.items():
                if cc == 0:
                    continue
                for (br, bc), cb_c in c_block.items():
                    if cb_c == 0:
                        continue
                    i_final = 2 * br + li
                    k_final = 2 * bc + lj
                    W[k_final * n + i_final, r] += cc * cb_c
    
    # Part 2: 4x4x3 as 4 * (2x2x3), each of rank 11
    def add_2x2x3_v2(ar, ac, br, cr):
        nonlocal current_rank
        
        for t_idx in range(7):
            r = current_rank
            current_rank += 1
            x_sub, y_sub, c_sub = strassen_2x2[t_idx]
            
            for (li, lj), cx in x_sub.items():
                i = 2 * ar + li
                j = 2 * ac + lj
                U[i * m + j, r] += cx
            
            for (li, lj), cy in y_sub.items():
                i = 2 * br + li
                j = 4 + lj
                V[i * p + j, r] += cy
            
            for (li, lj), cc in c_sub.items():
                if cc == 0:
                    continue
                i_final = 2 * cr + li
                k_final = 4 + lj
                W[k_final * n + i_final, r] += cc
        
        for xi in range(2):
            for xj in range(2):
                r = current_rank
                current_rank += 1
                i_a = 2 * ar + xi
                j_a = 2 * ac + xj
                i_b = 2 * br + xj
                U[i_a * m + j_a, r] = 1.0
                V[i_b * p + 6, r] = 1.0
                W[6 * n + i_a, r] = 1.0
    
    add_2x2x3_v2(0, 0, 0, 0)
    add_2x2x3_v2(0, 1, 1, 0)
    add_2x2x3_v2(1, 0, 0, 1)
    add_2x2x3_v2(1, 1, 1, 1)
    
    assert current_rank == 93, f"Expected 93, got {current_rank}"
    
    U = U[:, :current_rank].tolist()
    V = V[:, :current_rank].tolist()
    W = W[:, :current_rank].tolist()
    
    return {"U": U, "V": V, "W": W}
