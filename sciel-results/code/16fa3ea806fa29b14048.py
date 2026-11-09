import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Trivial rank
    R_trivial = n * m * p
    U_triv = np.zeros((n * m, R_trivial))
    V_triv = np.zeros((m * p, R_trivial))
    W_triv = np.zeros((p * n, R_trivial))
    
    # Fill trivial solution
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U_triv[i * m + j, r] = 1
                V_triv[j * p + k, r] = 1
                W_triv[k * n + i, r] = 1

    # Identify blocks to apply Strassen
    # Disjoint 2x2x2 blocks
    blocks = [
        (0, 0, 0), # i_s, j_s, k_s
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2)
    ]
    
    # Check if blocks fit
    valid_blocks = []
    for i_s, j_s, k_s in blocks:
        if i_s + 2 <= n and j_s + 2 <= m and k_s + 2 <= p:
            valid_blocks.append((i_s, j_s, k_s))
            
    if not valid_blocks:
        return {"U": U_triv.tolist(), "V": V_triv.tolist(), "W": W_triv.tolist()}

    # Create a set of blocked (i,j,k) tuples to exclude from trivial set
    blocked = set()
    for i_s, j_s, k_s in valid_blocks:
        for i in range(i_s, i_s+2):
            for j in range(j_s, j_s+2):
                for k in range(k_s, k_s+2):
                    blocked.add((i,j,k))
                    
    U_new = []
    V_new = []
    W_new = []
    
    # Add trivial terms not in blocks
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i,j,k) not in blocked:
                    col_U = np.zeros(n*m)
                    col_V = np.zeros(m*p)
                    col_W = np.zeros(p*n)
                    col_U[i*m+j] = 1
                    col_V[j*p+k] = 1
                    col_W[k*n+i] = 1
                    U_new.append(col_U)
                    V_new.append(col_V)
                    W_new.append(col_W)
                    
    # Add Strassen terms for each block
    # Strassen vectors (local)
    # u, v, w are length 4
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> (k,i)
    
    strassen_terms = [
        # u, v, w
        (np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, -1, -1, 1])), # M1
        (np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 0, 1, -1])), # M2
        (np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1]), np.array([0, 1, 0, 1])), # M3
        (np.array([0, 0, 0, 1]), np.array([-1, 0, 1, 0]), np.array([1, 0, 1, 0])), # M4
        (np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]), np.array([-1, 1, 0, 0])), # M5
        (np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1])), # M6
        (np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]))  # M7
    ]
    
    for i_s, j_s, k_s in valid_blocks:
        for u_loc, v_loc, w_loc in strassen_terms:
            u = np.zeros(n*m)
            v = np.zeros(m*p)
            w = np.zeros(p*n)
            
            # Map local A indices to global U
            # 0: (0,0) -> i_s, j_s
            # 1: (0,1) -> i_s, j_s+1
            # 2: (1,0) -> i_s+1, j_s
            # 3: (1,1) -> i_s+1, j_s+1
            u[i_s*m + j_s] = u_loc[0]
            u[i_s*m + (j_s+1)] = u_loc[1]
            u[(i_s+1)*m + j_s] = u_loc[2]
            u[(i_s+1)*m + (j_s+1)] = u_loc[3]
            
            # Map local B indices to global V
            # 0: (0,0) -> j_s, k_s
            # 1: (0,1) -> j_s, k_s+1
            # 2: (1,0) -> j_s+1, k_s
            # 3: (1,1) -> j_s+1, k_s+1
            v[j_s*p + k_s] = v_loc[0]
            v[j_s*p + (k_s+1)] = v_loc[1]
            v[(j_s+1)*p + k_s] = v_loc[2]
            v[(j_s+1)*p + (k_s+1)] = v_loc[3]
            
            # Map local C indices to global W
            # 0: (0,0) -> k_s, i_s
            # 1: (0,1) -> k_s, i_s+1
            # 2: (1,0) -> k_s+1, i_s
            # 3: (1,1) -> k_s+1, i_s+1
            w[k_s*n + i_s] = w_loc[0]
            w[k_s*n + (i_s+1)] = w_loc[1]
            w[(k_s+1)*n + i_s] = w_loc[2]
            w[(k_s+1)*n + (i_s+1)] = w_loc[3]
            
            U_new.append(u)
            V_new.append(v)
            W_new.append(w)
            
    U_final = np.column_stack(U_new)
    V_final = np.column_stack(V_new)
    W_final = np.column_stack(W_new)
    
    # Verify
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i*m+j, j*p+k, k*n+i] = 1
                
    T_calc = np.zeros((n*m, m*p, p*n))
    for r in range(U_final.shape[1]):
        T_calc += np.einsum('a,b,c->abc', U_final[:,r], V_final[:,r], W_final[:,r])
        
    if not np.allclose(T_calc, T_expected):
        # Fallback to trivial
        return {"U": U_triv.tolist(), "V": V_triv.tolist(), "W": W_triv.tolist()}
        
    # Try to reduce rank by finding linear dependencies
    # We have R_current terms. We want to find if any term can be removed or combined.
    # Since we are using exact integer arithmetic, we can check for linear dependencies in the columns.
    # However, the tensor decomposition is sum_r u_r v_r w_r.
    # If we have a linear relation sum c_r (u_r v_r w_r) = 0, we can't simply drop a term unless c_r is 0 for all but one.
    # But if we have a linear dependence among the vectors u_r, v_r, w_r such that the tensor sum remains unchanged...
    
    # A simpler approach: Check if the current set of vectors is linearly independent.
    # If we have R vectors in R-dimensional space (effectively), and they are independent, we can't reduce.
    # But we are in a high-dimensional space.
    
    # Let's try to see if we can merge any two terms? No, Strassen is optimal for 2x2x2.
    # The trivial terms are also independent.
    
    # Is there any overlap between Strassen blocks?
    # The blocks are disjoint.
    # So the tensors are supported on disjoint sets of (i,j,k) indices.
    # Therefore, the rank is simply the sum of the ranks of the blocks.
    # Rank = (total trivial terms) + 4 * 7 = (60 - 4*8) + 28 = 60 - 32 + 28 = 56.
    
    # Can we do better than 56?
    # The trivial rank is 60.
    # Strassen reduces 8 to 7 for each 2x2x2 block.
    # We have 4 disjoint 2x2x2 blocks.
    # Remaining trivial terms: 60 - 32 = 28.
    # Total rank: 28 + 28 = 56.
    
    # Is it possible to use a 3x3x3 block?
    # n=3, m=4, p=5.
    # We can fit one 3x3x3 block at (0,0,0).
    # The best known rank for 3x3x3 matrix multiplication is 23 (Landsberg, etc., but 23 is for 3x3x3? No, 3x3x3 is 3x3 matrices. Rank 23 is for 3x3x3? Actually, the rank of 3x3x3 matrix multiplication is 23? No, it's 23 for 3x3? Wait.
    # Rank of 2x2x2 is 7.
    # Rank of 3x3x3 is 23? No, the rank of 3x3x3 tensor is 23?
    # Actually, the rank of 3x3x3 matrix multiplication is 23?
    # Let's check. The trivial rank is 27.
    # Strassen-like algorithms for 3x3x3 exist.
    # But we have 3x4x5.
    
    # If we use a 3x3x3 block at (0,0,0):
    # It covers i=0,1,2; j=0,1,2; k=0,1,2.
    # The remaining parts are:
    # i=0,1,2; j=3; k=0,1,2,3,4 -> 3*1*5 = 15 terms
    # i=0,1,2; j=0,1,2; k=3,4 -> 3*3*2 = 18 terms
    # i=0,1,2; j=3; k=3,4 -> 3*1*2 = 6 terms (already counted in j=3)
    # Wait, let's count disjoint regions.
    # Block: i in [0,2], j in [0,2], k in [0,2].
    # Region 1: i in [0,2], j=3, k in [0,4] -> 3*1*5 = 15
    # Region 2: i in [0,2], j in [0,2], k in [3,4] -> 3*3*2 = 18
    # Total remaining: 15 + 18 = 33.
    # If rank of 3x3x3 is 23, total rank = 23 + 33 = 56.
    # Same as Strassen blocks!
    
    # What if we use a 3x2x2 block?
    # Rank of 3x2x2?
    # Trivial rank 12.
    # Can we do better?
    # If we treat it as 2x2x2 (rank 7) + 1x2x2 (rank 2) + 1x2x2 (rank 2)? No, that's not how it works.
    # The rank of 3x2x2 matrix multiplication is 10?
    # Let's assume the rank of 3x3x3 is 23.
    # Then 23 + 33 = 56.
    
    # So 56 seems to be a strong candidate.
    # The current program achieves 56.
    
    # Can we go lower?
    # Maybe by overlapping blocks?
    # Or using a more efficient algorithm for 3x4x5?
    
    # Let's stick with 56 for now, as it's a solid improvement over 60.
    # The current program already does this.
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
