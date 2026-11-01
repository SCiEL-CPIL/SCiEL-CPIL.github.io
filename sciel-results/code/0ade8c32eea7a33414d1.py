import time
import numpy as np

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices.
    Shapes:
    U: (n*m, rank) = (4, 35)
    V: (m*p, rank) = (20, 35)
    W: (p*n, rank) = (20, 35)
    
    The decomposition satisfies:
    C[i, k] = sum_{j=0}^{m-1} sum_{r=0}^{rank-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s limit
    
    n, m, p = 2, 2, 10
    num_blocks = p // 2
    rank_per_block = 7
    total_rank = num_blocks * rank_per_block
    
    # Initialize matrices with zeros
    # Using lists of lists for integer precision, converted to numpy array at the end?
    # The prompt says "integer/half-integer entries". Lists of ints are fine.
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Strassen's 7 products for 2x2 matrix multiplication
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = A * B
    #
    # Indices for 2x2 block:
    # A indices: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
    # B indices: (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h
    # C indices: (0,0), (0,1), (1,0), (1,1)
    #
    # Strassen terms (u, v, w) where:
    # u is a dict {(i,j): coeff}
    # v is a dict {(j,k): coeff}
    # w is a dict {(k,i): coeff}
    # Contribution to C[i,k] is sum of u[i,j]*v[j,k]*w[k,i] over j.
    # Note: w index is (k,i) corresponding to C[i,k] in the output tensor position W[k*n+i].
    
    strassen_terms = [
        # Term 0: M1 = (a+d)(e+h)
        # C[0,0] += M1, C[1,1] += M1
        ({(0,0): 1, (1,1): 1}, 
         {(0,0): 1, (1,1): 1}, 
         {(0,0): 1, (1,1): 1}),
         
        # Term 1: M2 = (c+d)e
        # C[1,0] += M2, C[1,1] -= M2
        ({(1,0): 1, (1,1): 1}, 
         {(0,0): 1}, 
         {(0,1): 1, (1,1): -1}),
         
        # Term 2: M3 = a(f-h)
        # C[0,1] += M3, C[1,1] += M3
        ({(0,0): 1}, 
         {(0,1): 1, (1,1): -1}, 
         {(1,0): 1, (1,1): 1}),
         
        # Term 3: M4 = d(g-e)
        # C[0,0] += M4, C[1,0] += M4
        ({(1,1): 1}, 
         {(1,0): 1, (0,0): -1}, 
         {(0,0): 1, (0,1): 1}),
         
        # Term 4: M5 = (a+b)h
        # C[0,0] -= M5, C[0,1] += M5
        ({(0,0): 1, (0,1): 1}, 
         {(1,1): 1}, 
         {(0,0): -1, (1,0): 1}),
         
        # Term 5: M6 = (c-a)(e+f)
        # C[1,1] += M6
        ({(1,0): 1, (0,0): -1}, 
         {(0,0): 1, (0,1): 1}, 
         {(1,1): 1}),
         
        # Term 6: M7 = (b-d)(g+h)
        # C[0,0] += M7
        ({(0,1): 1, (1,1): -1}, 
         {(1,0): 1, (1,1): 1}, 
         {(0,0): 1})
    ]
    
    # Fill in the factor matrices for each block
    for r_block in range(num_blocks):
        for t in range(rank_per_block):
            col = r_block * rank_per_block + t
            u_dict, v_dict, w_dict = strassen_terms[t]
            
            # Fill U: index [i*m + j]
            for (i, j), val in u_dict.items():
                U[i * m + j][col] = val
                
            # Fill V: index [j*p + k]
            # k is local column index (0 or 1)
            # Global column index is r_block * 2 + k
            for (j, k), val in v_dict.items():
                global_k = r_block * 2 + k
                V[j * p + global_k][col] = val
                
            # Fill W: index [k*n + i]
            # k is local column index (0 or 1)
            # Global column index is r_block * 2 + k
            for (k, i), val in w_dict.items():
                global_k = r_block * 2 + k
                W[global_k * n + i][col] = val
                
    # Check deadline
    if time.time() > deadline:
        # In this deterministic case, we are already done.
        # If we were doing search, we'd return best so far.
        pass

    return {"U": U, "V": V, "W": W}
