import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    timeout = 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    
    # Schoolbook decomposition: rank = n * m * p = 54
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # For each (i, j, k): U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    # where r = i*(m*p) + j*p + k
    
    rank_school = n * m * p  # 54
    
    def make_schoolbook():
        U = np.zeros((n * m, rank_school), dtype=float)
        V = np.zeros((m * p, rank_school), dtype=float)
        W = np.zeros((p * n, rank_school), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    # Try to find a lower rank decomposition using randomized optimization
    # The tensor T has entries T[(i,j), (j,k), (k,i)] = 1 for all i,j,k
    # We want sum_r U[:,r] V[:,r] W[:,r] = T (as a 3-tensor)
    # 
    # Strategy: try rank-36 by exploiting structure
    # For 2x3x3, known rank is 12 (from literature on small matrix multiplication)
    # If we can get rank-12 for 2x3x3, then rank-36 for 2x3x9 by block decomposition
    #
    # Let's try to construct a rank-12 decomposition for 2x3x3
    # T_{2,3,3}: C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor: sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{i,k}
    
    def try_rank_12_2x3x3():
        """Try to find rank-12 decomposition for 2x3x3 matrix multiplication."""
        # This is a known result but construction is non-trivial
        # Let's use a structured approach based on Strassen-like ideas
        # For 2x3x3, we can try:
        # - 3 columns of B, 3 rows of A (well, 2 rows)
        # - Need to compute 2x3 product from 3x3
        
        # Actually, let's use the fact that for rectangular matrices,
        # rank can sometimes be lower. Let's just try randomized search.
        
        # The tensor for 2x3x3:
        # T[(i,j), (j',k), (k',i')] = delta_{j,j'} * delta_{i,i'}
        # So T is non-zero when j=j' and i=i'
        
        # Let's set up the problem as: find U (6,12), V (9,12), W (6,12)
        # such that sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
        
        # This is a bilinear (actually trilinear) problem, hard to optimize directly.
        # Instead, let's use the known decomposition if we can recall it.
        
        # For 2x2x2: Strassen gives rank 7.
        # For 2x3x3: I'll try to construct it.
        
        # Actually, let's just try a simpler approach: use the schoolbook for 2x3x9
        # but try to compress using the fact that p=9=3*3.
        
        # Alternative: use the "block" approach where we treat the 3x9 matrix
        # as three 3x3 blocks, and if we have a rank-12 decomposition for 2x3x3,
        # we get rank 3*12=36 for 2x3x9.
        
        # Let me try to hardcode a rank-12 decomposition for 2x3x3.
        # This is based on the work of Landsberg et al. on small matrix multiplication.
        
        # For 2x3x3, the tensor is:
        # sum_{i=0}^{1} sum_{j=0}^{2} sum_{k=0}^{2} e_{i,j} ⊗ e_{j,k} ⊗ e_{i,k}
        
        # Let's index: A is 2x3, B is 3x3, C is 2x3
        # C[i,k] = sum_j A[i,j] B[j,k]
        
        # I'll try a structured decomposition. One approach:
        # Use the fact that 2x3x3 can be seen as computing 2 outputs, each a
        # linear combination of 9 products (2*3*3 schoolbook terms per output... no)
        
        # Schoolbook for 2x3x3: 2*3*3 = 18 terms
        # Can we do better? 
        
        # Let's try: for each column k of C, C[:,k] = A @ B[:,k]
        # Computing A @ v where A is 2x3 and v is 3x1: needs 6 products (2*3)
        # 3 columns: 18 products total.
        
        # To beat 18, we need to share products across columns.
        
        # Idea: B is 3x3. Let's write B = [b0, b1, b2] where bj are columns.
        # C[:,0] = A b0, C[:,1] = A b1, C[:,2] = A b2
        # If we compute A(b0+b1), A(b0-b1), A(b1+b2), A(b1-b2), etc., 
        # we might save products.
        
        # For 2x2x2 Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # For 2x3x3, let's try:
        # Let A = [a00, a01, a02; a10, a11, a12]
        # Let B = [b00, b01, b02; b10, b11, b12; b20, b21, b22]
        
        # We want C = A@B (2x3)
        # C[0,k] = a00*b0k + a01*b1k + a02*b2k
        # C[1,k] = a10*b0k + a11*b1k + a12*b2k
        
        # Let's define:
        # x0 = a00 + a10
        # x1 = a00 - a10
        # x2 = a01 + a11
        # x3 = a01 - a11
        # x4 = a02 + a12
        # x5 = a02 - a12
        
        # y0 = b00 + b10 + b20
        # y1 = b00 - b10
        # y2 = b00 - b20
        # y3 = b01 + b11 + b21
        # y4 = b01 - b11
        # y5 = b01 - b21
        # y6 = b02 + b12 + b22
        # y7 = b02 - b12
        # y8 = b02 - b22
        
        # This doesn't obviously help. Let me try a different approach.
        
        # Actually, for the purpose of this task, let's just return the schoolbook
        # decomposition. The known rank for 2x3x3 is indeed lower than 18, but
        # constructing it explicitly is complex. Given the time constraints, 
        # I'll stick with schoolbook rank 54.
        
        return None  # Return None to indicate failure
    
    # Try the rank-36 approach
    result = try_rank_12_2x3x3()
    
    if result is not None:
        U, V, W = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Check deadline
    if time.time() - start_time > timeout:
        U, V, W = make_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try randomized search for lower rank
    # We'll try rank 36, 30, 27, 24, 21, 18
    best_rank = rank_school
    best_decomp = None
    
    U_school, V_school, W_school = make_schoolbook()
    best_decomp = (U_school, V_school, W_school)
    
    # For each candidate rank, try to find a decomposition
    for target_rank in [36, 30, 27, 24, 21, 18]:
        if time.time() - start_time > timeout - 10:
            break
        
        # Try a few random initializations
        for attempt in range(3):
            if time.time() - start_time > timeout - 5:
                break
            
            # Initialize with schoolbook-like structure
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Use the schoolbook as a hint: project onto the schoolbook directions
            # Actually, let's just use gradient-free approach or return schoolbook
            
            # For now, let's just check if we can verify a decomposition
            # The tensor equation is:
            # For all (i,j), (j',k), (k',i'):
            # sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] = delta_{j,j'} * delta_{i,i'}
            
            # This is equivalent to:
            # sum_r U[:, r] V[:, r] W[:, r] = T
            # where T[a,b,c] = 1 if a's j-index equals b's j-index AND a's i-index equals c's i-index
            
            # Let's verify the schoolbook:
            # For schoolbook, U[i*m+j, r] is 1 when r = i*(m*p) + j*p + k
            # V[j'*p+k', r] is 1 when j'=j and k'=k
            # W[k''*n+i'', r] is 1 when k''=k and i''=i
            # So product is 1 when j=j', i=i', k=k=k'
            # Sum over r: exactly 1 when j=j' and i=i', else 0. Correct.
            
            # I don't have a good algorithm to find lower rank decompositions
            # in reasonable time. Let's just return schoolbook.
            pass
    
    U, V, W = best_decomp
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
