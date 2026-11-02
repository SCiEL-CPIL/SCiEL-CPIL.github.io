"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the tensor decomposition exactly reproduces matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The tensor T has entries T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    For matrix multiplication (n x m) * (m x p) -> (n x p),
    the tensor is T[i,j,k] = 1 if the output element (i,k) depends on input (i,j)*(j,k),
    which is always 1 for valid indices.
    
    Actually, the matrix multiplication tensor T_{i,j,k} = delta_{i,k_output} ... 
    Let me think about this more carefully.
    
    Matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    The tensor T has T[i,j,k] = 1 for all valid i,j,k (it's the standard bilinear form).
    
    A rank-r decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    So we need: for all i in [0,n), j in [0,m), k in [0,p):
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 4
    
    # The tensor T[i,j,k] = 1 for all valid i,j,k
    # We want rank-r decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Schoolbook gives rank = n*m*p = 16
    # Theoretically, rank of 2x2x4 tensor: 
    # For 2x2x2, the rank is 7 (Strassen-like)
    # For 2x2x4, we can think of it as two 2x2x2 tensors sharing the A matrix.
    # Actually, 2x2x4 = 2x2 * 2x4. The tensor decomposes into two 2x2x2 blocks.
    # Each 2x2x2 has rank at most 7, so total at most 14.
    # But can we do better by sharing? The A inputs are shared.
    # 
    # Let's try a randomized search for rank 13, 12, 11, etc.
    
    deadline = time.time() + 140  # Leave 10s margin
    
    def try_rank(r):
        """Try to find a rank-r decomposition using randomized approach."""
        if time.time() > deadline:
            return None
        
        # We need U (4 x r), V (8 x r), W (8 x r) such that
        # for all i in {0,1}, j in {0,1}, k in {0,1,2,3}:
        #   sum_c U[i*2+j, c] * V[j*4+k, c] * W[k*2+i, c] = 1
        
        # This is a system of 2*2*4 = 16 equations.
        # Unknowns: 4*r + 8*r + 8*r = 20*r
        # For r=13: 260 unknowns, 16 equations - very underdetermined.
        
        # Strategy: fix some structure and solve.
        # Let's use the block structure.
        # Split k into two groups: k=0,1 and k=2,3
        # For k=0,1: this is like 2x2x2 multiplication
        # For k=2,3: this is another 2x2x2 multiplication
        
        # Approach: Use random initialization and then try to solve via least squares
        # or use a structured approach.
        
        # Let's try: assume the decomposition has a block structure.
        # U has rows for (i,j) where i in {0,1}, j in {0,1}
        # V has rows for (j,k) where j in {0,1}, k in {0,1,2,3}
        # W has rows for (k,i) where k in {0,1,2,3}, i in {0,1}
        
        # Try rank r with random real entries, then check if we can adjust.
        # Actually, let's use a different approach: 
        # The tensor is T[i,j,k] = 1. 
        # A rank-1 term is U[i,j] * V[j,k] * W[k,i].
        # We need sum of r such terms to equal 1 for all (i,j,k).
        
        # Let's try to construct explicitly.
        # For 2x2x2, Strassen's algorithm gives rank 7.
        # The 7 rank-1 terms for 2x2x2:
        # Let's denote the 2x2x2 tensor as having indices (i,j,k) with i,j,k in {0,1}.
        # Strassen's 7 products:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b10-b11)
        # M4 = a11(b01-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b10)
        # M7 = (a01-a11)(b01+b11)
        
        # For 2x2x4, we have two such blocks (k=0,1 and k=2,3).
        # If we use independent Strassen decompositions, rank = 14.
        # Can we share some terms?
        
        # Let's try to find rank 13 by sharing one term.
        # Or try a fully random search.
        
        # Randomized approach: 
        # 1. Generate random U, V, W with rank r
        # 2. Compute the resulting tensor
        # 3. Check if it equals the target tensor
        # This won't work directly because the space of rank-r tensors is a variety,
        # and random points won't hit it.
        
        # Better: Use the fact that we have more unknowns than equations.
        # Fix r-1 columns randomly, solve for the last column.
        
        # Actually, let's use a simpler approach:
        # The equations are multilinear. Let's use a grid search or simulated annealing.
        
        # For now, let's try a structured construction.
        # Start with the schoolbook rank-16 and try to merge terms.
        
        # Schoolbook: for each (i,j,k), one term with U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
        # 16 terms. Can we combine any?
        
        # Two terms can be combined if they share the same (i,j) or (j,k) or (k,i) pattern
        # in a way that allows linearity.
        
        # Let's try: for fixed i, the terms involving that i are 2*4=8 terms.
        # For i=0: (i=0,j=0,k=0), (i=0,j=0,k=1), (i=0,j=0,k=2), (i=0,j=0,k=3),
        #            (i=0,j=1,k=0), (i=0,j=1,k=1), (i=0,j=1,k=2), (i=0,j=1,k=3)
        
        # This is getting complex. Let's just do a numerical search.
        
        # Use scipy to solve the system.
        # The system is: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # This is a system of 16 polynomial equations in 20*r variables.
        
        # For r=13, 260 variables, 16 equations. Very underdetermined.
        # We can fix 260-16=244 variables and solve for 16.
        # But the equations are trilinear, not linear.
        
        # Alternative: use random initialization and gradient descent on the loss
        # L = sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
        
        # Let's try this with multiple random restarts.
        
        best_loss = float('inf')
        best_solution = None
        
        for restart in range(50):
            if time.time() > deadline:
                break
            
            random.seed(restart * 1000 + r)
            np.random.seed(restart * 1000 + r)
            
            # Initialize with random values in [-1, 1]
            U_arr = np.random.uniform(-1, 1, (n*m, r))
            V_arr = np.random.uniform(-1, 1, (m*p, r))
            W_arr = np.random.uniform(-1, 1, (p*n, r))
            
            # Gradient descent
            lr = 0.01
            for step in range(2000):
                if time.time() > deadline:
                    break
                
                # Compute loss and gradients
                loss = 0.0
                grad_U = np.zeros_like(U_arr)
                grad_V = np.zeros_like(V_arr)
                grad_W = np.zeros_like(W_arr)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(r):
                                s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                            diff = s - 1.0
                            loss += diff * diff
                            for c in range(r):
                                grad_U[i*m+j, c] += 2 * diff * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                                grad_V[j*p+k, c] += 2 * diff * U_arr[i*m+j, c] * W_arr[k*n+i, c]
                                grad_W[k*n+i, c] += 2 * diff * U_arr[i*m+j, c] * V_arr[j*p+k, c]
                
                U_arr -= lr * grad_U
                V_arr -= lr * grad_V
                W_arr -= lr * grad_W
                
                if loss < best_loss:
                    best_loss = loss
                    best_solution = (U_arr.copy(), V_arr.copy(), W_arr.copy())
                
                if loss < 1e-12:
                    break
            
            if best_loss < 1e-12:
                break
        
        if best_loss < 1e-9:
            # Convert to list format
            U_list = [[float(U_arr[i, c]) for c in range(r)] for i in range(n*m)]
            V_list = [[float(V_arr[i, c]) for c in range(r)] for i in range(m*p)]
            W_list = [[float(W_arr[i, c]) for c in range(r)] for i in range(p*n)]
            
            # Verify
            if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from 13 down to 10
    for r in [13, 12, 11, 10]:
        if time.time() > deadline:
            break
        result = try_rank(r)
        if result is not None:
            return result
    
    # Fallback: schoolbook rank 16
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
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
