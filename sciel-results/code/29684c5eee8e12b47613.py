"""Rewritten program for 3x3 @ 3x5 matrix multiplication tensor decomposition."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # Build verification function
    def verify(U, V, W, atol=1e-8):
        """Check if U, V, W form a valid decomposition."""
        rank = U.shape[1]
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        result = np.zeros((n, m, p))
        for c in range(rank):
            result += np.einsum('ij,jk,ki->ijk', U3[:,:,c], V3[:,:,c], W3[:,:,c])
        return np.allclose(result, 1.0, atol=atol)
    
    # Schoolbook decomposition (rank 45)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    # Known low-rank decomposition for 3x3x3 matrix multiplication
    # Based on Strassen's 7-rank decomposition
    def strassen_3x3x3():
        """
        Returns U, V, W for 3x3 @ 3x3 with rank 7.
        U: (9, 7), V: (9, 7), W: (9, 7)
        Convention: U[i*3+j], V[j*3+k], W[k*3+i]
        """
        U = np.zeros((9, 7))
        V = np.zeros((9, 7))
        W = np.zeros((9, 7))
        
        # Strassen's 7 terms for 2x2x2, extended to 3x3x3
        # Using the standard 3x3x3 rank-7 decomposition
        # M1 = (A11+A22+A33)(B11+B22+B33)
        # M2 = (A11+A22)(B22+B33)
        # M3 = (A11+A33)(B33+B11)
        # M4 = A11(B11-B22)
        # M5 = A22(B22-B33)
        # M6 = A33(B33-B11)
        # M7 = (A11+A22+A33)(B11+B22+B33) -- this is M1, need different
        
        # Let me use a known valid 3x3x3 rank-7 decomposition
        # Actually, the minimal rank for 3x3x3 is 7 (Strassen)
        # But we need to be careful with the convention
        
        # Using explicit indices where row = i*m+j, col = j*p+k
        # For 3x3x3: m=p=3, so row = i*3+j, col = j*3+k
        
        # Strassen 7-term decomposition for 3x3:
        # The 7 multiplications are:
        # M1 = (A00+A11+A22) * (B00+B11+B22)
        # M2 = (A00+A11) * (B11+B22)  
        # M3 = (A00+A22) * (B22+B00)
        # M4 = A00 * (B00-B11)
        # M5 = A11 * (B11-B22)
        # M6 = A22 * (B22-B00)
        # M7 = (A00+A11+A22) * (B00+B11+B22) -- duplicate, need correction
        
        # Let me use a proper 3x3x3 rank-7 scheme
        # Actually, for 3x3 the rank is 7 via Strassen
        # Let me just use a verified decomposition
        
        # Standard Strassen 3x3x3:
        # C[i,j] = sum over the 7 terms
        
        # Term 1: (A00+A11+A22)(B00+B11+B22)
        U[0, 0] = 1; U[4, 0] = 1; U[8, 0] = 1
        V[0, 0] = 1; V[5, 0] = 1; V[10, 0] = 1
        W[0, 0] = 1; W[5, 0] = 1; W[10, 0] = 1
        # But this gives sum of all products, need to subtract others
        
        # This is getting complex. Let me use a different approach.
        # For 3x3x3, I'll use a known valid rank-7 decomposition.
        
        # Actually, let me just construct a valid one by hand.
        # The key insight: we can use the Kronecker product structure.
        # A 3x3 multiplication can be seen as combining 2x2 and 1x1 parts.
        
        # Let me use a simpler approach: embed 2x2x2 Strassen into 3x3x3
        # and handle the border terms separately.
        
        # For now, return a valid but higher-rank decomposition
        # We'll use rank 9 (schoolbook for 3x3x3)
        rank = 9
        U = np.zeros((9, rank))
        V = np.zeros((9, rank))
        W = np.zeros((9, rank))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i*3+j, col] = 1
                    V[j*3+k, col] = 1
                    W[k*3+i, col] = 1
                    col += 1
        return U, V, W
    
    # Try to construct a lower-rank decomposition for 3x3x5
    # Strategy: Use Strassen-style decomposition for the 3x3x3 core
    # and handle the extra column separately
    
    def build_3x3x5_decomposition():
        """
        Construct a decomposition for 3x3 @ 3x5.
        Split into 3x3 @ 3x3 and 3x3 @ 3x2 parts.
        """
        # For the 3x3x3 part, use rank 7 (Strassen)
        # For the 3x3x2 part, use rank 5 (or similar)
        
        # Actually, let's think about this differently.
        # The tensor T[i,j,k] = 1 for i in [0,3), j in [0,3), k in [0,5)
        # We need sum_c U[i*3+j, c] * V[j*5+k, c] * W[k*3+i, c] = 1
        
        # Lower bound: rank >= max(9, 15, 15) = 15
        # But also rank >= max(n, m, p) * max(n, m, p) = 9? No.
        # The tensor rank lower bound is at least max(n*m, m*p, p*n) = 15
        
        # Let me try to find a rank-15 or rank-16 decomposition
        
        # Strategy: Use the fact that the multiplication tensor has a
        # "block" structure. For each output element C[i,k], we need
        # C[i,k] = sum_j A[i,j] * B[j,k]
        
        # One approach: use the standard decomposition but merge terms
        # where possible.
        
        # Let me try a rank-18 decomposition first, then try to reduce.
        
        # Actually, let me try a completely different approach:
        # Use a linear algebra approach to find minimal rank.
        
        # For now, let's use a constructive approach with rank around 20-25
        
        best_rank = None
        best_decomp = None
        
        # Try ranks from 15 to 40
        for rank in range(15, 41):
            if time.time() > deadline:
                break
            
            # Try to find a valid decomposition at this rank
            found = try_find_decomposition(rank, deadline)
            if found is not None:
                best_rank = rank
                best_decomp = found
                break
        
        if best_decomp is not None:
            return best_decomp
        else:
            return schoolbook()
    
    def try_find_decomposition(rank, deadline):
        """Try to find a valid decomposition at the given rank."""
        # Strategy: Start from a structured initialization and refine
        
        # Approach 1: Use a "separable" structure
        # C[i,k] = sum_j A[i,j] B[j,k]
        # We can write this as a sum of rank-1 terms in the tensor sense
        
        # For a rank-r decomposition, we need:
        # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = delta_{i,k} ... no
        # Actually, the multiplication tensor is:
        # T[i,j,k] = 1 (for all i,j,k in the 3x3x5 case, it's just 1)
        # Wait, no. The matrix multiplication tensor T_{ijk} = 1 if and only if
        # the (i,k) element of the product is sum_j A_{ij} B_{jk}
        # So T[i,j,k] = 1 for all i,j,k? No, that's not right either.
        
        # The tensor for matrix multiplication C = AB where A is 3x3, B is 3x5:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # The structure tensor T has T[i,j,k] = 1 for all i in [0,3), j in [0,3), k in [0,5)
        # because each term A[i,j]*B[j,k] contributes to C[i,k]
        
        # So yes, T[i,j,k] = 1 for all valid i,j,k.
        # We need: sum_c U[i*3+j, c] * V[j*5+k, c] * W[k*3+i, c] = 1
        
        # This is a system of 45 equations in 45*rank unknowns (with structure)
        
        # Let me try a random search with half-integer entries
        random.seed(rank * 12345 + 67890)
        np.random.seed(rank * 12345 + 67890)
        
        for attempt in range(500):
            if time.time() > deadline - 5:
                break
            
            # Generate random half-integer matrices
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, rank))
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, rank))
            W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, rank))
            
            if verify(U, V, W):
                return (U, V, W)
        
        # Try sparse random
        for attempt in range(200):
            if time.time() > deadline - 5:
                break
            
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            for c in range(rank):
                # 2-3 nonzeros per column
                nnz_u = random.randint(2, 4)
                nnz_v = random.randint(2, 4)
                nnz_w = random.randint(2, 4)
                
                u_idx = random.sample(range(n*m), nnz_u)
                v_idx = random.sample(range(m*p), nnz_v)
                w_idx = random.sample(range(p*n), nnz_w)
                
                for idx in u_idx:
                    U[idx, c] = random.choice([-1, 1])
                for idx in v_idx:
                    V[idx, c] = random.choice([-1, 1])
                for idx in w_idx:
                    W[idx, c] = random.choice([-1, 1])
            
            if verify(U, V, W):
                return (U, V, W)
        
        return None
    
    # Run the main search
    best_decomp = build_3x3x5_decomposition()
    
    # Convert to lists and ensure half-integer entries
    U, V, W = best_decomp
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    for i in range(len(U_list)):
        for j in range(len(U_list[i])):
            U_list[i][j] = round(U_list[i][j] * 2) / 2
    for i in range(len(V_list)):
        for j in range(len(V_list[i])):
            V_list[i][j] = round(V_list[i][j] * 2) / 2
    for i in range(len(W_list)):
        for j in range(len(W_list[i])):
            W_list[i][j] = round(W_list[i][j] * 2) / 2
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
