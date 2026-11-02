import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 7]

def _strassen_2x2x2(n, m, p):
    # Returns U, V, W for 2x2x2 Strassen
    # n=2, m=2, p=2
    rank = 7
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Indices for A: 0,1,2,3 (0:00, 1:01, 2:10, 3:11)
    # Indices for B: 0,1,2,3 (0:00, 1:01, 2:10, 3:11)
    # Indices for C: 0,1,2,3 (0:00, 1:01, 2:10, 3:11)
    
    # M1 = (A00+A11)(B00+B11)
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[1,0]=1; W[2,0]=1; W[3,0]=1
    
    # M2 = (A10+A11)B00
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[2,1]=1; W[3,1]=1
    
    # M3 = A00(B01-B11)
    U[0,2]=1
    V[1,2]=1; V[3,2]=-1
    W[1,2]=1; W[3,2]=-1
    
    # M4 = A11(B10-B00)
    U[3,3]=1
    V[2,3]=1; V[0,3]=-1
    W[2,3]=1; W[0,3]=-1
    
    # M5 = (A00+A01)B11
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[1,4]=1; W[3,4]=1
    
    # M6 = (A10-A00)(B00+B01)
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[1,5]=1
    W[2,5]=1; W[3,5]=1
    
    # M7 = (A01-A11)(B10+B11)
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1; V[3,6]=1
    W[1,6]=1; W[3,6]=1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=7
    
    # We will build the decomposition by splitting B into 3 parts:
    # Part 1: cols 0,1 (2x3x2)
    # Part 2: cols 2,3 (2x3x2)
    # Part 3: cols 4,5,6 (2x3x3)
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    # Helper to add a block
    def add_block(b_start, b_end):
        # b_start, b_end are column indices in B
        p_block = b_end - b_start + 1
        # A is 2x3, B_block is 3xp_block
        # We use Strassen on the 2x2 part of A (cols 0,1) and B (rows 0,1)
        # and schoolbook for the rest.
        
        U_block = np.zeros((n*m, 0), dtype=float)
        V_block = np.zeros((m*p_block, 0), dtype=float)
        W_block = np.zeros((p_block*n, 0), dtype=float)
        
        # Get Strassen 2x2x2
        U_s, V_s, W_s = _strassen_2x2x2(2, 2, 2)
        
        # Map Strassen indices to block indices
        # A indices: 0,1,2,3 correspond to A[0,0], A[0,1], A[1,0], A[1,1]
        # In our A (2x3), these are A[0,0], A[0,1], A[1,0], A[1,1]
        # B indices: 0,1,2,3 correspond to B[0,0], B[0,1], B[1,0], B[1,1]
        # In our B_block (3xp), these are B[0,0], B[0,1], B[1,0], B[1,1]
        # C indices: 0,1,2,3 correspond to C[0,0], C[0,1], C[1,0], C[1,1]
        # In our C_block (2xp), these are C[0,0], C[0,1], C[1,0], C[1,1]
        
        # Add Strassen terms
        U_block = np.hstack([U_block, U_s])
        V_block = np.hstack([V_block, V_s])
        W_block = np.hstack([W_block, W_s])
        
        # Add schoolbook terms for the rest
        # A col 2 (index 2) and B row 2 (index 2)
        # Terms: A[0,2]B[2,k], A[1,2]B[2,k]
        # Also A[0,0]B[0,k] for k>=2? No, Strassen covers k=0,1 for rows 0,1.
        # We need to cover all terms not covered by Strassen.
        
        # Strassen covers:
        # A[0,0], A[0,1], A[1,0], A[1,1] with B[0,0], B[0,1], B[1,0], B[1,1]
        # So we need:
        # 1. A[0,2] with all B[2,k]
        # 2. A[1,2] with all B[2,k]
        # 3. A[0,0], A[0,1], A[1,0], A[1,1] with B[2,k] (if p_block > 2)
        
        # Let's just iterate over all terms (i,j,k) in the block and add them if not covered by Strassen.
        # But Strassen is a linear combination. It's easier to just add the missing scalar products.
        
        # Missing products:
        # For i in 0..1, j in 0..2, k in 0..p_block-1:
        # If (j,k) is not in the 2x2 Strassen part, add A[i,j]B[j,k].
        # Strassen part covers j in 0,1 and k in 0,1.
        
        for i in range(2):
            for j in range(3):
                for k in range(p_block):
                    # Check if this term is covered by Strassen
                    if j < 2 and k < 2:
                        continue # Covered by Strassen
                    # Add schoolbook term
                    U_new = np.zeros((n*m, 1), dtype=float)
                    V_new = np.zeros((m*p_block, 1), dtype=float)
                    W_new = np.zeros((p_block*n, 1), dtype=float)
                    
                    U_new[i*m + j, 0] = 1
                    V_new[j*p_block + k, 0] = 1
                    W_new[k*n + i, 0] = 1
                    
                    U_block = np.hstack([U_block, U_new])
                    V_block = np.hstack([V_block, V_new])
                    W_block = np.hstack([W_block, W_new])
        
        # Adjust V and W indices to global B and C indices
        # V_block has shape (3*p_block, rank). Global B has shape (3*7, rank).
        # We need to place V_block into the correct columns of V_global.
        # V_global index is j*7 + k_global.
        # V_block index is j*p_block + k_local.
        # k_global = b_start + k_local.
        
        V_global_block = np.zeros((m*p, U_block.shape[1]), dtype=float)
        for r in range(U_block.shape[1]):
            for j in range(3):
                for k_local in range(p_block):
                    k_global = b_start + k_local
                    V_global_block[j*p + k_global, r] = V_block[j*p_block + k_local, r]
        
        # W_block has shape (p_block*2, rank). Global C has shape (7*2, rank).
        # W_global index is k_global*2 + i.
        # W_block index is k_local*2 + i.
        
        W_global_block = np.zeros((p*n, U_block.shape[1]), dtype=float)
        for r in range(U_block.shape[1]):
            for k_local in range(p_block):
                k_global = b_start + k_local
                for i in range(2):
                    W_global_block[k_global*n + i, r] = W_block[k_local*2 + i, r]
        
        return U_block, V_global_block, W_global_block

    U1, V1, W1 = add_block(0, 1)
    U2, V2, W2 = add_block(2, 3)
    U3, V3, W3 = add_block(4, 6)
    
    U = np.hstack([U1, U2, U3])
    V = np.hstack([V1, V2, V3])
    W = np.hstack([W1, W2, W3])
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

construct()
