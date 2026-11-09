"""Matrix multiplication 2,3,6 - rank-minimal tensor decomposition."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 36
    
    # Build the target tensor T
    # T[x, y, z] = 1 if x = i*m+j, y = j*p+k, z = k*n+i for some i,j,k, else 0
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook decomposition
    def schoolbook():
        U = np.zeros((n*m, schoolbook_rank))
        V = np.zeros((m*p, schoolbook_rank))
        W = np.zeros((p*n, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    U, V, W = schoolbook()
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U, V, W
    
    # Verification function
    def verify(U, V, W, tol=1e-9):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        try:
            T_recon = np.einsum('xr,yr,zr->xyz', U, V, W)
            if np.max(np.abs(T_recon - T)) < tol:
                return True
        except:
            pass
        return False
    
    # Convert to clean list with integer/half-integer values
    def to_clean_list(arr):
        result = []
        for row in arr:
            clean_row = []
            for val in row:
                if abs(val - round(val)) < 1e-9:
                    clean_row.append(int(round(val)))
                elif abs(val*2 - round(val*2)) < 1e-9:
                    clean_row.append(round(val*2)/2.0)
                else:
                    clean_row.append(val)
            result.append(clean_row)
        return result
    
    if verify(U, V, W):
        best_U, best_V, best_W = U, V, W
        best_rank = schoolbook_rank
    
    # Strategy: Exploit structure of 2x3 x 3x6 matrix multiplication
    # C[i,k] = sum_j A[i,j]*B[j,k] for i in {0,1}, j in {0,1,2}, k in {0..5}
    
    # Known results: For small matrix multiplication, the rank is often equal to 
    # the number of output entries when the matrices are "small" in one dimension.
    # For 2x3 x 3x6, we have 12 output entries.
    
    # Let's try to find a decomposition using the structure.
    # Each output C[i,k] involves 3 products.
    # Total 36 products in schoolbook.
    
    # Key insight: If we can share some multiplications, we can reduce rank.
    # For example, if we compute (A[i,0] + A[i,1]) * B[j,k] for some j,
    # we can save one multiplication.
    
    # Let's try a structured search for rank < 36.
    # We'll use the fact that entries should be small integers or half-integers.
    
    # Approach: Try to find U, V, W with entries in {-1, 0, 1} or {-1/2, 0, 1/2}
    # that decompose T.
    
    # For rank r, each column of U, V, W is a vector.
    # The constraint is that sum_r U[x,r]*V[y,r]*W[z,r] = T[x,y,z]
    
    # Let's try a greedy approach: start with rank 12 (one per output) and see if we can
    # make it work, then try 13, 14, etc.
    
    # Actually, let's think about this differently.
    # The tensor rank of matrix multiplication <n,m,p> is denoted R(n,m,p).
    # Known values:
    # R(2,2,2) = 7 (Strassen)
    # R(2,2,3) = ?
    # R(2,3,3) = ?
    # R(2,3,6) = ?
    
    # For our case, let's try to find a decomposition with rank < 36.
    
    # Strategy: Use random search with structured initialization.
    # For each candidate rank r, try to find U, V, W.
    
    random.seed(42)
    
    # Lower bound: rank >= max(n*m, m*p, p*n) = max(6, 18, 12) = 18
    # But actually, for matrix multiplication, the rank is at least n*p = 12
    # (since each output entry requires at least one multiplication in any algorithm).
    # Actually, the rank is at least the number of output entries = n*p = 12.
    
    # Let's try ranks from 35 down to 12, but with limited attempts per rank.
    
    min_possible = 12
    
    # Try to find a lower rank decomposition
    for target_rank in range(best_rank - 1, min_possible - 1, -1):
        if time.time() > deadline - 5:
            break
        
        # For each target rank, try a few random initializations
        max_attempts = 20
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Initialize U, V, W with sparse random patterns
            U_try = np.zeros((n*m, target_rank))
            V_try = np.zeros((m*p, target_rank))
            W_try = np.zeros((p*n, target_rank))
            
            for r in range(target_rank):
                # For U: each row i*m+j should have some entries
                # For V: each row j*p+k should have some entries
                # For W: each row k*n+i should have some entries
                
                # Random sparse initialization
                num_u = random.randint(1, min(4, n*m))
                idx_u = random.sample(range(n*m), num_u)
                for idx in idx_u:
                    U_try[idx, r] = random.choice([-1, 1])
                
                num_v = random.randint(1, min(4, m*p))
                idx_v = random.sample(range(m*p), num_v)
                for idx in idx_v:
                    V_try[idx, r] = random.choice([-1, 1])
                
                num_w = random.randint(1, min(4, p*n))
                idx_w = random.sample(range(p*n), num_w)
                for idx in idx_w:
                    W_try[idx, r] = random.choice([-1, 1])
            
            if verify(U_try, V_try, W_try):
                best_rank = target_rank
                best_U, best_V, best_W = U_try, V_try, W_try
                found = True
                break
        
        if found:
            break
    
    # If no lower rank found, try a more structured approach
    # based on the actual matrix multiplication structure.
    
    # For 2x3 x 3x6:
    # A = [[a00, a01, a02], [a10, a11, a12]]
    # B = [[b00, b01, b02, b03, b04, b05],
    #      [b10, b11, b12, b13, b14, b15],
    #      [b20, b21, b22, b23, b24, b25]]
    # C[i,k] = sum_j A[i,j]*B[j,k]
    
    # C[0,0] = a00*b00 + a01*b10 + a02*b20
    # C[0,1] = a00*b01 + a01*b11 + a02*b21
    # ...
    # C[1,5] = a10*b05 + a11*b15 + a12*b25
    
    # Can we share multiplications?
    # For example, compute s0 = a00 + a01, then:
    # C[0,0] = s0*b00 + a02*b20 - a01*b00
    # This doesn't save us.
    
    # Let's try a different approach: use the fact that B has 3 rows.
    # We can think of B as a 3x6 matrix, and A as a 2x3 matrix.
    
    # Alternative: try to find a decomposition with rank 18 or so.
    # Each of the 18 entries of B could be multiplied by a linear combination of A entries.
    
    # Actually, let's just return the schoolbook solution if we can't find better.
    
    # Return the best solution found
    U_out = to_clean_list(best_U)
    V_out = to_clean_list(best_V)
    W_out = to_clean_list(best_W)
    
    return {"U": U_out, "V": V_out, "W": W_out}
