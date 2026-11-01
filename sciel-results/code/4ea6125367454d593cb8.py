import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Try rank 36 construction
    rank = 36
    U = np.zeros((n * m, rank))  # (8, 36)
    V = np.zeros((m * p, rank))  # (20, 36)
    W = np.zeros((p * n, rank))  # (10, 36)
    
    # Indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for all j
    # We need: sum_j sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k
    
    # Strategy: Use Strassen-like combination for a 2x2x2x2 sub-block
    # For i in {0,1}, j in {0,1}, k in {0,1}: use 7 terms instead of 8
    # For the remaining 32 terms, use schoolbook
    
    col = 0
    
    # First, handle the 2x2x2x2 sub-block (i=0,1; j=0,1; k=0,1) with 7 Strassen terms
    # A sub-block: rows 0-1, cols 0-1 of A (which is 2x2)
    # B sub-block: rows 0-1, cols 0-1 of B (which is 2x2)
    
    # Strassen for 2x2 x 2x2:
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    # M3 = A[0,0] * (B[1,0]-B[0,0])
    # M4 = A[1,1] * (B[1,0]-B[0,0])
    # M5 = (A[0,0]+A[0,1]) * B[1,1]
    # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1]) * (B[0,0]+B[0,1])
    
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # Term 0: M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # U[0*4+0, 0] = 1, U[1*4+1, 0] = 1  (A[0,0] and A[1,1])
    # V[0*5+0, 0] = 1, V[1*5+1, 0] = 1  (B[0,0] and B[1,1])
    # W[0*2+0, 0] = 1, W[0*2+1, 0] = 1, W[1*2+0, 0] = 1, W[1*2+1, 0] = 1
    # Wait, W[k*n+i, r] where k is column of B, i is row of A
    # For C[0,0]: k=0, i=0 -> W[0*2+0, r] = W[0, r]
    # For C[0,1]: k=1, i=0 -> W[1*2+0, r] = W[2, r]
    # For C[1,0]: k=0, i=1 -> W[0*2+1, r] = W[1, r]
    # For C[1,1]: k=1, i=1 -> W[1*2+1, r] = W[3, r]
    
    # M1 contributes to C[0,0] and C[1,1] with coefficient +1
    # So W[0, 0] = 1, W[3, 0] = 1
    U[0*4+0, 0] = 1  # A[0,0]
    U[1*4+1, 0] = 1  # A[1,1]
    V[0*5+0, 0] = 1  # B[0,0]
    V[1*5+1, 0] = 1  # B[1,1]
    W[0*2+0, 0] = 1  # C[0,0]
    W[1*2+1, 0] = 1  # C[1,1]
    col += 1
    
    # Term 1: M2 = (A[1,0]+A[1,1]) * B[0,0]
    # Contributes to C[1,0] with +1
    U[1*4+0, 1] = 1  # A[1,0]
    U[1*4+1, 1] = 1  # A[1,1]
    V[0*5+0, 1] = 1  # B[0,0]
    W[0*2+1, 1] = 1  # C[1,0]
    col += 1
    
    # Term 2: M3 = A[0,0] * (B[1,0]-B[0,0])
    # Contributes to C[0,1] with +1 and C[1,1] with +1
    U[0*4+0, 2] = 1  # A[0,0]
    V[1*5+0, 2] = 1  # B[1,0]
    V[0*5+0, 2] = -1  # B[0,0]
    W[1*2+0, 2] = 1  # C[0,1]
    W[1*2+1, 2] = 1  # C[1,1]
    col += 1
    
    # Term 3: M4 = A[1,1] * (B[1,0]-B[0,0])
    # Contributes to C[0,0] with +1 and C[1,0] with +1
    U[1*4+1, 3] = 1  # A[1,1]
    V[1*5+0, 3] = 1  # B[1,0]
    V[0*5+0, 3] = -1  # B[0,0]
    W[0*2+0, 3] = 1  # C[0,0]
    W[0*2+1, 3] = 1  # C[1,0]
    col += 1
    
    # Term 4: M5 = (A[0,0]+A[0,1]) * B[1,1]
    # Contributes to C[0,0] with -1 and C[0,1] with +1
    U[0*4+0, 4] = 1  # A[0,0]
    U[0*4+1, 4] = 1  # A[0,1]
    V[1*5+1, 4] = 1  # B[1,1]
    W[0*2+0, 4] = -1  # C[0,0]
    W[1*2+0, 4] = 1   # C[0,1]
    col += 1
    
    # Term 5: M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # Contributes to C[1,1] with +1
    U[1*4+0, 5] = 1  # A[1,0]
    U[0*4+0, 5] = -1  # A[0,0]
    V[0*5+0, 5] = 1  # B[0,0]
    V[0*5+1, 5] = 1  # B[0,1]
    W[1*2+1, 5] = 1  # C[1,1]
    col += 1
    
    # Term 6: M7 = (A[0,1]-A[1,1]) * (B[0,0]+B[0,1])
    # Contributes to C[0,0] with +1
    U[0*4+1, 6] = 1  # A[0,1]
    U[1*4+1, 6] = -1  # A[1,1]
    V[0*5+0, 6] = 1  # B[0,0]
    V[0*5+1, 6] = 1  # B[0,1]
    W[0*2+0, 6] = 1  # C[0,0]
    col += 1
    
    # Now verify the Strassen part
    # C[0,0] should get: M1(+1) + M4(+1) + M5(-1) + M7(+1) = 1+1-1+1 = 2? That's wrong!
    
    # I think I made an error in the Strassen formulas. Let me re-derive.
    
    # Standard Strassen for 2x2 x 2x2:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(g-h)
    # M4 = d(f-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C[0,0] = M1+M4-M5+M7
    # C[0,1] = M3+M5
    # C[1,0] = M2+M4
    # C[1,1] = M1-M2+M3+M6
    
    # Let me re-implement with correct formulas.
    
    # Reset
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    
    # a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    
    # M1 = (a+d)(e+h): U[a]=1,U[d]=1; V[e]=1,V[h]=1; W[C[0,0]]=1,W[C[1,1]]=1
    U[0*4+0, 0] = 1; U[1*4+1, 0] = 1
    V[0*5+0, 0] = 1; V[1*5+1, 0] = 1
    W[0*2+0, 0] = 1; W[1*2+1, 0] = 1
    col += 1
    
    # M2 = (c+d)e: U[c]=1,U[d]=1; V[e]=1; W[C[1,0]]=1
    U[1*4+0, 1] = 1; U[1*4+1, 1] = 1
    V[0*5+0, 1] = 1
    W[0*2+1, 1] = 1
    col += 1
    
    # M3 = a(g-h): U[a]=1; V[g]=1,V[h]=-1; W[C[0,1]]=1,W[C[1,1]]=1
    U[0*4+0, 2] = 1
    V[1*5+0, 2] = 1; V[1*5+1, 2] = -1
    W[1*2+0, 2] = 1; W[1*2+1, 2] = 1
    col += 1
    
    # M4 = d(f-e): U[d]=1; V[f]=1,V[e]=-1; W[C[0,0]]=1,W[C[1,0]]=1
    U[1*4+1, 3] = 1
    V[0*5+1, 3] = 1; V[0*5+0, 3] = -1
    W[0*2+0, 3] = 1; W[0*2+1, 3] = 1
    col += 1
    
    # M5 = (a+b)h: U[a]=1,U[b]=1; V[h]=1; W[C[0,0]]=-1,W[C[0,1]]=1
    U[0*4+0, 4] = 1; U[0*4+1, 4] = 1
    V[1*5+1, 4] = 1
    W[0*2+0, 4] = -1; W[1*2+0, 4] = 1
    col += 1
    
    # M6 = (c-a)(e+f): U[c]=1,U[a]=-1; V[e]=1,V[f]=1; W[C[1,1]]=1
    U[1*4+0, 5] = 1; U[0*4+0, 5] = -1
    V[0*5+0, 5] = 1; V[0*5+1, 5] = 1
    W[1*2+1, 5] = 1
    col += 1
    
    # M7 = (b-d)(g+h): U[b]=1,U[d]=-1; V[g]=1,V[h]=1; W[C[0,0]]=1
    U[0*4+1, 6] = 1; U[1*4+1, 6] = -1
    V[1*5+0, 6] = 1; V[1*5+1, 6] = 1
    W[0*2+0, 6] = 1
    col += 1
    
    # Now add schoolbook terms for the remaining (i,j,k) combinations
    # We've handled: i in {0,1}, j in {0,1}, k in {0,1} (8 combinations, 7 terms)
    # Remaining: 
    # i in {0,1}, j in {0,1}, k in {2,3,4} (6 combinations)
    # i in {0,1}, j in {2,3}, k in {0,1,2,3,4} (20 combinations)
    # Total remaining: 26 combinations, 26 terms
    # Total: 7 + 26 = 33 terms. But we want 36, so we have 3 extra.
    
    # Wait, let me recount. Total (i,j,k) = 2*4*5 = 40.
    # Handled by Strassen: 2*2*2 = 8 combinations.
    # Remaining: 40 - 8 = 32 combinations.
    # So total terms: 7 + 32 = 39. Still not 36.
    
    # I need to find a way to save 3 more terms.
    
    # Let me try applying Strassen to another 2x2x2x2 block.
    # For i in {0,1}, j in {2,3}, k in {0,1}: another 8 combinations, 7 terms.
    # Total: 7 + 7 + (40-16) = 14 + 24 = 38. Still not 36.
    
    # Apply Strassen to i in {0,1}, j in {0,1}, k in {2,3}: 8 more, 7 terms.
    # Total: 7 + 7 + 7 + (40-24) = 21 + 16 = 37. Close!
    
    # Apply Strassen to i in {0,1}, j in {2,3}, k in {2,3}: 8 more, 7 terms.
    # Total: 7*4 + (40-32) = 28 + 8 = 36!
    
    # So: apply Strassen to all four 2x2x2x2 blocks:
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    # Block 2: i in {0,1}, j in {0,1}, k in {2,3}
    # Block 3: i in {0,1}, j in {2,3}, k in {0,1}
    # Block 4: i in {0,1}, j in {2,3}, k in {2,3}
    # Remaining: i in {0,1}, j in {0,1,2,3}, k in {4} (10 combinations)
    # Total: 7*4 + 10 = 28 + 10 = 38. Hmm, still 38.
    
    # Wait, 2*4*5 = 40. Four 2x2x2 blocks cover 4*8 = 32 combinations.
    # Remaining: 40-32 = 8 combinations (k=4 for all i,j).
    # Total: 7*4 + 8 = 28 + 8 = 36!
    
    # Great! Let me implement this.
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    
    # Define Strassen terms for a 2x2x2x2 block
    # Parameters: i_offset, j_offset, k_offset
    def add_strassen_block(i_off, j_off, k_off):
        nonlocal col
        # a=A[i_off, j_off], b=A[i_off, j_off+1], c=A[i_off+1, j_off], d=A[i_off+1, j_off+1]
        # e=B[j_off, k_off], f=B[j_off, k_off+1], g=B[j_off+1, k_off], h=B[j_off+1, k_off+1]
        
        # M1 = (a+d)(e+h)
        U[i_off*m+j_off, col] = 1; U[(i_off+1)*m+(j_off+1), col] = 1
        V[j_off*p+k_off, col] = 1; V[(j_off+1)*p+(k_off+1), col] = 1
        W[k_off*n+i_off, col] = 1; W[(k_off+1)*n+(i_off+1), col] = 1
        col += 1
        
        # M2 = (c+d)e
        U[(i_off+1)*m+j_off, col] = 1; U[(i_off+1)*m+(j_off+1), col] = 1
        V[j_off*p+k_off, col] = 1
        W[k_off*n+(i_off+1), col] = 1
        col += 1
        
        # M3 = a(g-h)
        U[i_off*m+j_off, col] = 1
        V[(j_off+1)*p+k_off, col] = 1; V[(j_off+1)*p+(k_off+1), col] = -1
        W[(k_off+1)*n+i_off, col] = 1; W[(k_off+1)*n+(i_off+1), col] = 1
        col += 1
        
        # M4 = d(f-e)
        U[(i_off+1)*m+(j_off+1), col] = 1
        V[j_off*p+(k_off+1), col] = 1; V[j_off*p+k_off, col] = -1
        W[k_off*n+i_off, col] = 1; W[k_off*n+(i_off+1), col] = 1
        col += 1
        
        # M5 = (a+b)h
        U[i_off*m+j_off, col] = 1; U[i_off*m+(j_off+1), col] = 1
        V[(j_off+1)*p+(k_off+1), col] = 1
        W[k_off*n+i_off, col] = -1; W[(k_off+1)*n+i_off, col] = 1
        col += 1
        
        # M6 = (c-a)(e+f)
        U[(i_off+1)*m+j_off, col] = 1; U[i_off*m+j_off, col] = -1
        V[j_off*p+k_off, col] = 1; V[j_off*p+(k_off+1), col] = 1
        W[(k_off+1)*n+(i_off+1), col] = 1
        col += 1
        
        # M7 = (b-d)(g+h)
        U[i_off*m+(j_off+1), col] = 1; U[(i_off+1)*m+(j_off+1), col] = -1
        V[(j_off+1)*p+k_off, col] = 1; V[(j_off+1)*p+(k_off+1), col] = 1
        W[k_off*n+i_off, col] = 1
        col += 1
    
    # Add four Strassen blocks
    add_strassen_block(0, 0, 0)  # i={0,1}, j={0,1}, k={0,1}
    add_strassen_block(0, 0, 2)  # i={0,1}, j={0,1}, k={2,3}
    add_strassen_block(0, 2, 0)  # i={0,1}, j={2,3}, k={0,1}
    add_strassen_block(0, 2, 2)  # i={0,1}, j={2,3}, k={2,3}
    
    # Add schoolbook terms for k=4 (all i, j)
    for i in range(n):
        for j in range(m):
            U[i*m+j, col] = 1
            V[j*p+4, col] = 1
            W[4*n+i, col] = 1
            col += 1
    
    # Verify: col should be 36
    assert col == rank, f"Expected {rank} terms, got {col}"
    
    # Verify numerically
    # C[i,k] = sum_j sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    C = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            total = 0
            for j in range(m):
                for r in range(rank):
                    total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            C[i, k] = total
    
    # Check if C is all 1s
    if np.allclose(C, np.ones((n, p)), atol=1e-10):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fall back to schoolbook
        rank_school = n * m * p
        U_s = np.zeros((n * m, rank_school))
        V_s = np.zeros((m * p, rank_school))
        W_s = np.zeros((p * n, rank_school))
        col_s = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j, col_s] = 1
                    V_s[j * p + k, col_s] = 1
                    W_s[k * n + i, col_s] = 1
                    col_s += 1
        return {
            "U": U_s.tolist(),
            "V": V_s.tolist(),
            "W": W_s.tolist()
        }
