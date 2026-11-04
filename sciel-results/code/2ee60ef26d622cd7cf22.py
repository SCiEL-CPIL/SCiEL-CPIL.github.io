import numpy as np
import time

def construct():
    n, m, p = 2, 3, 6
    deadline = time.time() + 140
    
    # Strategy: decompose 2x3x6 as three 2x3x2 blocks
    # Each block handles B columns (2t, 2t+1) for t=0,1,2
    # Need rank of 2x3x2 matmul tensor
    
    # First, let me find a valid decomposition for 2x3x2
    # T[i,j,k] = 1 for i in {0,1}, j in {0,1,2}, k in {0,1}
    # Try rank 8
    
    # Let me construct it systematically
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # For 2x3x2: i in {0,1}, j in {0,1,2}, k in {0,1}
    
    # Try to find U (2xr), V (3xr), W (2xr) with sum_r U[i,r]*V[j,r]*W[k,r] = 1
    
    # Let me try rank 7 first (optimistic)
    # If that fails, try rank 8
    
    def find_2x3x2_decomposition(max_rank=10):
        """Find a decomposition for 2x3x2 matmul tensor."""
        n2, m2, p2 = 2, 3, 2
        
        for rank in range(7, max_rank + 1):
            # Try a structured decomposition
            # Idea: use schoolbook for some terms and Strassen-like for others
            
            # Schoolbook would be 2*3*2 = 12 terms
            # Try to do better
            
            # Let me try: express as combination of rank-1 terms
            # T[i,j,k] = 1
            
            # Approach: fix W to be identity-like and solve for U, V
            # Or use a known pattern
            
            # Try: use 7 terms with a Strassen-like pattern
            # For 2x2x2, Strassen uses 7 terms. For 2x3x2, we have an extra j dimension.
            
            # Let me try a direct construction:
            # Term 1: i=0, j=0, k=0 -> U[0,0]=1, V[0,0]=1, W[0,0]=1
            # This is getting complex. Let me just use a numerical approach.
            
            # Actually, let me try a specific known decomposition.
            # The 2x3x2 tensor can be decomposed as:
            # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
            # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
            # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
            # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
            
            # Try rank 8:
            # Use Strassen for the "middle" part and schoolbook for edges
            
            # Let me try a different approach: just verify if a specific pattern works
            # I'll use a greedy/structured approach
            
            U2 = np.zeros((n2, rank))
            V2 = np.zeros((m2, rank))
            W2 = np.zeros((p2, rank))
            
            # Try a structured decomposition
            # Terms 0-3: schoolbook for j=0 (A[:,0] * B[0,:])
            # C[i,k] gets contribution A[i,0]*B[0,k]
            for k in range(p2):
                for i in range(n2):
                    U2[i, k] = 1.0
                    V2[0, k] = 1.0
                    W2[k, i] = 1.0
            
            # Terms 4-7: handle j=1,2 with Strassen-like
            # Remaining: C[i,k] needs A[i,1]*B[1,k] + A[i,2]*B[2,k]
            # This is a 2x2x2 problem (i in {0,1}, j in {1,2}, k in {0,1})
            # Use Strassen (7 terms) but we only have rank-4 remaining if rank=8
            
            # So rank 8 won't work with this split. Let me try rank 12 (schoolbook) first,
            # then see if we can do better.
            
            # Actually, let me just go with the known approach:
            # For 2x3x2, the rank is 8. Let me construct it.
            
            if rank == 8:
                # Construction for 2x3x2 rank 8:
                # Use 4 terms for j=0 (schoolbook: A[i,0]*B[0,k])
                # Use 4 terms for j=1,2 combined
                
                # Terms 0-3: j=0
                for k in range(2):
                    for i in range(2):
                        U2[i, k] = 1.0
                        V2[0, k] = 1.0
                        W2[k, i] = 1.0
                
                # Terms 4-7: j=1,2 with a 2x2x2 decomposition using 4 terms?
                # That's not possible (2x2x2 needs 7). 
                # So this split doesn't work for rank 8.
                
                # Alternative: use a more clever decomposition
                # Let me try:
                # M1 = A[0,0]*(B[0,0]+B[0,1]) + ... no, this is getting complicated
                
                # Let me just use schoolbook for now (rank 12) and see if the total is better
                pass
            
            # For now, let me just use schoolbook for each 2x3x2 block
            # That gives 3 * 12 = 36, which is worse than 33.
            
            # So the block approach with schoolbook doesn't help.
            # Let me stick with the current approach but try to optimize.
            
            return None
        
        return None
    
    # The current approach (rank 33) is:
    # 12 terms for A[:,0] * B[0,:] (schoolbook)
    # 21 terms for A[:,1:3] * B[1:3,:] (3 Strassen 2x2x2 blocks)
    
    # Can we do better? Let me try a different split.
    # What if I use Strassen on a 2x3x3 block?
    
    # Actually, let me try: split B into two parts:
    # B_left = B[:,0:2] (3x2), B_right = B[:,2:6] (3x4)
    # C_left = A * B_left (2x2), C_right = A * B_right (2x4)
    
    # For C_right: A (2x3) * B_right (3x4)
    # Split B_right into two 3x2 blocks: B_right = [B_r1, B_r2]
    # C_right = A * B_r1 + A * B_r2, but that's not how it works.
    # Actually C_right[:,k] = A * B_right[:,k] for k=2,3,4,5
    
    # This is getting circular. Let me just verify the current solution works and try
    # small improvements.
    
    # Current rank 33 solution:
    rank = 33
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: A[:, 0] * B[0, :]
    for k in range(p):
        for i in range(n):
            U[i * m + 0, col] = 1.0
            V[0 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1
            
    # Part 2: Three 2x2x2 Strassen blocks
    for t in range(3):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        if time.time() > deadline:
            break
        
        # M1 = (A00 + A11)(B00 + B11)
        U[0 * m + 1, col] += 1.0; U[1 * m + 2, col] += 1.0
        V[1 * p + k0, col] += 1.0; V[2 * p + k1, col] += 1.0
        W[k0 * n + 0, col] += 1.0; W[k1 * n + 1, col] += 1.0
        col += 1
        
        # M2 = (A10 + A11)B00
        U[1 * m + 1, col] += 1.0; U[1 * m + 2, col] += 1.0
        V[1 * p + k0, col] += 1.0
        W[k0 * n + 1, col] += 1.0; W[k1 * n + 1, col] += -1.0
        col += 1
        
        # M3 = A00(B10 - B11)
        U[0 * m + 1, col] += 1.0
        V[1 * p + k1, col] += 1.0; V[2 * p + k1, col] += -1.0
        W[k1 * n + 0, col] += 1.0; W[k1 * n + 1, col] += 1.0
        col += 1
        
        # M4 = A11(B01 - B00)
        U[1 * m + 2, col] += 1.0
        V[2 * p + k0, col] += 1.0; V[1 * p + k0, col] += -1.0
        W[k0 * n + 0, col] += 1.0; W[k0 * n + 1, col] += 1.0
        col += 1
        
        # M5 = (A00 + A01)B11
        U[0 * m + 1, col] += 1.0; U[0 * m + 2, col] += 1.0
        V[2 * p + k1, col] += 1.0
        W[k0 * n + 0, col] += -1.0; W[k1 * n + 0, col] += 1.0
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[1 * m + 1, col] += 1.0; U[0 * m + 1, col] += -1.0
        V[1 * p + k0, col] += 1.0; V[1 * p + k1, col] += 1.0
        W[k1 * n + 1, col] += 1.0
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[0 * m + 2, col] += 1.0; U[1 * m + 2, col] += -1.0
        V[2 * p + k0, col] += 1.0; V[2 * p + k1, col] += 1.0
        W[k0 * n + 0, col] += 1.0
        col += 1

    # Verify
    U_r = U.reshape(n, m, -1)
    V_r = V.reshape(m, p, -1)
    W_r = W.reshape(p, n, -1)
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_r[i, j, :] * V_r[j, k, :] * W_r[k, i, :])
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
        rank_school = n * m * p
        U_school = np.zeros((n * m, rank_school))
        V_school = np.zeros((m * p, rank_school))
        W_school = np.zeros((p * n, rank_school))
        col_s = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_school[ii * m + jj, col_s] = 1.0
                    V_school[jj * p + kk, col_s] = 1.0
                    W_school[kk * n + ii, col_s] = 1.0
                    col_s += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
