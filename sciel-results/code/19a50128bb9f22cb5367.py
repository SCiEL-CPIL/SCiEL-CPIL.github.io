import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 decomposition
    # U[i,j,r], V[j,k,r], W[i,k,r]
    # C[i,k] = sum_r U[i,j,r]*A[i,j] * V[j,k,r]*B[j,k] * W[i,k,r]
    # Actually: C[i,k] = sum_r (sum_j U[i,j,r]*A[i,j]) * (sum_j V[j,k,r]*B[j,k]) * W[i,k,r]
    # No, the standard form is:
    # C[i,k] = sum_r W[i,k,r] * M_r
    # where M_r = (sum_{i,j} U[i,j,r] * A[i,j]) * (sum_{j,k} V[j,k,r] * B[j,k])
    
    # Strassen for 2x2:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e  
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C[0,0] = M1 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    U_s = np.zeros((2,2,7), dtype=int)
    V_s = np.zeros((2,2,7), dtype=int)
    W_s = np.zeros((2,2,7), dtype=int)
    
    # r=0: M1=(a+d)(e+h)
    U_s[0,0,0]=1; U_s[1,1,0]=1
    V_s[0,0,0]=1; V_s[1,1,0]=1
    W_s[0,0,0]=1; W_s[1,1,0]=1
    
    # r=1: M2=(c+d)e
    U_s[1,0,1]=1; U_s[1,1,1]=1
    V_s[0,0,1]=1
    W_s[1,0,1]=1; W_s[1,1,1]=-1
    
    # r=2: M3=a(f-h)
    U_s[0,0,2]=1
    V_s[0,1,2]=1; V_s[1,1,2]=-1
    W_s[0,1,2]=1; W_s[1,1,2]=1
    
    # r=3: M4=d(g-e)
    U_s[1,1,3]=1
    V_s[1,0,3]=1; V_s[0,0,3]=-1
    W_s[0,0,3]=1; W_s[1,0,3]=1
    
    # r=4: M5=(a+b)h
    U_s[0,0,4]=1; U_s[0,1,4]=1
    V_s[1,1,4]=1
    W_s[0,0,4]=-1; W_s[0,1,4]=1
    
    # r=5: M6=(c-a)(e+f)
    U_s[1,0,5]=1; U_s[0,0,5]=-1
    V_s[0,0,5]=1; V_s[0,1,5]=1
    W_s[1,1,5]=1
    
    # r=6: M7=(b-d)(g+h)
    U_s[0,1,6]=1; U_s[1,1,6]=-1
    V_s[1,0,6]=1; V_s[1,1,6]=1
    W_s[0,0,6]=1
    
    # Find all valid 2x2x2 block positions
    blocks = []
    for i_s in range(0, n-1):
        for j_s in range(0, m-1):
            for k_s in range(0, p-1):
                blocks.append((i_s, j_s, k_s))
    
    # Select non-overlapping blocks greedily (maximize count)
    # For n=4,m=5,p=6: max disjoint = 2*2*3 = 12
    # Use systematic grid: i_s in {0,2}, j_s in {0,2,4->no, only 0,2}, k_s in {0,2,4}
    selected_blocks = []
    used = set()
    for i_s in [0, 2]:
        for j_s in [0, 2]:
            for k_s in [0, 2, 4]:
                if i_s+2 <= n and j_s+2 <= m and k_s+2 <= p:
                    key = (i_s, j_s, k_s)
                    if key not in used:
                        selected_blocks.append(key)
                        used.add(key)
    
    num_blocks = len(selected_blocks)
    
    # Indices covered by blocks
    covered = set()
    for (i_s, j_s, k_s) in selected_blocks:
        for i in range(i_s, i_s+2):
            for j in range(j_s, j_s+2):
                for k in range(k_s, k_s+2):
                    covered.add((i, j, k))
    
    # Schoolbook indices (not covered)
    schoolbook = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    schoolbook.append((i, j, k))
    
    total_rank = num_blocks * 7 + len(schoolbook)
    
    U = np.zeros((n*m, total_rank), dtype=int)
    V = np.zeros((m*p, total_rank), dtype=int)
    W = np.zeros((p*n, total_rank), dtype=int)
    
    col = 0
    # Schoolbook terms
    for (i, j, k) in schoolbook:
        U[i*m+j, col] = 1
        V[j*p+k, col] = 1
        W[k*n+i, col] = 1
        col += 1
    
    # Strassen terms
    for (i_s, j_s, k_s) in selected_blocks:
        for r in range(7):
            for i_loc in range(2):
                for j_loc in range(2):
                    val = U_s[i_loc, j_loc, r]
                    if val != 0:
                        U[(i_s+i_loc)*m + (j_s+j_loc), col] += val
            for j_loc in range(2):
                for k_loc in range(2):
                    val = V_s[j_loc, k_loc, r]
                    if val != 0:
                        V[(j_s+j_loc)*p + (k_s+k_loc), col] += val
            for i_loc in range(2):
                for k_loc in range(2):
                    val = W_s[i_loc, k_loc, r]
                    if val != 0:
                        W[(k_s+k_loc)*n + (i_s+i_loc), col] += val
            col += 1
    
    # Verify: check that the decomposition is correct
    # C[i,k] should equal sum_j A[i,j]*B[j,k]
    # Using the decomposition: C[i,k] = sum_r W[k*n+i, r] * (sum_{i',j'} U[i'*m+j', r]*A[i',j']) * (sum_{j',k'} V[j'*p+k', r]*B[j',k'])
    # This is correct by construction.
    
    elapsed = time.time() - start_time
    if elapsed > 100:
        # Safety: return what we have
        pass
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
