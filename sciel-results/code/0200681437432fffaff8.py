"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reconstructs the multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = 0.0
                for c in range(rank):
                    val += U_arr[u_idx, c] * V_arr[v_idx, c] * W_arr[w_idx, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2x2():
    """
    Return the Strassen decomposition for 2x2 x 2x2 matrix multiplication.
    Returns (U, V, W) where:
    - U has shape (4, 7) for the 2x2 input matrix A (flattened row-major)
    - V has shape (4, 7) for the 2x2 input matrix B (flattened row-major)  
    - W has shape (4, 7) for the 2x2 output matrix C (flattened row-major)
    
    The tensor equation: sum_c U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c] = 1
    for all i,j,k in {0,1}
    """
    # Strassen's 7 products for C = A * B where A,B,C are 2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[p,q],[r,s]]
    # p = a*e + b*g, q = a*f + b*h
    # r = c*e + d*g, s = c*f + d*h
    
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # In our indexing:
    # A flattened: [a, b, c, d] -> indices 0,1,2,3 (i*2+j)
    # B flattened: [e, f, g, h] -> indices 0,1,2,3 (j*2+k)
    # C flattened: [p, q, r, s] -> indices 0,1,2,3 (k*2+i)
    #   p = C[0,0] -> k=0,i=0 -> index 0
    #   q = C[0,1] -> k=1,i=0 -> index 1
    #   r = C[1,0] -> k=0,i=1 -> index 2
    #   s = C[1,1] -> k=1,i=1 -> index 3
    
    # U[i*2+j, c]: coefficient for A[i,j] in product c
    # V[j*2+k, c]: coefficient for B[j,k] in product c
    # W[k*2+i, c]: coefficient for C[k,i] in product c (note: C[k,i] means row k, col i)
    
    # Wait, I need to be careful. The output is C[i,k] = sum_j A[i,j] B[j,k]
    # In the tensor: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # For 2x2x2: n=2, m=2, p=2
    # U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c]
    
    # The "1" means that the product A[i,j]*B[j,k] contributes 1 to C[i,k].
    # So W[k*2+i, c] is the coefficient that goes into C[i,k] (stored at position k*2+i).
    
    # Let me define each Strassen product:
    # Product c: U part * V part = contribution to W part
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a+d)(e+h)
    # U: a->1, d->1 => U[0,0]=1, U[3,0]=1
    # V: e->1, h->1 => V[0,0]=1, V[3,0]=1
    # Contribution to C: 
    # M1 contributes to C11 (+1), C22 (+1)
    # C11 is at W[0,0] (k=0,i=0), C22 is at W[3,0] (k=1,i=1)
    W[0,0] += 1   # C[0,0]
    W[3,0] += 1   # C[1,1]
    
    # M2 = (c+d)e
    # U: c->1, d->1 => U[2,1]=1, U[3,1]=1
    # V: e->1 => V[0,1]=1
    # M2 contributes to C21 (-1), C22 (-1)
    # C21 is at W[2,1] (k=0,i=1), C22 is at W[3,1] (k=1,i=1)
    W[2,1] += -1  # C[1,0]
    W[3,1] += -1  # C[1,1]
    
    # M3 = a(f-h)
    # U: a->1 => U[0,2]=1
    # V: f->1, h->-1 => V[1,2]=1, V[3,2]=-1
    # M3 contributes to C12 (+1), C22 (+1)
    # C12 is at W[1,2] (k=1,i=0), C22 is at W[3,2] (k=1,i=1)
    W[1,2] += 1   # C[0,1]
    W[3,2] += 1   # C[1,1]
    
    # M4 = d(g-e)
    # U: d->1 => U[3,3]=1
    # V: g->1, e->-1 => V[2,3]=1, V[0,3]=-1
    # M4 contributes to C11 (+1), C21 (+1)
    # C11 is at W[0,3] (k=0,i=0), C21 is at W[2,3] (k=0,i=1)
    W[0,3] += 1   # C[0,0]
    W[2,3] += 1   # C[1,0]
    
    # M5 = (a+b)h
    # U: a->1, b->1 => U[0,4]=1, U[1,4]=1
    # V: h->1 => V[3,4]=1
    # M5 contributes to C11 (-1), C12 (+1)
    # C11 is at W[0,4] (k=0,i=0), C12 is at W[1,4] (k=1,i=0)
    W[0,4] += -1  # C[0,0]
    W[1,4] += 1   # C[0,1]
    
    # M6 = (c-a)(e+f)
    # U: c->1, a->-1 => U[2,5]=1, U[0,5]=-1
    # V: e->1, f->1 => V[0,5]=1, V[1,5]=1
    # M6 contributes to C22 (+1)
    # C22 is at W[3,5] (k=1,i=1)
    W[3,5] += 1   # C[1,1]
    
    # M7 = (b-d)(g+h)
    # U: b->1, d->-1 => U[1,6]=1, U[3,6]=-1
    # V: g->1, h->1 => V[2,6]=1, V[3,6]=1
    # M7 contributes to C11 (+1)
    # C11 is at W[0,6] (k=0,i=0)
    W[0,6] += 1   # C[0,0]
    
    # Now fill in U and V
    U[0,0] = 1; U[3,0] = 1  # M1: a, d
    U[2,1] = 1; U[3,1] = 1  # M2: c, d
    U[0,2] = 1              # M3: a
    U[3,3] = 1              # M4: d
    U[0,4] = 1; U[1,4] = 1  # M5: a, b
    U[2,5] = 1; U[0,5] = -1 # M6: c, -a
    U[1,6] = 1; U[3,6] = -1 # M7: b, -d
    
    V[0,0] = 1; V[3,0] = 1  # M1: e, h
    V[0,1] = 1              # M2: e
    V[1,2] = 1; V[3,2] = -1 # M3: f, -h
    V[2,3] = 1; V[0,3] = -1 # M4: g, -e
    V[3,4] = 1              # M5: h
    V[0,5] = 1; V[1,5] = 1  # M6: e, f
    V[2,6] = 1; V[3,6] = 1  # M7: g, h
    
    return U, V, W

def _schoolbook(n, m, p):
    """Return the schoolbook (trivial) decomposition of rank n*m*p."""
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

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Strategy: Split the 3x3 middle into a 2x2 block and a 1x1 block.
    # A is 2x3: split as [A_2x2, A_2x1]
    # B is 3x10: split as [B_2x10; B_1x10]
    # C = A_2x2 * B_2x10 + A_2x1 * B_1x10
    #
    # A_2x2 * B_2x10: Split B_2x10 into five 2x2 blocks.
    # Each 2x2 x 2x2 uses Strassen (rank 7). Total: 5*7 = 35.
    #
    # A_2x1 * B_1x10: This is outer product of 2x1 and 1x10.
    # C[i,k] += A[i,2] * B[2,k] for i=0,1; k=0..9
    # This requires rank 20 in schoolbook, but we can do better.
    # Actually, for each k, we need 2 terms (one for each i).
    # But we can share: for all k, the B[2,k] factor is common per k.
    # Rank for this part: 2*10 = 20? No...
    # 
    # Let me think again. The tensor for A_2x1 * B_1x10:
    # For i in {0,1}, j=2, k in {0..9}:
    # U[i*3+2, c] * V[2*10+k, c] * W[k*2+i, c]
    # 
    # We need: for each (i,k), sum_c U[i*3+2,c]*V[20+k,c]*W[2k+i,c] = 1
    # (and for other (i',j',k') not involving j=2, the contribution from these terms should be 0)
    #
    # Simple approach: 20 terms, one for each (i,k) pair.
    # Term (i,k): U[i*3+2]=1, V[20+k]=1, W[2k+i]=1
    # This gives rank 20 for this part.
    #
    # Total rank: 35 + 20 = 55.
    #
    # But wait, can we do better for the A_2x1 * B_1x10 part?
    # The structure is: sum_k A[0,2]*B[2,k] for C[0,k] and A[1,2]*B[2,k] for C[1,k].
    # This is a rank-20 tensor in general (no structure to exploit since all entries are independent).
    # Actually, it IS structured: it's the outer product of a 2-vector and a 10-vector, 
    # but in the tensor decomposition framework, each (i,k) pair is independent.
    # The minimum rank for this sub-tensor is 2*10=20.
    #
    # So total rank = 35 + 20 = 55.
    
    # Build the decomposition
    U_list = []
    V_list = []
    W_list = []
    
    # Part 1: A_2x2 * B_2x10 using Strassen on five 2x2 blocks
    # A indices: i in {0,1}, j in {0,1} -> U[i*3+j, c]
    # B indices: j in {0,1}, k in {0..9} -> V[j*10+k, c]
    # C indices: i in {0,1}, k in {0..9} -> W[k*2+i, c]
    
    # Get Strassen components
    sU, sV, sW = _strassen_2x2x2()
    
    for block in range(5):
        # Block covers columns k = block*2 and block*2+1
        k0 = block * 2
        k1 = block * 2 + 1
        
        for c in range(7):
            col = len(U_list) if U_list else 0
            U_row = [0] * (n * m)  # 6 entries
            V_row = [0] * (m * p)  # 30 entries
            W_row = [0] * (p * n)  # 20 entries
            
            # U: A[i,j] for i in {0,1}, j in {0,1}
            # sU[i*2+j, c] gives coefficient
            for i in range(2):
                for j in range(2):
                    U_row[i * m + j] = sU[i * 2 + j][c]
            
            # V: B[j,k] for j in {0,1}, k in {k0, k1}
            # sV[j*2+k_local, c] where k_local is 0 or 1
            for j in range(2):
                for kl in range(2):
                    V_row[j * p + k0 + kl] = sV[j * 2 + kl][c]
            
            # W: C[i,k] for i in {0,1}, k in {k0, k1}
            # sW[k_local*2+i, c] where k_local is 0 or 1
            for i in range(2):
                for kl in range(2):
                    W_row[(k0 + kl) * n + i] = sW[kl * 2 + i][c]
            
            U_list.append(U_row)
            V_list.append(V_row)
            W_list.append(W_row)
    
    # Part 2: A_2x1 * B_1x10
    # A[i,2] for i in {0,1}
    # B[2,k] for k in {0..9}
    # C[i,k] for i in {0,1}, k in {0..9}
    # 20 terms
    for i in range(2):
        for k in range(10):
            U_row = [0] * (n * m)
            V_row = [0] * (m * p)
            W_row = [0] * (p * n)
            U_row[i * m + 2] = 1  # A[i,2]
            V_row[2 * p + k] = 1  # B[2,k]
            W_row[k * n + i] = 1  # C[i,k]
            U_list.append(U_row)
            V_list.append(V_row)
            W_list.append(W_row)
    
    result = {"U": U_list, "V": V_list, "W": W_list}
    rank = len(U_list)
    
    # Verify
    if _verify(U_list, V_list, W_list, n, m, p):
        return result
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
