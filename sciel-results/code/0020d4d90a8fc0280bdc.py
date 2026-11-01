import time
import numpy as np
import random

DIMENSIONS = [2, 5, 6]
DEADLINE = 115.0  # Safety margin before 120s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # 1. Construct a Rank 59 Decomposition using Strassen on a 2x2 sub-block
    # The tensor is defined by T[i, j, k] where:
    # i in [0, n) -> row of A (2)
    # j in [0, m) -> col of A / row of B (5)
    # k in [0, p) -> col of B (6)
    # Output C[i, k] = sum_j A[i, j] B[j, k]
    
    # We decompose the multiplication into terms.
    # Schoolbook has n*m*p = 2*5*6 = 60 terms.
    # We can replace the terms corresponding to a 2x2 sub-matrix multiplication with 7 terms (Strassen) instead of 4.
    # Let's pick the first 2 rows of A (i=0,1) and the first 2 columns of B (k=0,1) and the corresponding intermediate indices j=0,1.
    # Wait, the intermediate index j is shared.
    # The term for C[i, k] is sum_j A[i, j] B[j, k].
    # If we isolate the contribution of j=0,1 to C[0,0], C[0,1], C[1,0], C[1,1], we are effectively computing the product of a 2x2 matrix A_sub and 2x2 matrix B_sub.
    # A_sub = A[0:2, 0:2]
    # B_sub = B[0:2, 0:2]
    # C_sub = A_sub @ B_sub
    
    # The other terms in the sum for C[i,k] involve j=2,3,4.
    # For C[0,0], C[0,1], C[1,0], C[1,1], the remaining sum is over j=2,3,4.
    # There are 4 outputs * 3 inputs = 12 terms from the "tail" j=2,3,4 for these 4 outputs.
    # For the other outputs C[0,2..5] and C[1,2..5] (8 outputs), the sum is over all j=0..4.
    # That's 8 outputs * 5 inputs = 40 terms.
    
    # Total terms:
    # 7 (Strassen for A_sub @ B_sub)
    # 12 (Tail for C_sub)
    # 40 (Full sum for C_rest)
    # Total = 59.
    
    # Let's construct the matrices U, V, W.
    # Rank R = 59.
    R = 59
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    # Indices:
    # U rows: i*m + j (i in 0..1, j in 0..4)
    # V rows: j*p + k (j in 0..4, k in 0..5)
    # W rows: k*n + i (k in 0..5, i in 0..1)
    
    # Strassen's Algorithm for 2x2 matrices:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+cg, bf+dh], [ce+ag, bf+dh]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # Mapping to our indices:
    # a = A[0,0] -> U row 0*5+0 = 0
    # b = A[0,1] -> U row 0*5+1 = 1
    # c = A[1,0] -> U row 1*5+0 = 5
    # d = A[1,1] -> U row 1*5+1 = 6
    
    # e = B[0,0] -> V row 0*6+0 = 0
    # f = B[0,1] -> V row 0*6+1 = 1
    # g = B[1,0] -> V row 1*6+0 = 6
    # h = B[1,1] -> V row 1*6+1 = 7
    
    # Outputs:
    # C[0,0] -> W row 0*2+0 = 0
    # C[0,1] -> W row 1*2+0 = 2
    # C[1,0] -> W row 0*2+1 = 1
    # C[1,1] -> W row 1*2+1 = 3
    
    # We need to define 7 terms for the Strassen part.
    # Each term r corresponds to a product U_r * V_r * W_r.
    # The contribution to C[i,k] is sum_r U[i,j1,r] V[j2,k,r] W[k,i,r] ... wait.
    # The tensor decomposition is T = sum_r U_r \otimes V_r \otimes W_r.
    # T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
    # We want T to be 1 if j1=j2 (identity for the multiplication tensor? No, the tensor for matrix multiplication is defined such that C[i,k] = sum_j A[i,j]B[j,k].
    # The tensor entries are T[i,j,k] = 1 if we are multiplying A[i,j] and B[j,k] to contribute to C[i,k].
    # Actually, the standard definition is:
    # The multiplication tensor T has T[i,j,k] = 1.
    # The decomposition is sum_r u_r \otimes v_r \otimes w_r.
    # u_r[i,j], v_r[j,k], w_r[k,i].
    # Then C[i,k] = sum_j A[i,j] B[j,k] = sum_j sum_r u_r[i,j] v_r[j,k] w_r[k,i] A[i,j] B[j,k]?
    # No. The identity is:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = delta_{j,j'}? No.
    # Let's check the check function in the seed:
    # "sum_r U[i*m+j1, r] V[j2*p+k, r] W[k*n+i, r] = delta(j1, j2)"
    # This implies that the tensor being decomposed is the identity tensor in the middle index?
    # Let's re-read carefully.
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 5 matrix by a 5 by 6 matrix."
    # The output factor follows the official trace convention W[k*n+i,r].
    # The check function in the seed code verifies:
    # for i in range(n):
    #   for k in range(p):
    #     U_slice = U[i*m:(i+1)*m, :] # (m, rank)
    #     V_slice: rows j*p+k for j=0..m-1
    #     w_val = W[k*n + i, :]
    #     prod = U_slice * w_val[None, :]
    #     mat = prod @ V_slice.T
    #     if not np.array_equal(mat, np.eye(m, dtype=int)): return False
    
    # This check verifies that:
    # sum_r U[i, j1, r] W[k, i, r] V[j2, k, r] = delta_{j1, j2}
    # This is exactly the condition for the tensor decomposition of the matrix multiplication map.
    # Let A be n x m, B be m x p.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # If we have U, V, W such that sum_r U[i, j, r] W[k, i, r] V[j', k, r] = delta_{j, j'},
    # Then sum_j A[i, j] B[j, k] = sum_j A[i, j] B[j, k] * 1
    # = sum_j A[i, j] B[j, k] * sum_r U[i, j, r] W[k, i, r] V[j, k, r]  (using j=j')
    # = sum_r (sum_j A[i, j] U[i, j, r]) * W[k, i, r] * (sum_k? No, k is fixed)
    # Wait, the V index is [j, k].
    # So term r contributes:
    # (sum_j A[i, j] U[i, j, r]) * W[k, i, r] * (sum_k? No, V is indexed by j and k.
    # Let's look at the term: U[i, j, r] V[j, k, r] W[k, i, r].
    # If we contract with A[i, j] and B[j, k]:
    # sum_{i,j,k} A[i, j] B[j, k] (sum_r U[i, j, r] V[j, k, r] W[k, i, r])
    # This is not the standard way to view it.
    
    # The standard view:
    # C[i, k] = sum_j A[i, j] B[j, k].
    # We want to compute C using rank R.
    # C[i, k] = sum_r (sum_j A[i, j] U[i, j, r]) * (sum_k? No)
    # Let's define:
    # M1[i, r] = sum_j A[i, j] U[i, j, r]
    # M2[k, r] = sum_j B[j, k] V[j, k, r]
    # Then C[i, k] = sum_r M1[i, r] * W[k, i, r] * M2[k, r]?
    # Let's check the identity: sum_r U[i, j, r] V[j, k, r] W[k, i, r] = delta_{j, j'}?
    # If we set j'=j, we get sum_r U[i, j, r] V[j, k, r] W[k, i, r] = 1.
    # Then sum_j A[i, j] B[j, k] = sum_j A[i, j] B[j, k] sum_r U[i, j, r] V[j, k, r] W[k, i, r]
    # = sum_r W[k, i, r] (sum_j A[i, j] U[i, j, r]) (sum_j B[j, k] V[j, k, r])
    # Yes! This works.
    # So the condition is:
    # For all i, k, j1, j2:
    # sum_r U[i, j1, r] V[j2, k, r] W[k, i, r] = delta_{j1, j2}
    
    # This is a linear system.
    # We can construct a solution.
    # The Strassen construction provides a solution for the 2x2 block.
    # Let's implement the Strassen terms.
    
    # Term 0: M1 = (a+d)(e+h)
    # U: a=1, d=1 -> U[0,0]=1, U[1,1]=1 (indices 0 and 6)
    # V: e=1, h=1 -> V[0,0]=1, V[1,1]=1 (indices 0 and 7)
    # W: C11 += 1, C22 += 1 -> W[0,0]=1, W[1,1]=1 (indices 0 and 3)
    # Note: W index is k*n+i.
    # C[0,0] is k=0, i=0 -> index 0.
    # C[1,1] is k=1, i=1 -> index 3.
    
    U[0, 0] = 1; U[6, 0] = 1
    V[0, 0] = 1; V[7, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 1: M2 = (c+d)e
    # U: c=1, d=1 -> U[1,0]=1, U[1,1]=1 (indices 5 and 6)
    # V: e=1 -> V[0,0]=1 (index 0)
    # W: C21 += 1 -> W[0,1]=1 (index 1)
    U[5, 1] = 1; U[6, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1
    
    # Term 2: M3 = a(f-h)
    # U: a=1 -> U[0,0]=1 (index 0)
    # V: f=1, h=-1 -> V[0,1]=1, V[1,1]=-1 (indices 1 and 7)
    # W: C12 += 1, C22 += 1 -> W[1,0]=1, W[1,1]=1 (indices 2 and 3)
    U[0, 2] = 1
    V[1, 2] = 1; V[7, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # Term 3: M4 = d(g-e)
    # U: d=1 -> U[1,1]=1 (index 6)
    # V: g=1, e=-1 -> V[1,0]=1, V[0,0]=-1 (indices 6 and 0)
    # W: C11 += 1, C21 += 1 -> W[0,0]=1, W[0,1]=1 (indices 0 and 1)
    U[6, 3] = 1
    V[6, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # Term 4: M5 = (a+b)h
    # U: a=1, b=1 -> U[0,0]=1, U[0,1]=1 (indices 0 and 1)
    # V: h=1 -> V[1,1]=1 (index 7)
    # W: C11 -= 1, C12 += 1 -> W[0,0]=-1, W[1,0]=1 (indices 0 and 2)
    U[0, 4] = 1; U[1, 4] = 1
    V[7, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # Term 5: M6 = (c-a)(e+f)
    # U: c=1, a=-1 -> U[1,0]=1, U[0,0]=-1 (indices 5 and 0)
    # V: e=1, f=1 -> V[0,0]=1, V[0,1]=1 (indices 0 and 1)
    # W: C22 += 1 -> W[1,1]=1 (index 3)
    U[5, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Term 6: M7 = (b-d)(g+h)
    # U: b=1, d=-1 -> U[0,1]=1, U[1,1]=-1 (indices 1 and 6)
    # V: g=1, h=1 -> V[1,0]=1, V[1,1]=1 (indices 6 and 7)
    # W: C11 += 1 -> W[0,0]=1 (index 0)
    U[1, 6] = 1; U[6, 6] = -1
    V[6, 6] = 1; V[7, 6] = 1
    W[0, 6] = 1
    
    # Now add the tail terms for C[0,0], C[0,1], C[1,0], C[1,1] involving j=2,3,4.
    # For each i in [0,1], k in [0,1], j in [2,3,4]:
    # We need a term such that U[i,j,r]=1, V[j,k,r]=1, W[k,i,r]=1.
    # This is just schoolbook for these specific indices.
    
    col = 7
    for i in range(2):
        for k in range(2):
            for j in range(2, 5):
                U[i*5+j, col] = 1
                V[j*6+k, col] = 1
                W[k*2+i, col] = 1
                col += 1
                
    # Now add the full sum terms for C[0,2..5] and C[1,2..5] involving all j=0..4.
    # For each i in [0,1], k in [2,5], j in [0,4]:
    # We need a term such that U[i,j,r]=1, V[j,k,r]=1, W[k,i,r]=1.
    
    for i in range(2):
        for k in range(2, 6):
            for j in range(5):
                U[i*5+j, col] = 1
                V[j*6+k, col] = 1
                W[k*2+i, col] = 1
                col += 1
                
    # Total columns used: 7 + 12 + 40 = 59.
    # Check if col == 59.
    if col != R:
        # If not, we have a bug.
        pass
        
    # Verify the solution
    def check_solution(U, V, W, rank):
        for i in range(n):
            for k in range(p):
                U_slice = U[i*m:(i+1)*m, :]
                V_slice = np.empty((m, rank), dtype=int)
                for j in range(m):
                    V_slice[j, :] = V[j*p + k, :]
                w_val = W[k*n + i, :]
                prod = U_slice * w_val[None, :]
                mat = prod @ V_slice.T
                if not np.array_equal(mat, np.eye(m, dtype=int)):
                    return False
        return True

    if not check_solution(U, V, W, R):
        # Fallback to schoolbook if Strassen construction fails (should not happen)
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
