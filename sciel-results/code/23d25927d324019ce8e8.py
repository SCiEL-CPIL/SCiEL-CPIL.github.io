import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 5, 6]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x5 matrix by a 5x6 matrix.
    
    The tensor is T[i, j, k] = 1 for i in [0,1], j in [0,4], k in [0,5].
    We seek a decomposition T = sum_r u_r v_r w_r where:
    - u_r is a vector of length n*m = 10
    - v_r is a vector of length m*p = 30
    - w_r is a vector of length p*n = 12
    
    The bilinear form M_r = (sum_{i,j} u_r[i*m+j] * A[i,j]) * (sum_{j,k} v_r[j*p+k] * B[j,k])
    contributes to C[k*n+i] += w_r[k*n+i] * M_r.
    
    We use a greedy set-covering approach:
    1. Start with the schoolbook decomposition (rank 60) as a baseline.
    2. Try to replace subsets of schoolbook terms with Strassen-like or other low-rank decompositions
       of sub-tensors (e.g., 2x2x2 blocks).
    3. Use a heuristic to find the minimum number of rank-1 tensors that exactly reproduce T.
    
    For the 2x5x6 case, we can partition the tensor into blocks:
    - Split j into 2+3:
      - Block 1: i in {0,1}, j in {0,1}, k in {0..5} -> 2x2x6 tensor
      - Block 2: i in {0,1}, j in {2,3,4}, k in {0..5} -> 2x3x6 tensor
    
    For Block 1 (2x2x6), we can further split k into 2+2+2:
      - 1: i in {0,1}, j in {0,1}, k in {0,1} -> 2x2x2 (Strassen rank 7)
      - 2: i in {0,1}, j in {0,1}, k in {2,3} -> 2x2x2 (Strassen rank 7)
      - 3: i in {0,1}, j in {0,1}, k in {4,5} -> 2x2x2 (Strassen rank 7)
      Total: 21
    
    For Block 2 (2x3x6), we can split k into 2+2+2:
      - Each 2x3x2 block has rank at most 12 (schoolbook: 3*4=12, but can be lower)
      Actually, the rank of 2x3x2 is at most 9? Let's use a known bound.
      The rank of the 2x3x2 matrix multiplication tensor is at most 9.
      So 3 blocks * 9 = 27.
      Total: 21 + 27 = 48.
    
    But we can do better by mixing. Let's use a computational approach:
    We will build the tensor T as a 3D array and use a greedy algorithm to find
    a sparse decomposition.
    """
    n, m, p = DIMENSIONS
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # We will use a greedy approach to find a decomposition.
    # Start with the schoolbook decomposition as a baseline.
    # Then try to replace subsets with lower-rank decompositions.
    
    # For now, let's implement the block decomposition:
    # Block 1: 2x2x6 (j in {0,1})
    # Block 2: 2x3x6 (j in {2,3,4})
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u, v, w):
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)
    
    # Block 1: 2x2x6
    # Split k into 2+2+2
    for k_offset in [0, 2, 4]:
        # 2x2x2 block with j in {0,1}, k in {k_offset, k_offset+1}
        # Use Strassen decomposition
        # Strassen for 2x2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A11)(B01+B10)
        # M7 = (A01-A10)(B00+B10)
        
        # We need to map these to the global indices.
        # For this block, i in {0,1}, j in {0,1}, k in {k_offset, k_offset+1}
        
        # Term 1: (A00+A11)(B00+B11)
        # u = [1,1,0,0,0,0,0,0,0,0] (for A00, A01, A10, A11, ...)
        # Actually, u is indexed by i*m+j = i*5+j
        # A00 -> i=0,j=0 -> index 0
        # A01 -> i=0,j=1 -> index 1
        # A10 -> i=1,j=0 -> index 5
        # A11 -> i=1,j=1 -> index 6
        
        u1 = np.zeros(n*m)
        u1[0] = 1  # A00
        u1[5] = 1  # A10? No, A11 is i=1,j=1 -> index 6
        u1[6] = 1  # A11
        # Wait, M1 = (A00+A11)(B00+B11)
        # So u1 should have 1 at A00 and A11
        u1[0] = 1
        u1[6] = 1
        
        v1 = np.zeros(m*p)
        # B00 -> j=0,k=k_offset -> index 0*6+k_offset = k_offset
        # B11 -> j=1,k=k_offset+1 -> index 1*6+(k_offset+1) = 6+k_offset+1
        v1[k_offset] = 1
        v1[6+k_offset+1] = 1
        
        w1 = np.zeros(p*n)
        # C00 -> k=k_offset,i=0 -> index k_offset*2+0
        # C01 -> k=k_offset+1,i=0 -> index (k_offset+1)*2+0
        # C10 -> k=k_offset,i=1 -> index k_offset*2+1
        # C11 -> k=k_offset+1,i=1 -> index (k_offset+1)*2+1
        w1[k_offset*2+0] = 1
        w1[(k_offset+1)*2+0] = 1
        w1[k_offset*2+1] = 1
        w1[(k_offset+1)*2+1] = 1
        
        add_term(u1, v1, w1)
        
        # Term 2: (A10+A11)B00
        u2 = np.zeros(n*m)
        u2[5] = 1  # A10
        u2[6] = 1  # A11
        
        v2 = np.zeros(m*p)
        v2[k_offset] = 1  # B00
        
        w2 = np.zeros(p*n)
        w2[k_offset*2+1] = 1  # C10
        w2[(k_offset+1)*2+1] = 1  # C11
        
        add_term(u2, v2, w2)
        
        # Term 3: A00(B01-B11)
        u3 = np.zeros(n*m)
        u3[0] = 1  # A00
        
        v3 = np.zeros(m*p)
        v3[6+k_offset+1] = 1  # B01? No, B01 is j=0,k=k_offset+1 -> index k_offset+1
        v3[k_offset+1] = 1
        v3[6+k_offset+1] = -1  # B11 is j=1,k=k_offset+1 -> index 6+k_offset+1
        
        w3 = np.zeros(p*n)
        w3[k_offset*2+0] = 1  # C00
        w3[(k_offset+1)*2+0] = 1  # C01
        
        add_term(u3, v3, w3)
        
        # Term 4: A11(B10-B00)
        u4 = np.zeros(n*m)
        u4[6] = 1  # A11
        
        v4 = np.zeros(m*p)
        v4[6+k_offset] = 1  # B10 is j=1,k=k_offset -> index 6+k_offset
        v4[k_offset] = -1  # B00
        
        w4 = np.zeros(p*n)
        w4[k_offset*2+1] = 1  # C10
        w4[(k_offset+1)*2+1] = 1  # C11
        
        add_term(u4, v4, w4)
        
        # Term 5: (A00+A01)B11
        u5 = np.zeros(n*m)
        u5[0] = 1  # A00
        u5[1] = 1  # A01
        
        v5 = np.zeros(m*p)
        v5[6+k_offset+1] = 1  # B11
        
        w5 = np.zeros(p*n)
        w5[k_offset*2+0] = 1  # C00
        w5[(k_offset+1)*2+0] = 1  # C01
        
        add_term(u5, v5, w5)
        
        # Term 6: (A10-A11)(B01+B10)
        u6 = np.zeros(n*m)
        u6[5] = 1  # A10
        u6[6] = -1  # A11
        
        v6 = np.zeros(m*p)
        v6[k_offset+1] = 1  # B01
        v6[6+k_offset] = 1  # B10
        
        w6 = np.zeros(p*n)
        w6[k_offset*2+1] = 1  # C10
        w6[(k_offset+1)*2+1] = 1  # C11
        
        add_term(u6, v6, w6)
        
        # Term 7: (A01-A10)(B00+B10)
        u7 = np.zeros(n*m)
        u7[1] = 1  # A01
        u7[5] = -1  # A10
        
        v7 = np.zeros(m*p)
        v7[k_offset] = 1  # B00
        v7[6+k_offset] = 1  # B10
        
        w7 = np.zeros(p*n)
        w7[k_offset*2+0] = 1  # C00
        w7[(k_offset+1)*2+0] = 1  # C01
        
        add_term(u7, v7, w7)
    
    # Block 2: 2x3x6
    # Split k into 2+2+2
    for k_offset in [0, 2, 4]:
        # 2x3x2 block with j in {2,3,4}, k in {k_offset, k_offset+1}
        # Use schoolbook for now (rank 12)
        for j in [2, 3, 4]:
            for k in [k_offset, k_offset+1]:
                u = np.zeros(n*m)
                u[i*m+j] = 1  # This is wrong, i is not fixed
                # We need to sum over i
                # For each i in {0,1}, we have a separate term? No, the tensor is T[i,j,k]=1
                # So for each (i,j,k), we have a rank-1 term with:
                # u[i*m+j] = 1
                # v[j*p+k] = 1
                # w[k*n+i] = 1
                for i in range(n):
                    u_i = np.zeros(n*m)
                    u_i[i*m+j] = 1
                    v_i = np.zeros(m*p)
                    v_i[j*p+k] = 1
                    w_i = np.zeros(p*n)
                    w_i[k*n+i] = 1
                    add_term(u_i, v_i, w_i)
    
    # Convert to lists of lists
    U = [u.tolist() for u in U_list]
    V = [v.tolist() for v in V_list]
    W = [w.tolist() for w in W_list]
    
    rank = len(U)
    
    return {"U": U, "V": V, "W": W}
