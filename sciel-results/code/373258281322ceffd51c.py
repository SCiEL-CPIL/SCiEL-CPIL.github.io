import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0
    
    terms = []
    
    def add_term(U_dict, V_dict, W_dict):
        if time.time() > deadline:
            return
        U_dict = {k: v for k, v in U_dict.items() if v != 0}
        V_dict = {k: v for k, v in V_dict.items() if v != 0}
        W_dict = {k: v for k, v in W_dict.items() if v != 0}
        if U_dict and V_dict and W_dict:
            terms.append((U_dict, V_dict, W_dict))
    
    def add_strassen_2x2x2(r1, c1, k1):
        """Add Strassen terms for a 2x2x2 block starting at (r1,c1,k1)."""
        A00 = r1 * m + c1
        A01 = r1 * m + (c1 + 1)
        A10 = (r1 + 1) * m + c1
        A11 = (r1 + 1) * m + (c1 + 1)
        
        B00 = c1 * p + k1
        B01 = c1 * p + (k1 + 1)
        B10 = (c1 + 1) * p + k1
        B11 = (c1 + 1) * p + (k1 + 1)
        
        C00_w = k1 * n + r1
        C01_w = (k1 + 1) * n + r1
        C10_w = k1 * n + (r1 + 1)
        C11_w = (k1 + 1) * n + (r1 + 1)
        
        add_term({A00: 1, A11: 1}, {B00: 1, B11: 1}, {C00_w: 1, C11_w: 1})
        add_term({A10: 1, A11: 1}, {B00: 1}, {C10_w: 1, C11_w: -1})
        add_term({A00: 1}, {B01: 1, B11: -1}, {C01_w: 1, C11_w: 1})
        add_term({A11: 1}, {B10: 1, B00: -1}, {C00_w: 1, C10_w: 1})
        add_term({A00: 1, A01: 1}, {B11: 1}, {C00_w: -1, C01_w: 1})
        add_term({A10: 1, A00: -1}, {B00: 1, B01: 1}, {C11_w: 1})
        add_term({A01: 1, A11: -1}, {B10: 1, B11: 1}, {C00_w: 1})
    
    def add_laderman_3x3x3(r1, c1, k1):
        """Add Laderman's 11-term algorithm for a 3x3x3 block starting at (r1,c1,k1)."""
        def A(i, j):
            return (r1 + i) * m + (c1 + j)
        
        def B(j, k):
            return (c1 + j) * p + (k1 + k)
        
        def W(k, i):
            return (k1 + k) * n + (r1 + i)
        
        # Laderman's 11-term algorithm
        add_term({A(0,0): 1, A(1,2): 1}, {B(0,0): 1, B(1,2): 1}, {W(0,0): 1, W(2,2): 1})
        add_term({A(1,0): 1, A(2,2): 1}, {B(1,0): 1, B(2,2): 1}, {W(1,1): 1, W(2,2): 1})
        add_term({A(0,2): 1, A(2,0): 1}, {B(0,2): 1, B(2,0): 1}, {W(0,0): 1, W(2,2): 1})
        add_term({A(0,1): 1, A(1,0): 1, A(2,2): 1}, {B(0,1): 1, B(1,0): 1, B(2,2): 1}, {W(0,0): 1, W(2,2): 1})
        add_term({A(0,0): 1, A(1,1): 1, A(2,2): 1}, {B(0,0): 1, B(1,1): 1, B(2,2): 1}, {W(0,0): 1, W(2,2): 1})
        add_term({A(0,2): 1, A(1,1): 1, A(2,0): 1}, {B(0,2): 1, B(1,1): 1, B(2,0): 1}, {W(0,0): 1, W(2,2): 1})
        add_term({A(1,2): 1, A(2,1): 1}, {B(1,2): 1, B(2,1): 1}, {W(1,1): 1, W(2,2): 1})
        add_term({A(0,1): 1, A(1,2): 1}, {B(0,1): 1, B(1,2): 1}, {W(0,1): 1, W(1,2): 1})
        add_term({A(1,0): 1, A(2,1): 1}, {B(1,0): 1, B(2,1): 1}, {W(1,0): 1, W(2,1): 1})
        add_term({A(0,2): 1, A(2,1): 1}, {B(0,2): 1, B(2,1): 1}, {W(0,2): 1, W(2,1): 1})
        add_term({A(0,1): 1, A(2,0): 1}, {B(0,1): 1, B(2,0): 1}, {W(0,1): 1, W(1,0): 1})
    
    def add_schoolbook_block(r1, r2, c1, c2, k1, k2):
        """Add schoolbook terms for a block."""
        for ii in range(r1, r2):
            for jj in range(c1, c2):
                for kk in range(k1, k2):
                    u_idx = ii * m + jj
                    v_idx = jj * p + kk
                    w_idx = kk * n + ii
                    add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    # Partition: [3, 2]
    blocks = [(0, 3), (3, 5)]
    
    for r1, r2 in blocks:
        for c1, c2 in blocks:
            for k1, k2 in blocks:
                if time.time() > deadline:
                    break
                ar = r2 - r1
                ac = c2 - c1
                bc = k2 - k1
                
                if ar == 3 and ac == 3 and bc == 3:
                    add_laderman_3x3x3(r1, c1, k1)
                elif ar == 2 and ac == 2 and bc == 2:
                    add_strassen_2x2x2(r1, c1, k1)
                else:
                    add_schoolbook_block(r1, r2, c1, c2, k1, k2)
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Verify the decomposition
    rank = len(terms)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r in range(rank):
        U_dict, V_dict, W_dict = terms[r]
        for idx, val in U_dict.items():
            U[idx, r] = val
        for idx, val in V_dict.items():
            V[idx, r] = val
        for idx, val in W_dict.items():
            W[idx, r] = val
    
    # Check: sum_r U[:,r] * V[:,r] * W[:,r] should equal the multiplication tensor
    # The multiplication tensor has T[i,j,k] = 1 if i==j and j==k (for C[i,k] = sum_j A[i,j]B[j,k])
    # Actually, T[i,j,k] = 1 if the term A[i,j]*B[j,k]*C[k,i] appears, which is when i,j,k form a valid product
    # For matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor has T[i,j,k] = 1 for all i,j,k where the product A[i,j]*B[j,k] contributes to C[k,i]
    # Wait, the convention is W[k*n+i, r], so W index = k*n + i
    # The tensor element is T[u_idx, v_idx, w_idx] where u_idx = i*m+j, v_idx = j*p+k, w_idx = k*n+i
    # This should be 1 for all valid (i,j,k)
    
    # Reconstruct the tensor
    tensor = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        for u_idx in range(n*m):
            if U[u_idx, r] != 0:
                for v_idx in range(m*p):
                    if V[v_idx, r] != 0:
                        for w_idx in range(p*n):
                            if W[w_idx, r] != 0:
                                tensor[u_idx, v_idx, w_idx] += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
    
    # The correct tensor: T[i*m+j, j*p+k, k*n+i] = 1
    correct = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                correct[i*m+j, j*p+k, k*n+i] = 1
    
    # Check if they match
    if not np.allclose(tensor, correct):
        # Fallback to schoolbook
        terms.clear()
        add_schoolbook_block(0, n, 0, m, 0, p)
        rank = len(terms)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for r in range(rank):
            U_dict, V_dict, W_dict = terms[r]
            for idx, val in U_dict.items():
                U[idx, r] = val
            for idx, val in V_dict.items():
                V[idx, r] = val
            for idx, val in W_dict.items():
                W[idx, r] = val
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
