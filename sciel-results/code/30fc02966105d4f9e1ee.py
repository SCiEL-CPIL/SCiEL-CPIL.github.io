import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n, m, p = DIMENSIONS
    
    # Helper to create a rank-1 tensor term
    def add_term(U, V, W, u, v, w):
        idx = len(U)
        U.append(u)
        V.append(v)
        W.append(w)

    # Recursive decomposition function
    # Returns (U, V, W) for M_{n,m,p}
    def decompose(n, m, p):
        # Base cases
        if p == 1:
            # M_{n,m,1} rank nm
            U = []
            V = []
            W = []
            for i in range(n):
                for j in range(m):
                    u = [0] * (n*m)
                    v = [0] * (m*p)
                    w = [0] * (p*n)
                    u[i*m+j] = 1
                    v[j*p] = 1
                    w[i] = 1
                    add_term(U, V, W, u, v, w)
            return U, V, W
            
        if n == 2 and m == 2 and p == 2:
            # Strassen for 2x2 by 2x2
            U = []
            V = []
            W = []
            # Indices for A (2x2): 0,1,2,3
            # Indices for B (2x2): 0,1,2,3
            # Indices for C (2x2): 0,1,2,3
            # Strassen terms
            # M1 = (a11+a22)(b11+b22)
            u1 = [1,0,0,1]
            v1 = [1,0,0,1]
            # C11 += M1, C22 -= M1
            w1 = [1,0,0,-1]
            add_term(U, V, W, u1, v1, w1)
            
            # M2 = (a21+a22)(b11)
            u2 = [0,0,1,1]
            v2 = [1,0,0,0]
            w2 = [0,0,1,0]
            add_term(U, V, W, u2, v2, w2)
            
            # M3 = (a11)(b21-b22)
            u3 = [1,0,0,0]
            v3 = [0,1,0,-1]
            w3 = [0,1,0,0]
            add_term(U, V, W, u3, v3, w3)
            
            # M4 = (a22)(b12-b11)
            u4 = [0,0,0,1]
            v4 = [0,1,-1,0]
            w4 = [0,0,0,1]
            add_term(U, V, W, u4, v4, w4)
            
            # M5 = (a11+a12)(b22)
            u5 = [1,1,0,0]
            v5 = [0,0,0,1]
            w5 = [1,0,0,0]
            add_term(U, V, W, u5, v5, w5)
            
            # M6 = (a21-a11)(b11+b12)
            u6 = [-1,0,1,0]
            v6 = [1,1,0,0]
            w6 = [0,0,1,0]
            add_term(U, V, W, u6, v6, w6)
            
            # M7 = (a12-a22)(b21+b22)
            u7 = [0,1,0,-1]
            v7 = [0,0,1,1]
            w7 = [0,1,0,0]
            add_term(U, V, W, u7, v7, w7)
            
            return U, V, W

        # Recursive splits
        # Try splitting p
        if p > 1:
            p1 = p // 2
            p2 = p - p1
            U1, V1, W1 = decompose(n, m, p1)
            U2, V2, W2 = decompose(n, m, p2)
            
            # Combine
            # V1 is for B[:, :p1], V2 is for B[:, p1:]
            # We need to map indices
            U = []
            V = []
            W = []
            
            # Process first part
            for u, v, w in zip(U1, V1, W1):
                # v is length m*p1, need to extend to m*p
                v_ext = [0] * (m*p)
                for j in range(m):
                    for k in range(p1):
                        v_ext[j*p + k] = v[j*p1 + k]
                # w is length p1*n, need to extend to p*n
                w_ext = [0] * (p*n)
                for k in range(p1):
                    for i in range(n):
                        w_ext[k*n + i] = w[k*n + i]
                add_term(U, V, W, u, v_ext, w_ext)
                
            # Process second part
            for u, v, w in zip(U2, V2, W2):
                # v is length m*p2, need to shift
                v_ext = [0] * (m*p)
                for j in range(m):
                    for k in range(p2):
                        v_ext[j*p + (p1+k)] = v[j*p2 + k]
                # w is length p2*n, need to shift
                w_ext = [0] * (p*n)
                for k in range(p2):
                    for i in range(n):
                        w_ext[(p1+k)*n + i] = w[k*n + i]
                add_term(U, V, W, u, v_ext, w_ext)
                
            return U, V, W
            
        # If p==1 handled above.
        # What if n,m are large?
        # For this task, n=2, m=3.
        # If we reach here with p>1, we split p.
        # But we might want to split m if p is small?
        # For M_{2,3,1}, we handled p=1.
        # For M_{2,3,5}, we split p into 2,3.
        # M_{2,3,2} -> split p into 1,1?
        # M_{2,3,1} is rank 6.
        # M_{2,3,2} rank 12?
        # But we can split m=3 into 2,1 for M_{2,3,2}.
        # M_{2,2,2} rank 7.
        # M_{2,1,2} rank 4.
        # Total 11 < 12.
        
        # Let's add m-splitting logic if p is small or if it's better.
        # Heuristic: Split m if m > 2 and p <= 3?
        if m > 2 and p <= 3:
            m1 = 2
            m2 = m - 2
            U1, V1, W1 = decompose(n, m1, p)
            U2, V2, W2 = decompose(n, m2, p)
            
            U = []
            V = []
            W = []
            
            # Part 1: A[:, :m1], B[:m1, :]
            for u, v, w in zip(U1, V1, W1):
                # v is length m1*p
                v_ext = [0] * (m*p)
                for j in range(m1):
                    for k in range(p):
                        v_ext[j*p + k] = v[j*p + k]
                add_term(U, V, W, u, v_ext, w)
                
            # Part 2: A[:, m1:], B[m1:, :]
            for u, v, w in zip(U2, V2, W2):
                # v is length m2*p
                v_ext = [0] * (m*p)
                for j in range(m2):
                    for k in range(p):
                        v_ext[(m1+j)*p + k] = v[j*p + k]
                add_term(U, V, W, u, v_ext, w)
                
            return U, V, W
            
        # Fallback: Schoolbook
        rank = n * m * p
        U = []
        V = []
        W = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = [0] * (n*m)
                    v = [0] * (m*p)
                    w = [0] * (p*n)
                    u[i*m+j] = 1
                    v[j*p+k] = 1
                    w[k*n+i] = 1
                    add_term(U, V, W, u, v, w)
        return U, V, W

    U, V, W = decompose(n, m, p)
    
    # Convert to lists of lists if needed, but numpy arrays are fine?
    # The interface expects shapes (n*m, rank), etc.
    # My U, V, W are lists of vectors (rank x dim).
    # I need to transpose them.
    
    rank = len(U)
    U_mat = np.array(U).T
    V_mat = np.array(V).T
    W_mat = np.array(W).T
    
    # Verify validity quickly
    # T = sum U[:,r] V[:,r] W[:,r]
    # Check if it matches the tensor
    # For safety, just return
    
    return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
# EVOLVE-BLOCK-END
