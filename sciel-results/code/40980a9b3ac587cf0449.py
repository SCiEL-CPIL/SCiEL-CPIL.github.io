import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 100  # leave margin for evaluator kill at 150s
    
    # Strassen's rank-7 decomposition for 2x2 matrix multiplication
    # Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # Must satisfy: sum_c U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c] == delta_{ik} for all i,j,k
    
    # Standard Strassen with:
    # m1 = (a+d)(e+h), m2 = (c+d)e, m3 = a(f-h), m4 = d(g-h)
    # m5 = (a+b)(e+h), m6 = (c-d)(g+h), m7 = (a-b)(f+h)
    # p11 = m1+m3-m5-m7, p12 = m4+m5, p21 = m2+m3, p22 = m1-m2+m4+m6
    
    # Mapping to indices (i=row, j=col):
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # i=0,j=0: a; i=0,j=1: b; i=1,j=0: c; i=1,j=1: d
    # j=0,k=0: e; j=0,k=1: f; j=1,k=0: g; j=1,k=1: h
    # Output: k=0,i=0: p11; k=1,i=0: p12; k=0,i=1: p21; k=1,i=1: p22
    
    U2 = np.array([
        # i=0,j=0 (a)
        [1, 0, 1, 0, 1, 0, 1],
        # i=0,j=1 (b)
        [0, 0, 0, 0, 1, 0, -1],
        # i=1,j=0 (c)
        [0, 1, 0, 0, 0, 1, 0],
        # i=1,j=1 (d)
        [1, 1, 0, 1, 0, 1, 0],
    ], dtype=int)
    
    V2 = np.array([
        # j=0,k=0 (e)
        [1, 1, 0, 0, 1, 0, 0],
        # j=0,k=1 (f)
        [0, 0, 1, 0, 0, 0, 1],
        # j=1,k=0 (g)
        [1, 0, 0, 1, 0, 0, 0],
        # j=1,k=1 (h)
        [1, 0, 0, 0, 1, 1, 1],
    ], dtype=int)
    
    W2 = np.array([
        # k=0,i=0 (p11)
        [1, 0, 0, 0, -1, 0, -1],
        # k=1,i=0 (p12)
        [0, 0, 0, 1, 1, 0, 0],
        # k=0,i=1 (p21)
        [0, 1, 1, 0, 0, 0, 0],
        # k=1,i=1 (p22)
        [1, -1, 0, 1, 0, 1, 0],
    ], dtype=int)
    
    # Verify Strassen 2x2x2 decomposition
    T2 = U2[:, :, np.newaxis] * V2[np.newaxis, :, :]  # will do full check
    valid2 = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = sum(U2[i*2+j, c] * V2[j*2+k, c] * W2[k*2+i, c] for c in range(7))
                if s != (1 if i == k else 0):
                    valid2 = False
                    break
            if not valid2:
                break
        if not valid2:
            break
    
    if not valid2:
        # Try alternative Strassen formulation
        U2b = np.array([
            [1, 0, 1, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 0, -1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 1, 0],
        ], dtype=int)
        V2b = np.array([
            [1, 1, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 1, 1, 1],
        ], dtype=int)
        W2b = np.array([
            [1, 0, 0, 0, -1, 0, -1],
            [0, 0, 0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0, 0, 0],
            [1, -1, 0, 1, 0, 1, 0],
        ], dtype=int)
        
        valid2b = True
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = sum(U2b[i*2+j, c] * V2b[j*2+k, c] * W2b[k*2+i, c] for c in range(7))
                    if s != (1 if i == k else 0):
                        valid2b = False
                        break
                if not valid2b:
                    break
            if not valid2b:
                break
        
        if not valid2b:
            # Fallback: schoolbook rank-64
            n, m, p = 4, 4, 4
            rank = 64
            U = [[0.0] * rank for _ in range(n * m)]
            V = [[0.0] * rank for _ in range(m * p)]
            W = [[0.0] * rank for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m + j][col] = 1.0
                        V[j*p + k][col] = 1.0
                        W[k*n + i][col] = 1.0
                        col += 1
            return {"U": U, "V": V, "W": W}
        
        U2, V2, W2 = U2b, V2b, W2b
    
    # Build rank-49 decomposition via Kronecker/tensor product of two Strassen
    # 4x4x4 = (2x2x2) ⊗ (2x2x2)
    # Rank = 7 * 7 = 49
    n, m, p = 4, 4, 4
    rank = 49
    
    # U: shape (n*m, rank) = (16, 49)
    # V: shape (m*p, rank) = (16, 49)
    # W: shape (p*n, rank) = (16, 49)
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            u_val = U2[i1*2 + j1, c1] * U2[i2*2 + j2, c2]
                            U[i*4 + j, c] = u_val
                            
                            for k1 in range(2):
                                for k2 in range(2):
                                    k = k1 * 2 + k2
                                    V[j*4 + k, c] = V2[j1*2 + k1, c1] * V2[j2*2 + k2, c2]
                                    W[k*4 + i, c] = W2[k1*2 + i1, c1] * W2[k2*2 + i2, c2]
    
    # Verify rank-49 decomposition
    valid4 = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = sum(U[i*4 + j, c] * V[j*4 + k, c] * W[k*4 + i, c] for c in range(49))
                if abs(s - (1.0 if i == k else 0.0)) > 1e-9:
                    valid4 = False
                    break
            if not valid4:
                break
        if not valid4:
            break
    
    if not valid4:
        # Fallback to schoolbook rank-64
        rank = 64
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1.0
                    V[j*p + k][col] = 1.0
                    W[k*n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Attempt to reduce rank by finding zero/dependent columns via rank analysis
    if time.time() < deadline - 5:
        # Check column-wise: if a column of U is all zeros, or combinations
        # Try to find a rank-r sub-decomposition by checking which columns are "active"
        # Actually, for Kronecker of two rank-7, all 49 columns are generically needed.
        # But let's try a linear algebra approach: can we combine columns?
        
        # More sophisticated: try random linear combinations to find redundancies
        # The tensor T4x4 = T2x2 ⊗ T2x2 has border rank at most 49.
        # Known best bound for 4x4x4 multiplication is 49 (Landsberg).
        # Actually, the multiplication tensor M_{4,4,4} has rank at most 49 via this construction.
        # Lower bound is 49 as well? Let me check: the border rank of M_{n,n,n} >= n^2 for n >= 2.
        # For 4x4x4, we know rank <= 49 and lower bound is at least 49? Actually 49 is the 
        # Kronecker bound. The actual rank might be lower but is unknown.
        # 
        # Let's try a different approach: use a known rank-47 or rank-48 algorithm if it exists.
        # Strassen's algorithm for 4x4: rank <= 49.
        # There's a rank-47 algorithm? I'm not sure.
        
        # Try to exploit symmetry: the matrix multiplication tensor for 4x4x4
        # might have a lower rank decomposition that doesn't come from Kronecker.
        # 
        # For now, return the rank-49 decomposition which is valid and well-known.
        pass
    
    # Convert to lists for the return format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Ensure integer values where possible
    U_list = [[float(x) if x != int(x) else int(x) for x in row] for row in U_list]
    V_list = [[float(x) if x != int(x) else int(x) for x in row] for row in V_list]
    W_list = [[float(x) if x != int(x) else int(x) for x in row] for row in W_list]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
