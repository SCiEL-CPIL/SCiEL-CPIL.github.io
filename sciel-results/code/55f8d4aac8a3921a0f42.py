import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin within 120s
    
    n, m, p = DIMENSIONS
    total_rank = 46
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx, col] = u_val
        V[v_idx, col] = v_val
        W[w_idx, col] = w_val
        col += 1
    
    # Strassen for M_{2,2,2}
    # A = [[a11, a12], [a21, a22]]
    # B = [[b11, b12], [b21, b22]]
    # C = A*B
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # Indices for A (2x2): a11=0, a12=1, a21=2, a22=3
    # Indices for B (2x2): b11=0, b12=1, b21=2, b22=3
    # Indices for C (2x2): c11=0, c12=1, c21=2, c22=3
    
    # We need to map these to global indices.
    # Global A index: i*m + j
    # Global B index: j*p + k
    # Global C index: k*n + i
    
    # Let's define a helper to add a Strassen block at specific offsets
    def add_strassen(a_off, b_off, c_off, a_m, b_p, c_n, scale=1.0):
        # a_off: offset in A (row-major)
        # b_off: offset in B (row-major)
        # c_off: offset in C (row-major)
        # a_m: number of rows in A block
        # b_p: number of cols in B block (which is rows in B block)
        # c_n: number of rows in C block (which is cols in C block)
        
        # A block indices:
        # a11: a_off + 0*a_m + 0
        # a12: a_off + 0*a_m + 1
        # a21: a_off + 1*a_m + 0
        # a22: a_off + 1*a_m + 1
        
        # B block indices:
        # b11: b_off + 0*b_p + 0
        # b12: b_off + 0*b_p + 1
        # b21: b_off + 1*b_p + 0
        # b22: b_off + 1*b_p + 1
        
        # C block indices:
        # c11: c_off + 0*c_n + 0
        # c12: c_off + 0*c_n + 1
        # c21: c_off + 1*c_n + 0
        # c22: c_off + 1*c_n + 1
        
        # M1 = (a11+a22)(b11+b22)
        # U: a11, a22
        # V: b11, b22
        # W: c11, c22
        # This is not a single term, it's a product of sums.
        # We need to expand this into linear terms.
        # Actually, the tensor decomposition is linear in A, B, C.
        # M1 contributes to C11 and C22.
        # C11 += M1
        # C22 += M1
        # M1 = (a11+a22)(b11+b22) = a11*b11 + a11*b22 + a22*b11 + a22*b22
        # This is 4 terms.
        # But wait, Strassen's algorithm uses 7 multiplications.
        # The decomposition is:
        # C = sum_{r=1}^7 M_r * T_r
        # where M_r are the products of linear forms.
        # So for each r, we have a term:
        # U[:, r] * V[:, r] * W[:, r]
        # where U[:, r] is the vector of coefficients for A in M_r
        # V[:, r] is the vector of coefficients for B in M_r
        # W[:, r] is the vector of coefficients for C in M_r
        
        # Let's define the 7 terms for Strassen.
        # Each term has non-zero entries in U, V, W.
        
        # Term 1: M1 = (a11+a22)(b11+b22)
        # U: a11: 1, a22: 1
        # V: b11: 1, b22: 1
        # W: c11: 1, c22: 1
        # But this is not a rank-1 term. It's a product of two linear forms.
        # In the tensor decomposition T = sum_r U_r V_r W_r, each r is a rank-1 term.
        # So M1 is not a single rank-1 term.
        # Wait, the tensor T for matrix multiplication is:
        # T[i,j,k,l,m,n] = delta_{j,l} delta_{m,i}
        # The decomposition is T = sum_r U_{i,j,r} V_{l,m,r} W_{k,n,r}
        # Here, i,j are A indices, l,m are B indices, k,n are C indices.
        # In our case, A is (n,m), B is (m,p), C is (n,p).
        # So A indices: i, j (i=0..n-1, j=0..m-1)
        # B indices: j, k (j=0..m-1, k=0..p-1)
        # C indices: i, k (i=0..n-1, k=0..p-1)
        # The term is A[i,j] * B[j,k] -> C[i,k]
        
        # For Strassen, we are decomposing a sub-problem.
        # Let's say we have a sub-problem with A' (2x2), B' (2x2), C' (2x2).
        # The decomposition of this sub-problem has rank 7.
        # So we add 7 terms to the global U, V, W.
        
        # Let's define the 7 terms for the 2x2x2 sub-problem.
        # Term 1: M1 = (a11+a22)(b11+b22)
        # This contributes to C11 and C22.
        # C11 += M1, C22 += M1
        # So for term 1:
        # U: a11=1, a22=1
        # V: b11=1, b22=1
        # W: c11=1, c22=1
        # But this is not a valid rank-1 term because U and V have multiple non-zero entries.
        # A rank-1 term U_r V_r W_r means U_r is a vector, V_r is a vector, W_r is a vector.
        # The product U_r[i] V_r[j] W_r[k] is the coefficient of A[i] B[j] C[k].
        # For Strassen, the 7 multiplications are:
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b12-b22)
        # M4 = a22(b21-b11)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        
        # Each M_r is a product of a linear form in A and a linear form in B.
        # So for each r, we have:
        # U_r: vector of coefficients for A in the linear form
        # V_r: vector of coefficients for B in the linear form
        # W_r: vector of coefficients for C (how M_r contributes to C)
        
        # Term 1:
        # U1: a11=1, a22=1
        # V1: b11=1, b22=1
        # W1: c11=1, c22=1
        # This is valid. U1 is a vector, V1 is a vector, W1 is a vector.
        
        # Let's add these 7 terms.
        
        # Term 1
        U[a_off + 0, col] = 1  # a11
        U[a_off + a_m + 1, col] = 1  # a22
        V[b_off + 0, col] = 1  # b11
        V[b_off + b_p + 1, col] = 1  # b22
        W[c_off + 0, col] = 1  # c11
        W[c_off + c_n + 1, col] = 1  # c22
        col += 1
        
        # Term 2: M2 = (a21+a22)b11
        # U2: a21=1, a22=1
        # V2: b11=1
        # W2: c21=1, c22=1
        U[a_off + a_m + 0, col] = 1  # a21
        U[a_off + a_m + 1, col] = 1  # a22
        V[b_off + 0, col] = 1  # b11
        W[c_off + c_n + 0, col] = 1  # c21
        W[c_off + c_n + 1, col] = 1  # c22
        col += 1
        
        # Term 3: M3 = a11(b12-b22)
        # U3: a11=1
        # V3: b12=1, b22=-1
        # W3: c12=1, c22=1
        U[a_off + 0, col] = 1  # a11
        V[b_off + 1, col] = 1  # b12
        V[b_off + b_p + 1, col] = -1  # b22
        W[c_off + 1, col] = 1  # c12
        W[c_off + c_n + 1, col] = 1  # c22
        col += 1
        
        # Term 4: M4 = a22(b21-b11)
        # U4: a22=1
        # V4: b21=1, b11=-1
        # W4: c11=1, c21=1
        U[a_off + a_m + 1, col] = 1  # a22
        V[b_off + b_p + 0, col] = 1  # b21
        V[b_off + 0, col] = -1  # b11
        W[c_off + 0, col] = 1  # c11
        W[c_off + c_n + 0, col] = 1  # c21
        col += 1
        
        # Term 5: M5 = (a11+a12)b22
        # U5: a11=1, a12=1
        # V5: b22=1
        # W5: c11=-1, c12=1
        U[a_off + 0, col] = 1  # a11
        U[a_off + 1, col] = 1  # a12
        V[b_off + b_p + 1, col] = 1  # b22
        W[c_off + 0, col] = -1  # c11
        W[c_off + 1, col] = 1  # c12
        col += 1
        
        # Term 6: M6 = (a21-a11)(b11+b12)
        # U6: a21=1, a11=-1
        # V6: b11=1, b12=1
        # W6: c22=1
        U[a_off + a_m + 0, col] = 1  # a21
        U[a_off + 0, col] = -1  # a11
        V[b_off + 0, col] = 1  # b11
        V[b_off + 1, col] = 1  # b12
        W[c_off + c_n + 1, col] = 1  # c22
        col += 1
        
        # Term 7: M7 = (a12-a22)(b21+b22)
        # U7: a12=1, a22=-1
        # V7: b21=1, b22=1
        # W7: c11=1
        U[a_off + 1, col] = 1  # a12
        U[a_off + a_m + 1, col] = -1  # a22
        V[b_off + b_p + 0, col] = 1  # b21
        V[b_off + b_p + 1, col] = 1  # b22
        W[c_off + 0, col] = 1  # c11
        col += 1
        
    # Schoolbook for M_{2,2,3} (A 2x2, B 2x3, C 2x3)
    # Rank 6 (schoolbook)
    # But we want rank 11 using Strassen for first 2 cols + schoolbook for last 1 col.
    # Actually, let's just use schoolbook for simplicity first, then optimize.
    # Schoolbook for M_{n,m,p} has rank n*m*p.
    
    def add_schoolbook(a_off, b_off, c_off, a_n, a_m, b_m, b_p, c_n, c_p):
        # A is a_n x a_m
        # B is b_m x b_p (b_m should equal a_m)
        # C is c_n x c_p (c_n should equal a_n, c_p should equal b_p)
        for i in range(a_n):
            for j in range(a_m):
                for k in range(b_p):
                    U[a_off + i * a_m + j, col] = 1
                    V[b_off + j * b_p + k, col] = 1
                    W[c_off + k * c_n + i, col] = 1
                    col += 1
    
    # Now, decompose M_{2,5,5}
    # A (2x5), B (5x5), C (2x5)
    # Split A into A1 (2x2), A2 (2x3)
    # Split B into B1 (2x5), B2 (3x5)
    # C = A1*B1 + A2*B2
    # A1*B1 is M_{2,2,5}
    # A2*B2 is M_{2,3,5}
    
    # A1*B1: A1 (2x2), B1 (2x5), C1 (2x5)
    # Split B1 into B1a (2x2), B1b (2x3)
    # C1 = A1*B1a + A1*B1b
    # A1*B1a is M_{2,2,2} -> Strassen (7)
    # A1*B1b is M_{2,2,3} -> Schoolbook (12) or Strassen+Schoolbook (7+4=11)
    # Let's use Strassen for M_{2,2,2} and Schoolbook for M_{2,2,3}
    
    # A2*B2: A2 (2x3), B2 (3x5), C2 (2x5)
    # Split A2 into A2a (2x2), A2b (2x1)
    # Split B2 into B2a (2x5), B2b (1x5)
    # C2 = A2a*B2a + A2b*B2b
    # A2a*B2a is M_{2,2,5}
    # A2b*B2b is M_{2,1,5}
    
    # This is getting complex. Let's just use schoolbook for everything first to ensure correctness, then optimize.
    # Schoolbook rank: 2*5*5 = 50.
    
    # Let's implement schoolbook first.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    if time.time() > deadline:
        pass
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

