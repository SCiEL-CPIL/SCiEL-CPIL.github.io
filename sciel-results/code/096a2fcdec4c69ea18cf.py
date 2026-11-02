import time
import numpy as np
import itertools

def construct():
    start_time = time.time()
    deadline = 145  # safety margin before 150s limit

    n, m, p = 2, 2, 13
    N, M, P = n * m, m * p, p * n  # 4, 26, 26

    def build_strassen_2x2_2x13():
        """
        Strassen-based decomposition for 2x2 * 2x13.
        
        The 2x2 * 2x2 block uses Strassen (7 mults).
        Each 2x2 * 2x13 multiplication is split into 7 Strassen products,
        but since the right side is 2x13, we need to be more careful.
        
        Actually, let's think of it differently:
        A is 2x2, B is 2x13.
        
        Standard Strassen for 2x2 * 2x2:
        M1 = (A11+A22)*(B11+B22)
        M2 = (A21+A22)*B11
        M3 = A11*(B12-B22)
        M4 = A22*(B21-B11)
        M5 = (A11+A12)*B22
        M6 = (A21-A11)*(B11+B12)
        M7 = (A12-A22)*(B21+B22)
        
        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6
        
        For 2x2 * 2x13, we can view B as 7 blocks of 2x2 (since 13 = 7*2 - 1, not quite).
        Actually 13 = 6*2 + 1. So we have 6 full 2x2 blocks and one 2x1 block.
        
        Better approach: treat the 2x13 as 13 columns. We can group them.
        Actually, the cleanest approach: use the fact that matrix multiplication is linear.
        
        Let me use a direct Strassen-like approach for the 2x2 * 2x13 case.
        
        A = [[a, b], [c, d]] (2x2)
        B = 2x13 matrix with rows B[0,:] and B[1,:]
        
        C[0,:] = a*B[0,:] + b*B[1,:]
        C[1,:] = c*B[0,:] + d*B[1,:]
        
        Using Strassen's idea with 7 products:
        M1 = (a+d) * (B[0,:] + B[1,:])
        M2 = (c+d) * B[0,:]
        M3 = a * (B[1,:] - B[0,:])  -- wait, this isn't right for rectangular
        
        Actually for 2x2 * 2xp, we can still use 7 scalar multiplications per "column block"
        but since p=13 is odd, let's just use the standard 8 (schoolbook) for the full thing
        but see if we can do better.
        
        Alternative: Since n=m=2, the 2x2 part can use Strassen (7 instead of 8).
        For each of the 13 columns, we do a 2x2 * 2x1 multiplication, which is just 4 mults.
        Total: 13 * 4 = 52 (schoolbook).
        
        But if we use Strassen for the 2x2 * 2x2 blocks and handle the extra column:
        13 = 6*2 + 1
        6 blocks of 2x2: each uses 7 Strassen mults = 42
        1 block of 2x1: uses 4 mults = 4
        Total: 46
        
        Let me implement this.
        """
        U = np.zeros((N, 46))
        V = np.zeros((M, 46))
        W = np.zeros((P, 46))
        
        col = 0
        # Process 6 blocks of 2 columns each, then 1 block of 1 column
        # B columns 0,1 -> block 0; 2,3 -> block 1; ...; 10,11 -> block 5; 12 -> block 6
        
        # Strassen for 2x2 * 2x2:
        # A = [[a, b], [c, d]]
        # B = [[e, f], [g, h]]
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
        
        for blk in range(6):
            c0 = blk * 2
            c1 = blk * 2 + 1
            
            # M1 = (a+d)(e+h)
            U[0, col] = 1; U[3, col] = 1  # a is index 0, d is index 3
            V[0*p+c0, col] = 1; V[0*p+c1, col] = 1  # e is B[0,c0], h is B[1,c1]
            W[0*n+0, col] = 1; W[1*n+0, col] = 1; W[0*n+1, col] = 1; W[1*n+1, col] = 1
            # C11 += M1, C22 += M1
            col += 1
            
            # M2 = (c+d)e
            U[2, col] = 1; U[3, col] = 1  # c is index 2, d is index 3
            V[0*p+c0, col] = 1  # e is B[0,c0]
            W[1*n+0, col] = 1  # C21 += M2
            col += 1
            
            # M3 = a(f-h)
            U[0, col] = 1  # a is index 0
            V[0*p+c1, col] = 1; V[1*p+c0, col] = -1  # f is B[0,c1], h is B[1,c0]
            W[0*n+1, col] = 1; W[1*n+1, col] = 1  # C12 += M3, C22 += M3
            col += 1
            
            # M4 = d(g-e)
            U[3, col] = 1  # d is index 3
            V[1*p+c0, col] = 1; V[0*p+c0, col] = -1  # g is B[1,c0], e is B[0,c0]
            W[0*n+0, col] = 1; W[1*n+0, col] = 1  # C11 += M4, C21 += M4
            col += 1
            
            # M5 = (a+b)h
            U[0, col] = 1; U[1, col] = 1  # a is index 0, b is index 1
            V[1*p+c1, col] = 1  # h is B[1,c1]
            W[0*n+0, col] = -1; W[0*n+1, col] = 1  # C11 -= M5, C12 += M5
            col += 1
            
            # M6 = (c-a)(e+f)
            U[2, col] = 1; U[0, col] = -1  # c is index 2, a is index 0
            V[0*p+c0, col] = 1; V[0*p+c1, col] = 1  # e is B[0,c0], f is B[0,c1]
            W[1*n+1, col] = 1  # C22 += M6
            col += 1
            
            # M7 = (b-d)(g+h)
            U[1, col] = 1; U[3, col] = -1  # b is index 1, d is index 3
            V[1*p+c0, col] = 1; V[1*p+c1, col] = 1  # g is B[1,c0], h is B[1,c1]
            W[0*n+0, col] = 1  # C11 += M7
            col += 1
        
        # Last column (column 12) - schoolbook 4 mults
        c = 12
        # C[0,12] = a*B[0,12] + b*B[1,12]
        # C[1,12] = c*B[0,12] + d*B[1,12]
        
        # M1 = a * B[0,12]
        U[0, col] = 1
        V[0*p+c, col] = 1
        W[0*n+1, col] = 1  # Wait, W index: output is 2x13, so W[k*n+i] where k is col, i is row
        # Actually W index: k'*n + i' where k' is the output column, i' is the output row
        # C[i', k'] = sum over t of U[i*m+j, t] * V[j'*p+k, t] * W[k'*n+i', t]
        # For output C[0, 12]: i'=0, k'=12, so W index = 12*2 + 0 = 24
        # For output C[1, 12]: i'=1, k'=12, so W index = 12*2 + 1 = 25
        W[12*n+0, col] = 1
        col += 1
        
        # M2 = b * B[1,12]
        U[1, col] = 1
        V[1*p+c, col] = 1
        W[12*n+0, col] = 1
        col += 1
        
        # M3 = c * B[0,12]
        U[2, col] = 1
        V[0*p+c, col] = 1
        W[12*n+1, col] = 1
        col += 1
        
        # M4 = d * B[1,12]
        U[3, col] = 1
        V[1*p+c, col] = 1
        W[12*n+1, col] = 1
        col += 1
        
        return U, V, W

    def verify_decomposition(U, V, W, tol=1e-10):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            for ip in range(n):
                                expected = 1.0 if (j == jp and i == ip and k == kp) else 0.0
                                actual = float(np.dot(U[i*m+j, :], V[jp*p+k, :] * W[kp*n+ip, :]))
                                err = abs(actual - expected)
                                if err > max_err:
                                    max_err = err
        return max_err < tol, max_err

    # Build the Strassen-based decomposition
    U, V, W = build_strassen_2x2_2x13()
    
    # Verify
    valid, max_err = verify_decomposition(U, V, W)
    
    if valid:
        # Convert to lists for return
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U_fb = np.zeros((N, rank))
        V_fb = np.zeros((M, rank))
        W_fb = np.zeros((P, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
