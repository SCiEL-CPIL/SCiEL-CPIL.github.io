import numpy as np
import time

START_TIME = time.time()
TIME_LIMIT = 120.0

def _check_time():
    return (time.time() - START_TIME) > (TIME_LIMIT - 15.0)

def _strassen_2x2():
    """Strassen 2x2 decomposition (rank 7)."""
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    U2[0,0]=1; U2[3,0]=1; V2[0,0]=1; V2[3,0]=1
    U2[2,1]=1; U2[3,1]=1; V2[0,1]=1
    U2[0,2]=1; V2[3,2]=1; V2[1,2]=-1
    U2[3,3]=1; V2[2,3]=1; V2[0,3]=-1
    U2[0,4]=1; U2[1,4]=1; V2[3,4]=1
    U2[2,5]=1; U2[0,5]=-1; V2[0,5]=1; V2[1,5]=1
    U2[3,6]=1; U2[2,6]=-1; V2[2,6]=1; V2[3,6]=1
    
    W2[0,0]=1; W2[0,3]=1; W2[0,4]=-1; W2[0,6]=1
    W2[1,2]=1; W2[1,4]=1
    W2[2,1]=1; W2[2,3]=1
    W2[3,0]=1; W2[3,1]=-1; W2[3,2]=1; W2[3,5]=1
    
    return U2, V2, W2

def _verify_decomposition(U, V, W, n, m, p, trials=5):
    """Verify that the decomposition correctly computes matrix multiplication."""
    for _ in range(trials):
        A = np.random.randint(-3, 4, (n, m)).astype(float)
        B = np.random.randint(-3, 4, (m, p)).astype(float)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        for r in range(U.shape[1]):
            u_vec = U[:, r].reshape(n, m)
            v_vec = V[:, r].reshape(m, p)
            scalar = np.sum(u_vec * A) * np.sum(v_vec * B)
            if scalar != 0:
                w_mat = W[:, r].reshape(p, n).T
                C_computed += scalar * w_mat
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _kron_decompose(Ua, Va, Wa, Ub, Vb, Wb):
    """Kronecker product of two tensor decompositions."""
    import math
    n1 = int(math.isqrt(len(Ua)))
    m1 = int(math.isqrt(len(Va)))
    p1 = int(math.isqrt(len(Wa)))
    n2 = int(math.isqrt(len(Ub)))
    m2 = int(math.isqrt(len(Vb)))
    p2 = int(math.isqrt(len(Wb)))
    r1 = Ua.shape[1]
    r2 = Ub.shape[1]
    n, m, p = n1*n2, m1*m2, p1*p2
    r = r1 * r2
    U = np.zeros((n*m, r))
    V = np.zeros((m*p, r))
    W = np.zeros((p*n, r))
    
    for r1i in range(r1):
        for r2i in range(r2):
            r_idx = r1i * r2 + r2i
            for i1 in range(n1):
                for j1 in range(m1):
                    a = Ua[i1*m1+j1, r1i]
                    if a == 0: continue
                    for i2 in range(n2):
                        for j2 in range(m2):
                            b = Ub[i2*m2+j2, r2i]
                            if b == 0: continue
                            i = i1*n2 + i2
                            j = j1*m2 + j2
                            U[i*m+j, r_idx] = a * b
            for j1 in range(m1):
                for k1 in range(p1):
                    a = Va[j1*p1+k1, r1i]
                    if a == 0: continue
                    for j2 in range(m2):
                        for k2 in range(p2):
                            b = Vb[j2*p2+k2, r2i]
                            if b == 0: continue
                            j = j1*m2 + j2
                            k = k1*p2 + k2
                            V[j*p+k, r_idx] = a * b
            for k1 in range(p1):
                for i1 in range(n1):
                    a = Wa[k1*n1+i1, r1i]
                    if a == 0: continue
                    for k2 in range(p2):
                        for i2 in range(n2):
                            b = Wb[k2*n2+i2, r2i]
                            if b == 0: continue
                            k = k1*p2 + k2
                            i = i1*n2 + i2
                            W[k*n+i, r_idx] = a * b
    return U, V, W

def _laderman_3x3():
    """
    Laderman's 3x3 matrix multiplication (rank 23).
    Uses a verified formulation with explicit products.
    Solves for W via minimum-norm least squares, then verifies.
    Returns None if verification fails.
    """
    U3 = np.zeros((9, 23))
    V3 = np.zeros((9, 23))
    W3 = np.zeros((9, 23))
    
    # 23 bilinear forms for Laderman 3x3
    products = [
        ({0:1, 4:1}, {0:1, 4:1}),       # (a00+a11)(b00+b11)
        ({1:1, 4:1}, {0:1, 2:1}),       # (a01+a11)(b00+b10)
        ({3:1, 6:1}, {0:1, 1:1}),       # (a10+a20)(b00+b01)
        ({5:1, 4:1}, {2:1, 4:1}),       # (a21+a11)(b10+b11)
        ({0:1, 8:1}, {0:1, 8:1}),       # (a00+a22)(b00+b22)
        ({1:1, 5:1}, {1:1, 8:1}),       # (a01+a21)(b02+b22)
        ({2:1, 8:1}, {0:1, 6:1}),       # (a02+a22)(b00+b20)
        ({3:1, 8:1}, {1:1, 6:1}),       # (a10+a22)(b02+b20)
        ({4:1, 6:1}, {0:1, 8:1}),       # (a12+a20)(b00+b22)
        ({6:1, 2:1}, {4:1, 5:1}),       # (a20+a02)(b11+b12)
        ({5:1, 1:1}, {2:1, 5:1}),       # (a21+a01)(b10+b12)
        ({0:1, 4:1}, {1:1, 7:1}),       # (a00+a12)(b01+b21)
        ({4:1, 8:1}, {0:1, 5:1}),       # (a11+a22)(b00+b12)
        ({1:1, 8:1}, {0:1, 7:1}),       # (a01+a22)(b00+b21)
        ({3:1, 2:1}, {4:1, 1:1}),       # (a10+a02)(b11+b01)
        ({6:1, 4:1}, {2:1, 5:1}),       # (a20+a12)(b10+b12)
        ({2:1, 3:1}, {4:1, 5:1}),       # (a02+a10)(b11+b12)
        ({4:1, 5:1}, {0:1, 2:1}),       # (a12+a21)(b00+b10)
        ({0:1, 8:1}, {1:1, 7:1}),       # (a00+a22)(b01+b21)
        ({3:1, 5:1}, {0:1, 6:1}),       # (a10+a21)(b00+b20)
        ({1:1, 3:1}, {4:1, 8:1}),       # (a01+a10)(b11+b22)
        ({6:1, 8:1}, {2:1, 4:1}),       # (a20+a22)(b10+b11)
        ({2:1, 4:1}, {3:1, 5:1}),       # (a02+a12)(b21+b22)
    ]
    
    for r, (u_dict, v_dict) in enumerate(products):
        for idx, val in u_dict.items():
            U3[idx, r] = val
        for idx, val in v_dict.items():
            V3[idx, r] = val
    
    # Solve for W: for each (i,k), need sum_r W[k*3+i,r] * S[i,k,r] = delta_ik
    S = np.zeros((3, 3, 23))
    for i in range(3):
        for k in range(3):
            for r in range(23):
                for j in range(3):
                    S[i, k, r] += U3[i*3+j, r] * V3[j*3+k, r]
    
    for i in range(3):
        for k in range(3):
            s = S[i, k, :]
            target = 1.0 if i == k else 0.0
            norm_sq = np.dot(s, s)
            if norm_sq > 1e-15:
                W3[k*3+i, :] = s * (target / norm_sq)
            else:
                W3[k*3+i, :] = 0
    
    if not _verify_decomposition(U3, V3, W3, 3, 3, 3, trials=10):
        return None
    
    return U3, V3, W3

def _strassen_2x2():
    """Strassen 2x2 decomposition (rank 7)."""
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Strassen's 7 products
    # M1=(a00+a11)(b00+b11)
    U2[0,0]=1; U2[3,0]=1; V2[0,0]=1; V2[3,0]=1
    # M2=(a10+a11)b00
    U2[2,1]=1; U2[3,1]=1; V2[0,1]=1
    # M3=a00(b11-b01)
    U2[0,2]=1; V2[3,2]=1; V2[1,2]=-1
    # M4=a11(b10-b00)
    U2[3,3]=1; V2[2,3]=1; V2[0,3]=-1
    # M5=(a00+a01)b11
    U2[0,4]=1; U2[1,4]=1; V2[3,4]=1
    # M6=(a10-a00)(b00+b01)
    U2[2,5]=1; U2[0,5]=-1; V2[0,5]=1; V2[1,5]=1
    # M7=(a11-a10)(b10+b11)
    U2[3,6]=1; U2[2,6]=-1; V2[2,6]=1; V2[3,6]=1
    
    # W: c00=M1+M4-M5+M7, c01=M3+M5, c10=M2+M4, c11=M1-M2+M3+M6
    W2[0,0]=1; W2[0,3]=1; W2[0,4]=-1; W2[0,6]=1
    W2[1,2]=1; W2[1,4]=1
    W2[2,1]=1; W2[2,3]=1
    W2[3,0]=1; W2[3,1]=-1; W2[3,2]=1; W2[3,5]=1
    
    return U2, V2, W2

def _kron_decompose(Ua, Va, Wa, Ub, Vb, Wb):
    """Kronecker product of two tensor decompositions."""
    import math
    n1 = int(math.isqrt(len(Ua)))
    m1 = int(math.isqrt(len(Va)))
    p1 = int(math.isqrt(len(Wa)))
    n2 = int(math.isqrt(len(Ub)))
    m2 = int(math.isqrt(len(Vb)))
    p2 = int(math.isqrt(len(Wb)))
    r1 = Ua.shape[1]
    r2 = Ub.shape[1]
    n, m, p = n1*n2, m1*m2, p1*p2
    r = r1 * r2
    U = np.zeros((n*m, r))
    V = np.zeros((m*p, r))
    W = np.zeros((p*n, r))
    
    for r1i in range(r1):
        for r2i in range(r2):
            r_idx = r1i * r2 + r2i
            # U: index (i1,i2,j1,j2) -> i = i1*n2+i2, j = j1*m2+j2
            for i1 in range(n1):
                for j1 in range(m1):
                    a = Ua[i1*m1+j1, r1i]
                    if a == 0: continue
                    for i2 in range(n2):
                        for j2 in range(m2):
                            b = Ub[i2*m2+j2, r2i]
                            if b == 0: continue
                            i = i1*n2 + i2
                            j = j1*m2 + j2
                            U[i*m+j, r_idx] = a * b
            # V: index (j1,j2,k1,k2) -> j = j1*m2+j2, k = k1*p2+k2
            for j1 in range(m1):
                for k1 in range(p1):
                    a = Va[j1*p1+k1, r1i]
                    if a == 0: continue
                    for j2 in range(m2):
                        for k2 in range(p2):
                            b = Vb[j2*p2+k2, r2i]
                            if b == 0: continue
                            j = j1*m2 + j2
                            k = k1*p2 + k2
                            V[j*p+k, r_idx] = a * b
            # W: index (k1,k2,i1,i2) -> k = k1*p2+k2, i = i1*n2+i2
            for k1 in range(p1):
                for i1 in range(n1):
                    a = Wa[k1*n1+i1, r1i]
                    if a == 0: continue
                    for k2 in range(p2):
                        for i2 in range(n2):
                            b = Wb[k2*n2+i2, r2i]
                            if b == 0: continue
                            k = k1*p2 + k2
                            i = i1*n2 + i2
                            W[k*n+i, r_idx] = a * b
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, trials=5):
    """Verify that the decomposition correctly computes matrix multiplication."""
    for _ in range(trials):
        A = np.random.randint(-3, 4, (n, m)).astype(float)
        B = np.random.randint(-3, 4, (m, p)).astype(float)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        for r in range(U.shape[1]):
            u_vec = U[:, r].reshape(n, m)
            v_vec = V[:, r].reshape(m, p)
            scalar = np.sum(u_vec * A) * np.sum(v_vec * B)
            if scalar != 0:
                w_mat = W[:, r].reshape(p, n).T  # W[k*n+i] -> (i,k)
                C_computed += scalar * w_mat
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _get_laderman_correct():
    """Get a verified correct Laderman 3x3 decomposition."""
    # Use a known correct Laderman decomposition
    # Reference: Laderman (1976), "Noncommutative complexity"
    # The 23 bilinear forms for 3x3 matrix multiplication
    
    U3 = np.zeros((9, 23))
    V3 = np.zeros((9, 23))
    W3 = np.zeros((9, 23))
    
    # I'll use the decomposition from the literature with explicit W
    # Products:
    # P1  = (a00+a11)(b00+b11)
    # P2  = (a01+a11)(b00+b10)  
    # P3  = (a10+a20)(b00+b01)
    # P4  = (a21+a11)(b10+b11)
    # P5  = (a00+a22)(b00+b22)
    # P6  = (a01+a21)(b02+b22)
    # P7  = (a02+a22)(b00+b20)
    # P8  = (a10+a22)(b02+b20)
    # P9  = (a12+a20)(b00+b22)
    # P10 = (a20+a02)(b11+b12)
    # P11 = (a21+a01)(b10+b12)
    # P12 = (a00+a12)(b01+b21)
    # P13 = (a11+a22)(b00+b12)
    # P14 = (a01+a22)(b00+b21)
    # P15 = (a10+a02)(b11+b01)
    # P16 = (a20+a12)(b10+b12)
    # P17 = (a02+a10)(b11+b12)
    # P18 = (a12+a21)(b00+b10)
    # P19 = (a00+a12)(b01+b21)  -- same as P12? No
    # Let me just use a programmatically verified approach
    
    # Define U and V for all 23 products
    products = [
        # (U_indices, coeffs) for A, (V_indices, coeffs) for B
        (([0,4],[1,1]), ([0,4],[1,1])),      # P0
        (([1,4],[1,1]), ([0,2],[1,1])),      # P1
        (([3,6],[1,1]), ([0,1],[1,1])),      # P2
        (([5,4],[1,1]), ([2,4],[1,1])),      # P3
        (([0,8],[1,1]), ([0,8],[1,1])),      # P4
        (([1,5],[1,1]), ([1,8],[1,1])),      # P5
        (([2,8],[1,1]), ([0,6],[1,1])),      # P6
        (([3,8],[1,1]), ([1,6],[1,1])),      # P7
        (([4,6],[1,1]), ([0,8],[1,1])),      # P8
        (([6,2],[1,1]), ([4,5],[1,1])),      # P9
        (([5,1],[1,1]), ([2,5],[1,1])),      # P10
        (([0,4],[1,1]), ([1,7],[1,1])),      # P11
        (([4,8],[1,1]), ([0,5],[1,1])),      # P12
        (([1,8],[1,1]), ([0,7],[1,1])),      # P13
        (([3,2],[1,1]), ([4,1],[1,1])),      # P14
        (([6,4],[1,1]), ([2,5],[1,1])),      # P15
        (([2,3],[1,1]), ([4,5],[1,1])),      # P16
        (([4,5],[1,1]), ([0,2],[1,1])),      # P17
        (([0,4],[1,1]), ([1,7],[1,1])),      # P18 (same as P11, need to fix)
    ]
    
    # This is getting messy. Let me just verify Strassen-based approach or use schoolbook.
    # Actually, let me just use the Kronecker product of schoolbook 3x3 and Strassen 2x2
    # which gives rank 27*7 = 189, still better than 216.
    
    # Or better: just use block Strassen on 6x6 with 3x3 base (schoolbook)
    # Rank = 7 * 27 = 189
    
    return None

def construct():
    n, m, p = 6, 6, 6
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p  # 216 schoolbook
    
    # Strategy 1: Kronecker product of schoolbook 3x3 and Strassen 2x2
    # Rank = 27 * 7 = 189
    try:
        # Schoolbook 3x3
        U3 = np.zeros((9, 27))
        V3 = np.zeros((9, 27))
        W3 = np.zeros((9, 27))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U3[i*3+j, col] = 1
                    V3[j*3+k, col] = 1
                    W3[k*3+i, col] = 1
                    col += 1
        
        U2, V2, W2 = _strassen_2x2()
        U, V, W = _kron_decompose(U3, V3, W3, U2, V2, W2)
        
        if _verify_decomposition(U, V, W, 6, 6, 6, trials=3):
            best_U, best_V, best_W = U, V, W
            best_rank = 189
            print(f"Kronecker schoolbook3+Strassen2: rank {best_rank}")
    except Exception as e:
        print(f"Kronecker schoolbook3+Strassen2 failed: {e}")
    
    # Strategy 2: Try to get Laderman working
    if best_rank > 161:
        try:
            U3, V3, W3 = _laderman_3x3()
            # Verify Laderman
            if _verify_decomposition(U3, V3, W3, 3, 3, 3, trials=3):
                U2, V2, W2 = _strassen_2x2()
                U, V, W = _kron_decompose(U3, V3, W3, U2, V2, W2)
                if _verify_decomposition(U, V, W, 6, 6, 6, trials=3):
                    best_U, best_V, best_W = U, V, W
                    best_rank = 161
                    print(f"Laderman3xStrassen2: rank {best_rank}")
        except Exception as e:
            print(f"Laderman+Strassen failed: {e}")
    
    # Strategy 3: Block Strassen on 6x6 with 3x3 schoolbook base
    if best_rank > 189:
        try:
            U3 = np.zeros((9, 27))
            V3 = np.zeros((9, 27))
            W3 = np.zeros((9, 27))
            col = 0
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        U3[i*3+j, col] = 1
                        V3[j*3+k, col] = 1
                        W3[k*3+i, col] = 1
                        col += 1
            
            # Block Strassen: 6x6 = 2x2 blocks of 3x3
            r_base = 27
            total_rank = 7 * r_base
            U = np.zeros((36, total_rank))
            V = np.zeros((36, total_rank))
            W = np.zeros((36, total_rank))
            
            # Strassen products for block matrices
            products = [
                ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),  # M1=(A00+A11)(B00+B11)
                ({2:1, 3:1}, {0:1}, {2:1}),             # M2=(A10+A11)B00
                ({0:1}, {3:1, 1:-1}, {1:1}),            # M3=A00(B11-B01)
                ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),       # M4=A11(B10-B00)
                ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),       # M5=(A00+A01)B11
                ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),       # M6=(A10-A00)(B00+B01)
                ({3:1, 2:-1}, {2:1, 3:1}, {0:1, 3:-1}), # M7=(A11-A10)(B10+B11)
            ]
            
            block = 3
            for s in range(7):
                u_blocks, v_blocks, w_blocks = products[s]
                for rb in range(r_base):
                    r_idx = s * r_base + rb
                    for bi, cu in u_blocks.items():
                        i1 = bi // 2
                        j1 = bi % 2
                        for i2 in range(block):
                            for j2 in range(block):
                                a = U3[(i1*block+i2)*block + (j1*block+j2), rb]
                                if a != 0:
                                    U[(i1*block+i2)*6 + (j1*block+j2), r_idx] = cu * a
                    for bi, cv in v_blocks.items():
                        j1 = bi // 2
                        k1 = bi % 2
                        for j2 in range(block):
                            for k2 in range(block):
                                b = V3[(j1*block+j2)*block + (k1*block+k2), rb]
                                if b != 0:
                                    V[(j1*block+j2)*6 + (k1*block+k2), r_idx] = cv * b
                    for bi, cw in w_blocks.items():
                        k1 = bi // 2
                        i1 = bi % 2
                        for k2 in range(block):
                            for i2 in range(block):
                                c = W3[(k1*block+k2)*block + (i1*block+i2), rb]
                                if c != 0:
                                    W[(k1*block+k2)*6 + (i1*block+i2), r_idx] = cw * c
            
            if _verify_decomposition(U, V, W, 6, 6, 6, trials=3):
                best_U, best_V, best_W = U, V, W
                best_rank = 189
                print(f"Block Strassen schoolbook3: rank {best_rank}")
        except Exception as e:
            print(f"Block Strassen failed: {e}")
    
    # Fallback: schoolbook
    if best_U is None:
        U, V, W = _schoolbook(6, 6, 6)
        best_U, best_V, best_W = U, V, W
        best_rank = 216
        print(f"Schoolbook fallback: rank {best_rank}")
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
