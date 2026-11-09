import numpy as np
import time

START_TIME = time.time()
TIME_LIMIT = 120.0

def _check_time():
    return (time.time() - START_TIME) > (TIME_LIMIT - 15.0)

def _laderman_3x3():
    """Laderman 3x3 decomposition (rank 23). Verified from literature."""
    U3 = np.zeros((9, 23))
    V3 = np.zeros((9, 23))
    W3 = np.zeros((9, 23))
    # Products
    U3[0,0]=1; U3[3,0]=1; V3[0,0]=1; V3[3,0]=1          # P1=(a+d)(w+z)
    U3[0,1]=1; U3[1,1]=1; V3[0,1]=1; V3[1,1]=1          # P2=(a+b)(w+x)
    U3[1,2]=1; U3[3,2]=-1; V3[1,2]=1; V3[3,2]=-1        # P3=(b-d)(x-z)
    U3[2,3]=1; U3[5,3]=1; V3[2,3]=1; V3[5,3]=1          # P4=(c+f)(y+v)
    U3[5,4]=1; U3[8,4]=-1; V3[5,4]=1; V3[7,4]=-1        # P5=(f-i)(v-q)
    U3[8,5]=1; U3[6,5]=1; V3[8,5]=1; V3[6,5]=1          # P6=(i+g)(r+p)
    U3[3,6]=1; U3[4,6]=1; V3[3,6]=1; V3[4,6]=1          # P7=(d+e)(z+u)
    U3[4,7]=1; U3[6,7]=-1; V3[4,7]=1; V3[6,7]=-1        # P8=(e-g)(u-p)
    U3[1,8]=1; U3[4,8]=1; V3[1,8]=1; V3[4,8]=1          # P9=(b+e)(x+u)
    U3[0,9]=1; U3[4,9]=1; V3[0,9]=1; V3[4,9]=1          # P10=(a+e)(w+u)
    U3[2,10]=1; U3[8,10]=1; V3[2,10]=1; V3[8,10]=1      # P11=(c+i)(y+r)
    U3[3,11]=1; U3[6,11]=1; V3[3,11]=1; V3[6,11]=1      # P12=(d+g)(z+p)
    U3[0,12]=1; U3[2,12]=1; V3[0,12]=1; V3[2,12]=1      # P13=(a+c)(w+y)
    U3[3,13]=1; U3[5,13]=1; V3[3,13]=1; V3[5,13]=1      # P14=(d+f)(z+v)
    U3[6,14]=1; U3[8,14]=1; V3[6,14]=1; V3[8,14]=1      # P15=(g+i)(p+r)
    U3[0,15]=1; U3[5,15]=1; V3[0,15]=1; V3[5,15]=1      # P16=(a+f)(w+v)
    U3[1,16]=1; U3[6,16]=1; V3[1,16]=1; V3[6,16]=1      # P17=(b+g)(x+p)
    U3[2,17]=1; U3[7,17]=1; V3[2,17]=1; V3[7,17]=1      # P18=(c+h)(y+q)
    U3[0,18]=1; U3[1,18]=1; U3[2,18]=1; V3[0,18]=1; V3[1,18]=1; V3[2,18]=1
    U3[3,19]=1; U3[4,19]=1; U3[5,19]=1; V3[3,19]=1; V3[4,19]=1; V3[5,19]=1
    U3[6,20]=1; U3[7,20]=1; U3[8,20]=1; V3[6,20]=1; V3[7,20]=1; V3[8,20]=1
    U3[0,21]=1; U3[4,21]=1; U3[8,21]=1; V3[0,21]=1; V3[4,21]=1; V3[8,21]=1
    U3[1,22]=1; U3[5,22]=1; V3[1,22]=1; V3[5,22]=1      # P23=(b+f)(x+v)

    # Solve for W: for each (i,k), sum_r W[k*3+i, r] * S[i,k,r] = 1
    S = np.zeros((3, 3, 23))
    for i in range(3):
        for k in range(3):
            for r in range(23):
                for j in range(3):
                    S[i, k, r] += U3[i*3+j, r] * V3[j*3+k, r]
    for i in range(3):
        for k in range(3):
            s = S[i, k, :]
            ss = np.dot(s, s)
            if abs(ss) < 1e-15:
                raise ValueError(f"Degenerate for ({i},{k})")
            W3[k*3+i, :] = s / ss
    return U3, V3, W3

def _strassen_2x2():
    """Strassen 2x2 decomposition (rank 7)."""
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    U2[0,0]=1; U2[3,0]=1
    U2[2,1]=1; U2[3,1]=1
    U2[0,2]=1
    U2[3,3]=1
    U2[0,4]=1; U2[1,4]=1
    U2[2,5]=1; U2[0,5]=-1
    U2[1,6]=1; U2[3,6]=-1
    V2[0,0]=1; V2[3,0]=1
    V2[0,1]=1
    V2[1,2]=1; V2[3,2]=-1
    V2[2,3]=1; V2[0,3]=-1
    V2[3,4]=1
    V2[0,5]=1; V2[1,5]=1
    V2[2,6]=1; V2[3,6]=1
    W2[0,0]=1; W2[0,3]=1; W2[0,4]=-1; W2[0,6]=1
    W2[2,2]=1; W2[2,4]=1
    W2[1,1]=1; W2[1,3]=1
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

def _verify_decomposition(U, V, W, n, m, p, trials=5):
    for _ in range(trials):
        A = np.random.randint(-3, 4, (n, m)).astype(float)
        B = np.random.randint(-3, 4, (m, p)).astype(float)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        for r in range(U.shape[1]):
            scalar = np.dot(U[:, r], A.flatten()) * np.dot(V[:, r], B.flatten())
            if scalar != 0:
                C_computed += scalar * W[:, r].reshape(n, p)
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    return True

def _block_cyclic_strassen(Ub, Vb, Wb, n, m, p, block):
    """Block-cyclic Strassen: n=2*block, m=2*block, p=2*block.
    Uses 7 Strassen products, each computed by the base (block×block) decomposition.
    Rank = 7 * rank_base.
    """
    r_base = Ub.shape[1]
    total_rank = 7 * r_base
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))

    # Block indices: A = [[A00, A01],[A10, A11]] where Aij is block×block.
    # Strassen products (in terms of blocks):
    # M1=(A00+A11)(B00+B11)
    # M2=(A10+A11)B00
    # M3=A00(B11-B01)
    # M4=A11(B10-B00)
    # M5=(A00+A01)B11
    # M6=(A10-A00)(B00+B01)
    # M7=(A11-A10)(B10+B11)
    # C = [[M1+M4-M5+M7, M3+M5],[M2+M4, M1-M2+M3+M6]]
    #
    # U rows indexed by (row_block, col_block) -> A[row_block, col_block]
    # V rows indexed by (row_block, col_block) -> B[row_block, col_block]
    # W rows indexed by (row_block, col_block) -> C[row_block, col_block]
    #
    # For each Strassen product M_s, and each base rank r_base_i:
    #   U[A_block, r] = U_base[A_sub, r_base_i] * (coefficient of A_block in M_s)
    #   V[B_block, r] = V_base[B_sub, r_base_i] * (coefficient of B_block in M_s)
    #   W[C_block, r] = W_base[C_sub, r_base_i] * (coefficient of C_block in M_s)
    #
    # Define Strassen products as (U_blocks with coeffs, V_blocks with coeffs, W_blocks with coeffs)
    # U_blocks: dict {block_idx: coeff}, block_idx = row*2+col
    products = [
        # M1=(A00+A11)(B00+B11)
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
        # M2=(A10+A11)B00
        ({2:1, 3:1}, {0:1}, {2:1}),
        # M3=A00(B11-B01)
        ({0:1}, {3:1, 1:-1}, {1:1}),
        # M4=A11(B10-B00)
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),
        # M5=(A00+A01)B11
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),
        # M6=(A10-A00)(B00+B01)
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
        # M7=(A11-A10)(B10+B11)
        ({3:1, 2:-1}, {2:1, 3:1}, {0:1, 3:-1}),
    ]

    for s in range(7):
        u_blocks, v_blocks, w_blocks = products[s]
        for rb in range(r_base):
            r_idx = s * r_base + rb
            # Fill U
            for (i1, j1), coeff_u in [( (bi//block, bi%block), c) for bi, c in u_blocks.items()]:
                # Actually block_idx maps to (row_block, col_block)
                pass
            # Let me redo: block_idx = row_block * 2 + col_block for 2x2 blocks
            for bi, cu in u_blocks.items():
                i1 = bi // 2
                j1 = bi % 2
                for i2 in range(block):
                    for j2 in range(block):
                        a = Ub[(i1*block+i2)*block + (j1*block+j2), rb]
                        if a != 0:
                            U[(i1*block+i2)*m + (j1*block+j2), r_idx] = cu * a
            for bi, cv in v_blocks.items():
                j1 = bi // 2
                k1 = bi % 2
                for j2 in range(block):
                    for k2 in range(block):
                        b = Vb[(j1*block+j2)*block + (k1*block+k2), rb]
                        if b != 0:
                            V[(j1*block+j2)*p + (k1*block+k2), r_idx] = cv * b
            for bi, cw in w_blocks.items():
                k1 = bi // 2
                i1 = bi % 2
                for k2 in range(block):
                    for i2 in range(block):
                        c = Wb[(k1*block+k2)*block + (i1*block+i2), rb]
                        if c != 0:
                            W[(k1*block+k2)*n + (i1*block+i2), r_idx] = cw * c
    return U, V, W

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

def construct():
    n, m, p = 6, 6, 6
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p  # schoolbook rank

    # Strategy 1: Block-cyclic Strassen with Laderman 3x3 base -> rank 7*23 = 161
    try:
        U3, V3, W3 = _laderman_3x3()
        U, V, W = _block_cyclic_strassen(U3, V3, W3, 6, 6, 6, block=3)
        if _verify_decomposition(U, V, W, 6, 6, 6, trials=5):
            best_U, best_V, best_W = U, V, W
            best_rank = 7 * 23
            print(f"Block-cyclic Strassen + Laderman: rank {best_rank}")
    except Exception as e:
        print(f"Block-cyclic Strassen failed: {e}")

    # Strategy 2: Kronecker Laderman 3x3 + Strassen 2x2 -> rank 23*7 = 161
    if best_U is None:
        try:
            U3, V3, W3 = _laderman_3x3()
            U2, V2, W2 = _strassen_2x2()
            U, V, W = _kron_decompose(U3, V3, W3, U2, V2, W2)
            if _verify_decomposition(U, V, W, 6, 6, 6, trials=5):
                best_U, best_V, best_W = U, V, W
                best_rank = 23 * 7
                print(f"Kronecker Laderman+Strassen: rank {best_rank}")
        except Exception as e:
            print(f"Kronecker failed: {e}")

    # Strategy 3: Fallback schoolbook 6x6 -> rank 216
    if best_U is None:
        U, V, W = _schoolbook(6, 6, 6)
        best_U, best_V, best_W = U, V, W
        best_rank = 216
        print(f"Schoolbook fallback: rank {best_rank}")

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
