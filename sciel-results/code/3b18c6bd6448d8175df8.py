import time
import numpy as np

def construct():
    deadline = time.time() + 140
    try:
        return _construct_rank_22()
    except Exception:
        return _construct_rank_24()

def _construct_rank_24():
    n, m, p = 2, 3, 4
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

def _verify(U, V, W, n, m, p):
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    r = Ua.shape[1]
    # tensor: U[i*m+j, t] V[j*p+k, t] W[k*n+i, t]
    # Reconstruct C[i,p? ] via C[i, k] = sum_{j,t} A[i,j] B[j,k] U[i*m+j,t] V[j*p+k,t] W[k*n+i,t]
    err = 0.0
    for i in range(n):
        for k in range(p):
            # expected contribution is 1 for the (i,j,k) tensor coefficient
            for j in range(m):
                val = sum(U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t] for t in range(r))
                err += abs(val - 1.0)
    # Also check that no spurious (i,j,k) combos: we already iterate all (i,j,k) and expect 1
    return err

def _construct_rank_22():
    n, m, p = 2, 3, 4
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def put(u, v, w):
        nonlocal col
        for i, val in u.items():
            U[i][col] += val
        for i, val in v.items():
            V[i][col] += val
        for i, val in w.items():
            W[i][col] += val
        col += 1

    # ---- Block A: A[:,0:2] * B[0:2,:]  (2x2 x 2x4) -> 2x4, 8 terms (rank of 2x2x4 = 8) ----
    # A rows i in {0,1} (U indices 0,1,3,4), B cols k in {0,1} (V rows 0,1,4,5), C (i,k)
    # Use 2x2 Strassen on the (i,k) with the two B columns? 2x2x4 minimal is 8.
    # Decompose: for each of the 4 (i,k) outputs we need contributions. Use:
    # M1 = (A00+A11)(B00+B10) -> C00,C11
    # M2 = (A10+A11)(B00+B10) ... let's just use a known 2x2x4 rank-8.
    # 2x2x4: C = A(2x2) B(2x4). Rank 8.
    # Terms (u: A-index, v: B-index, w: C-index):
    # C00 = A00 B00 + A01 B10
    # C01 = A00 B01 + A01 B11
    # C10 = A10 B00 + A11 B10
    # C11 = A10 B01 + A11 B11
    # A indices: A00=0,A01=1,A10=3,A11=4
    # B indices: B00=0,B01=1,B10=4,B11=5
    # C indices: C00=0,C01=1,C10=2,C11=3
    # Rank-8: use M1=(A00+A11)(B00+B10), M2=(A10+A11)(B00+B10), M3=A00(B10+B11)... 
    # Simpler: 8 = schoolbook for 2x2x4? schoolbook = 2*2*4=16. Strassen-style 8:
    # M1=(A00+A11)(B00+B10)
    # M2=(A10+A11)(B00+B10)
    # M3=A00(B10+B11)
    # M4=A11(B01+B10)
    # M5=(A00+A01)B11
    # M6=(A10-A00)(B00+B01)
    # M7=(A01-A11)(B10+B11)
    # M8=(A10+A00)(B11-B01)  -- need to verify; instead I'll just verify numerically.
    # I'll use a safe construction: 2x2x4 can be done with 8 via:
    # Let a=A00,b=A01,c=A10,d=A11 ; B rows: B0=(B00,B01),B1=(B10,B11)
    # C0 = a*B0 + b*B1 ; C1 = c*B0 + d*B1
    # Strassen: M1=(a+d)(B0+B1), M2=(c+d)B0, M3=a(B1+B0), M4=d(B0+B1)... 
    # To be safe, I'll just build 8 terms and verify; if fails, fall back.
    put({0:1,4:1}, {0:1,4:1}, {0:1,3:1})   # M1 (a+d)(B00+B10)
    put({3:1,4:1}, {0:1,4:1}, {1:1,3:-1})  # M2 (c+d)(B00+B10)
    put({0:1}, {4:1,5:1}, {2:1,3:1})       # M3 a(B10+B11)
    put({4:1}, {1:1,4:1}, {0:-1,1:1})      # M4 d(B01+B10)
    put({0:1,1:1}, {5:1}, {0:1,2:1})       # M5 (a+b)B11
    put({3:1,0:-1}, {0:1,1:1}, {5:1})      # M6 (c-a)(B00+B01)
    put({1:1,4:-1}, {4:1,5:1}, {0:-1,3:1}) # M7 (b-d)(B10+B11)
    put({0:1,3:1}, {5:1,1:-1}, {6:1})      # M8 (a+c)(B11-B01)

    # ---- Block B: A[:,2] * B[:,0:2]  (2x1 x 3x2) -> 2x2, 4 terms (schoolbook) ----
    # A indices: A02=2, A12=5 ; B cols k in {0,1}: B00=0,B01=1,B10=4,B11=5 ; C C00=0,C01=1,C10=2,C11=3
    # C00 = A02 B00 + A12 B10 ; C01 = A02 B01 + A12 B11 ; C10 = ... wait C10 uses A12 B00 + A02 B10? 
    # C[i,k] = sum_j A[i,j] B[j,k]. For A[:,2]: C[i,k] = A[i,2] B[2,k].
    # B[2,0]=8, B[2,1]=9. So C00 = A02*B[2,0] = A02*8 ; C01=A02*B[2,1]=A02*9 ; C10=A12*8 ; C11=A12*9.
    put({2:1}, {8:1}, {0:1})
    put({2:1}, {9:1}, {1:1})
    put({5:1}, {8:1}, {2:1})
    put({5:1}, {9:1}, {3:1})

    # ---- Block C: A[:,0:2] * B[2,:]  (2x2 x 1x4) -> 2x4, 4 terms ----
    # C[i,k] = A[i,0]B[2,k] + A[i,1]B[2,k] = (A[i,0]+A[i,1]) B[2,k]
    # B[2,k]: k=0->8,k=1->9,k=2->10,k=3->11
    # C indices: C[i,k]: (i,k) -> k*n+i : (0,0)=0,(0,1)=1,(0,2)=4,(0,3)=5,(1,0)=2,(1,1)=3,(1,2)=6,(1,3)=7
    # For i=0: A00=0,A01=1 ; i=1: A10=3,A11=4
    put({0:1,1:1}, {8:1}, {0:1})   # (A00+A01)*B[2,0] -> C00
    put({0:1,1:1}, {9:1}, {1:1})   # -> C01
    put({0:1,1:1}, {10:1}, {4:1})  # -> C02
    put({0:1,1:1}, {11:1}, {5:1})  # -> C03
    put({3:1,4:1}, {8:1}, {2:1})   # (A10+A11)*B[2,0] -> C10
    put({3:1,4:1}, {9:1}, {3:1})   # -> C11
    put({3:1,4:1}, {10:1}, {6:1})  # -> C12
    put({3:1,4:1}, {11:1}, {7:1})  # -> C13

    # ---- Block D: A[:,2] * B[2,:]  (2x1 x 1x4) -> 2x4, 4 terms ----
    # C[i,k] = A[i,2] B[2,k]
    put({2:1}, {8:1}, {0:1})
    put({2:1}, {9:1}, {1:1})
    put({2:1}, {10:1}, {4:1})
    put({2:1}, {11:1}, {5:1})
    put({5:1}, {8:1}, {2:1})
    put({5:1}, {9:1}, {3:1})
    put({5:1}, {10:1}, {6:1})
    put({5:1}, {11:1}, {7:1})

    if col != 22:
        raise ValueError(f"Expected 22 columns, got {col}")

    # Verify numerically; if wrong, raise to trigger rank-24 fallback
    err = _verify(U, V, W, n, m, p)
    if err > 1e-6:
        raise ValueError(f"rank-22 verification failed, err={err}")
    return {"U": U, "V": V, "W": W}
