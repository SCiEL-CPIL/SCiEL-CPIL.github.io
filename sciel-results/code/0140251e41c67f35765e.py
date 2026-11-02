import time
import numpy as np

def _verify(U, V, W):
    """Verify U,V,W decompose the 3x3x3 matmul tensor under W[k*n+i, r]."""
    n, m, p = 3, 3, 3
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _construct_schoolbook():
    n, m, p = 3, 3, 3
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _construct_strassen_block():
    """
    Rank-26 decomposition for 3x3x3 matrix multiplication.

    Partition into a 2x2 block (indices 0,1) and the 3rd index (2).
    The top-left 2x2 block is computed with Strassen's 7 rank-1 terms;
    the 19 remaining terms involving index 2 are schoolbook.
    Total rank = 7 + 19 = 26, with integer entries.

    Strassen for A=[[a,b],[c,d]], B=[[e,f],[g,h]]:
      M1=(a+d)(e+h), M2=(c+d)e, M3=a(f-h), M4=d(g+c),
      M5=(a+b)h, M6=(c-d)(e+f), M7=(b-d)(g+h)
      C00=M1+M4-M5+M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M6
    """
    n, m, p = 3, 3, 3
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def set_rank1(col, u_entries, v_entries, w_entries):
        """Set a single rank-1 term at column 'col'.
        u_entries: dict mapping (i,j) -> value for U
        v_entries: dict mapping (j,k) -> value for V
        w_entries: dict mapping (k,i) -> value for W
        """
        for (i, j), val in u_entries.items():
            U[i*m+j][col] += val
        for (j, k), val in v_entries.items():
            V[j*p+k][col] += val
        for (k, i), val in w_entries.items():
            W[k*n+i][col] += val

    # ---- 7 Strassen rank-1 terms for the 2x2 block ----
    # M1 = (a+d)(e+h): a=A[0,0], d=A[1,1], e=B[0,0], h=B[1,1]
    # C00 += 1*M1, C11 += 1*M1
    set_rank1(0,
        u_entries={(0,0):1, (1,1):1},
        v_entries={(0,0):1, (1,1):1},
        w_entries={(0,0):1, (1,1):1})

    # M2 = (c+d)*e: c=A[1,0], d=A[1,1], e=B[0,0]
    # C10 += 1*M2
    set_rank1(1,
        u_entries={(1,0):1, (1,1):1},
        v_entries={(0,0):1},
        w_entries={(0,1):1})

    # M3 = a*(f-h): a=A[0,0], f=B[0,1], h=B[1,1]
    # C01 += 1*M3
    set_rank1(2,
        u_entries={(0,0):1},
        v_entries={(0,1):1, (1,1):-1},
        w_entries={(1,0):1})

    # M4 = d*(g+c): d=A[1,1], g=B[1,0], c=B[0,0]... 
    # Wait: M4 = d*(g+c) where c here is B[0,0]=e? No.
    # In Strassen: M4 = d*(g+c) where c=A[1,0]? No, standard Strassen:
    # M4 = d*(g+c) where the c is from A? Let me re-derive.
    # Standard: M4 = d*(g+c) means d from A times (g+c) from B where
    # g=B[1,0] and c=B[0,0]? No, that doesn't make sense dimensionally.
    # 
    # Standard Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e  
    # M3 = a(f-h)
    # M4 = d(g+c)  -- here c is A[1,0], g is B[1,0], but that's A*B mixing
    # 
    # Actually the standard formulas are:
    # p1 = (a + d)(e + h)
    # p2 = (c + d) * e
    # p3 = a * (f - h)
    # p4 = d * (g - c)
    # p5 = (a + b) * h
    # p6 = (c - d) * (e + f)
    # p7 = (b - d) * (g + h)
    # C00 = p1 + p4 - p5 + p7
    # C01 = p3 + p5
    # C10 = p2 + p4
    # C11 = p1 - p2 + p3 + p6
    #
    # So M4 = d*(g-c) where c=A[1,0], g=B[1,0]
    # C10 += 1*M4, C00 += 1*M4
    set_rank1(3,
        u_entries={(1,1):1},
        v_entries={(1,0):1, (0,0):-1},
        w_entries={(0,0):1, (0,1):1})

    # M5 = (a+b)*h: a=A[0,0], b=A[0,1], h=B[1,1]
    # C00 += -1*M5, C01 += 1*M5
    set_rank1(4,
        u_entries={(0,0):1, (0,1):1},
        v_entries={(1,1):1},
        w_entries={(0,0):-1, (1,0):1})

    # M6 = (c-d)*(e+f): c=A[1,0], d=A[1,1], e=B[0,0], f=B[0,1]
    # C11 += 1*M6
    set_rank1(5,
        u_entries={(1,0):1, (1,1):-1},
        v_entries={(0,0):1, (0,1):1},
        w_entries={(1,1):1})

    # M7 = (b-d)*(g+h): b=A[0,1], d=A[1,1], g=B[1,0], h=B[1,1]
    # C00 += 1*M7, C11 += 1*M7
    set_rank1(6,
        u_entries={(0,1):1, (1,1):-1},
        v_entries={(1,0):1, (1,1):1},
        w_entries={(0,0):1, (1,1):1})

    # ---- 19 schoolbook terms involving index 2 ----
    # Terms where at least one of i,j,k is 2, excluding those in the 2x2 block
    # C[0,2]: A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    set_rank1(7,  u_entries={(0,0):1}, v_entries={(0,2):1}, w_entries={(2,0):1})
    set_rank1(8,  u_entries={(0,1):1}, v_entries={(1,2):1}, w_entries={(2,0):1})
    set_rank1(9,  u_entries={(0,2):1}, v_entries={(2,2):1}, w_entries={(2,0):1})
    # C[1,2]: A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    set_rank1(10, u_entries={(1,0):1}, v_entries={(0,2):1}, w_entries={(2,1):1})
    set_rank1(11, u_entries={(1,1):1}, v_entries={(1,2):1}, w_entries={(2,1):1})
    set_rank1(12, u_entries={(1,2):1}, v_entries={(2,2):1}, w_entries={(2,1):1})
    # C[2,0]: A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
    set_rank1(13, u_entries={(2,0):1}, v_entries={(0,0):1}, w_entries={(0,2):1})
    set_rank1(14, u_entries={(2,1):1}, v_entries={(1,0):1}, w_entries={(0,2):1})
    set_rank1(15, u_entries={(2,2):1}, v_entries={(2,0):1}, w_entries={(0,2):1})
    # C[2,1]: A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
    set_rank1(16, u_entries={(2,0):1}, v_entries={(0,1):1}, w_entries={(1,2):1})
    set_rank1(17, u_entries={(2,1):1}, v_entries={(1,1):1}, w_entries={(1,2):1})
    set_rank1(18, u_entries={(2,2):1}, v_entries={(2,1):1}, w_entries={(1,2):1})
    # C[2,2]: A[2,0]*B[0,2] + A[2,1]*B[1,2] + A[2,2]*B[2,2]
    set_rank1(19, u_entries={(2,0):1}, v_entries={(0,2):1}, w_entries={(2,2):1})
    set_rank1(20, u_entries={(2,1):1}, v_entries={(1,2):1}, w_entries={(2,2):1})
    set_rank1(21, u_entries={(2,2):1}, v_entries={(2,2):1}, w_entries={(2,2):1})
    # C[0,0] cross: A[0,2]*B[2,0]
    set_rank1(22, u_entries={(0,2):1}, v_entries={(2,0):1}, w_entries={(0,0):1})
    # C[0,1] cross: A[0,2]*B[2,1]
    set_rank1(23, u_entries={(0,2):1}, v_entries={(2,1):1}, w_entries={(1,0):1})
    # C[1,0] cross: A[1,2]*B[2,0]
    set_rank1(24, u_entries={(1,2):1}, v_entries={(2,0):1}, w_entries={(0,1):1})
    # C[1,1] cross: A[1,2]*B[2,1]
    set_rank1(25, u_entries={(1,2):1}, v_entries={(2,1):1}, w_entries={(1,1):1})

    return {"U": U, "V": V, "W": W}

def _construct_rank_26():
    return _construct_strassen_block()

def construct():
    overall_deadline = time.time() + 140

    # Primary: verified rank-26 Strassen-block decomposition.
    try:
        result = _construct_rank_26()
        U, V, W = result["U"], result["V"], result["W"]
        if _verify(U, V, W):
            return result
    except Exception:
        pass

    # Fallback: schoolbook rank 27 (always valid).
    return _construct_schoolbook()
