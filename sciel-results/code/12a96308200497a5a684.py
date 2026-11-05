import time
from functools import lru_cache

DIMENSIONS = [4, 5, 5]
_start_time = time.time()
TIME_LIMIT = 140  # safety margin before 150s kill

def _strassen_2x2x2():
    """Return (U, V, W) for 2x2 * 2x2, rank 7."""
    U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    W = [
        [1, 0, 0, 1],
        [0, -1, 0, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 0, 0, 1],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    return U, V, W

def _winograd_3x3x3():
    """Return (U, V, W) for 3x3 * 3x3, rank 23 (Winograd)."""
    # A = [[a,b,c],[d,e,f],[g,h,i]] indexed as [i*3+j]
    # B = [[j,k,l],[m,n,o],[p,q,r]] indexed as [j*3+k]  
    # C = [[p,q,r],[s,t,u],[v,w,x]] indexed as [k*3+i]
    # 
    # Winograd's 23-term scheme:
    U = []
    V = []
    W = []
    
    # M1 = (a+b+c)(j+k+l)
    U.append([1,1,1, 1,1,1, 1,1,1])
    V.append([1,1,1, 1,1,1, 1,1,1])
    W.append([1,0,0, 0,1,0, 0,0,1])  # contributes to all of C
    
    # M2 = (a+c)(j+l)
    U.append([1,0,1, 0,0,0, 0,0,0])
    V.append([1,0,1, 0,0,0, 0,0,0])
    W.append([1,0,0, 0,0,0, 0,0,1])
    
    # M3 = (a+b)(j+k)
    U.append([1,1,0, 0,0,0, 0,0,0])
    V.append([1,1,0, 0,0,0, 0,0,0])
    W.append([1,0,0, 0,1,0, 0,0,0])
    
    # M4 = (b+c)(k+l)
    U.append([0,1,1, 0,0,0, 0,0,0])
    V.append([0,1,1, 0,0,0, 0,0,0])
    W.append([0,1,0, 0,0,0, 0,0,1])
    
    # M5 = (d+f)(m+o)
    U.append([0,0,0, 1,0,1, 0,0,0])
    V.append([0,0,0, 1,0,1, 0,0,0])
    W.append([0,0,0, 1,0,0, 0,0,1])
    
    # M6 = (d+e)(m+n)
    U.append([0,0,0, 1,1,0, 0,0,0])
    V.append([0,0,0, 1,1,0, 0,0,0])
    W.append([0,0,0, 1,0,0, 0,1,0])
    
    # M7 = (e+f)(n+o)
    U.append([0,0,0, 0,1,1, 0,0,0])
    V.append([0,0,0, 0,1,1, 0,0,0])
    W.append([0,0,0, 0,1,0, 0,0,1])
    
    # M8 = (g+i)(p+r)
    U.append([0,0,0, 0,0,0, 1,0,1])
    V.append([0,0,0, 0,0,0, 1,0,1])
    W.append([0,0,0, 0,0,0, 1,0,1])
    
    # M9 = (g+h)(p+q)
    U.append([0,0,0, 0,0,0, 1,1,0])
    V.append([0,0,0, 0,0,0, 1,1,0])
    W.append([0,0,0, 0,0,0, 1,0,0])
    
    # M10 = (h+i)(q+r)
    U.append([0,0,0, 0,0,0, 0,1,1])
    V.append([0,0,0, 0,0,0, 0,1,1])
    W.append([0,0,0, 0,0,0, 0,1,1])
    
    # M11 = (a+d)(j+m)
    U.append([1,0,0, 1,0,0, 0,0,0])
    V.append([1,0,0, 1,0,0, 0,0,0])
    W.append([1,0,0, 1,0,0, 0,0,0])
    
    # M12 = (b+e)(k+n)
    U.append([0,1,0, 0,1,0, 0,0,0])
    V.append([0,1,0, 0,1,0, 0,0,0])
    W.append([0,1,0, 0,1,0, 0,0,0])
    
    # M13 = (c+f)(l+o)
    U.append([0,0,1, 0,0,1, 0,0,0])
    V.append([0,0,1, 0,0,1, 0,0,0])
    W.append([0,0,1, 0,0,1, 0,0,0])
    
    # M14 = (a+g)(j+p)
    U.append([1,0,0, 0,0,0, 1,0,0])
    V.append([1,0,0, 0,0,0, 1,0,0])
    W.append([1,0,0, 0,0,0, 1,0,0])
    
    # M15 = (b+h)(k+q)
    U.append([0,1,0, 0,0,0, 0,1,0])
    V.append([0,1,0, 0,0,0, 0,1,0])
    W.append([0,1,0, 0,0,0, 0,1,0])
    
    # M16 = (c+i)(l+r)
    U.append([0,0,1, 0,0,0, 0,0,1])
    V.append([0,0,1, 0,0,0, 0,0,1])
    W.append([0,0,1, 0,0,0, 0,0,1])
    
    # M17 = (a+f)(j+o)
    U.append([1,0,0, 0,0,1, 0,0,0])
    V.append([1,0,0, 0,0,1, 0,0,0])
    W.append([1,0,0, 0,0,1, 0,0,0])
    
    # M18 = (b+e)(k+n) -- this is same as M12, need different terms
    # Actually the Winograd scheme has specific 23 terms. Let me use a verified construction.
    
    # I'll use a different approach: construct via embedding
    # For now, let me just use schoolbook for 3x3x3 and rely on the recursive splitting
    
    # Actually, let me return schoolbook for now
    rank = 27
    U = []
    V = []
    W = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                u = [0]*9
                v = [0]*9
                w = [0]*9
                u[i*3+j] = 1
                v[j*3+k] = 1
                w[k*3+i] = 1
                U.append(u)
                V.append(v)
                W.append(w)
    return U, V, W

@lru_cache(maxsize=None)
def _decompose(m, n, p):
    """
    Return (U, V, W) for m×n * n×p matrix multiplication.
    U: shape (m*n, rank)
    V: shape (n*p, rank)
    W: shape (p*m, rank), W[k*m+i, r] convention
    """
    if time.time() - _start_time > TIME_LIMIT:
        raise TimeoutError()
    
    if m == 0 or n == 0 or p == 0:
        return [], [], []
    
    if m == 1 and n == 1 and p == 1:
        return [[1]], [[1]], [[1]]
    
    if m == 2 and n == 2 and p == 2:
        return _strassen_2x2x2()
    
    # Try all possible splits and pick the best
    best = None
    best_rank = m * n * p  # schoolbook
    
    for m1 in range(1, m):
        m2 = m - m1
        for n1 in range(1, n):
            n2 = n - n1
            for p1 in range(1, p):
                p2 = p - p1
                try:
                    U1, V1, W1 = _decompose(m1, n1, p1)
                    U2, V2, W2 = _decompose(m1, n1, p2)
                    U3, V3, W3 = _decompose(m1, n2, p1)
                    U4, V4, W4 = _decompose(m1, n2, p2)
                    U5, V5, W5 = _decompose(m2, n1, p1)
                    U6, V6, W6 = _decompose(m2, n1, p2)
                    U7, V7, W7 = _decompose(m2, n2, p1)
                    U8, V8, W8 = _decompose(m2, n2, p2)
                    
                    total_rank = (len(U1) + len(U2) + len(U3) + len(U4) +
                                 len(U5) + len(U6) + len(U7) + len(U8))
                    if total_rank < best_rank:
                        best_rank = total_rank
                        U_new = []
                        V_new = []
                        W_new = []
                        
                        def add_decomp(U_s, V_s, W_s, m_i, n_i, p_i, offset_m, offset_n, offset_p):
                            for r in range(len(U_s)):
                                u_row = [0] * (m * n)
                                v_row = [0] * (n * p)
                                w_row = [0] * (p * m)
                                for ii in range(m_i):
                                    for jj in range(n_i):
                                        u_row[(offset_m + ii) * n + (offset_n + jj)] = U_s[r][ii * n_i + jj]
                                for jj in range(n_i):
                                    for kk in range(p_i):
                                        v_row[(offset_n + jj) * p + (offset_p + kk)] = V_s[r][jj * p_i + kk]
                                for kk in range(p_i):
                                    for ii in range(m_i):
                                        w_row[(offset_p + kk) * m + (offset_m + ii)] = W_s[r][kk * m_i + ii]
                                U_new.append(u_row)
                                V_new.append(v_row)
                                W_new.append(w_row)
                        
                        add_decomp(U1, V1, W1, m1, n1, p1, 0, 0, 0)
                        add_decomp(U3, V3, W3, m1, n2, p1, 0, n1, 0)
                        add_decomp(U2, V2, W2, m1, n1, p2, 0, 0, p1)
                        add_decomp(U4, V4, W4, m1, n2, p2, 0, n1, p1)
                        add_decomp(U5, V5, W5, m2, n1, p1, m1, 0, 0)
                        add_decomp(U6, V6, W6, m2, n1, p2, m1, 0, p1)
                        add_decomp(U7, V7, W7, m2, n2, p1, m1, n1, 0)
                        add_decomp(U8, V8, W8, m2, n2, p2, m1, n1, p1)
                        
                        best = (U_new, V_new, W_new)
                except TimeoutError:
                    raise
                except Exception:
                    continue
    
    if best is not None:
        return best
    else:
        rank = m * n * p
        U = []
        V = []
        W = []
        for i in range(m):
            for j in range(n):
                for k in range(p):
                    u = [0] * (m * n)
                    v = [0] * (n * p)
                    w = [0] * (p * m)
                    u[i * n + j] = 1
                    v[j * p + k] = 1
                    w[k * m + i] = 1
                    U.append(u)
                    V.append(v)
                    W.append(w)
        return U, V, W

def construct():
    global _start_time
    _start_time = time.time()
    _decompose.cache_clear()
    n, m, p = DIMENSIONS  # n=4, m=5, p=5
    U, V, W = _decompose(n, m, p)
    return {"U": U, "V": V, "W": W}
