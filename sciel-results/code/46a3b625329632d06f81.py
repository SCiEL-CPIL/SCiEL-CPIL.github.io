import numpy as np
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = 4, 4, 7

    def strassen_2x2_block(i_off, j_off, k_off, a_scale=1, b_scale=1, c_scale=1):
        terms = []
        # M1 = (A11+A22)*(B11+B22)
        u = [0]*(n*m); u[i_off*m+j_off]=a_scale; u[(i_off+1)*m+(j_off+1)]=a_scale
        v = [0]*(m*p); v[j_off*p+k_off]=b_scale; v[(j_off+1)*p+(k_off+1)]=b_scale
        w = [0]*(p*n); w[k_off*n+i_off]=c_scale; w[(k_off+1)*n+(i_off+1)]=c_scale
        terms.append((u,v,w))
        # M2 = (A21+A22)*B11
        u = [0]*(n*m); u[(i_off+1)*m+j_off]=a_scale; u[(i_off+1)*m+(j_off+1)]=a_scale
        v = [0]*(m*p); v[j_off*p+k_off]=b_scale
        w = [0]*(p*n); w[k_off*n+(i_off+1)]=c_scale; w[(k_off+1)*n+(i_off+1)]=-c_scale
        terms.append((u,v,w))
        # M3 = A11*(B12-B22)
        u = [0]*(n*m); u[i_off*m+j_off]=a_scale
        v = [0]*(m*p); v[j_off*p+(k_off+1)]=b_scale; v[(j_off+1)*p+(k_off+1)]=-b_scale
        w = [0]*(p*n); w[(k_off+1)*n+i_off]=c_scale; w[(k_off+1)*n+(i_off+1)]=c_scale
        terms.append((u,v,w))
        # M4 = A22*(B21-B11)
        u = [0]*(n*m); u[(i_off+1)*m+(j_off+1)]=a_scale
        v = [0]*(m*p); v[(j_off+1)*p+k_off]=b_scale; v[j_off*p+k_off]=-b_scale
        w = [0]*(p*n); w[k_off*n+i_off]=c_scale; w[k_off*n+(i_off+1)]=c_scale
        terms.append((u,v,w))
        # M5 = (A11+A12)*B22
        u = [0]*(n*m); u[i_off*m+j_off]=a_scale; u[i_off*m+(j_off+1)]=a_scale
        v = [0]*(m*p); v[(j_off+1)*p+(k_off+1)]=b_scale
        w = [0]*(p*n); w[(k_off+1)*n+i_off]=c_scale; w[k_off*n+i_off]=-c_scale
        terms.append((u,v,w))
        # M6 = (A21-A11)*(B11+B12)
        u = [0]*(n*m); u[(i_off+1)*m+j_off]=a_scale; u[i_off*m+j_off]=-a_scale
        v = [0]*(m*p); v[j_off*p+k_off]=b_scale; v[j_off*p+(k_off+1)]=b_scale
        w = [0]*(p*n); w[(k_off+1)*n+(i_off+1)]=c_scale
        terms.append((u,v,w))
        # M7 = (A12-A22)*(B21+B22)
        u = [0]*(n*m); u[i_off*m+(j_off+1)]=a_scale; u[(i_off+1)*m+(j_off+1)]=-a_scale
        v = [0]*(m*p); v[(j_off+1)*p+k_off]=b_scale; v[(j_off+1)*p+(k_off+1)]=b_scale
        w = [0]*(p*n); w[k_off*n+i_off]=c_scale
        terms.append((u,v,w))
        return terms

    def strassen_4x4_head():
        # 4x4 * 4x4 using 2x2 Strassen blocks
        terms = []
        for i_b in range(2):
            for j_b in range(2):
                i_off = i_b * 2
                j_off = j_b * 2
                k_off = 0
                terms.extend(strassen_2x2_block(i_off, j_off, k_off))
        return terms

    def strassen_4x4x2_tail():
        # 4x4 * 4x2 (cols 4,5)
        terms = []
        for i_b in range(2):
            for j_b in range(2):
                i_off = i_b * 2
                j_off = j_b * 2
                k_off = 4
                terms.extend(strassen_2x2_block(i_off, j_off, k_off))
        return terms

    def schoolbook_4x4x1_tail():
        # 4x4 * 4x1 (col 6)
        terms = []
        for i in range(n):
            for j in range(m):
                u = [0]*(n*m); u[i*m+j] = 1
                v = [0]*(m*p); v[j*p+6] = 1
                w = [0]*(p*n); w[6*n+i] = 1
                terms.append((u,v,w))
        return terms

    # Generate all terms
    all_terms = strassen_4x4_head() + strassen_4x4x2_tail() + schoolbook_4x4x1_tail()

    if time.time() > deadline:
        # Fallback: return empty (invalid)
        return {"U": [], "V": [], "W": []}

    rank = len(all_terms)
    U = np.zeros((n*m, rank), dtype=object)
    V = np.zeros((m*p, rank), dtype=object)
    W = np.zeros((p*n, rank), dtype=object)

    for c, (u, v, w) in enumerate(all_terms):
        for r in range(n*m):
            if u[r] != 0:
                U[r, c] = u[r]
        for r in range(m*p):
            if v[r] != 0:
                V[r, c] = v[r]
        for r in range(p*n):
            if w[r] != 0:
                W[r, c] = w[r]

    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
