import numpy as np
import time

# EVOLVE-BLOCK-START
_n, _m, _p = 4, 4, 4
_rank = 49

def _build_decomposition():
    """
    Build rank-49 decomposition for 4x4 matrix multiplication.
    
    Strategy: Recursive Strassen.
    4x4 matmul -> 7 block products of 2x2 blocks (Strassen at block level)
    Each 2x2 block product -> 7 scalar products (Strassen at element level)
    Total: 7 * 7 = 49 rank-1 terms.
    
    Convention:
      U[i*m+j, r] : coefficient of A[i,j] in r-th rank-1 term
      V[j*p+k, r] : coefficient of B[j,k] in r-th rank-1 term  
      W[k*n+i, r] : coefficient for output C[i,k] in r-th rank-1 term
                    (trace convention: W[k*n+i, r])
    
    The tensor identity is:
      For all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
      (This equals C[i,k] = sum_j A[i,j]*B[j,k])
    """
    U = [[0] * _rank for _ in range(_n * _m)]
    V = [[0] * _rank for _ in range(_m * _p)]
    W = [[0] * _rank for _ in range(_p * _n)]
    
    def u_idx(i, j):
        return i * _m + j
    
    def v_idx(j, k):
        return j * _p + k
    
    def w_idx(k, i):
        return k * _n + i
    
    # Strassen for 2x2: given A=[[a,b],[c,d]], B=[[e,f],[g,h]]
    # M1=(a+d)(e+h), M2=(c+d)e, M3=a(f-h), M4=d(g-e)
    # M5=(a+b)h, M6=(c-a)(e+f), M7=(b-d)(g+h)
    # C11=M1+M4-M5+M7, C12=M3+M5, C21=M2+M4, C22=M1-M2+M3+M6
    
    # Sub-contributions: for each M_k, list of (out_row, out_col, coeff)
    # where out_row/out_col are positions in the 2x2 result
    strassen_sub = [
        [(0,0,1), (1,1,1)],     # M1 -> C11:+1, C22:+1
        [(1,0,1), (1,1,-1)],    # M2 -> C21:+1, C22:-1
        [(0,1,1), (1,1,1)],     # M3 -> C12:+1, C22:+1
        [(0,0,1), (1,0,1)],     # M4 -> C11:+1, C21:+1
        [(0,0,-1), (0,1,1)],    # M5 -> C11:-1, C12:+1
        [(1,1,1)],              # M6 -> C22:+1
        [(0,0,1)],              # M7 -> C11:+1
    ]
    
    # Block-level Strassen for 4x4:
    # A = [[A11,A12],[A21,A22]], B = [[B11,B12],[B21,B22]]
    # P1=(A11+A22)(B11+B22), P2=(A21+A22)B11, P3=A11(B12-B22)
    # P4=A22(B21-B11), P5=(A11+A12)B22, P6=(A21-A11)(B11+B12)
    # P7=(A12-A22)(B21+B22)
    # C11=P1+P4-P5+P7, C12=P3+P5, C21=P2+P4, C22=P1-P2+P3+P6
    
    # For each block-level term t, define:
    #   L[t]: 2x2 array of dicts {(global_row, global_col): coeff} for A
    #   R[t]: 2x2 array of dicts {(global_row, global_col): coeff} for B
    #   C[t]: dict mapping (r,c) -> list of (out_row, out_col, coeff)
    #         where (r,c) is position in the 2x2 block result P_t,
    #         and (out_row, out_col) is the global C position
    
    def a_glob(bi, bj, r, c):
        return (2*bi + r, 2*bj + c)
    
    def b_glob(bi, bj, r, c):
        return (2*bi + r, 2*bj + c)
    
    def c_glob(bi, bj, r, c):
        return (2*bi + r, 2*bj + c)
    
    L_terms = []
    R_terms = []
    C_terms = []
    
    # Term 0: P1 = (A11 + A22)(B11 + B22)
    L0 = [[{} for _ in range(2)] for _ in range(2)]
    R0 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            L0[r][c] = {a_glob(0,0,r,c): 1, a_glob(1,1,r,c): 1}
            R0[r][c] = {b_glob(0,0,r,c): 1, b_glob(1,1,r,c): 1}
    C0 = {}
    for r in range(2):
        for c in range(2):
            C0[(r,c)] = [c_glob(0,0,r,c) + (1,), c_glob(1,1,r,c) + (1,)]
    L_terms.append(L0); R_terms.append(R0); C_terms.append(C0)
    
    # Term 1: P2 = (A21 + A22)B11
    L1 = [[{} for _ in range(2)] for _ in range(2)]
    R1 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            L1[r][c] = {a_glob(1,0,r,c): 1, a_glob(1,1,r,c): 1}
            R1[r][c] = {b_glob(0,0,r,c): 1}
    C1 = {}
    for r in range(2):
        for c in range(2):
            C1[(r,c)] = [c_glob(1,0,r,c) + (1,), c_glob(1,1,r,c) + (-1,)]
    L_terms.append(L1); R_terms.append(R1); C_terms.append(C1)
    
    # Term 2: P3 = A11(B12 - B22)
    L2 = [[{} for _ in range(2)] for _ in range(2)]
    R2 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            L2[r][c] = {a_glob(0,0,r,c): 1}
            R2[r][c] = {b_glob(0,1,r,c): 1, b_glob(1,1,r,c): -1}
    C2 = {}
    for r in range(2):
        for c in range(2):
            C2[(r,c)] = [c_glob(0,1,r,c) + (1,), c_glob(1,1,r,c) + (1,)]
    L_terms.append(L2); R_terms.append(R2); C_terms.append(C2)
    
    # Term 3: P4 = A22(B21 - B11)
    L3 = [[{} for _ in range(2)] for _ in range(2)]
    R3 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            L3[r][c] = {a_glob(1,1,r,c): 1}
            R3[r][c] = {b_glob(1,0,r,c): 1, b_glob(0,0,r,c): -1}
    C3 = {}
    for r in range(2):
        for c in range(2):
            C3[(r,c)] = [c_glob(0,0,r,c) + (1,), c_glob(1,0,r,c) + (1,)]
    L_terms.append(L3); R_terms.append(R3); C_terms.append(C3)
    
    # Term 4: P5 = (A11 + A12)B22
    L4 = [[{} for _ in range(2)] for _ in range(2)]
    R4 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            L4[r][c] = {a_glob(0,0,r,c): 1, a_glob(0,1,r,c): 1}
            R4[r][c] = {b_glob(1,1,r,c): 1}
    C4 = {}
    for r in range(2):
        for c in range(2):
            C4[(r,c)] = [c_glob(0,0,r,c) + (-1,), c_glob(0,1,r,c) + (1,)]
    L_terms.append(L4); R_terms.append(R4); C_terms.append(C4)
    
    # Term 5: P6 = (A21 - A11)(B11 + B12)
    L5 = [[{} for _ in range(2)] for _ in range(2)]
    R5 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            L5[r][c] = {a_glob(1,0,r,c): 1, a_glob(0,0,r,c): -1}
            R5[r][c] = {b_glob(0,0,r,c): 1, b_glob(0,1,r,c): 1}
    C5 = {}
    for r in range(2):
        for c in range(2):
            C5[(r,c)] = [c_glob(1,1,r,c) + (1,)]
    L_terms.append(L5); R_terms.append(R5); C_terms.append(C5)
    
    # Term 6: P7 = (A12 - A22)(B21 + B22)
    L6 = [[{} for _ in range(2)] for _ in range(2)]
    R6 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            L6[r][c] = {a_glob(0,1,r,c): 1, a_glob(1,1,r,c): -1}
            R6[r][c] = {b_glob(1,0,r,c): 1, b_glob(1,1,r,c): 1}
    C6 = {}
    for r in range(2):
        for c in range(2):
            C6[(r,c)] = [c_glob(0,0,r,c) + (1,)]
    L_terms.append(L6); R_terms.append(R6); C_terms.append(C6)
    
    # For each block-level term t, apply element-level Strassen
    for t in range(7):
        Lt = L_terms[t]
        Rt = R_terms[t]
        Ct = C_terms[t]
        
        # Extract 2x2 elements (each is a dict of A/B element coefficients)
        a, b = Lt[0][0], Lt[0][1]
        c, d = Lt[1][0], Lt[1][1]
        e, f = Rt[0][0], Rt[0][1]
        g, h = Rt[1][0], Rt[1][1]
        
        # Build the 7 Strassen sub-products
        def add_dicts(d1, d2, sign=1):
            result = {}
            for key, coeff in d1.items():
                result[key] = result.get(key, 0) + coeff
            for key, coeff in d2.items():
                result[key] = result.get(key, 0) + sign * coeff
            return result
        
        def sub_dicts(d1, d2):
            return add_dicts(d1, d2, -1)
        
        sub_prods = [
            (add_dicts(a, d), add_dicts(e, h)),           # M1: (a+d)(e+h)
            (add_dicts(c, d), dict(e)),                   # M2: (c+d)e
            (dict(a), sub_dicts(f, h)),                   # M3: a(f-h)
            (dict(d), sub_dicts(g, e)),                   # M4: d(g-e)
            (add_dicts(a, b), dict(h)),                   # M5: (a+b)h
            (sub_dicts(c, a), add_dicts(e, f)),           # M6: (c-a)(e+f)
            (sub_dicts(b, d), add_dicts(g, h)),           # M7: (b-d)(g+h)
        ]
        
        for k in range(7):
            x_k, y_k = sub_prods[k]
            rank_idx = t * 7 + k
            
            # Fill U from x_k (A coefficients)
            for (row, col), coeff in x_k.items():
                U[u_idx(row, col)][rank_idx] += coeff
            
            # Fill V from y_k (B coefficients)
            for (row, col), coeff in y_k.items():
                V[v_idx(row, col)][rank_idx] += coeff
            
            # Fill W from sub-contributions
            for (pr, pc, sub_coeff) in strassen_sub[k]:
                if (pr, pc) in Ct:
                    for (out_row, out_col, block_coeff) in Ct[(pr, pc)]:
                        W[w_idx(out_col, out_row)][rank_idx] += sub_coeff * block_coeff
    
    return U, V, W

# Build once at import time
_U, _V, _W = _build_decomposition()

def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    Returns precomputed factor matrices.
    """
    return {"U": _U, "V": _V, "W": _W}
# EVOLVE-BLOCK-END
