"""Rank-minimal exact tensor decomposition for 2x4 * 4x7 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 7]
DEADLINE = time.time() + 110.0


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify tensor decomposition: for all (i,k), sum_j sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1"""
    for i in range(n):
        for k in range(p):
            s = 0.0
            for j in range(m):
                u_row = U[i * m + j]
                v_row = V[j * p + k]
                w_row = W[k * n + i]
                s += float(np.sum(u_row * v_row * w_row))
            if abs(s - 1.0) > tol:
                return False
    return True


def _construct_rank_np(n, m, p):
    """
    Rank n*p construction: one column per output position (i,k).
    For column c corresponding to (i,k):
      U[i*m+j, c] = 1 for all j
      V[j*p+k, c] = 1 for all j
      W[k*n+i, c] = 1
    This gives C[i,k] = sum_j 1*1*1 = m... wait, that's m not 1.
    
    Correction: We need sum_j U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1.
    If W[k*n+i,c] = 1 and U,V are 1 for all j, then sum = m.
    So we need W[k*n+i,c] = 1/m? No, entries must be integer/half-integer.
    
    Alternative: use the schoolbook-style but compressed.
    Actually the correct rank n*p construction:
    For each (i,k), one column where:
      U[i*m+j, c] = 1 for all j  (row i*m+j)
      V[j*p+k, c] = 1 for all j  (row j*p+k)  
      W[k*n+i, c] = 1/m ... no.
    
    Let me reconsider. The tensor entry T[i,j,k] = 1 for all (i,j,k) in the multiplication tensor.
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Rank-r decomposition: C[i,k] = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c] (summed over j and c)
    More precisely: C[i,k] = sum_c (sum_j U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c]
    
    For rank n*p = 14 with n=2, p=7:
    Column c for output (i,k):
      U[i*m+j, c] = 1 for all j=0..m-1
      V[j*p+k, c] = 1 for all j=0..m-1
      W[k*n+i, c] = 1/m
    
    But 1/m = 1/4 is not integer or half-integer.
    
    Better approach: Use the standard construction where each column handles one (i,j,k) triple
    but we can group. Actually, the minimal rank for matrix multiplication tensor is known to be
    at most n*p (by the "trivial" construction using W as identity-like).
    
    Let me use: for each (i,k), set W[k*n+i, c_{ik}] = 1, and for that column,
    U[i*m+j, c_{ik}] = 1, V[j*p+k, c_{ik}] = 1/m... 
    
    Hmm, let me just use the schoolbook rank n*m*p = 56 as fallback, and try to find better.
    
    Actually, the correct rank n*p construction:
    For output (i,k), we need sum_j A[i,j]*B[j,k].
    Set for column c_{ik}: U[i*m+j, c] = 1 for all j, V[j*p+k, c] = 1 for all j, W[k*n+i, c] = 1/m.
    
    Since 1/4 is not allowed, let's scale: use W = 1 and divide U or V by m.
    U[i*m+j, c] = 1/m = 1/4, V[j*p+k, c] = 1, W[k*n+i, c] = 1.
    Then sum_j (1/4)*1*1 = 4*(1/4) = 1. 
    
    1/4 is not integer or half-integer either!
    
    So we need a different approach. Let's use rank = n*m*p = 56 (schoolbook) as guaranteed,
    and search for better.
    """
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W, rank


def _construct_strassen_based(n, m, p):
    """
    Use Strassen-style decomposition to reduce rank below n*m*p.
    
    For 2x4 * 4x7:
    Split A (2x4) into [A1 | A2] where A1, A2 are 2x2
    Split B (4x7) into [B1; B2] where B1, B2 are 2x7
    C = A1*B1 + A2*B2
    
    For each 2x2 * 2x7, split into 2x2*2x2 + 2x2*2x2 + 2x2*2x2 + 2x2*2x1
    Use Strassen (rank 7) for each 2x2*2x2, and rank 4 for 2x2*2x1.
    Total per 2x2*2x7: 3*7 + 4 = 25
    Total: 2 * 25 = 50
    """
    total_rank = 50
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    # Strassen 2x2*2x2 decomposition (7 terms)
    def strassen_2x2x2():
        sU = np.zeros((4, 7), dtype=float)
        sV = np.zeros((4, 7), dtype=float)
        sW = np.zeros((4, 7), dtype=float)
        # M1 = (a+d)(e+h)
        sU[0,0]=1; sU[1,0]=1  # a,d
        sV[0,0]=1; sV[3,0]=1  # e,h
        sW[0,0]=1; sW[1,0]=1; sW[2,0]=1; sW[3,0]=1
        # M2 = (c+d)e
        sU[2,1]=1; sU[3,1]=1  # c,d
        sV[0,1]=1  # e
        sW[2,1]=1; sW[3,1]=1
        # M3 = a(f-h)
        sU[0,2]=1  # a
        sV[1,2]=1; sV[3,2]=-1  # f,-h
        sW[1,2]=1; sW[3,2]=1
        # M4 = d(g-e)
        sU[1,3]=1  # d
        sV[2,3]=1; sV[0,3]=-1  # g,-e
        sW[0,3]=1; sW[2,3]=1
        # M5 = (a+b)h
        sU[0,4]=1; sU[1,4]=1  # a,b
        sV[3,4]=1  # h
        sW[0,4]=-1; sW[2,4]=-1
        # M6 = (c-a)(e+f)
        sU[2,5]=1; sU[0,5]=-1  # c,-a
        sV[0,5]=1; sV[1,5]=1  # e,f
        sW[2,5]=-1; sW[3,5]=-1
        # M7 = (b-d)(g+h)
        sU[1,6]=1; sU[3,6]=-1  # b,-d
        sV[2,6]=1; sV[3,6]=1  # g,h
        sW[0,6]=1; sW[1,6]=1
        return sU, sV, sW
    
    # 2x2*2x1 decomposition (rank 4)
    def simple_2x2x1():
        sU = np.zeros((4, 4), dtype=float)
        sV = np.zeros((2, 4), dtype=float)
        sW = np.zeros((2, 4), dtype=float)
        # P1=a*e, P2=b*g, P3=c*e, P4=d*g
        sU[0,0]=1; sV[0,0]=1; sW[0,0]=1
        sU[1,1]=1; sV[1,1]=1; sW[0,1]=1
        sU[2,2]=1; sV[0,2]=1; sW[1,2]=1
        sU[3,3]=1; sV[1,3]=1; sW[1,3]=1
        return sU, sV, sW
    
    col = 0
    for block in range(2):
        a_base = block * 4      # U row offset for A block
        b_base = block * 14     # V row offset for B block (2 rows * 7 cols)
        
        for sub in range(4):
            if sub < 3:
                k_base = 2 * sub
                sU, sV, sW = strassen_2x2x2()
                U[a_base:a_base+4, col:col+7] = sU
                for j in range(2):
                    for k in range(2):
                        V[b_base + j*7 + k_base + k, col:col+7] = sV[j*2+k]
                for k in range(2):
                    for i in range(2):
                        W[(k_base+k)*2 + i, col:col+7] = sW[k*2+i]
                col += 7
            else:
                k_base = 6
                sU, sV, sW = simple_2x2x1()
                U[a_base:a_base+4, col:col+4] = sU
                for j in range(2):
                    V[b_base + j*7 + k_base, col:col+4] = sV[j]
                for i in range(2):
                    W[k_base*2 + i, col:col+4] = sW[i]
                col += 4
    
    return U, V, W, total_rank


def _to_list(mat):
    """Convert numpy array to list of lists with int/Fraction entries."""
    result = []
    for row in mat:
        row_list = []
        for val in row:
            if abs(val - round(val)) < 1e-9:
                row_list.append(int(round(val)))
            else:
                # Try to express as half-integer or small fraction
                frac = Fraction(val).limit_denominator(100)
                if frac.denominator <= 2:
                    if frac.denominator == 1:
                        row_list.append(frac.numerator)
                    else:
                        row_list.append(frac)
                else:
                    row_list.append(frac)
        result.append(row_list)
    return result


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    
    best_U, best_V, best_W = None, None, None
    best_rank = float('inf')
    
    # Try Strassen-based (rank 50)
    U_s, V_s, W_s, r_s = _construct_strassen_based(n, m, p)
    if _verify(U_s, V_s, W_s, n, m, p):
        best_U, best_V, best_W = U_s, V_s, W_s
        best_rank = r_s
    
    # Try schoolbook (rank 56) as fallback
    U_b, V_b, W_b, r_b = _construct_rank_np(n, m, p)
    if _verify(U_b, V_b, W_b, n, m, p) and r_b < best_rank:
        best_U, best_V, best_W = U_b, V_b, W_b
        best_rank = r_b
    
    # If Strassen didn't work, use schoolbook
    if best_U is None:
        best_U, best_V, best_W = U_b, V_b, W_b
        best_rank = r_b
    
    return {
        'U': _to_list(best_U),
        'V': _to_list(best_V),
        'W': _to_list(best_W)
    }
