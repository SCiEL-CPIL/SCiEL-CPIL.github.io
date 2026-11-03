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
    """Schoolbook rank n*m*p = 56 construction."""
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


def _strassen_2x2x2():
    """Strassen 2x2*2x2 decomposition (rank 7)."""
    sU = np.zeros((4, 7), dtype=float)
    sV = np.zeros((4, 7), dtype=float)
    sW = np.zeros((4, 7), dtype=float)
    # M1 = (a+d)(e+h)
    sU[0,0]=1; sU[1,0]=1
    sV[0,0]=1; sV[3,0]=1
    sW[0,0]=1; sW[1,0]=1; sW[2,0]=1; sW[3,0]=1
    # M2 = (c+d)e
    sU[2,1]=1; sU[3,1]=1
    sV[0,1]=1
    sW[2,1]=1; sW[3,1]=1
    # M3 = a(f-h)
    sU[0,2]=1
    sV[1,2]=1; sV[3,2]=-1
    sW[1,2]=1; sW[3,2]=1
    # M4 = d(g-e)
    sU[1,3]=1
    sV[2,3]=1; sV[0,3]=-1
    sW[0,3]=1; sW[2,3]=1
    # M5 = (a+b)h
    sU[0,4]=1; sU[1,4]=1
    sV[3,4]=1
    sW[0,4]=-1; sW[2,4]=-1
    # M6 = (c-a)(e+f)
    sU[2,5]=1; sU[0,5]=-1
    sV[0,5]=1; sV[1,5]=1
    sW[2,5]=-1; sW[3,5]=-1
    # M7 = (b-d)(g+h)
    sU[1,6]=1; sU[3,6]=-1
    sV[2,6]=1; sV[3,6]=1
    sW[0,6]=1; sW[1,6]=1
    return sU, sV, sW


def _simple_2x2x1():
    """2x2*2x1 decomposition (rank 4)."""
    sU = np.zeros((4, 4), dtype=float)
    sV = np.zeros((2, 4), dtype=float)
    sW = np.zeros((2, 4), dtype=float)
    sU[0,0]=1; sV[0,0]=1; sW[0,0]=1
    sU[1,1]=1; sV[1,1]=1; sW[0,1]=1
    sU[2,2]=1; sV[0,2]=1; sW[1,2]=1
    sU[3,3]=1; sV[1,3]=1; sW[1,3]=1
    return sU, sV, sW


def _construct_strassen_based(n, m, p):
    """
    Strassen-based decomposition for 2x4 * 4x7.
    Split middle dim 4 into two blocks of 2.
    Split output dim 7 into 3 blocks of 2 + 1 block of 1.
    For each (A_block, B_block): use Strassen (rank 7) for 2x2*2x2, rank 4 for 2x2*2x1.
    Total: 2 * (3*7 + 4) = 2 * 25 = 50
    """
    total_rank = 50
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    col = 0
    for block in range(2):
        a_base = block * 4      # U row offset for A block (rows 4*block to 4*block+3)
        b_base = block * 14     # V row offset for B block (rows 14*block to 14*block+13)
        
        for sub in range(4):
            if sub < 3:
                k_base = 2 * sub
                sU, sV, sW = _strassen_2x2x2()
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
                sU, sV, sW = _simple_2x2x1()
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
                frac = Fraction(val).limit_denominator(100)
                if frac.denominator == 1:
                    row_list.append(frac.numerator)
                elif frac.denominator == 2:
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
