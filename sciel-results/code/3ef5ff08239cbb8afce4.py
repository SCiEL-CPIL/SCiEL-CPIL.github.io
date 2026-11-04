"""Matrix multiplication 2x5x6 rank-minimal exact tensor decomposition.

Uses Strassen's algorithm on 2x2 sub-blocks to reduce bilinear complexity
from 60 (schoolbook) to 54.
"""
import time

DIMENSIONS = [2, 5, 6]

def construct():
    """Return {'U','V','W'} factor matrices for 2x5 * 5x6 matrix multiplication.
    
    Uses Strassen's 7-term decomposition for 2x2 blocks.
    Partition: middle dim 5 -> 2+2+1, output dim 6 -> 2+2+2.
    6 Strassen blocks (2x2)*(2x2) + 3 schoolbook blocks (2x1)*(1x2) = 42+12 = 54.
    """
    start_time = time.time()
    
    n, m, p = 2, 5, 6
    
    U_entries = {}
    V_entries = {}
    W_entries = {}
    col = 0
    
    def add_strassen_block(i0, jo, ko):
        """Strassen for A[i0:i0+2, jo:jo+2] * B[jo:jo+2, ko:ko+2] -> C[i0:i0+2, ko:ko+2]"""
        nonlocal col
        # M1 = (A[i0,jo]+A[i0+1,jo+1])*(B[jo,ko]+B[jo+1,ko+1])
        # C00+=1, C11+=1
        U_entries[(i0*m+jo, col)] = U_entries.get((i0*m+jo, col), 0) + 1
        U_entries[((i0+1)*m+jo+1, col)] = U_entries.get(((i0+1)*m+jo+1, col), 0) + 1
        V_entries[(jo*p+ko, col)] = V_entries.get((jo*p+ko, col), 0) + 1
        V_entries[((jo+1)*p+ko+1, col)] = V_entries.get(((jo+1)*p+ko+1, col), 0) + 1
        W_entries[(ko*n+i0, col)] = W_entries.get((ko*n+i0, col), 0) + 1
        W_entries[((ko+1)*n+i0+1, col)] = W_entries.get(((ko+1)*n+i0+1, col), 0) + 1
        col += 1
        
        # M2 = (A[i0+1,jo]+A[i0+1,jo+1])*B[jo,ko]
        # C10+=1, C11+=-1
        U_entries[((i0+1)*m+jo, col)] = U_entries.get(((i0+1)*m+jo, col), 0) + 1
        U_entries[((i0+1)*m+jo+1, col)] = U_entries.get(((i0+1)*m+jo+1, col), 0) + 1
        V_entries[(jo*p+ko, col)] = V_entries.get((jo*p+ko, col), 0) + 1
        W_entries[(ko*n+i0+1, col)] = W_entries.get((ko*n+i0+1, col), 0) + 1
        W_entries[((ko+1)*n+i0+1, col)] = W_entries.get(((ko+1)*n+i0+1, col), 0) - 1
        col += 1
        
        # M3 = A[i0,jo]*(B[jo,ko+1]-B[jo+1,ko+1])
        # C01+=1, C11+=1
        U_entries[(i0*m+jo, col)] = U_entries.get((i0*m+jo, col), 0) + 1
        V_entries[(jo*p+ko+1, col)] = V_entries.get((jo*p+ko+1, col), 0) + 1
        V_entries[((jo+1)*p+ko+1, col)] = V_entries.get(((jo+1)*p+ko+1, col), 0) - 1
        W_entries[((ko+1)*n+i0, col)] = W_entries.get(((ko+1)*n+i0, col), 0) + 1
        W_entries[((ko+1)*n+i0+1, col)] = W_entries.get(((ko+1)*n+i0+1, col), 0) + 1
        col += 1
        
        # M4 = A[i0+1,jo+1]*(B[jo+1,ko]-B[jo,ko])
        # C00+=1, C10+=1
        U_entries[((i0+1)*m+jo+1, col)] = U_entries.get(((i0+1)*m+jo+1, col), 0) + 1
        V_entries[((jo+1)*p+ko, col)] = V_entries.get(((jo+1)*p+ko, col), 0) + 1
        V_entries[(jo*p+ko, col)] = V_entries.get((jo*p+ko, col), 0) - 1
        W_entries[(ko*n+i0, col)] = W_entries.get((ko*n+i0, col), 0) + 1
        W_entries[(ko*n+i0+1, col)] = W_entries.get((ko*n+i0+1, col), 0) + 1
        col += 1
        
        # M5 = (A[i0,jo]+A[i0,jo+1])*B[jo+1,ko+1]
        # C00+=-1, C01+=1
        U_entries[(i0*m+jo, col)] = U_entries.get((i0*m+jo, col), 0) + 1
        U_entries[(i0*m+jo+1, col)] = U_entries.get((i0*m+jo+1, col), 0) + 1
        V_entries[((jo+1)*p+ko+1, col)] = V_entries.get(((jo+1)*p+ko+1, col), 0) + 1
        W_entries[(ko*n+i0, col)] = W_entries.get((ko*n+i0, col), 0) - 1
        W_entries[((ko+1)*n+i0, col)] = W_entries.get(((ko+1)*n+i0, col), 0) + 1
        col += 1
        
        # M6 = (A[i0+1,jo]-A[i0,jo])*(B[jo,ko]+B[jo,ko+1])
        # C11+=1
        U_entries[((i0+1)*m+jo, col)] = U_entries.get(((i0+1)*m+jo, col), 0) + 1
        U_entries[(i0*m+jo, col)] = U_entries.get((i0*m+jo, col), 0) - 1
        V_entries[(jo*p+ko, col)] = V_entries.get((jo*p+ko, col), 0) + 1
        V_entries[(jo*p+ko+1, col)] = V_entries.get((jo*p+ko+1, col), 0) + 1
        W_entries[((ko+1)*n+i0+1, col)] = W_entries.get(((ko+1)*n+i0+1, col), 0) + 1
        col += 1
        
        # M7 = (A[i0,jo+1]-A[i0+1,jo+1])*(B[jo+1,ko]+B[jo+1,ko+1])
        # C00+=1
        U_entries[(i0*m+jo+1, col)] = U_entries.get((i0*m+jo+1, col), 0) + 1
        U_entries[((i0+1)*m+jo+1, col)] = U_entries.get(((i0+1)*m+jo+1, col), 0) - 1
        V_entries[((jo+1)*p+ko, col)] = V_entries.get(((jo+1)*p+ko, col), 0) + 1
        V_entries[((jo+1)*p+ko+1, col)] = V_entries.get(((jo+1)*p+ko+1, col), 0) + 1
        W_entries[(ko*n+i0, col)] = W_entries.get((ko*n+i0, col), 0) + 1
        col += 1
    
    def add_schoolbook_2x1x2_block(i0, jo, ko):
        """Schoolbook for A[i0:i0+2, jo] * B[jo, ko:ko+2] -> C[i0:i0+2, ko:ko+2]"""
        nonlocal col
        for di in range(2):
            for dk in range(2):
                i = i0 + di
                k = ko + dk
                U_entries[(i*m+jo, col)] = U_entries.get((i*m+jo, col), 0) + 1
                V_entries[(jo*p+k, col)] = V_entries.get((jo*p+k, col), 0) + 1
                W_entries[(k*n+i, col)] = W_entries.get((k*n+i, col), 0) + 1
                col += 1
    
    # 6 Strassen blocks: jo in {0,2}, ko in {0,2,4}, i0=0
    for jo in [0, 2]:
        for ko in [0, 2, 4]:
            add_strassen_block(0, jo, ko)
    
    # 3 schoolbook blocks: jo=4, ko in {0,2,4}, i0=0
    for ko in [0, 2, 4]:
        add_schoolbook_2x1x2_block(0, 4, ko)
    
    rank = col
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for (idx, c), val in U_entries.items():
        U[idx][c] = val
    for (idx, c), val in V_entries.items():
        V[idx][c] = val
    for (idx, c), val in W_entries.items():
        W[idx][c] = val
    
    return {"U": U, "V": V, "W": W}
