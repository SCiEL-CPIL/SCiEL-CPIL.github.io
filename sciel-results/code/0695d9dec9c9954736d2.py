import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = 3, 4, 6
    
    # Split: i=3 as 2+1, j=4 as 2+2, k=6 as 2+2+2
    # i-blocks: I0={0,1}, I1={2}
    # j-blocks: J0={0,1}, J1={2,3}
    # k-blocks: K0={0,1}, K1={2,3}, K2={4,5}
    
    # For each (i-block, j-block, k-block), we have a sub-problem.
    # (2,2,2) sub-problems: 6 of them, rank 7 each
    # (1,2,2) sub-problems: 6 of them, rank 4 each
    # Total rank: 6*7 + 6*4 = 42 + 24 = 66
    
    rank = 66
    U = np.zeros((n*m, rank))  # (12, 66)
    V = np.zeros((m*p, rank))  # (24, 66)
    W = np.zeros((p*n, rank))  # (18, 66)
    
    col = 0
    
    # Strassen coefficients for (2,2,2):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # For a (2,2,2) sub-problem with i-offset io, j-offset jo, k-offset ko:
    # A indices: (io+0, jo+0), (io+0, jo+1), (io+1, jo+0), (io+1, jo+1)
    # B indices: (jo+0, ko+0), (jo+0, ko+1), (jo+1, ko+0), (jo+1, ko+1)
    # C indices: (io+0, ko+0), (io+0, ko+1), (io+1, ko+0), (io+1, ko+1)
    
    # U for A: row index = i*m + j
    # V for B: row index = j*p + k
    # W for C: row index = k*n + i (trace convention W[k*n+i, r])
    
    def add_strassen(io, jo, ko, col):
        # M1 = (A00+A11)(B00+B11)
        U[io*m + jo, col] += 1       # A[io,jo]
        U[(io+1)*m + (jo+1), col] += 1  # A[io+1,jo+1]
        V[jo*p + ko, col] += 1       # B[jo,ko]
        V[(jo+1)*p + (ko+1), col] += 1  # B[jo+1,ko+1]
        # C00 += M1, C11 += M1
        W[ko*n + io, col] += 1       # C[io,ko]
        W[(ko+1)*n + (io+1), col] += 1  # C[io+1,ko+1]
        
        col += 1
        # M2 = (A10+A11)B00
        U[io*m + (jo+1), col] += 1   # A[io,jo+1]
        U[(io+1)*m + (jo+1), col] += 1  # A[io+1,jo+1]
        V[jo*p + ko, col] += 1       # B[jo,ko]
        # C10 += M2
        W[ko*n + (io+1), col] += 1   # C[io+1,ko]
        
        col += 1
        # M3 = A00(B10-B11)
        U[io*m + jo, col] += 1       # A[io,jo]
        V[(jo+1)*p + ko, col] += 1   # B[jo+1,ko]
        V[(jo+1)*p + (ko+1), col] -= 1  # B[jo+1,ko+1]
        # C01 += M3
        W[(ko+1)*n + io, col] += 1   # C[io,ko+1]
        
        col += 1
        # M4 = A11(B01-B00)
        U[(io+1)*m + (jo+1), col] += 1  # A[io+1,jo+1]
        V[jo*p + (ko+1), col] += 1   # B[jo,ko+1]
        V[jo*p + ko, col] -= 1       # B[jo,ko]
        # C00 += M4, C10 += M4
        W[ko*n + io, col] += 1       # C[io,ko]
        W[ko*n + (io+1), col] += 1   # C[io+1,ko]
        
        col += 1
        # M5 = (A00+A01)B11
        U[io*m + jo, col] += 1       # A[io,jo]
        U[io*m + (jo+1), col] += 1   # A[io,jo+1]
        V[(jo+1)*p + (ko+1), col] += 1  # B[jo+1,ko+1]
        # C00 -= M5, C01 += M5
        W[ko*n + io, col] -= 1       # C[io,ko]
        W[(ko+1)*n + io, col] += 1   # C[io,ko+1]
        
        col += 1
        # M6 = (A10-A00)(B00+B01)
        U[io*m + (jo+1), col] += 1   # A[io,jo+1]
        U[io*m + jo, col] -= 1       # A[io,jo]
        V[jo*p + ko, col] += 1       # B[jo,ko]
        V[jo*p + (ko+1), col] += 1   # B[jo,ko+1]
        # C11 += M6
        W[(ko+1)*n + (io+1), col] += 1  # C[io+1,ko+1]
        
        col += 1
        # M7 = (A01-A11)(B10+B11)
        U[io*m + (jo+1), col] += 1   # A[io,jo+1]
        U[(io+1)*m + (jo+1), col] -= 1  # A[io+1,jo+1]
        V[(jo+1)*p + ko, col] += 1   # B[jo+1,ko]
        V[(jo+1)*p + (ko+1), col] += 1  # B[jo+1,ko+1]
        # C00 += M7
        W[ko*n + io, col] += 1       # C[io,ko]
        
        return col
    
    def add_rank4(io, jo, ko, col):
        # (1,2,2) sub-problem: C[io,ko] = A[io,jo]*B[jo,ko] + A[io,jo+1]*B[jo+1,ko]
        # C[io,ko+1] = A[io,jo]*B[jo,ko+1] + A[io,jo+1]*B[jo+1,ko+1]
        # P1 = A[io,jo]*B[jo,ko]
        U[io*m + jo, col] += 1
        V[jo*p + ko, col] += 1
        W[ko*n + io, col] += 1
        col += 1
        # P2 = A[io,jo]*B[jo,ko+1]
        U[io*m + jo, col] += 1
        V[jo*p + (ko+1), col] += 1
        W[(ko+1)*n + io, col] += 1
        col += 1
        # P3 = A[io,jo+1]*B[jo+1,ko]
        U[io*m + (jo+1), col] += 1
        V[(jo+1)*p + ko, col] += 1
        W[ko*n + io, col] += 1
        col += 1
        # P4 = A[io,jo+1]*B[jo+1,ko+1]
        U[io*m + (jo+1), col] += 1
        V[(jo+1)*p + (ko+1), col] += 1
        W[(ko+1)*n + io, col] += 1
        col += 1
        return col
    
    # (2,2,2) sub-problems: io in {0, 2}, jo in {0, 2}, ko in {0, 2, 4}
    for io in [0, 2]:
        for jo in [0, 2]:
            for ko in [0, 2, 4]:
                col = add_strassen(io, jo, ko, col)
    
    # (1,2,2) sub-problems: io=2, jo in {0, 2}, ko in {0, 2, 4}
    for jo in [0, 2]:
        for ko in [0, 2, 4]:
            col = add_rank4(2, jo, ko, col)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
