"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_block(U, V, W, i0, j0, k0, col_start, n, m, p):
    """
    Add Strassen decomposition for 2x2 A-block at (i0, j0), 2x2 B-block at (j0, k0).
    A indices: i0, i0+1 ; j0, j0+1
    B indices: j0, j0+1 ; k0, k0+1
    C indices: i0, i0+1 ; k0, k0+1
    W row index for C[i,k] is k*n+i
    """
    # Strassen: p1=(a+d)(e+h), p2=(c+d)(e+f), p3=(a+b)(g+h),
    # p4=d(h-f), p5=a(e-g), p6=(c-a)(f+h), p7=(b-d)(e+g)
    # C00=p1+p4-p5+p7, C01=p3+p5, C10=p2-p4, C11=p1-p2+p3+p6
    #
    # For tensor T[i,j,k]=1, each Strassen term is a rank-1 contribution.
    # Term t: U part = A_coeffs, V part = B_coeffs, W part = C_coeffs
    # where C_coeffs gives the signed contribution to each C output.
    
    # A local: [a=A(0,0), b=A(0,1), c=A(1,0), d=A(1,1)]
    # B local: [e=B(0,0), f=B(0,1), g=B(1,0), h=B(1,1)]
    # C local: [C00, C01, C10, C11]
    
    terms = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # p1 = (a+d)(e+h): C00+=1, C11+=1
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        # p2 = (c+d)(e+f): C10+=1, C11+=-1
        ([0,0,1,1], [1,1,0,0], [0,0,1,-1]),
        # p3 = (a+b)(g+h): C01+=1, C11+=1
        ([1,1,0,0], [0,0,1,1], [0,1,0,1]),
        # p4 = d(h-f): C00+=1, C10+=-1
        ([0,0,0,1], [0,-1,0,1], [1,0,-1,0]),
        # p5 = a(e-g): C00+=-1, C01+=1
        ([1,0,0,0], [1,0,-1,0], [-1,1,0,0]),
        # p6 = (c-a)(f+h): C11+=1
        ([-1,0,1,0], [0,1,0,1], [0,0,0,1]),
        # p7 = (b-d)(e+g): C00+=1
        ([0,1,0,-1], [1,0,1,0], [1,0,0,0]),
    ]
    
    col = col_start
    for t in range(7):
        a_coeffs = terms[t][0]
        b_coeffs = terms[t][1]
        c_coeffs = terms[t][2]
        
        for di in range(2):
            for dj in range(2):
                U[(i0+di)*m + (j0+dj)][col] = a_coeffs[di*2+dj]
        
        for dj in range(2):
            for dk in range(2):
                V[(j0+dj)*p + (k0+dk)][col] = b_coeffs[dj*2+dk]
        
        for di in range(2):
            for dk in range(2):
                W[(k0+dk)*n + (i0+di)][col] = c_coeffs[di*2+dk]
        
        col += 1
    
    return col

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 110

    # Group: i={0,1,2}, j={0,1,2,3}, k={0..7}
    # i-groups: {0,1} (size 2), {2} (size 1)
    # j-groups: {0,1} (size 2), {2,3} (size 2)
    # k-groups: {0,1}, {2,3}, {4,5}, {6,7} (size 2 each)
    #
    # For i-group size 2, j-group size 2, k-group size 2: Strassen rank 7
    # For i-group size 1, j-group size 2, k-group size 2: schoolbook rank 4
    #
    # Total: 2 (i-groups) * 2 (j-groups) * 4 (k-groups) = 16 sub-blocks
    # 8 Strassen (rank 7) + 8 schoolbook (rank 4) = 56 + 32 = 88

    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i0 in [0, 2]:
        i_size = 2 if i0 == 0 else 1
        for j0 in [0, 2]:
            for k0 in [0, 2, 4, 6]:
                if i_size == 2:
                    col = _strassen_block(U, V, W, i0, j0, k0, col, n, m, p)
                else:
                    for dj in range(2):
                        for dk in range(2):
                            U[i0*m + (j0+dj)][col] = 1
                            V[(j0+dj)*p + (k0+dk)][col] = 1
                            W[(k0+dk)*n + i0][col] = 1
                            col += 1
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
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
# EVOLVE-BLOCK-END
