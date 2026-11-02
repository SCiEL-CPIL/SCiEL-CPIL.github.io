"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 120  # safety margin
    n, m, p = DIMENSIONS
    
    def verify(U, V, W, rank):
        """Verify the decomposition is exact."""
        # Build the tensor T
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1
        
        # Compute the decomposition
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                    if abs(val - T[a,b,c]) > 1e-9:
                        return False
        return True
    
    # Try rank 53: 7 pairs of Strassen + 1 schoolbook column
    # Strassen for 2x2 x 2x2:
    # A=[[a,b],[c,d]], B=[[e,f],[g,h]]
    # P1=(a+d)(e+h), P2=(c+d)(e+f), P3=(a-b)(g+h), P4=(a+d)(f-h)
    # P5=(a+c)(e-g), P6=(b+d)(f-g), P7=(a-c)(e+h)
    # C00=P1-P4+P6-P7, C01=P3+P4, C10=P2-P5+P7, C11=P1-P2+P3+P5
    
    # Let me just verify Strassen numerically first
    # A=[[1,2],[3,4]], B=[[5,6],[7,8]], AB=[[19,22],[43,50]]
    a,b,c,d = 1,2,3,4
    e,f,g,h = 5,6,7,8
    P1=(a+d)*(e+h); P2=(c+d)*(e+f); P3=(a-b)*(g+h); P4=(a+d)*(f-h)
    P5=(a+c)*(e-g); P6=(b+d)*(f-g); P7=(a-c)*(e+h)
    C00=P1-P4+P6-P7; C01=P3+P4; C10=P2-P5+P7; C11=P1-P2+P3+P5
    # Expected: C00=19, C01=22, C10=43, C11=50
    # P1=65,P2=77,P3=-15,P4=-5,P5=3,P6=24,P7=-28
    # C00=65+5+24+28=122 (WRONG, should be 19)
    # So this Strassen formula is wrong.
    
    # Let me try to find the correct one by solving
    # We need 7 bilinear forms that reproduce the 4 outputs.
    # This is hard to do by hand. Let me just use schoolbook.
    
    # Schoolbook: rank = n*m*p = 60
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
