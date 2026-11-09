"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
from fractions import Fraction as F

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """Strassen rank-7 for 2x2. Returns (U, V, W) each 4x7."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    # Strassen's 7 products:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    # U rows: a00, a01, a10, a11
    # V rows: b00, b01, b10, b11
    # W rows: c00, c01, c10, c11
    # M1: U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,-1]
    U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=-1
    # M2: U=[0,0,1,1], V=[1,0,0,0], W=[0,0,0,1]
    U[2][1]=1; U[3][1]=1; V[0][1]=1; W[3][1]=1
    # M3: U=[1,0,0,0], V=[0,0,1,-1], W=[0,0,1,0]
    U[0][2]=1; V[2][2]=1; V[3][2]=-1; W[2][2]=1
    # M4: U=[0,0,0,1], V=[1,-1,0,0], W=[0,1,0,0]
    U[3][3]=1; V[0][3]=1; V[1][3]=-1; W[1][3]=1
    # M5: U=[1,1,0,0], V=[0,0,0,1], W=[1,0,0,0]
    U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=1
    # M6: U=[0,0,1,-1], V=[1,0,1,0], W=[0,0,-1,0]
    U[2][5]=1; U[3][5]=-1; V[0][5]=1; V[2][5]=1; W[2][5]=-1
    # M7: U=[0,1,0,-1], V=[0,1,0,1], W=[0,1,0,0]
    U[1][6]=1; U[3][6]=-1; V[1][6]=1; V[3][6]=1; W[1][6]=1
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matmul tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = F(0)
                for c in range(rank):
                    s += F(U[i*m+j][c]) * F(V[j*p+k][c]) * F(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def _try_block_decomposition(n, m, p):
    """
    Attempt to build a decomposition by splitting dimensions into 2x2 blocks
    and using Strassen for each block, plus handling remainders.
    This is a heuristic that may not yield a valid decomposition for 3x4x5,
    but we try it and verify.
    """
    # For 3x4x5, we can try:
    # Split n=3 into 2+1, m=4 into 2+2, p=5 into 2+3
    # This gets complicated. A simpler approach:
    # Use the known result that for small matrices, we can sometimes
    # combine Strassen with schoolbook for remainders.
    
    # Let's try a specific construction for 3x4x5:
    # We can view 3x4 as 2x4 + 1x4, and 4x5 as 4x2 + 4x3.
    # But this doesn't directly give a tensor decomposition.
    
    # Instead, let's try a rank-52 construction if known, or fall back.
    # Since I don't have a verified rank-52 decomposition hardcoded,
    # I'll return None to indicate failure.
    return None

def construct():
    """
    Constructs a tensor decomposition for matrix multiplication.
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    """
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = 145.0  # 5 seconds safety margin before 150s kill
    
    # First, try the block decomposition approach
    result = _try_block_decomposition(n, m, p)
    if result is not None:
        U, V, W = result
        # Verify the decomposition
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check time
                if time.time() > deadline:
                    # Return whatever we have so far? 
                    # For schoolbook, we must complete it to be valid.
                    # If we abort, it's invalid.
                    # Since it's fast, this won't trigger.
                    pass
                
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][column] = 1
                V[v_idx][column] = 1
                W[w_idx][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
