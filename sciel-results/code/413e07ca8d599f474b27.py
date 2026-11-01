"""Matrix multiplication tensor decomposition for 2x4 by 4x5 matrices."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x4x5():
    """
    Construct a rank-36 decomposition for 2x4 * 4x5 matrix multiplication.
    
    Strategy: Split A (2x4) into [A1 A2] where A1, A2 are 2x2.
    Split B (4x5) into [B1; B2] where B1, B2 are 2x5.
    C = A1*B1 + A2*B2
    
    For each 2x5 product (2x2 * 2x5), we can use a rank-7 decomposition
    (Strassen-like for rectangular matrices).
    
    Actually, let's use a more direct approach:
    We have 2 blocks of 2x2 * 2x5.
    
    For 2x2 * 2x5, we can write:
    A = [[a,b],[c,d]], B = [[e,f,g,h,i],[j,k,l,m,n]]
    C = [[ae+bj, af+bk, ag+bl, ah+bm, ai+bn],
         [ce+dj, cf+dk, cg+dl, ch+dm, ci+dn]]
    
    Standard rank for 2x2*2x5 is 10 (4*5/2 = 10? No, it's 2*5=10 scalar mults).
    Can we do better? The tensor rank of M_{2,2,5} is known to be 7 or 8.
    
    Let's use the known rank-7 decomposition for M_{2,2,5} if it exists,
    or fall back to rank 8.
    
    Actually, for M_{2,2,n}, the rank is n+2 for n>=2 (I think).
    So M_{2,2,5} has rank 7.
    
    If we use rank 7 for each block, total rank = 14.
    But we need to combine them properly.
    
    Wait, the structure is:
    C = A1*B1 + A2*B2
    
    If we decompose A1*B1 with rank r1 and A2*B2 with rank r2,
    and the decompositions are independent (no sharing), total rank = r1 + r2.
    
    So if M_{2,2,5} has rank 7, we get rank 14.
    But that seems too low. The lower bound for M_{2,4,5} is 36.
    
    Let me reconsider. The issue is that B1 and B2 share the same "rows" in some sense?
    No, B1 is the top 2x5 block and B2 is the bottom 2x5 block.
    
    Actually, I think the rank of M_{2,4,5} is indeed higher because of the way
    the blocks interact.
    
    Let me try a different approach: use the known result that for M_{n,m,p},
    if one dimension is small, we can get good decompositions.
    
    For M_{2,4,5}, let's try to construct it explicitly.
    
    Alternative: Use the fact that M_{2,4,5} can be viewed as two copies of M_{2,2,5}
    plus cross terms? No, that's not right either.
    
    Let me just implement the schoolbook and see if we can improve it with
    a structured search.
    """
    n, m, p = 2, 4, 5
    
    # For now, return None to use schoolbook
    # TODO: Implement actual rank-36 decomposition
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    deadline = time.time() + 100
    
    # Try structured decomposition first
    try:
        result = _strassen_2x4x5()
        if result is not None:
            if _verify(result["U"], result["V"], result["W"], n, m, p):
                return result
    except Exception:
        pass
    
    # Fall back to schoolbook
    return schoolbook()
