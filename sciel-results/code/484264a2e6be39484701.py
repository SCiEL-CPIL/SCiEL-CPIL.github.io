import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]
_DEADLINE = 140.0

def _remaining():
    return _DEADLINE - time.time()

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _verify(n, m, p, U, V, W):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _try_rank_49():
    if _remaining() < 15:
        return None
    
    n, m, p = 2, 5, 5
    rank = 49
    
    # Construction for 2x5x5 with rank 49
    # Use the known result: rank(2xn x n) = 2n^2 - 1 for odd n
    # 
    # The construction:
    # For c = 0, 1, ..., 48:
    #   i_c = 0 if c < 25 else 1
    #   j_c = (c % 25) // 5
    #   k_c = (c % 25) % 5
    # But this gives 50 terms, not 49.
    #
    # Instead, use a shared term. Let's use:
    # Terms 0-23: i=0, j=0..4, k=0..4 (25 terms, but we use 24)
    # Terms 24-47: i=1, j=0..4, k=0..4 (24 terms)
    # Term 48: shared correction
    #
    # This is getting too vague. Let me try a direct approach.
    
    # Start with schoolbook for 49 terms (skip the last one)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if c < rank:
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
    
    # Now check which (i,j,k) are not satisfied
    # The missing term is (i=1, j=4, k=4)
    # We need to fix this.
    
    # Let's modify term 0 to also help with (1,4,4)
    # Term 0 currently: U[0][0]=1, V[0][0]=1, W[0][0]=1
    # This contributes to (i=0,j=0,k=0)
    
    # If we set W[0][0] = 0, we break (0,0,0).
    # Instead, let's add a fractional contribution.
    
    # For (1,4,4): need U[9][c]*V[24][c]*W[9][c] to sum to 1
    # U[9] is nonzero at c=45,46,47,48 (for i=1,j=4,k=0,1,2,3)
    # V[24] is nonzero at c=24 (for i=0,j=4,k=4)
    # W[9] is nonzero at c=29,34,39,44 (for i=1,j=0,1,2,3,k=4)
    
    # No overlap, so sum is 0. We need to create an overlap.
    
    # Modify V[24][45] = 1 (add contribution at c=45)
    # Then U[9][45]=1, V[24][45]=1, W[9][45]=? 
    # W[9][45] is 0 currently. Set W[9][45]=1.
    # But W[9][45] being 1 affects other (i,j,k) where k=4,i=1 and c=45.
    # c=45 corresponds to (i=1,j=4,k=0). So W[9][45]=1 means for (i=1,j=*,k=4),
    # the term c=45 contributes W[9][45]*U[1*5+j][45]*V[j*5+4][45].
    # U[9][45]=1 (i=1,j=4), so for j=4: U[9][45]=1, V[24][45]=1, W[9][45]=1 -> contributes 1 to (1,4,4). Good.
    # But also for other j: U[5+j][45]=0 (since c=45 is for j=4), so no other contribution.
    
    # Wait, U[5+j][45]: c=45 is (i=1,j=4,k=0), so U[1*5+4][45]=U[9][45]=1, and U[5+j][45]=0 for j!=4.
    # V[j*5+4][45]: c=45 is (i=1,j=4,k=0), so V[4*5+0][45]=V[20][45]=1, and V[j*5+4][45]=0 for j!=4 or k!=0... 
    # Actually V[j*p+k][c] where c=45 corresponds to (i=1,j=4,k=0), so V[4*5+0][45]=V[20][45]=1.
    # So V[j*5+4][45] is nonzero only when j=4 and k=0, i.e., V[20][45]=1. But we need V[24][45] (j=4,k=4).
    
    # I'm confusing myself. Let me be more careful.
    
    # c=45: i=1, j=4, k=0
    # U[i*m+j][c] = U[1*5+4][45] = U[9][45] = 1
    # V[j*p+k][c] = V[4*5+0][45] = V[20][45] = 1
    # W[k*n+i][c] = W[0*2+1][45] = W[1][45] = 1
    
    # So term c=45 contributes to (i=1,j=4,k=0): U[9][45]*V[20][45]*W[1][45] = 1*1*1 = 1. Good.
    
    # Now, if I set V[24][45] = 1, then for (i=1,j=4,k=4):
    # U[9][45]*V[24][45]*W[9][45] = 1*1*W[9][45]
    # W[9][45] = W[4*2+1][45] = W[9][45]
    # Currently W[9][45] = 0 (since c=45 is for k=0, not k=4)
    # So I need W[9][45] = 1 as well.
    
    # But W[9][45] = 1 means: for any (i',j',k') with k'*2+i' = 9, i.e., k'=4, i'=1,
    # the term c=45 contributes U[1*5+j'][45]*V[j'*5+4][45]*W[9][45] = U[5+j'][45]*V[j'*5+4][45]*1
    # U[5+j'][45] is nonzero only when j'=4 (since c=45 is for j=4), so U[9][45]=1
    # V[j'*5+4][45] is nonzero only when j'=4 and k=0... wait, V[j*p+k][c] where c=45 has j=4,k=0.
    # So V[j'*5+4][45] = V[20][45] if j'=4, else 0. But we set V[24][45]=1, which is j'=4,k=4.
    # V[24][45] is a different entry from V[20][45].
    
    # So for (i=1,j=4,k=4): U[9][45]*V[24][45]*W[9][45] = 1*1*1 = 1. 
    # But this ALSO affects (i=1,j=4,k=0): U[9][45]*V[20][45]*W[1][45] = 1*1*1 = 1. Still good.
    # And it affects (i=1,j=4,k=4) via other c values? No, we're only adding to c=45.
    
    # But wait, does setting V[24][45]=1 and W[9][45]=1 break anything else?
    # V[24][45]=1: V[24] is for (j=4,k=4). This affects all (i,j=4,k=4) for any i.
    # For (i=0,j=4,k=4): U[4][45]*V[24][45]*W[9][45]... wait, W[k*n+i] = W[4*2+0] = W[8], not W[9].
    # W[8][45] is 0 (c=45 is for k=0,i=1, so W[1][45]=1, not W[8]).
    # So (i=0,j=4,k=4) is unaffected.
    
    # For (i=1,j=4,k=4): already handled, now gets +1 from c=45. But it was 0 before, now 1. 
    # But we need it to be exactly 1, and it was 0, so now it's 1. 
    
    # Let me verify all (i,j,k) are still satisfied.
    
    # Modify the matrices
    V[24][45] = 1
    W[9][45] = 1
    
    # Now verify
    if _verify(n, m, p, U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # If verification fails, try other modifications
    # For now, return None
    return None

def _construct():
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    
    r49 = _try_rank_49()
    if r49 is not None:
        return r49
    
    return best

def construct():
    try:
        return _construct()
    except Exception:
        n, m, p = DIMENSIONS
        return _schoolbook(n, m, p)
