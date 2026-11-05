import time
import numpy as np

DIMENSIONS = [2, 5, 5]
_DEADLINE = 135.0

def _remaining():
    return _DEADLINE - time.time()

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

def _strassen_block(j0, k0):
    """
    Returns list of (U, V, W) for a 2x2x2 Strassen block.
    j indices: j0, j0+1
    k indices: k0, k0+1
    i indices: 0, 1
    """
    terms = []
    # Strassen formulas for C = A*B where A is 2x2, B is 2x2, C is 2x2
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    # M1 = A[0,0]*(B[0,0]+B[1,1])
    # M2 = (A[0,0]+A[1,0])*B[1,1]
    # M3 = A[1,0]*(B[0,0]+B[0,1])
    # M4 = A[0,1]*(B[1,0]+B[1,1])
    # M5 = (A[0,0]+A[0,1])*B[0,0]
    # M6 = (A[1,0]+A[0,1])*(B[0,0]+B[1,1])
    # M7 = (A[0,0]+A[1,0])*(B[1,0]+B[0,1])
    
    # Each term: (u_dict, v_dict, w_dict)
    # u_dict: {(i,j): val}
    # v_dict: {(j,k): val}
    # w_dict: {(k,i): val}
    
    # M1
    u = {(0, j0): 1}
    v = {(j0, k0): 1, (j0+1, k0+1): 1}
    w = {(k0, 0): 1, (k0+1, 1): 1}
    terms.append((u, v, w))
    
    # M2
    u = {(0, j0): 1, (1, j0): 1}
    v = {(j0+1, k0+1): 1}
    w = {(k0+1, 1): 1}
    # C[1,0] = M2 + M4, so M2 contributes to C[1,0] => i=1,k=0 => W[k0,1]=1
    # C[1,1] = M1 - M2 + M3 + M6, so M2 contributes to C[1,1] => i=1,k=1 => W[k0+1,1]=-1
    w = {(k0, 1): 1, (k0+1, 1): -1}
    terms.append((u, v, w))
    
    # M3
    u = {(1, j0): 1}
    v = {(j0, k0): 1, (j0, k0+1): 1}
    # C[0,1] = M3 + M5, so M3 contributes to C[0,1] => i=0,k=1 => W[k0+1,0]=1
    # C[1,1] = M1 - M2 + M3 + M6, so M3 contributes to C[1,1] => i=1,k=1 => W[k0+1,1]=1
    w = {(k0+1, 0): 1, (k0+1, 1): 1}
    terms.append((u, v, w))
    
    # M4
    u = {(0, j0+1): 1}
    v = {(j0+1, k0): 1, (j0+1, k0+1): 1}
    # C[0,0] = M1 + M4 - M5 + M7, so M4 contributes to C[0,0] => i=0,k=0 => W[k0,0]=1
    # C[1,0] = M2 + M4, so M4 contributes to C[1,0] => i=1,k=0 => W[k0,1]=1
    w = {(k0, 0): 1, (k0, 1): 1}
    terms.append((u, v, w))
    
    # M5
    u = {(0, j0): 1, (0, j0+1): 1}
    v = {(j0, k0): 1}
    # C[0,0] = M1 + M4 - M5 + M7, so M5 contributes to C[0,0] => i=0,k=0 => W[k0,0]=-1
    # C[0,1] = M3 + M5, so M5 contributes to C[0,1] => i=0,k=1 => W[k0+1,0]=1
    w = {(k0, 0): -1, (k0+1, 0): 1}
    terms.append((u, v, w))
    
    # M6
    u = {(1, j0): 1, (0, j0+1): 1}
    v = {(j0, k0): 1, (j0+1, k0+1): 1}
    # C[1,1] = M1 - M2 + M3 + M6, so M6 contributes to C[1,1] => i=1,k=1 => W[k0+1,1]=1
    w = {(k0+1, 1): 1}
    terms.append((u, v, w))
    
    # M7
    u = {(0, j0): 1, (1, j0): 1}
    v = {(j0+1, k0): 1, (j0, k0+1): 1}
    # C[0,0] = M1 + M4 - M5 + M7, so M7 contributes to C[0,0] => i=0,k=0 => W[k0,0]=1
    w = {(k0, 0): 1}
    terms.append((u, v, w))
    
    return terms

def _construct():
    n, m, p = DIMENSIONS
    all_terms = []
    
    # Strassen blocks for (j0,j1)x(k0,k1) where j in {0,1}, k in {0,1}
    all_terms.extend(_strassen_block(0, 0))
    # Strassen blocks for (j2,j3)x(k2,k3) where j in {2,3}, k in {2,3}
    all_terms.extend(_strassen_block(2, 2))
    
    # 2x1x1 blocks: j in {0,1}, k=4
    # Term 1: U[0,0,c1]=1, V[0,4,c1]=1, W[4,0,c1]=1
    # Term 2: U[1,0,c2]=1, V[0,4,c2]=1, W[4,1,c2]=1
    # Wait, we need U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all i,j in block
    # For j in {0,1}, k=4, i in {0,1}:
    # We need 4 equations but can use 2 terms:
    # c1: U[0,0]=1, V[0,4]=1, W[4,0]=1; U[1,0]=0, V[0,4]=0? No...
    # Actually each term is a rank-1 slice. Let me think again.
    # For the sub-block j∈{0,1}, k={4}, i∈{0,1}:
    # We need: for each (i,j,k) in this sub-block, sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
    # That's 2*2*1 = 4 equations.
    # Term a: U[0,0,a]=1, V[0,4,a]=1, W[4,0,a]=1, others 0 => gives 1 at (i=0,j=0,k=4)
    # Term b: U[1,0,b]=1, V[0,4,b]=1, W[4,1,b]=1, others 0 => gives 1 at (i=1,j=0,k=4)
    # Term c: U[0,1,c]=1, V[1,4,c]=1, W[4,0,c]=1, others 0 => gives 1 at (i=0,j=1,k=4)
    # Term d: U[1,1,d]=1, V[1,4,d]=1, W[4,1,d]=1, others 0 => gives 1 at (i=1,j=1,k=4)
    # That's 4 terms. Can we do better?
    # Try 2 terms:
    # Term x: U[0,0,x]=1, U[0,1,x]=1, V[0,4,x]=1, V[1,4,x]=1, W[4,0,x]=1
    #   => (i=0,j=0,k=4): 1*1*1=1 ✓; (i=0,j=1,k=4): 1*1*1=1 ✓; (i=1,j=0,k=4): 0*1*0=0; (i=1,j=1,k=4): 0*1*0=0
    # Term y: U[1,0,y]=1, U[1,1,y]=1, V[0,4,y]=1, V[1,4,y]=1, W[4,1,y]=1
    #   => (i=1,j=0,k=4): 1*1*1=1 ✓; (i=1,j=1,k=4): 1*1*1=1 ✓; (i=0,j=0,k=4): 0*1*0=0; (i=0,j=1,k=4): 0*1*0=0
    # Total: all 4 are 1. 
    u = {(0,0): 1, (0,1): 1}
    v = {(0,4): 1, (1,4): 1}
    w = {(4,0): 1}
    all_terms.append((u, v, w))
    u = {(1,0): 1, (1,1): 1}
    v = {(0,4): 1, (1,4): 1}
    w = {(4,1): 1}
    all_terms.append((u, v, w))
    
    # 2x1x1 blocks: j in {2,3}, k=4
    u = {(0,2): 1, (0,3): 1}
    v = {(2,4): 1, (3,4): 1}
    w = {(4,0): 1}
    all_terms.append((u, v, w))
    u = {(1,2): 1, (1,3): 1}
    v = {(2,4): 1, (3,4): 1}
    w = {(4,1): 1}
    all_terms.append((u, v, w))
    
    # 1x2x1 blocks: j=4, k in {0,1}
    u = {(0,4): 1, (1,4): 1}
    v = {(4,0): 1, (4,1): 1}
    w = {(0,0): 1, (1,0): 1}
    # Check: (i=0,j=4,k=0): 1*1*1=1; (i=0,j=4,k=1): 1*1*1=1; (i=1,j=4,k=0): 1*1*1=1; (i=1,j=4,k=1): 1*1*1=1
    # But wait, this also contributes to other (i,k) pairs if W has non-zero there. W only has (0,0) and (1,0) so only i=0.
    # Hmm, I need W[4,0] for k=0,i=0 and W[4,1] for k=1,i=0? No, W[k,i].
    # W[(k,i)]: W[0,0]=1 means k=0,i=0. W[1,0]=1 means k=1,i=0.
    # So for i=1, W[k,1] is 0 for all k. That's wrong.
    # Let me redo:
    # For j=4, k in {0,1}, i in {0,1}:
    # Term: U[i,4,c]=1 for all i, V[4,k,c]=1 for k in {0,1}, W[k,i,c]=1 for all k,i
    # That's 1 term that covers all 4 points.
    u = {(0,4): 1, (1,4): 1}
    v = {(4,0): 1, (4,1): 1}
    w = {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}
    all_terms.append((u, v, w))
    
    # 1x2x1 blocks: j=4, k in {2,3}
    u = {(0,4): 1, (1,4): 1}
    v = {(4,2): 1, (4,3): 1}
    w = {(2,0): 1, (2,1): 1, (3,0): 1, (3,1): 1}
    all_terms.append((u, v, w))
    
    # 1x1x1 block: j=4, k=4
    u = {(0,4): 1, (1,4): 1}
    v = {(4,4): 1}
    w = {(4,0): 1, (4,1): 1}
    all_terms.append((u, v, w))
    
    R = len(all_terms)
    U = [[0]*R for _ in range(n*m)]
    V = [[0]*R for _ in range(m*p)]
    W = [[0]*R for _ in range(p*n)]
    
    for c, (u, v, w) in enumerate(all_terms):
        for (i, j), val in u.items():
            U[i*m + j][c] += val
        for (j, k), val in v.items():
            V[j*p + k][c] += val
        for (k, i), val in w.items():
            W[k*n + i][c] += val
    
    return {"U": U, "V": V, "W": W}

def construct():
    try:
        result = _construct()
        n, m, p = DIMENSIONS
        if _verify(n, m, p, result['U'], result['V'], result['W']):
            return result
        else:
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
    except Exception:
        n, m, p = DIMENSIONS
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
