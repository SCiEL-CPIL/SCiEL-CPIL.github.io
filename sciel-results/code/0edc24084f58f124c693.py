import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 8]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify tensor decomposition: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    where T[i*m+j, j'*p+k, k'*n+i'] = 1 iff j==j' and i==i' and k==k'
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for k2 in range(p):
                            a = i * m + j
                            b = j2 * p + k
                            c = k2 * n + i2
                            expected = 1.0 if (j == j2 and i == i2 and k == k2) else 0.0
                            actual = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                            if abs(actual - expected) > 1e-9:
                                return False
    return True

def _strassen_2x2():
    """Strassen 2x2x2: U(4,7), V(4,7), W(4,7) with trace convention W[k*2+i]."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    # M1=(A11-A10)(B00+B01): U[3]=1,U[2]=-1; V[0]=1,V[1]=1; C: W[0]=1,W[2]=-1,W[3]=1
    U[3][0]=1; U[2][0]=-1; V[0][0]=1; V[1][0]=1; W[0][0]=1; W[2][0]=-1; W[3][0]=1
    # M2=(A10+A11)(B10+B11): U[2]=1,U[3]=1; V[2]=1,V[3]=1; C: W[2]=1,W[3]=1
    U[2][1]=1; U[3][1]=1; V[2][1]=1; V[3][1]=1; W[2][1]=1; W[3][1]=1
    # M3=(A00+A10)(B00): U[0]=1,U[2]=1; V[0]=1; C: W[2]=1,W[3]=1
    U[0][2]=1; U[2][2]=1; V[0][2]=1; W[2][2]=1; W[3][2]=1
    # M4=(A11+A01)(B11): U[3]=1,U[1]=1; V[3]=1; C: W[0]=1,W[2]=-1
    U[3][3]=1; U[1][3]=1; V[3][3]=1; W[0][3]=1; W[2][3]=-1
    # M5=(A00-A10)(B01+B11): U[0]=1,U[2]=-1; V[1]=1,V[3]=1; C: W[0]=-1,W[2]=1
    U[0][4]=1; U[2][4]=-1; V[1][4]=1; V[3][4]=1; W[0][4]=-1; W[2][4]=1
    # M6=(A01+A11)(B10-B00): U[1]=1,U[3]=1; V[2]=1,V[0]=-1; C: W[2]=-1,W[3]=-1
    U[1][5]=1; U[3][5]=1; V[2][5]=1; V[0][5]=-1; W[2][5]=-1; W[3][5]=-1
    # M7=(A00+A01)(B00-B10): U[0]=1,U[1]=1; V[0]=1,V[2]=-1; C: W[0]=1,W[2]=-1
    U[0][6]=1; U[1][6]=1; V[0][6]=1; V[2][6]=-1; W[0][6]=1; W[2][6]=-1
    return U, V, W

def _kronecker_build(base_U, base_V, base_W, n, m, p, depth):
    """Build decomposition for (n,m,p) by Kronecker expansion of base (n0,m0,p0) decomposition.
    
    Splits into blocks: A is (n/n0) x (m/m0) blocks of size n0 x m0, etc.
    The multiplication tensor T[i,j,k] for the full problem factors as
    T[i,j,k] = T_block[ib,jb,kb] * T_sub[i0,j0,k0]
    where (ib,jb,kb) are block indices and (i0,j0,k0) are within-block indices.
    
    This gives rank = rank(base) * (n/n0) * (m/m0) * (p/p0) for the block part,
    but actually the Kronecker structure means:
    rank_full = rank_base * (n/n0) * (m/m0) * (p/p0)
    
    Wait, that's not right. Let me think more carefully.
    
    For 4x4x4 from 2x2x2: rank = 7 * 2 * 2 * 2 = 56? No, Strassen gives 49.
    The issue is that the Kronecker product of the tensors gives:
    T_full[a,b,c] = T_block[ab,bb,cb] * T_sub[as,bs,cs]
    
    A rank-r decomposition of T_block combined with a rank-s decomposition of T_sub
    gives a rank-r*s decomposition of T_full via tensor product of factors.
    
    For 2x2x2 Strassen (r=7) to 4x4x4: rank = 7*7 = 49. Correct!
    For 4x4x4 (r=49) to 4x4x8: we need to extend p from 4 to 8.
    
    Actually, for 4x4x8, we can view it as two 4x4x4 problems side by side.
    That gives rank 98. But can we do better?
    
    Alternative: view 4x4x8 as 2x2x4 blocks.
    A is 2x2 blocks of 2x4, B is 2x2 blocks of 4x4, C is 2x2 blocks of 4x2.
    This doesn't directly factor.
    
    Let me just use the straightforward Kronecker approach:
    For (n,m,p) = (n0*a, m0*b, p0*c) from base (n0,m0,p0):
    T[i,j,k] where i=ib*n0+i0, j=jb*m0+j0, k=kb*p0+k0
    T[i,j,k] = delta(jb,jb') * delta(ib,ib') * delta(kb,kb') * T_sub[i0,j0,k0]
    
    Wait, the full tensor is:
    T[a,b,c] = 1 iff (j==j' and i==i' and k==k')
    where a=i*m+j, b=j'*p+k, c=k'*n+i'
    
    In block form: i = ib*n0 + i0, j = jb*m0 + j0, k = kb*p0 + k0
    i' = ib'*n0 + i0', j' = jb'*m0 + j0', k' = kb'*p0 + k0'
    
    T = 1 iff jb==jb' AND ib==ib' AND kb==kb' AND j0==j0' AND i0==i0' AND k0==k0'
    
    So T_full = T_block ⊗ T_sub (Kronecker product of the two tensors)
    
    If T_block has rank rb and T_sub has rank rs, then T_full has rank rb*rs.
    
    For 4x4x4 from 2x2x2: T_block is 2x2x2 (rank 7), T_sub is 2x2x2 (rank 7), total rank 49. ✓
    For 4x4x8: We can split as (2,2,4) ⊗ (2,2,2)? No, dimensions don't work that way.
    
    Actually 4x4x8 = (2*2) x (2*2) x (2*4). 
    Split A into 2x2 blocks of 2x2, B into 2x2 blocks of 2x4, C into 2x2 blocks of 2x4.
    Hmm, this is getting complicated. Let me just use the simple approach.
    """
    n0, m0, p0 = len(base_U), len(base_V) // (len(base_U) // len(base_V) * len(base_U) // len(base_V)), 0
    # Actually let me just hardcode the known cases
    
    # For 4x4x4 from 2x2x2:
    if n == 4 and m == 4 and p == 4 and len(base_U) == 4:
        r_base = len(base_U[0])
        r_full = r_base * r_base
        U = [[0]*r_full for _ in range(16)]
        V = [[0]*r_full for _ in range(16)]
        W = [[0]*r_full for _ in range(16)]
        
        for rb in range(r_base):
            for rs in range(r_base):
                col = rb * r_base + rs
                # A: full index (ib*n0+i0)*m + (jb*m0+j0) = (ib*2+i0)*4 + (jb*2+j0)
                # base A index: ib*2+jb (for 2x2 block) -- wait no
                # base U index for 2x2: i0*2+j0 where i0,j0 in {0,1}
                # But the block structure: A is 2x2 blocks of 2x2
                # Full A[i,j] where i=ib*2+i0, j=jb*2+j0
                # U_full[i*4+j, col] = U_block[ib*2+jb, rb] * U_sub[i0*2+j0, rs]
                
                for ib in range(2):
                    for jb in range(2):
                        for i0 in range(2):
                            for j0 in range(2):
                                full_idx = (ib*2+i0)*4 + (jb*2+j0)
                                block_idx = ib*2 + jb  # This is wrong for the tensor structure
                                # Actually for the Kronecker product of tensors:
                                # T_full[a,b,c] = T_block[ab,bb,cb] * T_sub[as,bs,cs]
                                # where a = ab*4+as, b = bb*4+bs, c = cb*4+cs for 4x4x4
                                # ab = ib*2+jb (block row, block col), as = i0*2+j0 (within block)
                                # Similarly for b: bb = jb'*2+k0... 
                                
                                # This is getting too complex. Let me use the direct formula.
                                pass
        
        # Let me just use the direct Kronecker formula
        # For 4x4x4: a = i*4+j, b = j'*4+k, c = k'*4+i'
        # i = ib*2+i0, j = jb*2+j0, j' = jb'*2+j0', k = kb*2+k0, k' = kb'*2+k0', i' = ib'*2+i0'
        # a = (ib*2+i0)*4 + (jb*2+j0) = ib*8 + i0*4 + jb*2 + j0
        # b = (jb'*2+j0')*4 + (kb*2+k0) = jb'*8 + j0'*4 + kb*2 + k0
        # c = (kb'*2+k0')*4 + (ib'*2+i0') = kb'*8 + k0'*4 + ib'*2 + i0'
        
        # T[a,b,c] = 1 iff j==j' AND i==i' AND k==k'
        # = 1 iff jb==jb' AND j0==j0' AND ib==ib' AND i0==i0' AND kb==kb' AND k0==k0'
        
        # T_block[ab,bb,cb] where ab=(ib,jb), bb=(jb',kb), cb=(kb',ib')
        # T_block = 1 iff jb==jb' AND ib==ib' AND kb==kb' (this is 2x2x2 matmul tensor)
        # T_sub[as,bs,cs] where as=(i0,j0), bs=(j0',k0), cs=(k0',i0')
        # T_sub = 1 iff j0==j0' AND i0==i0' AND k0==k0' (this is also 2x2x2 matmul tensor)
        
        # So T_full = T_block ⊗ T_sub, both are 2x2x2 matmul tensors
        # Rank = 7 * 7 = 49
        
        U = [[0]*49 for _ in range(16)]
        V = [[0]*49 for _ in range(16)]
        W = [[0]*49 for _ in range(16)]
        
        for rb in range(7):
            for rs in range(7):
                col = rb * 7 + rs
                # U_full[a, col] = U_block[ab, rb] * U_sub[as, rs]
                # a = ib*8 + i0*4 + jb*2 + j0
                # ab = ib*2+jb (2x2 block index for A)
                # as = i0*2+j0 (2x2 within-block index for A)
                for ib in range(2):
                    for jb in range(2):
                        for i0 in range(2):
                            for j0 in range(2):
                                a = ib*8 + i0*4 + jb*2 + j0
                                ab = ib*2 + jb
                                as_idx = i0*2 + j0
                                U[a][col] = base_U[ab][rb] * base_U[as_idx][rs]
                
                # V_full[b, col] = V_block[bb, rb] * V_sub[bs, rs]
                # b = jb'*8 + j0'*4 + kb*2 + k0
                # bb = jb'*2+kb (2x2 block index for B)
                # bs = j0'*2+k0 (2x2 within-block index for B)
                for jb2 in range(2):
                    for j02 in range(2):
                        for kb in range(2):
                            for k0 in range(2):
                                b = jb2*8 + j02*4 + kb*2 + k0
                                bb = jb2*2 + kb
                                bs = j02*2 + k0
                                V[b][col] = base_V[bb][rb] * base_V[bs][rs]
                
                # W_full[c, col] = W_block[cb, rb] * W_sub[cs, rs]
                # c = kb'*8 + k0'*4 + ib'*2 + i0'
                # cb = kb'*2+ib' (2x2 block index for C, trace convention)
                # cs = k0'*2+i0' (2x2 within-block index for C, trace convention)
                for kb2 in range(2):
                    for k02 in range(2):
                        for ib2 in range(2):
                            for i02 in range(2):
                                c = kb2*8 + k02*4 + ib2*2 + i02
                                cb = kb2*2 + ib2
                                cs = k02*2 + i02
                                W[c][col] = base_W[cb][rb] * base_W[cs][rs]
        
        return U, V, W
    
    # For 4x4x8 from 4x4x4:
    if n == 4 and m == 4 and p == 8 and len(base_U) == 16:
        r_base = len(base_U[0])
        # 4x4x8 = two 4x4x4 side by side
        # rank = 2 * r_base
        r_full = 2 * r_base
        U = [[0]*r_full for _ in range(16)]
        V = [[0]*r_full for _ in range(32)]
        W = [[0]*r_full for _ in range(32)]
        
        for r in range(r_base):
            # First half: columns 0-3 of B
            for i in range(16):
                U[i][r] = base_U[i][r]
                U[i][r + r_base] = base_U[i][r]
            for j in range(4):
                for k in range(4):
                    V[j*8+k][r] = base_V[j*4+k][r]
                    V[j*8+(k+4)][r + r_base] = base_V[j*4+k][r]
            for k in range(4):
                for i in range(4):
                    W[k*4+i][r] = base_W[k*4+i][r]
                    W[(k+4)*4+i][r + r_base] = base_W[k*4+i][r]
        
        return U, V, W
    
    return None

def _try_kronecker_4x4x8():
    """Build 4x4x8 via Kronecker: 2x2x2 Strassen -> 4x4x4 -> 4x4x8"""
    U2, V2, W2 = _strassen_2x2()
    U4, V4, W4 = _kronecker_build(U2, V2, W2, 4, 4, 4, 1)
    if U4 is None:
        return None
    U8, V8, W8 = _kronecker_build(U4, V4, W4, 4, 4, 8, 2)
    if U8 is None:
        return None
    return U8, V8, W8

def _try_direct_2x2x4_kronecker():
    """Try building 4x4x8 directly as Kronecker of 2x2x2 and 2x2x4.
    
    4x4x8 = (2*2) x (2*2) x (2*4)
    Split: A into 2x2 blocks of 2x2, B into 2x2 blocks of 2x4, C into 2x2 blocks of 2x4
    
    Actually this doesn't factor cleanly. Let me try another split.
    
    4x4x8: A is 4x4, B is 4x8, C is 4x8
    Split A into 2x2 blocks of 2x2: 4 blocks
    Split B into 2x4 blocks of 2x4: 2 blocks  
    Split C into 2x4 blocks of 2x4: 2 blocks
    
    Hmm, this is asymmetric. Let me just use the 4x4x4 -> 4x4x8 approach.
    """
    return None

def _rank_reduce(U, V, W, max_iter=200):
    """Try to reduce rank by finding linear dependencies among columns.
    
    If column r can be expressed as a linear combination of other columns,
    we can eliminate it. We look for rank-deficient structure.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # The tensor is T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # If we can find a matrix M such that U*M, V*M, W*M give same tensor with fewer non-zero columns...
    # This is equivalent to finding a change of basis that makes some columns zero.
    
    # Alternative: check if any column is a linear combination of others
    # by checking if the tensor can be represented with fewer terms.
    
    # Simple approach: try removing each column and see if the tensor is preserved
    # This is expensive, so we'll do a randomized version.
    
    for _ in range(max_iter):
        # Pick a random column to try removing
        r = np.random.randint(rank)
        # Check if column r is in the span of others
        # This is equivalent to checking if the r-th "slice" is in the span of others
        
        # Build the tensor without column r
        # T_no_r[a,b,c] = sum_{r'!=r} U[a,r']*V[b,r']*W[c,r']
        # T_r[a,b,c] = U[a,r]*V[b,r]*W[c,r]
        # We need T_r to be expressible as sum of other slices
        
        # This is a tensor rank problem, which is NP-hard in general.
        # For now, skip this and just return the original.
        pass
    
    return U, V, W, rank

def schoolbook_decomposition(n, m, p):
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

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook (rank 128)
    best_result = schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Try Kronecker construction: 2x2x2 Strassen -> 4x4x4 (rank 49) -> 4x4x8 (rank 98)
    try:
        U8, V8, W8 = _try_kronecker_4x4x8()
        if U8 is not None:
            result = {"U": U8, "V": V8, "W": W8}
            if _verify_decomposition(U8, V8, W8, n, m, p):
                best_result = result
                best_rank = len(U8[0])
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_result
    
    return best_result

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
