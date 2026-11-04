import time
import numpy as np

START = time.time()
DEADLINE = START + 130.0

def _strassen_2x2x2():
    """Strassen 2x2x2 decomposition, rank 7."""
    U = np.zeros((4, 7), dtype=np.int16)
    V = np.zeros((4, 7), dtype=np.int16)
    W = np.zeros((4, 7), dtype=np.int16)
    # M1 = (A00+A11)(B00+B11)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # M2 = (A10+A11)B00
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[1,1]=1; W[3,1]=-1
    # M3 = A00(B01-B11)
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[2,2]=1; W[3,2]=1
    # M4 = A11(B10-B00)
    U[3,3]=1; V[0,3]=-1; V[2,3]=1; W[0,3]=1; W[1,3]=1
    # M5 = (A00+A01)B11
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[2,4]=1
    # M6 = (A10-A00)(B00+B01)
    U[0,5]=-1; U[2,5]=1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # M7 = (A01-A11)(B10+B11)
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W

def _build_4x4x4_tensors():
    """Build 4x4x4 via Kronecker product of two 2x2x2 Strassen."""
    U2, V2, W2 = _strassen_2x2x2()
    rank = 49
    U4 = np.zeros((16, rank), dtype=np.int16)
    V4 = np.zeros((16, rank), dtype=np.int16)
    W4 = np.zeros((16, rank), dtype=np.int16)
    for c0 in range(7):
        for c1 in range(7):
            c = c0*7 + c1
            for i in range(4):
                for j in range(4):
                    U4[i*4+j, c] = U2[(i//2)*2+(j//2), c0] * U2[(i%2)*2+(j%2), c1]
            for j in range(4):
                for k in range(4):
                    V4[j*4+k, c] = V2[(j//2)*2+(k//2), c0] * V2[(j%2)*2+(k%2), c1]
            for k in range(4):
                for i in range(4):
                    W4[k*4+i, c] = W2[(k//2)*2+(i//2), c0] * W2[(k%2)*2+(i%2), c1]
    return U4, V4, W4

def _build_shared_rank_decomposition():
    """
    Build a low-rank decomposition for 4x4 * 4x8 using shared U factors.
    
    Strategy: The 4x4x8 multiplication tensor T[i,j,k] where k in 0..7.
    We split into two 4x4x4 blocks. Instead of two independent rank-49 decompositions
    (total rank 98), we use a "shared" approach:
    
    For the first block (k=0..3): standard 4x4x4 decomposition, rank 49
    For the second block (k=4..7): We can try to reuse some of the U structure.
    
    Actually, let's use a different approach: since both blocks share the same A,
    we can write:
      C[:, 0:4] = A @ B[:, 0:4]  
      C[:, 4:8] = A @ B[:, 4:8]
    
    The key insight: we can factor this as a single rank-r decomposition where
    the U columns are shared between the two output blocks in a structured way.
    
    Let's use rank 49 + 49 = 98 as baseline, but try to find a lower rank by
    exploiting the shared A structure.
    
    Alternative: use the fact that 4x4x8 has a different border rank.
    Let's try rank 56 by using a mixed decomposition.
    """
    U4, V4, W4 = _build_4x4x4_tensors()
    
    # Standard approach: two blocks, rank 98
    # Let's try to find a lower rank decomposition.
    
    # For 4x4x8, the trivial rank is 128. Strassen blocks give 98.
    # Can we do better?
    
    # One approach: use the 2x2x8 tensor product structure
    # Split 4x4 into 2x2 blocks, and 4x8 into 2x8 blocks
    # The 2x2x8 tensor can be decomposed with rank 7*? 
    # Actually 2x2x8: A is 2x2, B is 2x8, C is 2x8
    # This is equivalent to 4 independent 2x2x2 multiplications sharing A
    # Rank would be 7*4 = 28 for this sub-tensor
    # Then 4x4x8 = tensor product of 4x4x4 and 2x2x8? No...
    
    # Let me think differently. Use block structure:
    # A = [[A00, A01], [A10, A11]] (each 2x2)
    # B = [[B00, B01, B02, B03], [B10, B11, B12, B13]] (each 2x2)
    # C = A @ B (4x8)
    
    # C[:, 0:2] = A @ B[:, 0:2]
    # C[:, 2:4] = A @ B[:, 2:4]  
    # C[:, 4:6] = A @ B[:, 4:6]
    # C[:, 6:8] = A @ B[:, 6:8]
    
    # For each 2-column output block, we have a 4x4x2 sub-problem
    # 4x4x2 = (4x4x4) with half the output, rank 49/2? No, rank is still ~49
    
    # Let's just use the rank-98 decomposition which is proven correct.
    total_rank = 98
    U = np.zeros((16, total_rank), dtype=np.int16)
    V = np.zeros((32, total_rank), dtype=np.int16)
    W = np.zeros((32, total_rank), dtype=np.int16)
    
    for block in range(2):
        col_start = block * 49
        col_end = (block + 1) * 49
        U[:, col_start:col_end] = U4
        for j in range(4):
            for k_local in range(4):
                k_global = block * 4 + k_local
                V[j*8 + k_global, col_start:col_end] = V4[j*4 + k_local, :]
        for k_local in range(4):
            k_global = block * 4 + k_local
            for i in range(4):
                W[k_global*4 + i, col_start:col_end] = W4[k_local*4 + i, :]
    
    # Verify
    n, m, p = 4, 4, 8
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(total_rank):
                    total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(total - 1.0) > 1e-9:
                    return None
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_lower_rank_56():
    """
    Try to construct a rank-56 decomposition using shared structure.
    
    Key idea: For 4x4x8, we can write the tensor as:
    T = sum_r U_r ⊗ V_r ⊗ W_r
    
    Since the output dimension is 8 = 4+4, and both halves use the same A,
    we might be able to use fewer terms by having V and W span both halves
    simultaneously.
    
    Let's try: use the 49-term 4x4x4 decomposition, but modify V and W to
    cover both output blocks at once, and add a small number of correction terms.
    
    Actually, let's try a different decomposition strategy:
    Use the 2x2x2 Strassen decomposition but applied at a higher level.
    
    Split 4x4x8 as:
    - Left: 4x4 (split into 2x2 blocks)
    - Right: 4x8 (split into 2x2 blocks, 4 of them)
    
    This gives a 2x2x(2x4) = 2x2x8 tensor for each "macro" position.
    The 2x2x8 tensor: A is 2x2, B is 2x8, C is 2x8
    This is 4 independent 2x2x2 multiplications sharing A.
    Rank of 2x2x8 = 4 * 7 = 28 (using Strassen for each 2x2x2 sub-block)
    
    Then 4x4x8 = tensor product of 2x2x8 and... hmm, this gets complicated.
    
    Let me try a direct approach: rank 56.
    """
    # This is speculative. Let's construct it carefully.
    # If we can't verify it, return None.
    
    U2, V2, W2 = _strassen_2x2x2()
    
    # 4x4x8 decomposition attempt:
    # Use 49 terms for the first 4 output columns, and try to share
    # some terms with the second 4 output columns.
    
    # Actually, let's use a different known result:
    # The rank of 4x4x8 is at most 98 (two 4x4x4 blocks)
    # Can we do better? The border rank might allow it.
    
    # For now, return None to indicate we couldn't find a lower rank.
    return None

def _schoolbook(n=4, m=4, p=8):
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

def construct():
    # Try rank-56 lower decomposition first
    if time.time() < DEADLINE - 5:
        try:
            res = _try_lower_rank_56()
            if res is not None:
                # Verify
                n, m, p = 4, 4, 8
                U, V, W = res['U'], res['V'], res['W']
                rank = len(U[0])
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for r in range(rank):
                                s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                            if abs(s - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid:
                    return res
        except Exception:
            pass
    
    # Fall back to rank-98
    if time.time() < DEADLINE - 5:
        try:
            res = _build_shared_rank_decomposition()
            if res is not None:
                return res
        except Exception:
            pass
    
    # Final fallback: schoolbook
    return _schoolbook()

if __name__ == "__main__":
    res = construct()
    rank = len(res['U'][0])
    print(f"Rank: {rank}")
    n, m, p = 4, 4, 8
    U, V, W = res['U'], res['V'], res['W']
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    print(f"Valid: {valid}")
