"""
Rank-minimal exact tensor decomposition for 4x5x6 matrix multiplication.

Strategy:
1. Build a strong initial decomposition using Strassen-like block structure.
   For 4x5x6, we can think of it as:
   - 4x4x4 core (Strassen rank 49) + 4x1x6 strip + 4x4x2 strip
   Or more carefully:
   - Use 2x2x2 Strassen blocks where possible, then handle the odd dimensions.
   
   Actually, let's use a cleaner approach:
   - 4x5x6 = (4x4x6) + (4x1x6)
   - 4x4x6 = (4x4x4) + (4x4x2)
   - 4x4x4: Strassen rank 49
   - 4x4x2: can be done with 2 Strassen 2x2x2 blocks on the j dimension? 
     Actually 4x4x2: split into (2x2x2) blocks: 4 blocks of 2x2x2, each rank 7 -> 28
     Or better: 4x4x2 = (4x2x2) + (4x2x2) = 2 * (4x2x2)
     4x2x2: split i into 2x2, so 2 blocks of 2x2x2, each rank 7 -> 14
     So 4x4x2 = 28 rank
   
   Wait, let me reconsider. The matrix is A (4x5) * B (5x6) = C (4x6).
   
   We can decompose as:
   - A = [A1 | A2] where A1 is 4x4, A2 is 4x1
   - B = [B1; B2] where B1 is 4x6, B2 is 1x6
   - C = A1*B1 + A2*B2
   
   For A1*B1 (4x4 * 4x6):
   - Split B1 into [B11 | B12] where B11 is 4x4, B12 is 4x2
   - A1*B11 (4x4*4x4): Strassen rank 49
   - A1*B12 (4x4*4x2): 
     Split A1 into 2x2 blocks, B12 into 2x2 blocks
     Actually 4x4*4x2: we can view this as 2 independent 4x2*4x2? No.
     Let's use: 4x4 * 4x2. Split i into 2 groups of 2, j into 2 groups of 2, k into 1 group of 2.
     That gives 4 blocks of 2x2x2, each rank 7 -> 28 rank.
   
   For A2*B2 (4x1 * 1x6):
   - This is just outer product: rank 4*1*6 = 24? No, it's rank 1 per output element.
   - Actually A2 is 4x1, B2 is 1x6. C2 = A2 * B2 is 4x6.
   - Each C[i,k] = A2[i,0] * B2[0,k]. This is rank 1 for each (i,k) pair.
   - But we can share: for fixed i, all k share A2[i,0]. For fixed k, all i share B2[0,k].
   - The tensor for 4x1x1x6... actually it's just a rank-4 decomposition if we think of it right.
   - C[i,k] = sum over j=0 to 0 of A[i,j]*B[j,k] = A[i,0]*B[0,k]
   - This is rank 1! Because it's an outer product.
   - Wait no, the rank of the 4x1x6 multiplication tensor: 
     T[i,j,k] = delta(j,0) * delta... 
     Actually for n=4, m=1, p=6: T[i,j,k] = 1 if j=0 else 0, for all i,k.
     This tensor has rank... let's think. 
     C[i,k] = A[i,0] * B[0,k]. 
     This is a rank-1 bilinear form in the sense that it's one term: u_i * v_j * w_k where u_i=1, v_j=delta(j,0), w_k=1? No.
     Actually: C[i,k] = A[i,0] * B[0,k]. 
     In tensor form: T[i,j,k] = delta_{j,0} * delta_{i,i} * delta_{k,k}... 
     The multiplication tensor for m=1: T[i,j,k] = 1 if j=0, 0 otherwise.
     This can be written as sum over c=1..4 of u_c[i]*v_c[j]*w_c[k] where:
     For c=i: u_c[i']=1 if i'=i else 0, v_c[j]=1 if j=0 else 0, w_c[k]=1 for all k.
     So rank = 4.
   
   Total: 49 + 28 + 4 = 81.
   
   But wait, we can do better. Let me reconsider.
   
   Actually, for 4x4x2: 
   A (4x4) * B (4x2) = C (4x2)
   We can split into 2 blocks along k: two 4x4x1 problems.
   4x4x1: rank? C[i,k] = sum_j A[i,j] B[j,k]. For fixed k, this is a 4x4 vector-matrix product.
   The tensor T[i,j,k] for 4x4x1 has rank 16 (schoolbook) but can be done with Strassen?
   Actually 4x4x1 is just 4 independent 4x1 vector-matrix products, each rank 4, so total 16.
   But Strassen doesn't directly help for x1.
   
   Let me try a different decomposition:
   4x5x6:
   - Use 2x2x2 Strassen blocks for the 4x4x4 part (4 blocks, rank 4*7=28)
   - Then handle the remaining parts.
   
   Actually, the cleanest approach might be:
   - 4x5x6 = (4x4x4) + (4x4x2) + (4x1x6)
   - 4x4x4: 4 blocks of 2x2x2 Strassen = 4*7 = 28
   - 4x4x2: 4 blocks of 2x2x2 Strassen = 4*7 = 28  
   - 4x1x6: rank 4
   Total: 28+28+4 = 60? That seems too low. Let me verify.
   
   Wait, 4x4x4 with 2x2x2 blocks: we have 2x2=4 blocks, each 2x2x2, each rank 7. Total 28. Correct.
   4x4x2: split i into 2 groups of 2, j into 2 groups of 2, k into 1 group of 2.
   That's 2*2*1 = 4 blocks of 2x2x2, each rank 7. Total 28. Correct.
   4x1x6: rank 4 as computed above.
   Total: 28+28+4 = 60.
   
   But is this correct? Let me verify the 4x1x6 part.
   A is 4x1, B is 1x6. C = A*B is 4x6.
   C[i,k] = A[i,0] * B[0,k].
   We need: C[i,k] = sum_c U[i,0,c] * V[0,k,c] * W[k,i,c]
   Wait, the indices are: U has shape (n*m, rank) = (4*1, rank) = (4, rank)
   V has shape (m*p, rank) = (1*6, rank) = (6, rank)
   W has shape (p*n, rank) = (6*4, rank) = (24, rank)
   
   C[i,k] = sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
   For m=1: C[i,k] = sum_c U[i, c] * V[k, c] * W[k*n+i, c]
   
   We want C[i,k] = A[i,0] * B[0,k].
   If we set rank=4, and for c=i:
   U[i, c] = 1 if c==i else 0
   V[k, c] = 1 for all k (since B[0,k] is the same for all i in this term)
   Wait that doesn't work because V[k,c] would need to depend on B[0,k].
   
   Let me reconsider. The tensor entries are:
   T[i,j,k] = 1 if j=0, 0 otherwise. (independent of i,k)
   
   We need: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
   With m=1, n=4, p=6:
   T[i,0,k] = sum_c U[i, c] * V[k, c] * W[k*4+i, c] = 1 for all i,k
   T[i,j,k] = 0 for j!=0 (but j is always 0 since m=1)
   
   So we need: for all i in 0..3, k in 0..5:
   sum_c U[i,c] * V[k,c] * W[k*4+i, c] = 1
   
   One solution: rank=1, U[i,0]=1, V[k,0]=1, W[k*4+i,0]=1 for all i,k.
   Then sum_c = U[i,0]*V[k,0]*W[k*4+i,0] = 1*1*1 = 1. 
   
   So 4x1x6 has rank 1! Not 4.
   
   So total: 28+28+1 = 57.
   
   Let me double-check: is the 4x1x6 multiplication tensor really rank 1?
   The tensor T has T[i,0,k] = 1 for all i,k.
   This is T = u * v * w where u[i]=1, v[0]=1 (and v[j]=0 for j!=0, but j is always 0), w[k]=1.
   Yes, rank 1.
   
   So we can achieve rank 57. Let me verify this is correct by construction.
"""

import time
import random
import numpy as np

DIMENSIONS = [4, 5, 6]
START_TIME = time.time()
TIME_LIMIT = 140.0

def remaining():
    return TIME_LIMIT - (time.time() - START_TIME)

def _strassen_2x2_terms():
    """Return the 7 rank-1 terms for 2x2x2 Strassen."""
    # Each term is (u, v, w) where u,v,w are length-4 vectors
    # u indexes (i,j) in row-major: [00, 01, 10, 11]
    # v indexes (j,k) in row-major: [00, 01, 10, 11]  
    # w indexes (k,i) in row-major: [00, 01, 10, 11]
    terms = [
        # P1 = (a00+a11)(b00+b11)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # P2 = (a10+a11)b00
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        # P3 = a00(b01-b11)
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        # P4 = a11(b10-b00)
        ([0, 0, 0, 1], [-1, 0, 1, 0], [-1, 1, 1, 0]),
        # P5 = (a00+a01)b11
        ([1, 1, 0, 0], [0, 0, 1, 0], [1, 1, 0, 0]),
        # P6 = (a10-a00)(b00+b01)
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, -1, 1]),
        # P7 = (a01-a11)(b10+b11)
        ([0, 1, 0, -1], [0, 1, 0, 1], [-1, 1, 1, 0]),
    ]
    return terms

def _embed_block_222(terms, i_base, j_base, k_base, n, m, p):
    """Embed 2x2x2 Strassen terms at position (i_base, j_base, k_base)."""
    full_terms = []
    for (u, v, w) in terms:
        ufull = [0] * (n * m)
        vfull = [0] * (m * p)
        wfull = [0] * (p * n)
        
        for a in range(2):
            for b in range(2):
                idx = (i_base + a) * m + (j_base + b)
                ufull[idx] = u[a * 2 + b]
        
        for b in range(2):
            for c in range(2):
                idx = (j_base + b) * p + (k_base + c)
                vfull[idx] = v[b * 2 + c]
        
        for c in range(2):
            for a in range(2):
                idx = (k_base + c) * n + (i_base + a)
                wfull[idx] = w[c * 2 + a]
        
        full_terms.append((ufull, vfull, wfull))
    return full_terms

def _build_57_rank():
    """
    Build a rank-57 decomposition for 4x5x6 matrix multiplication.
    
    Structure:
    - 4x4x4 part: 4 blocks of 2x2x2 Strassen (i:0-3, j:0-3, k:0-3) -> rank 28
    - 4x4x2 part: 4 blocks of 2x2x2 Strassen (i:0-3, j:0-3, k:4-5) -> rank 28
    - 4x1x6 part: rank 1 (i:0-3, j:4, k:0-5)
    Total: 57
    """
    n, m, p = DIMENSIONS
    strassen_terms = _strassen_2x2_terms()
    
    terms_U = []
    terms_V = []
    terms_W = []
    
    # 4x4x4 part: i in 0-3, j in 0-3, k in 0-3
    for ib in range(2):
        for jb in range(2):
            for kb in range(2):
                i_base = 2 * ib
                j_base = 2 * jb
                k_base = 2 * kb
                embedded = _embed_block_222(strassen_terms, i_base, j_base, k_base, n, m, p)
                terms_U.extend([t[0] for t in embedded])
                terms_V.extend([t[1] for t in embedded])
                terms_W.extend([t[2] for t in embedded])
    
    # 4x4x2 part: i in 0-3, j in 0-3, k in 4-5
    for ib in range(2):
        for jb in range(2):
            for kb in range(1):
                i_base = 2 * ib
                j_base = 2 * jb
                k_base = 4 + 2 * kb
                embedded = _embed_block_222(strassen_terms, i_base, j_base, k_base, n, m, p)
                terms_U.extend([t[0] for t in embedded])
                terms_V.extend([t[1] for t in embedded])
                terms_W.extend([t[2] for t in embedded])
    
    # 4x1x6 part: i in 0-3, j=4, k in 0-5
    # Rank 1: U[i,0]=1, V[k,0]=1, W[k*n+i,0]=1
    ufull = [0] * (n * m)
    vfull = [0] * (m * p)
    wfull = [0] * (p * n)
    for i in range(n):
        ufull[i * m + 4] = 1
    for k in range(p):
        vfull[4 * p + k] = 1
    for k in range(p):
        for i in range(n):
            wfull[k * n + i] = 1
    terms_U.append(ufull)
    terms_V.append(vfull)
    terms_W.append(wfull)
    
    rank = len(terms_U)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for i in range(n * m):
            U[i][c] = terms_U[c][i]
        for j in range(m * p):
            V[j][c] = terms_V[c][j]
        for k in range(p * n):
            W[k][c] = terms_W[c][k]
    
    return {"U": U, "V": V, "W": W, "rank": rank}

def _schoolbook():
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W, "rank": rank}

def _verify(U, V, W, rank, num_trials=10):
    """Verify exact decomposition using random integer matrices."""
    n, m, p = DIMENSIONS
    for _ in range(num_trials):
        if remaining() <= 0:
            return False
        A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
        C = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        for i in range(n):
            for k in range(p):
                acc = 0
                for j in range(m):
                    for c in range(rank):
                        u_val = U[i * m + j][c]
                        if u_val == 0:
                            continue
                        acc += u_val * W[k * n + i][c] * V[j * p + k][c]
                if acc != C[i][k]:
                    return False
    return True

def _try_remove_columns(U, V, W, rank, max_removals=10):
    """
    Try to remove columns that are linearly dependent or redundant.
    This is a heuristic approach: try removing each column and verify.
    """
    if remaining() < 5:
        return U, V, W, rank
    
    n, m, p = DIMENSIONS
    current_rank = rank
    U_copy = [row[:] for row in U]
    V_copy = [row[:] for row in V]
    W_copy = [row[:] for row in W]
    
    # Try removing columns one at a time
    for attempt in range(min(max_removals, current_rank)):
        if remaining() < 2:
            break
        
        removed = False
        for c in range(current_rank - 1, -1, -1):
            if remaining() < 2:
                break
            
            # Remove column c
            U_new = [[U_copy[i][cc] for cc in range(current_rank) if cc != c] for i in range(n * m)]
            V_new = [[V_copy[j][cc] for cc in range(current_rank) if cc != c] for j in range(m * p)]
            W_new = [[W_copy[k][cc] for cc in range(current_rank) if cc != c] for k in range(p * n)]
            
            if _verify(U_new, V_new, W_new, current_rank - 1, num_trials=3):
                U_copy = U_new
                V_copy = V_new
                W_copy = W_new
                current_rank -= 1
                removed = True
                break
        
        if not removed:
            break
    
    return U_copy, V_copy, W_copy, current_rank

def construct():
    n, m, p = DIMENSIONS
    
    candidates = []
    
    # Try the 57-rank decomposition
    if remaining() > 20:
        try:
            cand57 = _build_57_rank()
            candidates.append(cand57)
        except Exception as e:
            pass
    
    # Add schoolbook as fallback
    candidates.append(_schoolbook())
    
    best = None
    for cand in candidates:
        if remaining() <= 5:
            break
        rank = cand["rank"]
        U, V, W = cand["U"], cand["V"], cand["W"]
        
        if _verify(U, V, W, rank):
            if best is None or rank < best["rank"]:
                best = cand
    
    if best is None:
        best = candidates[-1]
    
    # Try to reduce rank further by removing redundant columns
    if remaining() > 30:
        try:
            U, V, W, rank = _try_remove_columns(best["U"], best["V"], best["W"], best["rank"])
            if _verify(U, V, W, rank, num_trials=5):
                best = {"U": U, "V": V, "W": W, "rank": rank}
        except Exception:
            pass
    
    return {"U": best["U"], "V": best["V"], "W": best["W"]}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    # Quick verification
    n, m, p = DIMENSIONS
    U, V, W = result["U"], result["V"], result["W"]
    rank = len(U[0])
    A = [[random.randint(-3, 3) for _ in range(m)] for _ in range(n)]
    B = [[random.randint(-3, 3) for _ in range(p)] for _ in range(m)]
    C_expected = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
    C_computed = [[sum(sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank)) for j in range(m)) for k in range(p)] for i in range(n)]
    print(f"Match: {C_expected == C_computed}")
