"""FMM-inspired approach: systematic bilinear reduction for matrix multiplication tensor."""
import time
import sys
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8

    # Schoolbook fallback
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    # FMM algorithm: Find minimal rank decomposition using systematic bilinear reduction
    # Key insight: The matrix multiplication tensor has structure that can be exploited
    # through recursive block decomposition with optimal substructure.
    
    # For 2x3x8, we can use the following strategy:
    # 1. Decompose into 2x2x2 blocks where possible (Strassen, rank 7 vs 8)
    # 2. Handle remaining dimensions optimally
    
    # Let's try a more systematic approach using the FMM framework:
    # The FMM algorithm works by finding a basis transformation that minimizes rank.
    
    # For 2x3x8, let's consider the tensor T[i,a,k] = 1
    # We want to find U, V, W such that sum_r U[i,a,r]*V[a,k,r]*W[k,i,r] = 1
    
    # Strategy: Use Strassen for the 2x2 part of the first two dimensions,
    # and handle the third dimension (size 8) with optimized decomposition.
    
    # Actually, let's try a different approach: 
    # Decompose 2x3x8 as (2x2x8) + (2x1x8)
    # For 2x2x8: Use Strassen on the 2x2 part, giving 7*4 = 28 for 4 blocks of size 2 in k
    # But wait, we can do better by sharing across k blocks.
    
    # Let's try: 2x3x8 = 2x(2+1)x8
    # = 2x2x8 + 2x1x8
    
    # For 2x2x8: Split k into 4 groups of 2, use Strassen for each: 4*7 = 28
    # For 2x1x8: Schoolbook: 2*1*8 = 16
    # Total: 44 (same as current)
    
    # Can we do better? Let's think about 2x3x8 differently.
    # 
    # Alternative: 3x8 can be seen as having structure.
    # 
    # Let's try: Decompose using the fact that rank(2x3x8) <= rank(2x2x8) + rank(2x1x8)
    # But maybe there's a better decomposition.
    #
    # Actually, let's try to find if rank < 44 is possible.
    # 
    # Theoretical lower bound: For matrix multiplication, the tensor rank R(n,m,p) satisfies
    # R(n,m,p) >= max(n*m, m*p, p*n) = max(6, 24, 16) = 24
    # Also R(n,m,p) >= n*m*p / max(n,m,p) = 48/8 = 6 (weak)
    # 
    # Known results: R(2,2,2) = 7, R(2,3,3) = ? 
    # 
    # Let's try a direct approach: Can we find a decomposition with rank 43 or less?
    # 
    # Actually, let's try a smarter decomposition:
    # 2x3x8: Note that 3 = 2+1, 8 = 4*2
    # 
    # Alternative decomposition:
    # A (2x3) * B (3x8)
    # Let A = [A1 | A2] where A1 is 2x2, A2 is 2x1
    # Let B = [B1; B2] where B1 is 2x8, B2 is 1x8
    # C = A1*B1 + A2*B2
    #
    # For A1*B1 (2x2x8): We can use Strassen-like decomposition.
    # Split B1 into 4 blocks of 2 columns: B1 = [B1_1, B1_2, B1_3, B1_4]
    # Each A1*B1_j is 2x2x2, which has rank 7 (Strassen).
    # Total for A1*B1: 4*7 = 28
    #
    # For A2*B2 (2x1x8): This is just outer product, rank 16.
    # Total: 44.
    #
    # Can we do better for A1*B1? 
    # Instead of treating each 2x2x2 block independently, can we share structure?
    # 
    # Actually, for 2x2x8, the tensor rank is at most 7*4 = 28, but could be lower.
    # Let's check: R(2,2,8) <= R(2,2,2)*4 = 28, but also R(2,2,8) <= 2*2*8 = 32.
    # Is there a known better bound? 
    #
    # Actually, R(2,2,8) = 14? No, that doesn't seem right.
    # 
    # Let's try a different approach: Use the FMM algorithm properly.
    # 
    # FMM approach: 
    # 1. Start with the standard basis.
    # 2. Apply row/column operations to reduce the tensor.
    # 3. Find the minimal number of rank-1 terms.
    #
    # For our case, let's try to construct a decomposition with rank < 44.
    # 
    # Idea: For the 2x1x8 part (A2*B2), instead of using 16 terms,
    # can we combine with some of the Strassen terms?
    #
    # Actually, let's try a completely different decomposition:
    # 2x3x8: Split the middle dimension differently.
    # 
    # A (2x3) * B (3x8)
    # Let's use the fact that we can choose any linear combination of rows/columns.
    #
    # Alternative: Use 3 = 2+1 but with overlapping structure.
    # 
    # Let's try: 
    # C[i,k] = sum_a A[i,a] * B[a,k]
    # = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    #
    # Define:
    # M1 = (A[i,0] + A[i,1]) * (B[0,k] + B[1,k])  -- but this doesn't work directly
    #
    # Let's use the standard Strassen approach for the 2x2 part and see if we can
    # optimize the 2x1 part.
    #
    # Actually, let's just try to verify if 44 is optimal or if we can do better.
    # 
    # For now, let's stick with 44 but verify it's correct.
    # If we can find a better decomposition, we'll use it.
    
    # Let's try to find a decomposition with rank 43.
    # 
    # Idea: In the Strassen decomposition for 2x2x2, one of the terms might be
    # redundant when combined with the 2x1x8 part.
    #
    # Actually, let's try a different block decomposition:
    # 2x3x8 = (2x3x4) + (2x3x4) where the two 4s are disjoint column blocks.
    # R(2,3,4) <= ? 
    # 
    # For 2x3x4: Split as 2x(2+1)x4 = 2x2x4 + 2x1x4
    # R(2,2,4) <= 7*2 = 14 (two 2x2x2 blocks)
    # R(2,1,4) = 8
    # Total: 22 per 4-block, so 44 total. Same as before.
    #
    # Can we do better for 2x2x4? 
    # R(2,2,4): The tensor is 2x2x4. 
    # Schoolbook: 16. Strassen-based: 14.
    # Is there a rank-13 decomposition? 
    #
    # Actually, for 2x2x4, we can write it as two 2x2x2 tensors, each rank 7, total 14.
    # But maybe there's a shared structure.
    #
    # Let's try to construct a rank-13 decomposition for 2x2x4.
    # 
    # T[i,a,k] for i in {0,1}, a in {0,1}, k in {0,1,2,3}
    # 
    # Using Strassen for k=0,1 and k=2,3 separately gives 14.
    # Can we do better? 
    #
    # I don't think so, as the two 2x2x2 blocks are independent.
    #
    # So 44 seems to be the best we can get with this approach.
    #
    # Let's verify the current decomposition is correct and return it.
    
    # Build the 44-rank decomposition
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2x2 decomposition
    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    # Verify Strassen
    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        acc += S_U[r][i*2 + a] * S_V[r][a*2 + k] * S_W[r][k*2 + i]
                    if acc != 1:
                        return False
        return True
    
    if not _verify_strassen():
        return _schoolbook()
    
    # Part 1: A1 (2x2) * B1 (2x8)
    for block_idx in range(4):
        k0 = block_idx * 2
        k1 = k0 + 1
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li*2 + la]
                    if u_val == 0:
                        continue
                    u_idx = li * m + la
                    U[u_idx][col] = u_val
            
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la*2 + lk]
                    if v_val == 0:
                        continue
                    v_idx = la * p + (k0 + lk)
                    V[v_idx][col] = v_val
            
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk*2 + li]
                    if w_val == 0:
                        continue
                    w_idx = (k0 + lk) * n + li
                    W[w_idx][col] = w_val
            
            col += 1
            if not _elapsed_ok():
                return _schoolbook()
    
    # Part 2: A2 (2x1) * B2 (1x8)
    for i in range(2):
        for k in range(8):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            if not _elapsed_ok():
                return _schoolbook()
    
    candidate = {"U": U, "V": V, "W": W}
    
    if _validate(candidate):
        return candidate
    else:
        return _schoolbook()
