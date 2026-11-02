import numpy as np
import time

def construct():
    start_time = time.time()
    
    n, m, p = 2, 4, 6
    deadline = start_time + 120  # safety margin
    
    # Schoolbook rank: n * m * p = 48
    # But we can do better using Strassen on 2x2x2 blocks
    # The matrix is 2x4 times 4x6.
    # We can split A into 2 blocks of 2x2 (vertically? no, A is 2x4, so horizontally into two 2x2)
    # B is 4x6. Split B into 2 blocks of 2x3 (vertically into two 2x3)
    # C = A * B is 2x6.
    
    # Let's use a known rank decomposition.
    # For 2x2x2, Strassen gives rank 7.
    # For 2x2x3, the minimal rank is 11 (known from literature? Or is it 10? 9?).
    # Actually, the rank of 2x2x3 tensor is known to be 11.
    # We have 2 such blocks (since A is 2x4 = 2*(2x2), B is 4x6 = 2*(2x3)).
    # Wait, the middle dimension is shared.
    # A = [A1, A2] where A1, A2 are 2x2.
    # B = [B1; B2] where B1, B2 are 2x3.
    # C = A1*B1 + A2*B2.
    # So we need to decompose two 2x2x3 multiplications and add them.
    # If we use rank 11 for each, total rank = 22.
    # Is there a better way?
    # Maybe we can combine them?
    # The tensor is M_{2,4,6} = M_{2,2,3} \oplus M_{2,2,3} (block diagonal in the middle index j).
    # So rank(M_{2,4,6}) = rank(M_{2,2,3}) + rank(M_{2,2,3}) = 11 + 11 = 22.
    # This is a valid upper bound.
    # Can we do better?
    # The trivial rank is 48. Strassen on whole thing?
    # If we treat it as 2x4x6, maybe we can use a rank 20 or 21 decomposition?
    # Let's implement the rank 22 decomposition first, as it's guaranteed to work if I can find the 2x2x3 rank 11 decomposition.
    
    # Do I have a rank 11 decomposition for 2x2x3?
    # I don't have it memorized.
    # Let's stick to the known Strassen 2x2x2 rank 7.
    # Can I split 2x4x6 into four 2x2x2? No, p=6 is not divisible by 2 in a way that makes 4 blocks?
    # A is 2x4, B is 4x6.
    # Split A into 2x2, 2x2.
    # Split B into 2x3, 2x3.
    # This gives 2 terms of 2x2x3.
    
    # Alternative: Split A into 1x4, 1x4? No.
    # Split B into 4x2, 4x2, 4x2?
    # A is 2x4. B is 4x6 = [B1, B2, B3] each 4x2.
    # C = [C1, C2, C3] each 2x2.
    # C1 = A*B1 (2x2 * 4x2 -> invalid. A is 2x4, B1 is 4x2. Result 2x2.)
    # So we have 3 blocks of 2x4x2.
    # Rank of 2x4x2?
    # M_{2,4,2} = M_{2,2,2} \oplus M_{2,2,2}. Rank = 7+7=14.
    # Total rank = 3 * 14 = 42.
    # This matches the "Strassen on 2x2x2 sub-blocks reduces rank from 48 to 42" hint.
    # So the hint suggests rank 42 is achievable.
    # And the previous best score was 1.143, which corresponds to 42? (48/42 = 1.1428...). Yes.
    # So the current island best is likely rank 42.
    # Can I beat 42?
    # The rank 22 construction above would be much better (score ~2.18).
    # But I need the 2x2x3 rank 11 decomposition.
    
    # Let's try to find or construct a rank 11 decomposition for 2x2x3.
    # If I can't find it, I will implement the rank 42 one (which I know how to do using Strassen).
    # Actually, I can implement the rank 42 one using the known Strassen 2x2x2 decomposition.
    # Let's do that first to ensure a valid non-zero score, then try to optimize.
    
    # Strassen 2x2x2 decomposition (Rank 7):
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # C = [[c11, c12], [c21, c22]]
    
    # M1 = (a+b)(e+g)
    # M2 = (c+d)(e+g)
    # M3 = (a-b)(f-h)
    # M4 = (c-d)(f-h)
    # M5 = (a+b)(i-f) -- wait, B is [[e,f],[g,h]].
    # Standard Strassen:
    # M1 = (a+b)(e+g)
    # M2 = (c+d)(e+g)
    # M3 = (a-b)(f-h)
    # M4 = (c-d)(f-h)
    # M5 = (a+b)(h-f)  # Note: some sources use (a+b)(i-f) if B is 2x2, but here indices matter.
    # Let's use the explicit tensor factors.
    
    # For 2x2x2:
    # U, V, W are 4x7 matrices.
    # Indices: U[0]=a, U[1]=b, U[2]=c, U[3]=d
    # V[0]=e, V[1]=f, V[2]=g, V[3]=h
    # W[0]=c11 (k=0,i=0), W[1]=c21 (k=0,i=1)? No, W[k*n+i]. n=2.
    # W[0] = W[0*2+0] = c11
    # W[1] = W[1*2+0] = c21 ? No.
    # Trace convention W[k*n+i].
    # C is 2x2. C[i,k].
    # W index = k*n + i.
    # W[0] -> k=0, i=0 -> C[0,0]
    # W[1] -> k=1, i=0 -> C[0,1]
    # W[2] -> k=0, i=1 -> C[1,0]
    # W[3] -> k=1, i=1 -> C[1,1]
    
    # Let's verify Strassen W vectors.
    # C[0,0] = M1 + M4 - M2 - M5
    # C[0,1] = M1 - M3 - M2 + M6
    # C[1,0] = M3 + M2 - M4 - M6
    # C[1,1] = M5 - M3 + M6 + M4
    
    # So:
    # W for M1: [1, 1, 0, 0] ? No.
    # W[r, c] where c is the product index.
    # For product M1:
    # Contribution to C[0,0] is 1. Index 0.
    # Contribution to C[0,1] is 1. Index 1.
    # Contribution to C[1,0] is 0.
    # Contribution to C[1,1] is 0.
    # Wait, my Strassen formulas above might be swapped or wrong.
    # Let's use a standard verified set.
    # Source: Wikipedia "Strassen algorithm"
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B10+B11)
    # M4 = (A11)(B01-B11)
    # M5 = (A00+A01)(B01)
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M6
    # C11 = M1 - M2 + M3 + M4
    
    # Let's map this to U, V, W.
    # A: A00(0), A01(1), A10(2), A11(3)
    # B: B00(0), B01(1), B10(2), B11(3)
    # C: C00(0), C01(1), C10(2), C11(3) -- Using W[k*n+i].
    # W[0] = C00 (k=0,i=0)
    # W[1] = C01 (k=1,i=0)
    # W[2] = C10 (k=0,i=1)
    # W[3] = C11 (k=1,i=1)
    
    # M1: (A00+A11)(B00+B11)
    # U: [1, 0, 0, 1]
    # V: [1, 0, 0, 1]
    # C00 += M1, C11 += M1.
    # W: [1, 0, 0, 1]
    
    # M2: (A10+A11)(B00)
    # U: [0, 0, 1, 1]
    # V: [1, 0, 0, 0]
    # C10 += M2, C11 -= M2.
    # W: [0, 0, 1, -1]
    
    # M3: (A00)(B10+B11)
    # U: [1, 0, 0, 0]
    # V: [0, 0, 1, 1]
    # C01 += M3, C11 += M3.
    # W: [0, 1, 0, 1]
    
    # M4: (A11)(B01-B11)
    # U: [0, 0, 0, 1]
    # V: [0, 1, 0, -1]
    # C00 += M4, C11 += M4.
    # W: [1, 0, 0, 1]
    
    # M5: (A00+A01)(B01)
    # U: [1, 1, 0, 0]
    # V: [0, 1, 0, 0]
    # C00 -= M5, C01 += M5.
    # W: [-1, 1, 0, 0]
    
    # M6: (A10-A00)(B00+B10)
    # U: [-1, 0, 1, 0]
    # V: [1, 0, 1, 0]
    # C10 += M6.
    # W: [0, 0, 1, 0]
    
    # M7: (A01-A11)(B10+B11)
    # U: [0, 1, 0, -1]
    # V: [0, 0, 1, 1]
    # C00 += M7, C01 -= M7?
    # Formula: C00 = ... + M7. C11 = ... ?
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M6
    # C11 = M1 - M2 + M3 + M4
    # So M7 only affects C00?
    # Let's check C01. M7 has B10, B11. A01, A11.
    # Term A01*B11 is in C01? No, C01 = A00*B01 + A01*B11.
    # So M7 contributes to C01?
    # (A01-A11)(B10+B11) = A01*B10 + A01*B11 - A11*B10 - A11*B11.
    # A01*B11 is in C01.
    # A11*B11 is in C11.
    # So M7 must contribute to C01 and C11?
    # But the formula C01 = M3 + M5 doesn't have M7.
    # Let's re-verify the formulas.
    # Wikipedia:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M6
    # C11 = M1 - M2 + M3 + M4
    
    # If C01 doesn't have M7, then M7's contribution to C01 must be cancelled?
    # No, it's a sum of products.
    # Let's check M7 again.
    # Maybe the formula is different.
    # Let's use the rank 42 approach which is safer.
    # Rank 42 comes from 3 blocks of 2x4x2.
    # Each 2x4x2 is two 2x2x2s.
    # So 3 * 2 * 7 = 42.
    
    # Let's implement the rank 42 decomposition.
    # It's guaranteed to work if the 2x2x2 Strassen is correct.
    # I will construct the 2x2x2 U, V, W and then assemble them.
    
    # 2x2x2 Strassen Factors (U, V, W)
    # U is 4x7, V is 4x7, W is 4x7.
    U222 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V222 = np.array([
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 1],
        [1, 0, 1, -1, 0, 0, 1]
    ])
    W222 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [1, -1, 1, 1, 0, 0, 0]
    ])
    
    # Verify 2x2x2
    # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c] ?
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i].
    # Let's check T[0,0,0] (i=0,j=0,k=0).
    # sum_c U[0,c] V[0,c] W[0,c]
    # c=0: 1*1*1 = 1
    # c=1: 0*1*0 = 0
    # c=2: 1*0*0 = 0
    # c=3: 0*0*1 = 0
    # c=4: 1*0*-1 = 0
    # c=5: -1*1*0 = 0
    # c=6: 0*0*1 = 0
    # Sum = 1. Correct (T[0,0,0]=1).
    
    # Check T[0,1,0] (i=0,j=1,k=0). C[0,0] term A[0,1]B[1,0].
    # sum_c U[1,c] V[2,c] W[0,c]
    # U[1] = [0,0,0,0,1,0,1]
    # V[2] = [0,0,1,0,0,1,1]
    # W[0] = [1,0,0,1,-1,0,1]
    # c=0: 0*0*1=0
    # c=1: 0*0*0=0
    # c=2: 0*1*0=0
    # c=3: 0*0*1=0
    # c=4: 1*0*-1=0
    # c=5: 0*1*0=0
    # c=6: 1*1*1=1
    # Sum = 1. Correct.
    
    # Check T[1,0,0] (i=1,j=0,k=0). C[1,0] term A[1,0]B[0,0].
    # sum_c U[2,c] V[0,c] W[2,c]
    # U[2] = [0,1,0,0,0,1,0]
    # V[0] = [1,1,0,0,0,1,0]
    # W[2] = [0,1,0,0,0,1,0]
    # c=0: 0*1*0=0
    # c=1: 1*1*1=1
    # c=2: 0*0*0=0
    # c=3: 0*0*0=0
    # c=4: 0*0*0=0
    # c=5: 1*1*1=1
    # c=6: 0*0*0=0
    # Sum = 2. Incorrect. Should be 1.
    
    # There is an error in my Strassen factors.
    # Let's use a reliable source or derive it carefully.
    # Given the time, I will implement the schoolbook rank 48.
    # It is 100% correct and scores 1.0.
    # The island best is 1.143 (rank 42).
    # I might not beat it, but I will get a valid score.
    # Then I can try to optimize.
    
    rank = 48
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1
                V[j * p + k, idx] = 1
                W[k * n + i, idx] = 1
                idx += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

BET: {mechanism: "Implementing correct schoolbook rank-48 decomposition which is guaranteed to satisfy the tensor identity, fixing the empty matrices that caused the ValueError", q: 0.95, delta: [0.5, 1.5]}
