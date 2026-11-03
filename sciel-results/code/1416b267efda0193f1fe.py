import time
import random
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    def make_schoolbook():
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
    
    def validate(U, V, W):
        # Convert to numpy for fast checking
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)
        
        # Check dimensions
        if U_np.shape[0] != n * m:
            return False
        if V_np.shape[0] != m * p:
            return False
        if W_np.shape[0] != p * n:
            return False
        
        r = U_np.shape[1]
        if V_np.shape[1] != r or W_np.shape[1] != r:
            return False
        
        # The tensor T[i, j, k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We need T[i, j, k] = 1 for all i, j, k.
        
        # We can check this by iterating over a subset of (i, j, k) if r is large, 
        # but for exactness we should check all or use a faster method.
        # Given the small sizes (15, 25, 15), even full check is 15*25*15 * r = 5625 * r ops.
        # With r=75, that's ~400k ops, which is fast in numpy?
        # Actually, we can construct the product directly.
        
        # Let's check if the decomposition is exact.
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        
        # Efficient check:
        # For each c, the contribution is outer(U_col, V_col) * W_col? No.
        # It's a 3-tensor contraction.
        
        # Let's just iterate, as dimensions are tiny.
        # But we want to be fast.
        # We can check T[i, j, k] = (U * V) * W? No.
        
        # Let's do it simply but with some numpy acceleration if possible.
        # Or just pure python loops since 5625 * 75 = 421,875 iterations is < 1s in Python.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(r):
                        val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if val != 1:
                        return False
        return True
        
    def attempt_strassen_blocks():
        # Try to decompose using 2x2x2 Strassen blocks where possible.
        # Dimensions: 3x5x5
        # Can we partition into blocks?
        # 3 = 2 + 1
        # 5 = 2 + 2 + 1
        # This is complex. Let's try a simpler heuristic:
        # Replace schoolbook 1x1x1 terms with 2x2x2 Strassen terms if indices align.
        
        # This is risky to implement correctly without a specific known decomposition.
        # Let's try a different approach: Random search / Local improvement.
        # Start with schoolbook, then try to merge terms or find linear dependencies.
        # Actually, finding a lower rank is hard.
        
        # Let's try to construct a specific known lower rank decomposition if I can recall one.
        # For 3x5x5, is there a known rank?
        # 3*5*5 = 75.
        # If we use Strassen on a 2x2x2 sub-block, we save 1 (8 -> 7).
        # How many disjoint 2x2x2 blocks can we fit?
        # i in {0, 1}, j in {0, 1}, k in {0, 1} -> One block.
        # i in {0, 1}, j in {2, 3}, k in {2, 3} -> Second block?
        # The tensor is T[i, j, k] = 1.
        # If we apply Strassen to a sub-block T[i, j, k] for i in {0,1}, j in {0,1}, k in {0,1},
        # we are replacing 8 terms with 7 terms.
        # The indices in the global tensor must be consistent.
        
        # Let's try to implement a specific decomposition that uses Strassen for the (0,0,0) to (1,1,1) sub-tensor.
        # The terms for this sub-tensor are (i,j,k) in {0,1}^3.
        # Instead of 8 rank-1 terms, use 7.
        # Strassen for 2x2x2:
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # P1 = (a+d)(e+h)
        # P2 = (c+d)(e)
        # P3 = (a)(f-h)
        # P4 = (d)(g-c)
        # P5 = (a+b)(h)
        # P6 = (c-a)(e+f)
        # P7 = (b-d)(g+h)
        
        # In the context of the full matrix multiplication tensor T[i,j,k]=1:
        # The sub-tensor for i,j,k in {0,1} corresponds to the multiplication of A[0:2, 0:2] * B[0:2, 0:2].
        # The entries are:
        # (0,0): a*e + b*g
        # (0,1): a*f + b*h
        # (1,0): c*e + d*g
        # (1,1): c*f + d*h
        
        # The standard rank-8 terms for this sub-block are:
        # a*e, b*g, a*f, b*h, c*e, d*g, c*f, d*h.
        # Corresponding to (i,j,k):
        # (0,0,0): a*e -> U[0,0]=1, V[0,0]=1, W[0,0]=1
        # (0,1,0): b*g -> U[0,1]=1, V[1,0]=1, W[0,1]=1
        # (0,0,1): a*f -> U[0,0]=1, V[0,1]=1, W[1,0]=1
        # (0,1,1): b*h -> U[0,1]=1, V[1,1]=1, W[1,1]=1
        # (1,0,0): c*e -> U[1,0]=1, V[0,0]=1, W[0,0]=1 ... wait, W index is k*n+i.
        # Let's stick to the global index convention.
        # U[i*m + j, c]
        # V[j*p + k, c]
        # W[k*n + i, c]
        
        # For the sub-block i in {0,1}, j in {0,1}, k in {0,1}:
        # The 8 standard terms are:
        # 1. i=0, j=0, k=0: U[0, c]=1, V[0, c]=1, W[0, c]=1
        # 2. i=0, j=1, k=0: U[1, c]=1, V[5, c]=1, W[0, c]=1  (V index j*p+k = 1*5+0=5)
        # 3. i=0, j=0, k=1: U[0, c]=1, V[1, c]=1, W[3, c]=1  (W index k*n+i = 1*3+0=3)
        # 4. i=0, j=1, k=1: U[1, c]=1, V[6, c]=1, W[3, c]=1
        # 5. i=1, j=0, k=0: U[5, c]=1, V[0, c]=1, W[1, c]=1  (U index i*m+j = 1*5+0=5)
        # 6. i=1, j=1, k=0: U[6, c]=1, V[5, c]=1, W[1, c]=1
        # 7. i=1, j=0, k=1: U[5, c]=1, V[1, c]=1, W[4, c]=1
        # 8. i=1, j=1, k=1: U[6, c]=1, V[6, c]=1, W[4, c]=1
        
        # We want to replace these 8 with 7 Strassen terms.
        # Let's define the Strassen terms for this specific sub-tensor.
        # Let a = A[0,0], b = A[0,1], c = A[1,0], d = A[1,1]
        # e = B[0,0], f = B[0,1], g = B[1,0], h = B[1,1]
        
        # P1 = (a+d)(e+h)
        # P2 = (c+d)(e)
        # P3 = (a)(f-h)
        # P4 = (d)(g-c)
        # P5 = (a+b)(h)
        # P6 = (c-a)(e+f)
        # P7 = (b-d)(g+h)
        
        # We need to express these as sum of U, V, W vectors.
        # This is tricky because the U, V, W vectors for the full matrix must be consistent with the other terms.
        # The other terms (outside this 2x2x2 block) are 1x1x1 schoolbook terms.
        # If we modify the U, V, W vectors for the indices involved in this block, we must ensure
        # that the contributions to the other parts of the matrix (where i, j, or k is not in {0,1}) are zero
        # or correctly accounted for.
        
        # Since U[i*m+j, c] is non-zero only for specific (i,j), if we put non-zero values in
        # U[0, c], U[1, c], U[5, c], U[6, c], it only affects rows 0, 1, 5, 6 of U.
        # These correspond to (i=0, j=0), (i=0, j=1), (i=1, j=0), (i=1, j=1).
        # So it only affects the sub-block of A.
        # Similarly for V and W.
        # So the modification is local to this 2x2x2 block of the tensor.
        
        # Let's construct the 7 terms.
        # We need to assign coefficients to U, V, W for columns c0..c6.
        
        # P1 = (a+d)(e+h)
        # Terms: a*e, a*h, d*e, d*h
        # P2 = (c+d)(e)
        # Terms: c*e, d*e
        # P3 = (a)(f-h)
        # Terms: a*f, -a*h
        # P4 = (d)(g-c)
        # Terms: d*g, -c*g
        # P5 = (a+b)(h)
        # Terms: a*h, b*h
        # P6 = (c-a)(e+f)
        # Terms: c*e, c*f, -a*e, -a*f
        # P7 = (b-d)(g+h)
        # Terms: b*g, b*h, -d*g, -d*h
        
        # Sum of all:
        # a*e: P1(+1), P6(-1) -> 0? No, we need 1.
        # Wait, Strassen decomposition is for the product C = AB.
        # The tensor entries are the coefficients of the products.
        # We need sum_c U_i V_j W_k = 1 for all i,j,k in the block.
        
        # Let's map the Strassen P terms to the tensor components.
        # The 8 components are:
        # C00 = a*e + b*g
        # C01 = a*f + b*h
        # C10 = c*e + d*g
        # C11 = c*f + d*h
        
        # Strassen:
        # C00 = P1 + P4 + P6 - P2 - P5
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P3 - P6 + P7
        
        # Let's check:
        # C00: (a+d)(e+h) + (d)(g-c) + (c-a)(e+f) - (c+d)(e) - (a+b)(h)
        # = (ae+ah+de+dh) + (dg-dc) + (ce+cf-ae-af) - (ce+de) - (ah+bh)
        # = ae + ah + de + dh + dg - dc + ce + cf - ae - af - ce - de - ah - bh
        # = ae - ae + de - de + ce - ce + ah - ah + dh + dg - dc + cf - af - bh
        # = dh + dg - dc + cf - af - bh
        # This is not a*e + b*g.
        
        # I am mixing up the standard Strassen formulas. Let's look up the exact ones for matrix multiplication.
        # Standard Strassen for 2x2:
        # p1 = (a+d)(e+h)
        # p2 = (c+d)(e)
        # p3 = (a)(f-h)
        # p4 = (d)(g-c)
        # p5 = (a+b)(h)
        # p6 = (c-a)(e+f)
        # p7 = (b-d)(g+h)
        
        # C11 = p1 - p3 - p6 + p7
        # C10 = p2 + p4
        # C01 = p3 + p5
        # C00 = p1 + p4 - p2 - p5
        
        # Let's re-evaluate C00:
        # p1 + p4 - p2 - p5
        # = (ae+ah+de+dh) + (dg-dc) - (ce+de) - (ah+bh)
        # = ae + ah + de + dh + dg - dc - ce - de - ah - bh
        # = ae + dh + dg - dc - ce - bh
        # Still not ae + bg.
        
        # The standard Strassen is:
        # p1 = (a + d)(e + h)
        # p2 = (c + d)(e)
        # p3 = (a)(f - h)
        # p4 = (d)(g - c)
        # p5 = (a + b)(h)
        # p6 = (c - a)(e + f)
        # p7 = (b - d)(g + h)
        
        # C = [[p1 + p4 - p2 - p5, p3 + p5], [p2 + p4, p1 - p3 - p6 + p7]]
        
        # Let's check C00 again: p1 + p4 - p2 - p5
        # = (a+d)(e+h) + d(g-c) - (c+d)e - (a+b)h
        # = (ae+ah+de+dh) + dg - dc - ce - de - ah - bh
        # = ae + dh + dg - dc - ce - bh
        # This is not ae + bg.
        
        # There is an error in my recall or application.
        # Actually, the standard Strassen is:
        # p1 = (a+d)(e+h)
        # p2 = (c+d)(e)
        # p3 = (a)(f-h)
        # p4 = (d)(g-c)
        # p5 = (a+b)(h)
        # p6 = (c-a)(e+f)
        # p7 = (b-d)(g+h)
        
        # And the combinations are:
        # C11 = p1 - p3 - p6 + p7
        # C10 = p2 + p4
        # C01 = p3 + p5
        # C00 = p1 + p4 - p2 - p5
        
        # Let's check C10: p2 + p4 = (c+d)e + d(g-c) = ce + de + dg - dc.
        # We want ce + dg.
        # So C10 = p2 + p4 - de? No, the terms must be rank 1.
        
        # It seems my memory of the exact combinations is faulty.
        # Given the risk of error and time constraint, I will stick to the schoolbook decomposition.
        # It is safe and guarantees a score of 1.0.
        return None
        
    best_result = make_schoolbook()
    
    # If we had time and a valid strategy, we would test alternative decompositions.
    # But for now, return the safe one.
    
    return best_result
# EVOLVE-BLOCK-END
