import numpy as np
import time

def construct():
    n, m, p = 2, 2, 4
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    # Schoolbook rank = 16
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    # Try to find a rank-10 decomposition using structured algebraic approach
    # For 2x2 * 2x4, we can try to exploit the rectangular structure
    
    # The key insight: we can use a modified Strassen-like approach
    # where we split the 2x4 matrix into two 2x2 blocks and use
    # cross-term optimizations to reduce the total rank.
    
    # Let's try a rank-10 construction based on the following idea:
    # A (2x2) times B (2x4) where B = [B1 | B2] with B1, B2 both 2x2
    # C = A*B = [A*B1 | A*B2]
    
    # Standard Strassen gives rank 7 for each 2x2*2x2, total 14.
    # To get rank 10, we need to share 4 products between the two blocks.
    
    # Let me try a direct algebraic construction for rank 10.
    # We'll define 10 rank-1 terms.
    
    # Actually, let me try a different approach: use the fact that 
    # for rectangular matrices, we can sometimes do better by combining
    # the computations.
    
    # Let's try rank 11, 12, 13, 14 in order, starting with the most promising.
    
    # First, let's try a rank-14 solution (two independent Strassen)
    # This is known to work.
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition is correct."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Reconstruct tensor
        T_recon = np.einsum('ad,bd,cd->abc', U_np, V_np, W_np)
        
        # True tensor
        T_true = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T_true[a, b, c] = 1.0
                    
        return np.allclose(T_recon, T_true, atol=1e-9)

    # Try rank-14 (two Strassen blocks)
    try:
        # Strassen coefficients for M_{2,2,2}
        u1 = [1, 0, 1, 1]
        u2 = [0, 0, 1, 1]
        u3 = [1, 0, 0, 0]
        u4 = [0, 0, 0, 1]
        u5 = [1, 1, 0, 0]
        u6 = [-1, 0, 1, 0]
        u7 = [0, 1, 0, -1]
        
        v1 = [1, 0, 0, 1]
        v2 = [1, 0, 0, 0]
        v3 = [0, 0, 1, -1]
        v4 = [-1, 1, 0, 0]
        v5 = [0, 0, 0, 1]
        v6 = [1, 1, 0, 0]
        v7 = [0, 0, 1, 1]
        
        w1 = [1, 0, 0, 1]
        w2 = [0, 1, 0, -1]
        w3 = [0, 0, 1, 1]
        w4 = [-1, 1, 0, 0]
        w5 = [-1, 0, 1, 0]
        w6 = [0, 0, 0, -1]
        w7 = [1, 0, 0, 0]
        
        U_strassen = [u1, u2, u3, u4, u5, u6, u7]
        V_strassen = [v1, v2, v3, v4, v5, v6, v7]
        W_strassen = [w1, w2, w3, w4, w5, w6, w7]
        
        rank = 14
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Left block
        for k in range(7):
            for i in range(4):
                U[i][k] = U_strassen[k][i]
            V[0][k] = V_strassen[k][0]
            V[1][k] = V_strassen[k][1]
            V[4][k] = V_strassen[k][2]
            V[5][k] = V_strassen[k][3]
            W[0][k] = W_strassen[k][0]
            W[1][k] = W_strassen[k][1]
            W[2][k] = W_strassen[k][2]
            W[3][k] = W_strassen[k][3]
            
        # Right block
        for k in range(7):
            for i in range(4):
                U[i][k + 7] = U_strassen[k][i]
            V[2][k + 7] = V_strassen[k][0]
            V[3][k + 7] = V_strassen[k][1]
            V[6][k + 7] = V_strassen[k][2]
            V[7][k + 7] = V_strassen[k][3]
            W[4][k + 7] = W_strassen[k][0]
            W[5][k + 7] = W_strassen[k][1]
            W[6][k + 7] = W_strassen[k][2]
            W[7][k + 7] = W_strassen[k][3]
            
        if verify_decomposition(U, V, W, rank):
            return {"U": U, "V": V, "W": W}
    except:
        pass

    # Now try to find a rank-10 decomposition using a more clever approach
    # The idea: we can potentially share some of the Strassen products
    # between the two 2x2 blocks.
    
    # Let me try a direct construction for rank 10.
    # We'll use 10 rank-1 terms with carefully chosen coefficients.
    
    # For the 2x2 * 2x4 case, let's think about what we need:
    # C00 = a*e + b*g
    # C10 = c*e + d*g
    # C01 = a*f + b*h
    # C11 = c*f + d*h
    # C02 = a*e' + b*g'
    # C12 = c*e' + d*g'
    # C03 = a*f' + b*h'
    # C13 = c*f' + d*h'
    
    # That's 8 outputs, each a sum of 2 products.
    # Total "work" is 16 products if done naively.
    
    # Strassen reduces each 2x2*2x2 to 7 products.
    # If we can share 4 products between the two blocks, we get 14 - 4 = 10.
    
    # Let me try to construct this explicitly.
    
    # Actually, let me try a different approach: use a numerical search
    # to find a rank-10 decomposition.
    
    # But first, let's check if we're running out of time.
    if time.time() > deadline - 20:
        return {"U": U_school, "V": V_school, "W": W_school}

    # Try a rank-10 construction based on shared Strassen terms
    # This is a heuristic approach - we'll define the 10 terms and verify.
    
    try:
        # Define 10 rank-1 terms
        # The idea is to use 7 terms for the left block and 7 for the right,
        # but share 4 of them.
        
        # Let me try a specific construction:
        # Terms 1-4: shared between both blocks
        # Terms 5-7: left block only
        # Terms 8-10: right block only
        
        # But this doesn't quite work because the U, V, W factors need to be consistent.
        
        # Let me try a different approach: use a linear algebra method to find
        # a rank-10 decomposition.
        
        # Actually, let me just try a few specific constructions and verify them.
        
        # Construction attempt 1:
        # Use the following 10 terms:
        
        # For the 2x2*2x4 case, let's define:
        # A = [[a, b], [c, d]]
        # B = [[e, f, e', f'], [g, h, g', h']]
        # C = [[c00, c01, c02, c03], [c10, c11, c12, c13]]
        
        # where c00 = a*e + b*g, etc.
        
        # Let me try to define 10 rank-1 terms that span the tensor.
        
        # Term 1: (a+d)(e+h) -> contributes to c00, c01, c10, c11
        # Term 2: (c+d)(e+h) -> contributes to c10, c11
        # Term 3: a*e -> contributes to c00
        # Term 4: d*h -> contributes to c11
        # Term 5: (a+b)(e+f) -> contributes to c00, c01
        # Term 6: (c-a)(e-g) -> contributes to c10, c00
        # Term 7: (b-d)(f-h) -> contributes to c01, c11
        # Term 8: (a+d)(e'+h') -> contributes to c02, c03, c12, c13
        # Term 9: (c+d)(e'+h') -> contributes to c12, c13
        # Term 10: a*e' -> contributes to c02
        
        # This doesn't quite work because we're missing some terms.
        
        # Let me try a more systematic approach.
        
        # I'll use a simple heuristic: start with the rank-14 solution and try
        # to merge some of the terms.
        
        # For now, let's just return the rank-14 solution if it works.
        
        # Actually, I already verified it above. Let me just return the schoolbook
        # if we can't find a better solution in time.
        
        pass
    except:
        pass

    # If we get here, return the best we have
    return {"U": U_school, "V": V_school, "W": W_school}
