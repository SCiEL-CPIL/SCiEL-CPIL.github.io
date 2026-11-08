"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # n=3, m=4, p=4
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Index convention: U[i*m+j], V[j*p+k], W[k*n+i]
    
    nm = n * m  # 12
    mp = m * p  # 16
    pn = p * n  # 12
    
    # The tensor T has T[(i,j),(j',k),(i'',k'')] = 1 iff i=i'', j=j', k=k'
    # So for each (i,j,k) with matching indices, T contributes to the product
    
    # Build the "flattened" tensor for verification
    # T[(i,j), (j',k), (i'',k'')] where the contraction gives sum over j of A[i,j]*B[j,k]
    # In the decomposition: sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i'', r]
    # We need this to equal delta(i,i'') * delta(j,j') * delta(k,k')
    
    # Let's try to find a low-rank decomposition
    # Start with schoolbook (rank 48) and try to reduce
    
    def build_tensors(U, V, W):
        """Verify that U, V, W give correct matrix multiplication."""
        # For each (i,j), (j',k), (i'',k'') check:
        # sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i'', r] == 1 if i==i'' and j==j' and k==k' else 0
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for ip in range(n):
                            for kp in range(p):
                                expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                actual = 0.0
                                for r in range(len(U[0])):
                                    actual += U[i*m+j][r] * V[jp*p+k][r] * W[kp*n+ip][r]
                                if abs(actual - expected) > 1e-9:
                                    return False
        return True
    
    def schoolbook():
        rank = n * m * p
        U = [[0.0] * rank for _ in range(nm)]
        V = [[0.0] * rank for _ in range(mp)]
        W = [[0.0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
        return U, V, W, rank
    
    # Try to find a better decomposition using a constructive approach
    # Idea: use the fact that we can share computations
    
    # Let's try rank 36: use a 3x3 block structure with sharing
    # Actually, let's try to use a known technique: for each pair (i, k), 
    # we need sum_j A[i,j] B[j,k]. 
    
    # Try rank 32: 
    # We can think of it as 3 rows of A, each multiplied by 4x4 B.
    # For each row i of A, we compute A[i,:] * B (1x4 times 4x4 = 1x4).
    # The 1x4 times 4x4 has rank at most 4*4=16 per row, so 3*16=48 total.
    # But we can share the B factor.
    
    # Let's try a more clever approach: 
    # Use the fact that we can write the multiplication as:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # Try rank 36: 
    # For each i (3) and k (4), we have 12 outputs.
    # Each output needs 4 multiplications naively.
    # With rank r, we have r "intermediate" values, each is a product of one U and one V.
    
    # Let me try to construct a rank-36 decomposition:
    # Split the 4 columns of B into two groups of 2.
    # For each i, compute partial products.
    
    # Actually, let's try a systematic search with small rank
    # using a greedy / iterative approach
    
    best = schoolbook()
    U_best, V_best, W_best, rank_best = best
    
    # Try rank 40, 36, 32, 28
    for target_rank in [40, 38, 36, 34, 32, 30]:
        if time.time() > deadline - 30:
            break
        
        # Initialize with random or structured values
        # Use a structured initialization based on schoolbook
        U = [[0.0] * target_rank for _ in range(nm)]
        V = [[0.0] * target_rank for _ in range(mp)]
        W = [[0.0] * target_rank for _ in range(pn)]
        
        # Initialize: use first target_rank terms of schoolbook
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col >= target_rank:
                        break
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
                if col >= target_rank:
                    break
            if col >= target_rank:
                break
        
        # Now try to optimize: use gradient-free iterative improvement
        # For each term r, adjust U[:, r], V[:, r], W[:, r] to better match
        
        # Use a simple iterative method:
        # For each (i,j,jp,k,kp,ip) where expected != 0, we want the contribution to be 1
        # For expected = 0, we want contribution to be 0
        
        # This is a rank-constrained tensor completion problem.
        # Use ALS (Alternating Least Squares) style updates.
        
        # The tensor equation is:
        # sum_r U[i*m+j, r] * V[jp*p+k, r] * W[kp*n+ip, r] = T[(i,j),(jp,k),(kp,ip)]
        
        # Let's flatten: for fixed r, we're solving for three vectors.
        # This is a 3-way tensor problem. Use a simple iterative approach.
        
        # Actually, let's use a different strategy: 
        # Fix V and W, solve for U; then fix U and W, solve for V; etc.
        # But the equations are nonlinear.
        
        # Simpler: use a perturbation approach.
        # Start from schoolbook truncated, then add small perturbations to satisfy constraints.
        
        # Let's try a direct algebraic construction for rank 36:
        # 
        # For 3x4 x 4x4, consider the following:
        # We have 3x4 = 12 elements in A, 4x4 = 16 in B, 3x4 = 12 in C.
        # 
        # Try to use a rank-12 decomposition for the "diagonal" part and 
        # handle off-diagonal with shared terms.
        
        # Let me try a specific construction:
        # For each i (0,1,2) and each k (0,1,2,3):
        # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
        #
        # Rank 48: one term per (i,j,k)
        # Can we merge terms?
        
        # Try rank 36: use the fact that B is 4x4, and we can use a rank-4 
        # decomposition of the identity-like structure... 
        
        # Actually, let's just try to verify if rank 48 is truly minimal or if we can do 44 or 40.
        # For now, return schoolbook which is guaranteed correct.
        
        # Let's try one more thing: use a 2x2 Strassen-like trick on sub-blocks.
        # Split A into 3 rows, each 1x4. Split B into 4x4.
        # For each row of A, we do 1x4 * 4x4. 
        # 1x4 * 4x4 can be done in rank 4 (schoolbook) or maybe less?
        # No, 1x4 * 4x4 requires rank 4 since it's a single row times full matrix.
        
        # So total is 3 * 4 = 12... wait, that's not right.
        # 1x4 * 4x4: the result is 1x4. Each output C[k] = sum_j A[j] * B[j,k].
        # This is 4 outputs, each a dot product of length 4.
        # Rank for 1x4 * 4x4: at least 4 (since there are 4 independent outputs each depending on 4 inputs).
        # Actually, the bilinear complexity of 1xn * nxn is n^2? No.
        # For 1x1 * 1x1: rank 1.
        # For 1x2 * 2x2: C0 = A0*B00 + A1*B10, C1 = A0*B01 + A1*B11. Rank 4? 
        # Can we do it in rank 3? 
        # T[(a),(b,c),(d,e)] ... let's see. The tensor has entries where the a-index matches with b in B and c in A... 
        # I think 1x2*2x2 has rank 4.
        
        # So 1x4*4x4 has rank 16? That gives 3*16=48 total. Same as schoolbook.
        
        # Hmm. Let me just return the schoolbook solution which is rank 48.
        # But wait - maybe we can do better by sharing between rows of A.
        
        # Key insight: all three rows of A share the same B matrix!
        # So we can potentially share the B factors.
        
        # Let's try rank 36: 
        # Use 12 "base" terms for the diagonal (i matches, j matches, k matches) 
        # and share some off-diagonal terms.
        
        # Actually, I realize I should just try to find ANY valid decomposition 
        # with rank less than 48. Let me try rank 44 first.
        
        pass  # Skip this target_rank for now
    
    # Return the schoolbook solution as it's guaranteed correct
    U, V, W, rank = schoolbook()
    
    # Convert to integer/half-integer format (they're already integers)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
