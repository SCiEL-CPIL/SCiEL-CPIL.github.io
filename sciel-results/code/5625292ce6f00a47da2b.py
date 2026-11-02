import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def verify_decomposition(U, V, W, n, m, p):
    """
    Verifies if U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    
    The tensor T is defined by T[a, b, c, d, i, k] = 1 if a==i, b==d, c==k else 0.
    The decomposition is sum_l U[a*m+b, l] * V[c*p+d, l] * W[k*n+i, l]
    """
    r = U.shape[1]
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    for i in range(n):
                        for k in range(p):
                            target = 1.0 if (a == i and c == k and b == d) else 0.0
                            s = np.dot(U_np[a * m + b, :], V_np[c * p + d, :] * W_np[k * n + i, :])
                            if not np.isclose(s, target, atol=1e-9):
                                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin

    n, m, p = DIMENSIONS
    
    # Strategy: Global Algebraic Elimination Search
    # 1. Initialize with Schoolbook decomposition (Rank 36).
    # 2. Iteratively attempt to eliminate one term (reduce rank by 1) by finding
    #    a linear combination of the remaining terms that matches the eliminated term's contribution.
    #    This is equivalent to checking if the tensor rank can be reduced.
    #    For M_{2,3,6}, we hope to reach Rank 33.
    
    # Schoolbook Decomposition
    # U index: a*m + b (0..5)
    # V index: c*p + d (0..17)
    # W index: k*n + i (0..11)
    # Term l corresponds to (a, b, c, d) such that it contributes to C[i, k] if a==i and c==k and b==d.
    # Actually, schoolbook terms are indexed by (b, c).
    # C[i, k] = sum_b A[i, b] B[b, k]
    # So for each b in 0..2, c in 0..2 (wait, B is 3x6, so c is row of B, d is col of B).
    # Let's use standard schoolbook:
    # C[i, k] = sum_{j=0}^{m-1} A[i, j] B[j, k]
    # Here m=3. So j=0,1,2.
    # B is 3x6. So B[j, k] where j in 0..2, k in 0..5.
    # A is 2x3. A[i, j] where i in 0..1, j in 0..2.
    # C is 2x6. C[i, k] where i in 0..1, k in 0..5.
    
    # Total terms = m * n * p? No.
    # Standard schoolbook rank is n*m*p? No, it's m * n * p? 
    # For C = AB, C_{ik} = sum_j A_{ij} B_{jk}.
    # Each term A_{ij} B_{jk} is a rank-1 tensor contribution?
    # The tensor is T_{a,b,c,d,i,k} = delta_{a,i} delta_{b,d} delta_{c,k}? No.
    # T is defined by the multiplication map.
    # The schoolbook decomposition has rank n*m*p? No.
    # The number of multiplications in schoolbook is m * n * p? No, it is m * (n*p)?
    # Let's count: For each j (0..m-1), for each i (0..n-1), for each k (0..p-1), we have a product A[i,j]*B[j,k].
    # This is n*m*p terms? No, the multiplication A[i,j]*B[j,k] is shared for all i,k? No.
    # In the tensor decomposition, a "term" is a rank-1 tensor u v w.
    # The schoolbook decomposition uses n*m*p terms?
    # Let's check M_{2,2,2}. Schoolbook rank 8. n*m*p = 8.
    # M_{2,3,6}. Schoolbook rank = 2*3*6 = 36.
    
    # So we start with Rank 36.
    U = np.zeros((n*m, n*m*p))
    V = np.zeros((m*p, n*m*p))
    W = np.zeros((p*n, n*m*p))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Term: A[i, j] * B[j, k] contributes to C[i, k]
                # U index for A[i, j] is i*m + j
                U[i*m + j, idx] = 1.0
                # V index for B[j, k] is j*p + k
                V[j*p + k, idx] = 1.0
                # W index for C[i, k] is k*n + i (per task: W[k*n+i, r])
                W[k*n + i, idx] = 1.0
                idx += 1
    
    current_rank = n*m*p
    target_rank = 33 # We aim for 33
    
    # Elimination Loop
    # We try to eliminate term `elim_idx` by expressing it as a linear combination of the others.
    # This is only possible if the tensor is dependent.
    # We iterate over all pairs of terms to try to eliminate one.
    
    # To speed up, we can use a heuristic: try eliminating terms that are "redundant" in the schoolbook sense?
    # Actually, in schoolbook, no term is linearly dependent on the others in the tensor space *if* we fix the other coefficients.
    # But we are allowed to change the coefficients of the remaining terms.
    # This is the "Rank Reduction" algorithm.
    
    # Let's try to eliminate one term at a time.
    # We will try all possible eliminations.
    # If successful, we reduce rank by 1 and repeat.
    
    # To make this feasible, we need a fast check.
    # Eliminating term t means:
    # There exist coefficients alpha_l for l != t such that:
    # U[:, t] = sum_{l!=t} alpha_l U[:, l]
    # V[:, t] = sum_{l!=t} alpha_l V[:, l]
    # W[:, t] = sum_{l!=t} alpha_l W[:, l]
    # This is a strict condition that is rarely met in schoolbook.
    
    # A more general approach: 
    # We want to find a new decomposition of rank r-1.
    # This is equivalent to finding a hyperplane in the space of tensors that contains the tensor.
    # This is complex.
    
    # Alternative Strategy: Use the known Rank 33 structure if we can find it.
    # Given the difficulty of general elimination, let's try a specific known structure for M_{2,3,6}.
    # Is Rank 33 possible?
    # Lower bound: max(2*6, 3*6, 2*3) = 18? No.
    # Lower bound: n*m*p / (n+m+p-2)? No.
    # Stothers bound for M_{2,3,6}?
    
    # Let's try to construct Rank 33 using a "Block" approach but with a valid Rank 11 block for M_{2,3,2} if possible, 
    # OR a valid Rank 11 block for M_{2,2,3}?
    # M_{2,3,6} can be seen as 3 blocks of M_{2,3,2}.
    # If Rank(M_{2,3,2}) = 11, then Rank(M_{2,3,6}) <= 33.
    # It is known that Rank(M_{2,3,2}) = 11.
    # The previous program failed to construct it correctly.
    # Let's try to construct a valid Rank 11 decomposition for M_{2,3,2} using a solver or known values.
    
    # Known Rank 11 decomposition for M_{2,3,2} (A: 2x3, B: 3x2, C: 2x2)
    # Source: "The rank of the 2x3x2 matrix multiplication tensor is 11"
    # We will hardcode a verified set of 11 terms.
    
    # Let's try to derive or find a valid set.
    # If we can't find one, we fall back to Rank 34 (2x17) which was working.
    
    # Let's try to use the "Strassen-like" extension.
    # Actually, let's try to use a simple heuristic search for Rank 33 on the full tensor.
    # We start with Rank 36.
    # We try to remove one term and solve for the rest.
    
    # This is too slow for 36 terms.
    
    # Let's go back to the Block Strategy but ensure correctness.
    # We will use the Rank 11 decomposition for M_{2,3,2}.
    # I will use a set of values from a reliable source or derive them.
    
    # Derivation of Rank 11 for M_{2,3,2}:
    # We know M_{2,2,2} has rank 7.
    # M_{2,3,2} = M_{2,2,2} + extra terms for A[:, 2].
    # A[:, 2] is a 2x1 vector. B is 3x2.
    # The extra part is A[:, 2] * B[2, :]? No.
    # C = A[:, 0:2] B[0:2, :] + A[:, 2] B[2, :]
    # The first part is 2x2 * 2x2 (Rank 7).
    # The second part is 2x1 * 1x2 (Rank 1? No, it's an outer product of a 2-vector and a 2-vector, which is rank 1 in the bilinear map sense? No.)
    # The term A[:, 2] B[2, :] contributes to C.
    # C[i, k] += A[i, 2] B[2, k].
    # This is a rank-1 tensor in the A-B-C space?
    # U vector: has 1 at A[i, 2].
    # V vector: has 1 at B[2, k].
    # W vector: has 1 at C[i, k].
    # So for each i, k, we have a term? No, it's a single term if we can factor it.
    # The tensor for A[:, 2] B[2, :] is:
    # T_extra[a, b, c, d, i, k] = 1 if a==i, b==2, c==2, d==k? No.
    # It is A[i, 2] * B[2, k].
    # This corresponds to a rank-1 tensor: u = e_{i,2} (for all i? No, u is a vector in A-space).
    # u = [0, 0, 1, 0, 0, 1]^T? (A[0,2]=1, A[1,2]=1)
    # v = [0, 0, 1, 0, 0, 1]^T? (B[2,0]=1, B[2,1]=1) -> B is 3x2. B[2,0] is index 2*2+0=4, B[2,1] is 5.
    # w = [1, 1, 1, 1]^T? (C[0,0], C[0,1], C[1,0], C[1,1])
    # Let's check:
    # Sum_l u_a v_c w_i = u_{i,2} v_{2,k} w_{i,k}?
    # If u has 1s at A[0,2] and A[1,2], v has 1s at B[2,0] and B[2,1], w has 1s at all C.
    # Then the product is A[i,2] B[2,k] * 1.
    # This exactly matches the extra term!
    # So the extra part is Rank 1.
    # Total Rank = 7 (for 2x2x2) + 1 (for extra) = 8?
    # Wait, the 2x2x2 part uses A[:, 0:2] and B[0:2, :].
    # The Strassen terms for 2x2x2 use specific linear combinations of A and B entries.
    # They do NOT use A[:, 2] or B[2, :].
    # So the supports are disjoint?
    # Strassen terms use A[0,0], A[0,1], A[1,0], A[1,1] and B[0,0], B[0,1], B[1,0], B[1,1].
    # The extra term uses A[0,2], A[1,2] and B[2,0], B[2,1].
    # Since the supports are disjoint, we can just add the rank-1 term to the Strassen decomposition.
    # Total Rank = 7 + 1 = 8?
    # But the rank of M_{2,3,2} is 11. Why?
    # Because the Strassen decomposition of 2x2x2 is not "compatible" with the extra term in a way that allows simple addition?
    # No, if the supports are disjoint, the rank is additive.
    # Let's check if M_{2,3,2} rank is actually 8?
    # Lower bound for M_{2,3,2}: max(2*2, 3*2, 2*3) = 6?
    # Actually, rank(M_{2,3,2}) is known to be 11.
    # Why is it not 8?
    # Because the Strassen terms for 2x2x2 involve linear combinations that might interfere?
    # No, if we define the decomposition as the union of the 7 Strassen terms and 1 extra term, it is a valid decomposition of rank 8.
    # Let's verify this hypothesis.
    # If Rank(M_{2,3,2}) = 8, then Rank(M_{2,3,6}) <= 3 * 8 = 24.
    # This would be a huge improvement!
    
    # Let's test the Rank 8 hypothesis for M_{2,3,2}.
    # U (6x8), V (6x8), W (4x8).
    # Terms 0-6: Strassen for A[:, 0:2], B[0:2, :].
    # Term 7: Extra for A[:, 2], B[2, :].
    
    U8 = np.zeros((6, 8))
    V8 = np.zeros((6, 8))
    W8 = np.zeros((4, 8))
    
    # Strassen for 2x2 (indices in U: 0,1,3,4; in V: 0,1,2,3; in W: 0,1,2,3)
    # Note: In M_{2,3,2}, A is 2x3. Indices: A[0,0]=0, A[0,1]=1, A[0,2]=2, A[1,0]=3, A[1,1]=4, A[1,2]=5.
    # B is 3x2. Indices: B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3, B[2,0]=4, B[2,1]=5.
    # C is 2x2. Indices: C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3. (W index k*n+i. n=2. C[i,k] -> W[k*2+i])
    # Wait, task says W[k*n+i]. For n=2, W[k*2+i].
    # C[0,0] -> W[0]. C[0,1] -> W[1]. C[1,0] -> W[2]. C[1,1] -> W[3].
    
    # Strassen Terms for A[:, 0:2] (cols 0,1) and B[0:2, :] (rows 0,1).
    # A entries: A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4.
    # B entries: B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3.
    
    # T0: (A00+A11)(B00+B11)
    U8[0, 0] = 1; U8[4, 0] = 1
    V8[0, 0] = 1; V8[3, 0] = 1
    W8[0, 0] = 1; W8[1, 0] = 1; W8[2, 0] = 1; W8[3, 0] = 1
    
    # T1: (A10+A11)B00
    U8[3, 1] = 1; U8[4, 1] = 1
    V8[0, 1] = 1
    W8[2, 1] = 1; W8[3, 1] = -1
    
    # T2: A00(B10-B11)
    U8[0, 2] = 1
    V8[2, 2] = 1; V8[3, 2] = -1
    W8[1, 2] = 1; W8[3, 2] = 1
    
    # T3: A11(B01-B00)
    U8[4, 3] = 1
    V8[1, 3] = 1; V8[0, 3] = -1
    W8[0, 3] = 1; W8[2, 3] = 1
    
    # T4: (A00+A01)B11
    U8[0, 4] = 1; U8[1, 4] = 1
    V8[3, 4] = 1
    W8[0, 4] = -1; W8[1, 4] = 1
    
    # T5: (A10-A00)(B00+B01)
    U8[3, 5] = 1; U8[0, 5] = -1
    V8[0, 5] = 1; V8[1, 5] = 1
    W8[2, 5] = 1; W8[3, 5] = 1
    
    # T6: (A01-A11)(B10+B11)
    U8[1, 6] = 1; U8[4, 6] = -1
    V8[2, 6] = 1; V8[3, 6] = 1
    W8[0, 6] = 1; W8[3, 6] = 1
    
    # T7: Extra term A[:, 2] B[2, :]
    # U: A[0,2]=2, A[1,2]=5.
    # V: B[2,0]=4, B[2,1]=5.
    # W: All 1s.
    U8[2, 7] = 1; U8[5, 7] = 1
    V8[4, 7] = 1; V8[5, 7] = 1
    W8[0, 7] = 1; W8[1, 7] = 1; W8[2, 7] = 1; W8[3, 7] = 1
    
    # Verify M_{2,3,2} with Rank 8
    if verify_decomposition(U8, V8, W8, 2, 3, 2):
        # If Rank 8 works for M_{2,3,2}, then Rank 24 works for M_{2,3,6}.
        # Construct M_{2,3,6} as 3 blocks.
        U_comb = np.zeros((6, 24))
        V_comb = np.zeros((18, 24))
        W_comb = np.zeros((12, 24))
        
        for b in range(3):
            U_comb[:, b*8:(b+1)*8] = U8
            # V mapping: B is 3x6.
            # Block b uses B[:, b*2 : b*2+2].
            # V index for B[j, k] is j*6 + k.
            # In U8/V8, V index is j*2 + k_local.
            for j in range(3):
                for k_local in range(2):
                    k_global = b*2 + k_local
                    V_comb[j*6 + k_global, b*8:(b+1)*8] = V8[j*2 + k_local, :]
            
            # W mapping: C is 2x6.
            # Block b uses C[:, b*2 : b*2+2].
            # W index for C[i, k] is k*2 + i.
            # In U8/W8, W index is k_local*2 + i.
            for i in range(2):
                for k_local in range(2):
                    k_global = b*2 + k_local
                    W_comb[k_global*2 + i, b*8:(b+1)*8] = W8[k_local*2 + i, :]
        
        if verify_decomposition(U_comb, V_comb, W_comb, n, m, p):
            return {
                "U": U_comb.tolist(),
                "V": V_comb.tolist(),
                "W": W_comb.tolist()
            }
    
    # If Rank 8 fails, fall back to Rank 34 (2x17) which was verified in previous runs.
    # Re-implement the Rank 34 solution.
    
    U233 = np.zeros((6, 17))
    V233 = np.zeros((9, 17))
    W233 = np.zeros((6, 17))
    
    # Hardcoded Rank 17 for M_{2,3,3}
    U233[0, 0] = 1; U233[4, 0] = 1
    V233[0, 0] = 1; V233[4, 0] = 1
    W233[0, 0] = 1; W233[3, 0] = 1
    
    U233[3, 1] = 1; U233[4, 1] = 1
    V233[0, 1] = 1
    W233[1, 1] = 1; W233[3, 1] = -1
    
    U233[0, 2] = 1
    V233[1, 2] = 1; V233[4, 2] = -1
    W233[2, 2] = 1; W233[3, 2] = 1
    
    U233[4, 3] = 1
    V233[3, 3] = 1; V233[0, 3] = -1
    W233[0, 3] = 1; W233[1, 3] = 1
    
    U233[0, 4] = 1; U233[1, 4] = 1
    V233[4, 4] = 1
    W233[0, 4] = -1; W233[2, 4] = 1
    
    U233[3, 5] = 1; U233[0, 5] = -1
    V233[0, 5] = 1; V233[1, 5] = 1
    W233[3, 5] = 1
    
    U233[1, 6] = 1; U233[4, 6] = -1
    V233[3, 6] = 1; V233[4, 6] = 1
    W233[0, 6] = 1
    
    U233[0, 7] = 1
    V233[2, 7] = 1
    W233[4, 7] = 1
    
    U233[1, 8] = 1
    V233[5, 8] = 1
    W233[4, 8] = 1
    
    U233[3, 9] = 1
    V233[2, 9] = 1
    W233[5, 9] = 1
    
    U233[4, 10] = 1
    V233[5, 10] = 1
    W233[5, 10] = 1
    
    U233[2, 11] = 1
    V233[6, 11] = 1
    W233[0, 11] = 1
    
    U233[2, 12] = 1
    V233[7, 12] = 1
    W233[2, 12] = 1
    
    U233[2, 13] = 1
    V233[8, 13] = 1
    W233[4, 13] = 1
    
    U233[5, 14] = 1
    V233[6, 14] = 1
    W233[1, 14] = 1
    
    U233[5, 15] = 1
    V233[7, 15] = 1
    W233[3, 15] = 1
    
    U233[5, 16] = 1
    V233[8, 16] = 1
    W233[5, 16] = 1
    
    U_comb = np.zeros((6, 34))
    V_comb = np.zeros((18, 34))
    W_comb = np.zeros((12, 34))
    
    U_comb[:, :17] = U233
    U_comb[:, 17:] = U233
    
    for j in range(3):
        for k in range(3):
            V_comb[j * 6 + k, :17] = V233[j * 3 + k]
            V_comb[j * 6 + 3 + k, 17:] = V233[j * 3 + k]
            
    for k in range(3):
        for i in range(2):
            W_comb[k * 2 + i, :17] = W233[k * 2 + i]
            W_comb[(k + 3) * 2 + i, 17:] = W233[k * 2 + i]
            
    if verify_decomposition(U_comb, V_comb, W_comb, n, m, p):
        return {
            "U": U_comb.tolist(),
            "V": V_comb.tolist(),
            "W": W_comb.tolist()
        }
    
    # Final fallback: Schoolbook Rank 36
    U_school = np.zeros((6, 36))
    V_school = np.zeros((18, 36))
    W_school = np.zeros((12, 36))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, idx] = 1.0
                V_school[j*p + k, idx] = 1.0
                W_school[k*n + i, idx] = 1.0
                idx += 1
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }

if __name__ == "__main__":
    construct()
