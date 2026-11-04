import numpy as np
import time

def construct():
    deadline = time.time() + 120
    n, m, p = 2, 3, 6
    rank = 33

    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)

    # Strassen factors for 2x2x2
    S_U = [
        np.array([[1, 0], [0, 1]]),
        np.array([[0, 0], [1, 1]]),
        np.array([[1, 0], [0, 0]]),
        np.array([[0, 0], [0, 1]]),
        np.array([[1, 1], [0, 0]]),
        np.array([[-1, 0], [1, 0]]),
        np.array([[0, 1], [0, -1]])
    ]
    S_V = [
        np.array([[1, 0], [0, 1]]),
        np.array([[1, 0], [0, 0]]),
        np.array([[0, 1], [0, -1]]),
        np.array([[-1, 0], [1, 0]]),
        np.array([[0, 0], [0, 1]]),
        np.array([[1, 1], [0, 0]]),
        np.array([[0, 0], [1, 1]])
    ]
    S_W = [
        np.array([[1, 0], [0, 1]]),
        np.array([[0, 1], [0, -1]]),
        np.array([[0, 0], [1, 1]]),
        np.array([[1, 1], [0, 0]]),
        np.array([[-1, 0], [1, 0]]),
        np.array([[0, 0], [0, 1]]),
        np.array([[1, 0], [0, 0]])
    ]

    # Standard factors for 2x1x2
    Std_U = [
        np.array([[1], [0]]),
        np.array([[1], [0]]),
        np.array([[0], [1]]),
        np.array([[0], [1]])
    ]
    Std_V = [
        np.array([[1, 0]]),
        np.array([[0, 1]]),
        np.array([[1, 0]]),
        np.array([[0, 1]])
    ]
    Std_W = [
        np.array([[1, 0], [0, 0]]),
        np.array([[0, 0], [1, 0]]),
        np.array([[0, 1], [0, 0]]),
        np.array([[0, 0], [0, 1]])
    ]

    col = 0
    # Process the three blocks corresponding to the 2-column chunks of B
    for b in range(3):
        # Strassen part for A[:, 0:2] * B[0:2, 2b:2b+2]
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[i * m + j, col] = S_U[r][i, j]
            
            for j in range(2):
                for kp in range(2):
                    k_global = 2 * b + kp
                    V[j * p + k_global, col] = S_V[r][j, kp]
            
            for kp in range(2):
                for i in range(2):
                    k_global = 2 * b + kp
                    W[k_global * n + i, col] = S_W[r][kp, i]
            
            col += 1

        # Standard part for A[:, 2] * B[2, 2b:2b+2]
        for r in range(4):
            for i in range(2):
                U[i * m + 2, col] = Std_U[r][i, 0]
            
            for kp in range(2):
                k_global = 2 * b + kp
                V[2 * p + k_global, col] = Std_V[r][0, kp]
            
            for kp in range(2):
                for i in range(2):
                    k_global = 2 * b + kp
                    W[k_global * n + i, col] = Std_W[r][kp, i]
            
            col += 1

    if time.time() > deadline - 5:
         return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # --- Rank Reduction Attempt ---
    # The standard decomposition is essentially a sum of rank-1 tensors.
    # We can check if the set of 33 vectors (u_i, v_i, w_i) is linearly dependent 
    # in the joint space in a way that allows cancellation or linear combinations.
    # Since they are {0, 1, -1}, we look for a small integer relation sum c_i T_i = 0.
    # This is a knapsack-like problem. We use a simple heuristic search for a relation
    # using vectors that are "close" or have simple structures.
    
    # Build the list of factor triples
    triples = []
    for r in range(rank):
        u = U[:, r].copy()
        v = V[:, r].copy()
        w = W[:, r].copy()
        triples.append((u, v, w))

    # We look for a relation sum d_r T_r = 0 where d_r are small integers.
    # This is equivalent to sum d_r u_r = 0 AND sum d_r v_r = 0 AND sum d_r w_r = 0.
    # Actually, it's sum d_r (u_r, v_r, w_r) = 0 in the product space.
    # We can try to find 3 or 4 triples that sum to 0.
    
    best_reduction = 0
    found_relation = None
    indices_to_remove = []

    # Simple search for 4-sum zero
    # Given the structure, the Strassen parts in different blocks are disjoint, so no cross-block cancellation.
    # Within a block (2x2x2 + 2x1x2), we have 7 + 4 = 11 terms.
    # Let's check within a block.
    
    block_size = 11
    for start in range(0, rank, block_size):
        if time.time() > deadline - 2:
            break
        block_triples = triples[start:start+block_size]
        
        # Search for a relation among these
        # We want d_1 T_1 + d_2 T_2 + d_3 T_3 + d_4 T_4 = 0
        # With small integers.
        
        # Since we don't have a robust fast solver here, we rely on the fact that
        # the decomposition we constructed is very "block diagonal" (in terms of support).
        # U and V have supports that don't overlap between the Strassen part (j<2) and Std part (j=2).
        # For a sum to be zero, the sum of U parts must be 0, sum of V parts must be 0, sum of W parts must be 0.
        
        # Strassen U parts: 7 vectors in {0,1,-1}^{2x2}
        # Std U parts: 4 vectors in {0,1}^{2x1} (actually 2x3 with only j=2 non-zero)
        # They live in orthogonal subspaces. So any relation must be zero in both subspaces.
        # Sum of Strassen U parts = 0 AND Sum of Std U parts = 0.
        # Same for V and W.
        
        # Within Strassen:
        # Are the 7 Strassen U vectors linearly dependent over the rationals? No, rank 7.
        # Within Std:
        # Are the 4 Std U vectors linearly dependent?
        # [1,0], [1,0], [0,1], [0,1] -> rank 2. Yes! 
        # T1: [1,0] U, [1,0] V, [1,0; 0,0] W
        # T2: [1,0] U, [0,1] V, [0,0; 1,0] W
        # T3: [0,1] U, [1,0] V, [0,1; 0,0] W
        # T4: [0,1] U, [0,1] V, [0,0; 0,1] W
        
        # Wait, U is (n*m, rank). The Std U parts have non-zeros only at indices 2, 3 (since j=2, i=0,1).
        # In a block, the U parts for Std are:
        # r0: U[2]=1, U[3]=0 (if mapping was U[i*3+2])
        # Let's check my loop:
        # U[i * m + 2, col] = Std_U[r][i, 0]
        # i=0: U[0*3+2]=U[2]. Std_U[0]=[1,0] -> U[2]=1.
        # i=1: U[1*3+2]=U[5]. Std_U[0]=[1,0] -> U[5]=0.
        # So U[2]=1, U[5]=0.
        # r1: U[2]=1, U[5]=0.
        # r2: U[2]=0, U[5]=1.
        # r3: U[2]=0, U[5]=1.
        
        # V parts:
        # V[2 * p + k, col]
        # r0: V[12+0]=1, V[12+1]=0 -> V[12]=1, V[13]=0
        # r1: V[12]=0, V[13]=1
        # r2: V[12]=1, V[13]=0
        # r3: V[12]=0, V[13]=1
        
        # W parts:
        # W[k*n+i]
        # r0: W[0]=1
        # r1: W[1]=1 (k=0, i=1 -> 0*2+1) -- wait.
        # Let's check W mapping in Std:
        # W[k_global * n + i, col] = Std_W[r][kp, i]
        # k_global = 2b + kp.
        # Std_W[0] = [[1,0],[0,0]] -> W[2b*2+0] = 1.
        # Std_W[1] = [[0,0],[1,0]] -> W[(2b+1)*2+0] = 1.
        # Std_W[2] = [[0,1],[0,0]] -> W[2b*2+1] = 1.
        # Std_W[3] = [[0,0],[0,1]] -> W[(2b+1)*2+1] = 1.
        
        # So for a fixed block b:
        # S1: U[2]=1, V[12]=1, W[4b]=1
        # S2: U[2]=1, V[13]=1, W[4b+2]=1  (k=2b+1 -> (2b+1)*2 = 4b+2)
        # S3: U[5]=1, V[12]=1, W[4b+1]=1  (k=2b -> 4b, i=1 -> 4b+1)
        # S4: U[5]=1, V[13]=1, W[4b+3]=1  (k=2b+1 -> 4b+2, i=1 -> 4b+3)
        
        # We want sum d_i T_i = 0.
        # U: d1 + d2 = 0 => d1 = -d2. d3 + d4 = 0 => d3 = -d4.
        # V: d1 + d3 = 0 => d3 = -d1. d2 + d4 = 0 => d4 = -d2.
        # Combining: d3 = -d1 and d3 = -d4 => d1 = d4.
        # d2 = -d1 and d4 = -d2 => d2 = d1.
        # So d1 = d2 = d3' = d4' where d3=-d1, d4=-d1.
        # Let d1 = 1. Then d2=1, d3=-1, d4=-1.
        # Check W:
        # 1*T1 - 1*T2 + (-1)*T3 - (-1)*T4? No.
        # Relation: 1*T1 - 1*T2 - 1*T3 + 1*T4 ?
        # Let's check coefficients:
        # U: 1(1) - 1(1) - 1(0) + 1(0) = 0. OK. (Using d1=1, d2=1, d3=-1, d4=-1 for U eqns? No)
        
        # Let's restart relation with d = (d1, d2, d3, d4).
        # U[2]: d1 + d2 = 0
        # U[5]: d3 + d4 = 0
        # V[12]: d1 + d3 = 0
        # V[13]: d2 + d4 = 0
        # W[4b]: d1 = 0
        # W[4b+1]: d3 = 0
        # W[4b+2]: d2 = 0
        # W[4b+3]: d4 = 0
        # This forces all d_i = 0.
        
        # Why? Because the W factors are linearly independent in the standard decomposition.
        # The tensor rank of matrix multiplication is not just the rank of the U/V/W sets independently,
        # it's the rank of the tensor in the tensor product space.
        # The "standard" decomposition is often rank-minimal for small sizes if no clever cancellation exists.
        # However, for 2x3x6, maybe there is a cancellation?
        
        # Let's check if we can merge Strassen and Standard parts?
        # No, they act on disjoint indices of U and V (Strassen on j<2, Std on j=2).
        # So any relation must satisfy the sum=0 condition on those subspaces separately.
        # Since Strassen is irreducible (rank 7 for 2x2x2) and Std is rank 4 (which equals n*p*1?),
        # we suspect no reduction.
        
        # However, let's double check the rank of the "Standard" part alone.
        # Multiplying 2x1 by 1x2 is equivalent to 2x2 matrix mult where the middle is a column/row vector.
        # Actually, it's just outer product of a 2x1 and 1x2 vector.
        # The resulting 2x2 matrix C has rank 1.
        # The 4 terms are:
        # A00 B00 -> C00
        # A00 B01 -> C01
        # A10 B00 -> C10
        # A10 B01 -> C11
        # This is just expanding the product. Is it reducible?
        # C = A B. A is 2x1, B is 1x2.
        # The tensor is simply the "diagonal" of the 2x2x2 tensor? No.
        # It is the subset of the 2x2x2 tensor where the middle index is 0?
        # If we had full 2x2x2, we would have 4 terms.
        # But here the "middle" dimension is 1.
        # So it's just A_i B_j C_ij.
        # The tensor is sum_i sum_j A_i B_j C_ij.
        # This is an identity tensor of size 2x2? No, it's 4 rank-1 tensors.
        # The rank of the identity tensor of size N is N (or close to it).
        # For 2x2, rank is 4?
        # Actually, the rank of the identity tensor e_1 f_1 g_1^* + e_2 f_2 g_2^* is 2? No.
        # The identity map I: V -> W^*.
        # If we view it as V x V^* -> K, it's the pairing.
        # Here we have A (2x1) and B (1x2).
        # C_ij = sum_k A_ik B_kj. Since k=0 only.
        # C_00 = A_00 B_00
        # C_01 = A_00 B_01
        # C_10 = A_10 B_00
        # C_11 = A_10 B_01
        # This is just the outer product of A and B.
        # Is there a rank 3 decomposition?
        # We need 3 tensors T1, T2, T3.
        # U1, U2, U3 (2x1 vectors)
        # V1, V2, V3 (1x2 vectors)
        # W1, W2, W3 (2x2 matrices)
        # Sum U_i V_i W_i = identity? No, A B.
        # For a general A, B, this must hold.
        # This is equivalent to decomposing the multiplication tensor for 2x1x2.
        # It is known that rank(2x1x2) = 4?
        # Actually, 2x1x2 is a sub-tensor of 2x2x2.
        # The rank of the 2x1x2 tensor is 4.
        # Proof: The 4 entries of C are linearly independent functions of the 4 products of entries of A and B.
        # A_00 B_00, A_00 B_01, A_10 B_00, A_10 B_01.
        # These 4 monomials are linearly independent.
        # Each rank-1 tensor provides 1 monomial (product of an A-entry and a B-entry) mapped to C-entry.
        # So we need at least 4 terms to cover 4 independent monomials.
        # Thus rank is 4.
        
        # So the Standard part is irreducible at rank 4.
        # The Strassen part is irreducible at rank 7.
        # They are disjoint.
        # Total rank 3 * (7+4) = 33.
        
        # Therefore, rank 33 is likely minimal for this partition.
        # Can we do better by NOT partitioning?
        # Multiplication of 2x3 and 3x6.
        # Lower bound: sum of max(n, p, m) ? No.
        # Rank of 2x3x6 multiplication tensor.
        # Known results:
        # Rank(2x2x2) = 7
        # Rank(2x2x3) = ?
        # If we could do 2x3x6 as two 2x2x3 blocks? No.
        
        # Given the constraint, I will stick with 33.

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
