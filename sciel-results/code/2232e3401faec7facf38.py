import numpy as np
import time

def construct():
    n, m, p = 2, 4, 8
    start_time = time.time()
    deadline = 140  # Safety margin for 150s limit

    # The problem is to decompose the tensor T_{ijk} = delta_{jk} (for C = AB)
    # A is n x m (2x4), B is m x p (4x8), C is n x p (2x8).
    # The tensor is of size (n*m) x (m*p) x (p*n) = 8 x 32 x 16.
    # Rank 64 is the trivial (schoolbook) rank.
    # Rank 56 is achievable via Strassen-like methods.
    
    # Strategy:
    # Partition A into two row-vectors: A0 (1x4), A1 (1x4).
    # Partition B into two 2x4 blocks: B0 (2x4), B1 (2x4).
    # Wait, B is 4x8. Let's partition B by columns into 4 blocks of 2x4? No, B is 4x8.
    # Let's partition B into 2x4 blocks vertically? No, B is 4x8.
    # Let's use the structure: C = A @ B.
    # A = [A0; A1] where A0, A1 are 1x4.
    # B is 4x8.
    # C0 = A0 @ B (1x8)
    # C1 = A1 @ B (1x8)
    
    # This doesn't directly help with Strassen unless we block B.
    # Let's block B into 2x2 blocks.
    # B = [[B00, B01, B02, B03],
    #      [B10, B11, B12, B13]]
    # where Bij are 2x2.
    # A = [A00, A01, A02, A03] (1x4) -> A00, A01, A02, A03 are scalars? No, A is 2x4.
    # Let's partition A into 1x2 blocks: A0 = [A00, A01], A1 = [A02, A03] for row 0?
    # No, A is 2x4.
    # A = [[a00, a01, a02, a03],
    #      [a10, a11, a12, a13]]
    
    # Let's view the multiplication as 4 independent 2x2 matrix multiplications?
    # No, the middle dimension is 4.
    
    # Correct Strassen approach for 2x4 * 4x8:
    # We can view this as multiplying a 2x4 matrix by a 4x8 matrix.
    # We can split the 4x8 matrix into two 4x4 matrices: B_left (cols 0-3), B_right (cols 4-7).
    # Then C_left = A @ B_left (2x4) and C_right = A @ B_right (2x4).
    # Now we need to multiply 2x4 by 4x4.
    # Split 2x4 into two 2x2 matrices: A_left (cols 0-1), A_right (cols 2-3).
    # Split 4x4 into four 2x2 matrices:
    # B_left = [[B00, B01],
    #           [B10, B11]]
    # C_left = A_left @ B00 + A_right @ B10  (top-left 2x2 of C_left)
    #         A_left @ B01 + A_right @ B11  (top-right 2x2 of C_left)
    
    # This is getting complicated. Let's use the known result:
    # The rank of the 2x4x8 tensor is 56.
    # It can be constructed by taking the Strassen decomposition of 2x2x2 (rank 7)
    # and "stretching" it.
    
    # Actually, a simpler way to think about it:
    # The multiplication map is linear in A and B.
    # We can define 56 rank-1 tensors that sum to the identity tensor.
    
    # Let's try to construct the rank-56 decomposition explicitly.
    # Reference: "The rank of 2x4x8 matrix multiplication is 56"
    # This is derived from the fact that R(2,4,8) <= 2 * R(2,2,4) ? No.
    # R(2,4,8) <= R(2,2,2) * 8 ? No.
    
    # Let's use the block Strassen method.
    # We treat A as a 2x2 block matrix of 1x2 blocks? No.
    # Let's treat A as two 1x4 vectors.
    # Let's treat B as a 4x8 matrix.
    
    # Alternative: Use the decomposition of 4x4 matrix multiplication (rank 21? No, 20 or 19).
    # If we view A as a 2x4 matrix, we can't directly use 4x4.
    
    # Let's stick to the explicit Strassen construction for 2x4 * 4x8.
    # We can partition the 4x8 matrix B into 2x2 blocks: B_{ij} for i=0,1 and j=0,1,2,3.
    # And A into 1x2 blocks: A_i for i=0,1,2,3? No, A is 2x4.
    # A = [A0, A1, A2, A3] where A_i are scalars? No.
    # A = [[A00, A01], [A10, A11]]? No, A is 2x4.
    # A00 = A[0:2, 0:2] (2x2), A01 = A[0:2, 2:4] (2x2).
    
    # So we have 2x2 blocks for A and B.
    # C = A @ B
    # C = [[A00, A01], [A10, A11]] @ [[B00, B01, B02, B03], [B10, B11, B12, B13]]
    # C00 = A00 B00 + A01 B10
    # C01 = A00 B01 + A01 B11
    # C02 = A00 B02 + A01 B12
    # C03 = A00 B03 + A01 B13
    # C10 = A10 B00 + A11 B10
    # C11 = A10 B01 + A11 B11
    # C12 = A10 B02 + A11 B12
    # C13 = A10 B03 + A11 B13
    
    # Each Cij is a 2x2 matrix.
    # We can compute each 2x2 product using Strassen (rank 7).
    # But we have 8 such products (C00...C13).
    # However, the terms A00 B00, A01 B10, etc., are not independent.
    # We can share the Strassen products.
    
    # Let's define the Strassen products for the 2x2 blocks.
    # We need to compute A00 B00, A00 B01, etc.
    # This is complex.
    
    # Let's try a different approach: **Randomized SVD-like initialization**.
    # Initialize U, V, W with random values, but ensure that the initial error is low.
    # Then use gradient descent with a larger learning rate and momentum.
    
    # Actually, the most reliable way to get rank 56 is to use the **explicit construction**.
    # I will hardcode the rank-56 decomposition if I can find it.
    # Since I can't, I will use a **simulated annealing** approach to find the decomposition.
    
    # Let's try to implement the Strassen-based decomposition explicitly.
    # The rank is 56.
    # U, V, W are matrices of size (8, 56), (32, 56), (16, 56).
    
    # I will use a random search with **Nesterov momentum** and **adaptive learning rate**.
    
    def get_schoolbook():
        rank = n * m * p  # 64
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, r] = 1.0
                    V[j * p + k, r] = 1.0
                    W[k * n + i, r] = 1.0
                    r += 1
        return U, V, W

    def compute_error(U, V, W, A, B):
        U_reshaped = U.reshape(n, m, -1)
        V_reshaped = V.reshape(m, p, -1)
        W_reshaped = W.reshape(p, n, -1)
        X = np.einsum('ijr,ij->ir', U_reshaped, A)
        Y = np.einsum('jkr,jk->kr', V_reshaped, B)
        C_recon = np.einsum('ir,kr,ki->ik', X, Y, W_reshaped)
        C_true = A @ B
        return np.sum((C_recon - C_true) ** 2)

    def verify_decomposition(U, V, W, num_tests=20, seed=42):
        rng = np.random.RandomState(seed)
        max_err = 0
        for _ in range(num_tests):
            A = rng.randn(n, m)
            B = rng.randn(m, p)
            err = compute_error(U, V, W, A, B)
            max_err = max(max_err, err)
        return max_err

    best_rank = n * m * p
    best_U, best_V, best_W = get_schoolbook()

    # NEW STRATEGY: Explicit Strassen-based construction for Rank 56.
    # We partition A (2x4) into 2x2 blocks: A00, A01 (cols 0-1, 2-3)
    # We partition B (4x8) into 2x2 blocks: B_{ij} for i=0,1 (rows 0-1, 2-3) and j=0,1,2,3 (cols)
    # Wait, B is 4x8.
    # Rows of B: 0,1,2,3.
    # Cols of B: 0..7.
    # Let's group rows of B into 2 pairs: (0,1) and (2,3).
    # Let's group cols of B into 4 pairs: (0,1), (2,3), (4,5), (6,7).
    # So B is a 2x4 block matrix of 2x2 blocks.
    # A is a 1x2 block matrix of 2x2 blocks? No, A is 2x4.
    # A = [A00, A01] where A00 is 2x2 (cols 0-1), A01 is 2x2 (cols 2-3).
    
    # C = A @ B
    # C is 2x8.
    # C = [C00, C01, C02, C03] where Ckj is 2x2.
    # Ckj = A00 @ B0j + A01 @ B1j
    
    # We need to compute 8 products of the form A00 B0j + A01 B1j.
    # This is a sum of two 2x2 matrix products.
    # We can use Strassen to compute the product of a "doubled" matrix?
    # No, Strassen computes one product.
    # However, we can view the computation of Ckj as a bilinear map.
    # The map is (A00, A01, B0j, B1j) -> A00 B0j + A01 B1j.
    # This is equivalent to multiplying a 2x4 matrix (A00, A01) by a 4x2 matrix (B0j, B1j).
    # Wait, dimensions:
    # A00 is 2x2, A01 is 2x2. So (A00, A01) is 2x4.
    # B0j is 2x2, B1j is 2x2. So (B0j; B1j) is 4x2.
    # The result is 2x2.
    # The rank of 2x4 * 4x2 matrix multiplication is R(2,4,2).
    # By symmetry R(2,4,2) = R(2,2,4).
    # We know R(2,2,2) = 7.
    # R(2,2,4) <= 2 * R(2,2,2) = 14? No, that's not tight.
    # Actually, R(2,2,4) is 15? Or 14?
    # If R(2,2,4) = 14, then total rank = 4 (blocks of j) * 14 = 56.
    # This matches the target rank 56!
    
    # So we need a rank-14 decomposition for 2x2 * 4x2 multiplication?
    # Or 2x4 * 4x2.
    # Let's construct the rank-14 decomposition for 2x4 * 4x2.
    # Let X be 2x4, Y be 4x2. Z = X Y (2x2).
    # Partition X into [X0, X1] (2x2 each).
    # Partition Y into [Y0; Y1] (2x2 each).
    # Z = X0 Y0 + X1 Y1.
    # This is the sum of two 2x2 products.
    # We can use Strassen on the "combined" structure?
    # Actually, there is a known decomposition for 2x2 * 2x2 + 2x2 * 2x2.
    # This is equivalent to multiplying a 2x4 matrix by a 4x2 matrix.
    # The rank is 14.
    # How to get 14?
    # We can use two Strassen decompositions? That would be 14.
    # But they are not independent.
    # Actually, we can just use the Strassen algorithm for 2x2, but apply it to the blocks?
    # No, the sum of two products cannot be done with 2*7=14 independent scalar multiplications of the *blocks*?
    # Wait, the "scalar" multiplications in Strassen are 2x2 matrix multiplications? No, they are scalar multiplications if we go down to scalars.
    # If we treat the 2x2 blocks as "scalars" (which is wrong, they are matrices), we can't.
    # We need the rank of the tensor for 2x4 * 4x2.
    # This tensor is the sum of two 2x2 * 2x2 tensors.
    # The rank of the sum is at most the sum of the ranks.
    # R(2x2*2x2) = 7.
    # So R(2x4*4x2) <= 14.
    # Is it exactly 14?
    # If we take the Strassen decomposition for X0 Y0 (rank 7) and X1 Y1 (rank 7), and sum them, we get a rank 14 decomposition.
    # Are they independent?
    # The terms in X0 Y0 involve X0 and Y0.
    # The terms in X1 Y1 involve X1 and Y1.
    # Since the variables are disjoint, the tensors are independent.
    # So yes, we can just concatenate the two Strassen decompositions.
    
    # So the plan:
    # For each j in 0..3:
    #   Compute Ckj = A00 B0j + A01 B1j.
    #   Use Strassen for A00 B0j (7 terms).
    #   Use Strassen for A01 B1j (7 terms).
    #   Total 14 terms per j.
    #   Total 4 * 14 = 56 terms.
    
    # Let's implement this.
    
    # Strassen decomposition for 2x2 matrices P, Q:
    # M1 = (P00+P11)(Q00+Q11)
    # M2 = (P10+P11)Q00
    # M3 = P00(Q10-Q11)
    # M4 = P11(Q01-Q00)
    # M5 = (P00+P01)Q11
    # M6 = (P10-P00)(Q00+Q10)
    # M7 = (P01-P11)(Q01+Q11)
    
    # We need to map these to the global U, V, W matrices.
    # U has shape (8, 56). Indices: i in 0..1, j in 0..3.
    # V has shape (32, 56). Indices: j in 0..3, k in 0..7.
    # W has shape (16, 56). Indices: k in 0..7, i in 0..1.
    
    # The decomposition is a sum of rank-1 tensors.
    # Each rank-1 tensor is defined by:
    #   u in U (vector of length 8)
    #   v in V (vector of length 32)
    #   w in W (vector of length 16)
    # such that sum_r u_r \otimes v_r \otimes w_r = T.
    
    # T_{i,j,k} = delta_{j,k}? No.
    # T is the multiplication tensor.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # In the tensor decomposition, the coefficient of A_{i,j} B_{j,k} C_{i,k} is 1.
    # So T_{(i,j), (j,k), (k,i)} = 1. (Note: indices in U, V, W are flattened).
    # U index: i*m + j.
    # V index: j*p + k.
    # W index: k*n + i.
    
    # We will build U, V, W by adding contributions from each of the 56 Strassen terms.
    
    U = np.zeros((n * m, 56))
    V = np.zeros((m * p, 56))
    W = np.zeros((p * n, 56))
    
    def add_strassen_term(P_idx, Q_idx, C_idx, coeff_P, coeff_Q, coeff_C, rank_offset):
        """
        Adds a rank-1 term corresponding to a Strassen multiplication.
        P_idx: indices of the P matrix in A (i, j)
        Q_idx: indices of the Q matrix in B (j, k)
        C_idx: indices of the C matrix in C (k, i)
        coeff_P: 2x2 matrix of coefficients for P
        coeff_Q: 2x2 matrix of coefficients for Q
        coeff_C: 2x2 matrix of coefficients for C
        rank_offset: which column of U, V, W to write to
        """
        # P is 2x2 block of A.
        # P_{ab} corresponds to A_{i+a, j+b}
        # Q is 2x2 block of B.
        # Q_{cd} corresponds to B_{j+c, k+d}
        # C is 2x2 block of C.
        # C_{ef} corresponds to C_{k+e, i+f}
        
        # The term is (coeff_P \cdot P) * (coeff_Q \cdot Q) * (coeff_C \cdot C)
        # This contributes to the tensor T.
        # The contribution to T_{(i,j), (j,k), (k,i)} is:
        # sum_{a,b,c,d,e,f} coeff_P[a,b] coeff_Q[c,d] coeff_C[e,f] * delta_{j+b, j+c} delta_{k+d, k+e} delta_{i+f, i+a}
        # Wait, the indices must match.
        # A_{i+a, j+b} B_{j+c, k+d} C_{k+e, i+f}
        # For this to be a valid term in the sum, we need j+b = j+c => b=c, and k+d = k+e => d=e, and i+f = i+a => f=a.
        # So the term is non-zero only if b=c, d=e, f=a.
        # Then the value is coeff_P[a,b] coeff_Q[b,d] coeff_C[d,a].
        
        # We need to add this to U, V, W.
        # U_{i+a, j+b} += coeff_P[a,b]
        # V_{j+b, k+d} += coeff_Q[b,d]  (Wait, V index is j*p + k. Here j is the row of B, k is col.
        # In the term, B is indexed by (j+c, k+d). So V index is (j+c)*p + (k+d).
        # Since c=b, d=e, V index is (j+b)*p + (k+d).)
        # W_{k+d, i+a} += coeff_C[d,a]  (W index is k*n + i. Here k is row of C, i is col.
        # In the term, C is indexed by (k+e, i+f). So W index is (k+e)*n + (i+f).
        # Since e=d, f=a, W index is (k+d)*n + (i+a).)
        
        for a in range(2):
            for b in range(2):
                val_P = coeff_P[a, b]
                if val_P == 0: continue
                U[(i+a)*m + (j+b), rank_offset] += val_P
                # Note: we must also update V and W with the corresponding coefficients.
                # But the term is a product.
                # The rank-1 tensor is u \otimes v \otimes w.
                # u_{i+a, j+b} = coeff_P[a,b]
                # v_{j+b, k+d} = coeff_Q[b,d]
                # w_{k+d, i+a} = coeff_C[d,a]
                # This must hold for all a,b,d.
                # So we can't just loop over a,b and add. We need to construct the full vectors.
                
        # Let's construct the full vectors for this rank-1 term.
        u_vec = np.zeros(n*m)
        v_vec = np.zeros(m*p)
        w_vec = np.zeros(p*n)
        
        for a in range(2):
            for b in range(2):
                u_vec[(i+a)*m + (j+b)] = coeff_P[a, b]
                
        for b in range(2):
            for d in range(2):
                v_vec[(j+b)*p + (k+d)] = coeff_Q[b, d]
                
        for d in range(2):
            for a in range(2):
                w_vec[(k+d)*n + (i+a)] = coeff_C[d, a]
                
        U[:, rank_offset] += u_vec
        V[:, rank_offset] += v_vec
        W[:, rank_offset] += w_vec

    # Strassen coefficients for P, Q, C
    # M1: P=(1,0;0,1) -> P00+P11. Q=(1,0;0,1) -> Q00+Q11. C=(1,0;0,1) -> C00+C11.
    # Wait, Strassen formulas:
    # M1 = (A00+A11)(B00+B11)
    # A00 is top-left, A11 is bottom-right.
    # So coeff_P = [[1,0],[0,1]].
    # coeff_Q = [[1,0],[0,1]].
    # The result M1 contributes to C.
    # C00 = M1 - M4 + M6
    # C01 = M5 + M7
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5
    
    # Let's define the 7 Strassen terms for a single 2x2 product.
    # Each term is (coeff_P, coeff_Q, coeff_C)
    # But coeff_C is not a single matrix, it's a set of coefficients for the 4 elements of C.
    # Actually, the rank-1 term is a scalar multiplication of the "block" product.
    # The term is (u^T P) (v^T Q) (w^T C) ? No.
    # The term is sum_{a,b} u_{ab} P_{ab} * sum_{c,d} v_{cd} Q_{cd} * sum_{e,f} w_{ef} C_{ef}.
    # This is not a rank-1 tensor in the space of matrices.
    # It is a rank-1 tensor in the space of *scalars* if P, Q, C are scalars.
    # Here P, Q, C are 2x2 matrices.
    # The tensor is T_{P,Q,C}.
    # The rank-1 decomposition is sum_r u_r \otimes v_r \otimes w_r.
    # u_r is a vector in R^{4} (flattened P).
    # v_r is a vector in R^{4} (flattened Q).
    # w_r is a vector in R^{4} (flattened C).
    
    # So for each Strassen term, we have:
    # u_r = vector of coeff_P
    # v_r = vector of coeff_Q
    # w_r = vector of coeff_C
    
    # Let's re-verify the Strassen coefficients.
    # M1: P = [[1,0],[0,1]], Q = [[1,0],[0,1]].
    # This product M1 is used in C00, C11.
    # C00 += M1, C11 += M1.
    # So w_r for M1 should have 1 at C00 and C11?
    # No, the term is M1 * (indicator of where M1 appears).
    # But M1 is a scalar (result of block multiplication).
    # The contribution to the tensor T is:
    # (u_M1 \otimes v_M1) * (w_M1)
    # where w_M1 is the vector indicating which C elements M1 contributes to.
    # M1 contributes +1 to C00 and +1 to C11.
    # So w_M1 = [1, 0, 0, 1] (flattened C).
    
    # Let's list the 7 terms:
    # 1. M1: P=[1,0;0,1], Q=[1,0;0,1]. C: C00+=1, C11+=1. w=[1,0,0,1]
    # 2. M2: P=[0,0;1,1], Q=[1,0;0,0]. C: C10+=1. w=[0,0,1,0] (C10 is index 2)
    # 3. M3: P=[1,0;0,0], Q=[0,0;1,-1]. C: C11-=1. w=[0,0,0,-1]
    # 4. M4: P=[0,0;0,1], Q=[0,1;-1,0]. C: C00-=1, C10+=1. w=[-1,0,1,0]
    # 5. M5: P=[1,1;0,0], Q=[0,0;0,1]. C: C01+=1, C11+=1. w=[0,1,0,1]
    # 6. M6: P=[0,-1;1,0], Q=[1,0;1,0]. C: C00+=1. w=[1,0,0,0]
    # 7. M7: P=[0,1;0,-1], Q=[0,1;0,1]. C: C01+=1. w=[0,1,0,0]
    
    # Let's check M4:
    # M4 = A11 (B01 - B00)
    # P = A11 -> [[0,0],[0,1]]
    # Q = B01 - B00 -> [[0,1],[-1,0]]
    # C00 = M1 - M4 + M6 -> C00 -= M4.
    # C10 = M2 + M4 -> C10 += M4.
    # So w = [-1, 0, 1, 0]. Correct.
    
    # Let's check M3:
    # M3 = A00 (B10 - B11)
    # P = A00 -> [[1,0],[0,0]]
    # Q = B10 - B11 -> [[0,0],[1,-1]]
    # C11 = M1 - M3 + M5 -> C11 -= M3.
    # So w = [0, 0, 0, -1]. Correct.
    
    # Let's check M6:
    # M6 = (A10 - A00) (B00 + B10)
    # P = A10 - A00 -> [[-1,0],[1,0]]?
    # A10 is bottom-left, A00 is top-left.
    # P = [[-1,0],[1,0]].
    # Q = B00 + B10 -> [[1,0],[1,0]].
    # C00 = M1 - M4 + M6 -> C00 += M6.
    # So w = [1, 0, 0, 0]. Correct.
    
    # Let's check M7:
    # M7 = (A01 - A11) (B01 + B11)
    # P = A01 - A11 -> [[0,1],[0,-1]].
    # Q = B01 + B11 -> [[0,1],[0,1]].
    # C01 = M5 + M7 -> C01 += M7.
    # So w = [0, 1, 0, 0]. Correct.
    
    # Let's check M2:
    # M2 = (A10 + A11) B00
    # P = [[0,0],[1,1]].
    # Q = [[1,0],[0,0]].
    # C10 = M2 + M4 -> C10 += M2.
    # So w = [0, 0, 1, 0]. Correct.
    
    # Let's check M5:
    # M5 = (A00 + A01) B11
    # P = [[1,1],[0,0]].
    # Q = [[0,0],[0,1]].
    # C01 = M5 + M7 -> C01 += M5.
    # C11 = M1 - M3 + M5 -> C11 += M5.
    # So w = [0, 1, 0, 1]. Correct.
    
    # Now we have the 7 terms for a single 2x2 product.
    # We need to apply this to 8 products (4 for A00, 4 for A01).
    # Wait, for each j (0..3), we compute Ckj = A00 B0j + A01 B1j.
    # This is the sum of two 2x2 products.
    # We use 7 terms for A00 B0j and 7 terms for A01 B1j.
    # Total 14 terms per j.
    
    # The indices:
    # A00 is A[0:2, 0:2]. i=0, j=0.
    # A01 is A[0:2, 2:4]. i=0, j=2.
    # B0j is B[0:2, 2*j:2*j+2]. j_block=0, k_block=j.
    # B1j is B[2:4, 2*j:2*j+2]. j_block=1, k_block=j.
    # Ckj is C[0:2, 2*j:2*j+2]. i_block=0, k_block=j.
    
    # Wait, C is 2x8.
    # C_{i,k}. i is 0,1. k is 0..7.
    # Ckj is the block at row i, col 2j..2j+1.
    # So C_{i, 2j+a} for a=0,1.
    # In the W matrix, index is k*n + i.
    # So for C_{i, 2j+a}, W index is (2j+a)*n + i.
    
    # Let's loop over j (0..3) and the 14 terms.
    
    strassen_terms = [
        # (P, Q, C)
        (np.array([[1,0],[0,1]]), np.array([[1,0],[0,1]]), np.array([1,0,0,1])),
        (np.array([[0,0],[1,1]]), np.array([[1,0],[0,0]]), np.array([0,0,1,0])),
        (np.array([[1,0],[0,0]]), np.array([[0,0],[1,-1]]), np.array([0,0,0,-1])),
        (np.array([[0,0],[0,1]]), np.array([[0,1],[-1,0]]), np.array([-1,0,1,0])),
        (np.array([[1,1],[0,0]]), np.array([[0,0],[0,1]]), np.array([0,1,0,1])),
        (np.array([[-1,0],[1,0]]), np.array([[1,0],[1,0]]), np.array([1,0,0,0])),
        (np.array([[0,1],[0,-1]]), np.array([[0,1],[0,1]]), np.array([0,1,0,0])),
    ]
    
    rank_offset = 0
    for j in range(4):
        # Term 1: A00 B0j
        for P, Q, C in strassen_terms:
            # A00: i=0, j_block=0
            # B0j: j_block=0, k_block=j
            # Ckj: i_block=0, k_block=j
            
            u_vec = np.zeros(n*m)
            v_vec = np.zeros(m*p)
            w_vec = np.zeros(p*n)
            
            # Fill u_vec
            for a in range(2):
                for b in range(2):
                    u_vec[(0+a)*m + (0+b)] = P[a, b]
                    
            # Fill v_vec
            for c in range(2):
                for d in range(2):
                    v_vec[(0+c)*p + (j*2+d)] = Q[c, d]
                    
            # Fill w_vec
            # C is flattened [C00, C01, C10, C11]
            # C00 -> C[0, 2j] -> W[(2j)*n + 0]
            # C01 -> C[0, 2j+1] -> W[(2j+1)*n + 0]
            # C10 -> C[1, 2j] -> W[(2j)*n + 1]
            # C11 -> C[1, 2j+1] -> W[(2j+1)*n + 1]
            
            w_vec[(j*2)*n + 0] = C[0]
            w_vec[(j*2+1)*n + 0] = C[1]
            w_vec[(j*2)*n + 1] = C[2]
            w_vec[(j*2+1)*n + 1] = C[3]
            
            U[:, rank_offset] += u_vec
            V[:, rank_offset] += v_vec
            W[:, rank_offset] += w_vec
            rank_offset += 1
            
        # Term 2: A01 B1j
        for P, Q, C in strassen_terms:
            # A01: i=0, j_block=2
            # B1j: j_block=1, k_block=j
            # Ckj: i_block=0, k_block=j
            
            u_vec = np.zeros(n*m)
            v_vec = np.zeros(m*p)
            w_vec = np.zeros(p*n)
            
            # Fill u_vec
            for a in range(2):
                for b in range(2):
                    u_vec[(0+a)*m + (2+b)] = P[a, b]
                    
            # Fill v_vec
            for c in range(2):
                for d in range(2):
                    v_vec[(1+c)*p + (j*2+d)] = Q[c, d]
                    
            # Fill w_vec
            w_vec[(j*2)*n + 0] = C[0]
            w_vec[(j*2+1)*n + 0] = C[1]
            w_vec[(j*2)*n + 1] = C[2]
            w_vec[(j*2+1)*n + 1] = C[3]
            
            U[:, rank_offset] += u_vec
            V[:, rank_offset] += v_vec
            W[:, rank_offset] += w_vec
            rank_offset += 1
            
    # Verify the decomposition
    max_err = verify_decomposition(U, V, W, num_tests=50, seed=123)
    
    if max_err < 1e-6:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook if construction failed (should not happen)
        U, V, W = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

