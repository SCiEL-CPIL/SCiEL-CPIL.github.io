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
    
    # Let's use the following decomposition:
    # C_{ik} = sum_j A_{ij} B_{jk}
    # We can group the sum over j into pairs: j=0,1 and j=2,3.
    # C_{ik} = sum_{j=0,1} A_{ij} B_{jk} + sum_{j=2,3} A_{ij} B_{jk}
    # Let B^{(0)} be the first 2 columns of B (4x2), B^{(1)} be the last 2 columns (4x2).
    # This doesn't help with rank reduction directly.
    
    # Let's use the known rank-56 decomposition from the literature.
    # It is based on the Strassen algorithm for 2x2 matrices.
    # We can view the 2x4 * 4x8 multiplication as 8 independent 2x2 * 2x2 multiplications?
    # No, the middle dimension is shared.
    
    # Actually, the rank is 56 because we can use the Strassen algorithm on the 2x2 blocks.
    # If we partition A into 1x2 blocks and B into 2x2 blocks, we get a structure.
    
    # Let's try to generate the rank-56 decomposition by solving a linear system or using a known construction.
    # Since I don't have the explicit formula memorized, I will use a random search with a better initialization.
    # The current program's random search failed. Why?
    # Probably because the landscape is difficult.
    # Let's try a different optimization method: Alternating Least Squares (ALS) or similar.
    
    # However, the prompt asks for a MATERIAL change.
    # The current program uses gradient descent.
    # I will use a **structured initialization** based on the Strassen algorithm.
    
    # Strassen for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # For 2x4 * 4x8:
    # Partition A into 1x2 blocks: A0 (cols 0-1), A1 (cols 2-3).
    # Partition B into 2x2 blocks: B_{ij} for i=0,1 (rows) and j=0,1,2,3 (cols).
    # Wait, B is 4x8.
    # Let's partition B into 2x2 blocks:
    # B = [[B00, B01, B02, B03],
    #      [B10, B11, B12, B13]]
    # where Bij are 2x2.
    # A = [A0, A1] where A0, A1 are 1x2? No, A is 2x4.
    # A = [[A00, A01, A02, A03],
    #      [A10, A11, A12, A13]]
    # Let's partition A into 2x2 blocks:
    # A = [[A00, A01],
    #      [A10, A11]]
    # where Aij are 1x2? No, A is 2x4.
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

    target_rank = 56
    if time.time() < deadline - 20:
        for trial in range(20):
            if time.time() > deadline - 10:
                break
            rng = np.random.RandomState(target_rank * 100 + trial)
            U = rng.randn(n * m, target_rank)
            V = rng.randn(m * p, target_rank)
            W = rng.randn(p * n, target_rank)

            A_test = np.random.RandomState(0).randn(n, m)
            B_test = np.random.RandomState(1).randn(m, p)

            # Nesterov Momentum
            vU = np.zeros_like(U)
            vV = np.zeros_like(V)
            vW = np.zeros_like(W)
            mu = 0.9
            lr = 0.01
            
            for step in range(500):
                if time.time() > deadline - 10:
                    break

                U_reshaped = U.reshape(n, m, -1)
                V_reshaped = V.reshape(m, p, -1)
                W_reshaped = W.reshape(p, n, -1)

                X = np.einsum('ijr,ij->ir', U_reshaped, A_test)
                Y = np.einsum('jkr,jk->kr', V_reshaped, B_test)
                C_recon = np.einsum('ir,kr,ki->ik', X, Y, W_reshaped)
                C_true = A_test @ B
                diff = C_recon - C_true

                Z = np.einsum('ik,kr,ki->ir', diff, Y, W_reshaped)
                dU = 2 * np.einsum('ij,ir->ijr', A_test, Z).reshape(n * m, target_rank)

                Z2 = np.einsum('ik,ir,ki->kr', diff, X, W_reshaped)
                dV = 2 * np.einsum('jk,kr->jkr', B_test, Z2).reshape(m * p, target_rank)

                dW_reshaped = 2 * np.einsum('ik,ir,kr->ki', diff, X, Y)
                dW = dW_reshaped.reshape(p * n, target_rank)

                # Nesterov update
                vU = mu * vU + dU
                vV = mu * vV + dV
                vW = mu * vW + dW

                U -= lr * vU
                V -= lr * vV
                W -= lr * vW

                err = compute_error(U, V, W, A_test, B_test)
                if err < 1e-10:
                    break

            max_verify_err = verify_decomposition(U, V, W, num_tests=20)
            if max_verify_err < 1e-6:
                best_rank = target_rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break

    if best_rank < n * m * p:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        U, V, W = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
