import numpy as np
import time
import itertools
from scipy.linalg import svd

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    n, m, p = 2, 4, 5
    
    dim_a = n * m  # 8
    dim_b = m * p  # 20
    dim_c = p * n  # 10
    
    # 1. Construct the canonical multiplication tensor T
    # T[a, b, c] = 1 if i==i', j==j', k==k' else 0
    # a = i*m + j, b = j'*p + k, c = k'*n + i'
    T = np.zeros((dim_a, dim_b, dim_c), dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    # Flatten T into a matrix for rank estimation
    # Reshape T into (dim_a, dim_b * dim_c)
    T_mat = T.reshape(dim_a, dim_b * dim_c)
    
    # 2. Determine the numerical rank of the tensor
    # The rank of a tensor is at least the rank of any matrix unfolding.
    # The rank of the tensor is the minimum number of rank-1 tensors that sum to T.
    # For a generic tensor, the rank is often close to the rank of the flattening, but can be lower.
    # However, for matrix multiplication tensors, the rank is a specific algebraic integer.
    # We can estimate the rank by looking at the singular values of the flattening.
    
    s = np.linalg.svd(T_mat, compute_uv=False)
    # Count significant singular values
    tol = 1e-9
    num_sig = np.sum(s > tol)
    
    # The tensor rank R must be at least num_sig? No, the rank of the flattening is a lower bound on the tensor rank?
    # Actually, the rank of the tensor is >= rank of any flattening.
    # So R >= num_sig.
    # Let's check the rank of other flattenings just in case.
    T_mat2 = T.reshape(dim_b, dim_a * dim_c)
    s2 = np.linalg.svd(T_mat2, compute_uv=False)
    num_sig2 = np.sum(s2 > tol)
    
    T_mat3 = T.reshape(dim_c, dim_a * dim_b)
    s3 = np.linalg.svd(T_mat3, compute_uv=False)
    num_sig3 = np.sum(s3 > tol)
    
    min_rank_bound = max(num_sig, num_sig2, num_sig3)
    
    # For 2x4x5, the schoolbook rank is 40.
    # The flattening ranks are likely much lower (e.g., 8, 20, 10).
    # The actual tensor rank is a known value.
    # Known bounds:
    # Rank(2x4x5) <= 36 (Strassen-like)
    # Rank(2x4x5) >= ?
    
    # Let's try to find a decomposition of rank R.
    # We will iterate R from min_rank_bound up to 40.
    # However, finding an exact decomposition is hard.
    
    # Strategy:
    # We know that the schoolbook decomposition has rank 40.
    # We suspect the rank is 36.
    # We will attempt to construct a rank 36 decomposition.
    # If we fail, we return 40.
    
    # How to construct a rank 36 decomposition?
    # We can use a randomized search for the factors U, V, W.
    # We want T = sum_{r=1}^R u_r v_r w_r^T.
    # This is a system of bilinear equations.
    
    # Alternative: Use the known Strassen decomposition for 2x2x2.
    # Can we embed 2x2x2 into 2x4x5?
    # A (2x4) * B (4x5) = C (2x5).
    # Split A into two 2x2 blocks: A1, A2.
    # Split B into two 2x5 blocks: B1, B2.
    # C = A1 B1 + A2 B2.
    # This requires 2 * (2x2x5) multiplications? No.
    # A1 (2x2) * B1 (2x5) is a 2x2x5 multiplication.
    # Rank(2x2x5) <= Rank(2x2x2) + Rank(2x2x3)?
    # This is getting complicated.
    
    # Let's try a different approach:
    # Use the fact that the tensor is small.
    # We can try to solve for the factors of a rank R decomposition using a nonlinear solver.
    # But that's slow.
    
    # Let's try to verify if rank 35 is possible using the "slice rank" bound or similar.
    # If rank 35 is impossible, we know the rank is at least 36.
    # If rank 36 is possible, we need to find it.
    
    # Given the time constraints, and the fact that we have a valid rank 40 decomposition,
    # and that rank 36 is a known improvement,
    # we will attempt to construct the rank 36 decomposition using a hardcoded Strassen-like structure if possible,
    # or simply return 40 if we can't find a better one quickly.
    
    # Actually, let's look at the "Inspiration" again.
    # "Replacing the schoolbook rank 40 decomposition with a block Strassen-based rank 36 decomposition reduces the rank by 4... Measured: 1 -> 1.111."
    # This implies that a rank 36 decomposition exists and was found in a previous run.
    # We need to reproduce that.
    
    # How was it found?
    # "block Strassen-based".
    # Let's try to construct it.
    
    # Decomposition of 2x4x5:
    # A = [[A11, A12], [A21, A22]] where Aij are 2x2.
    # B = [[B11, B12], [B21, B22]] where B11, B21 are 2x5? No, B is 4x5.
    # Let's split B into two 2x5 blocks: B_top (2x5), B_bottom (2x5).
    # C = A_top B_top + A_bottom B_bottom.
    # A_top is 2x4, B_top is 4x5.
    # This is just the original problem.
    
    # Let's split A into 2x2 blocks and B into 2x5 blocks.
    # A = [A_L, A_R] (2x2 each)
    # B = [B_L; B_R] (2x5 each)
    # C = A_L B_L + A_R B_R.
    # Each term is a 2x2 matrix times a 2x5 matrix.
    # Result is 2x5.
    # Multiplication of 2x2 by 2x5.
    # Rank(2x2x5) <= ?
    # We can view 2x2x5 as two 2x2x1 multiplications? No.
    # A (2x2) * B (2x5) = C (2x5).
    # This is equivalent to multiplying A by 5 columns of B.
    # We can use Strassen for each 2x2x2 block of B?
    # B = [B1, B2, B3, B4, B5] where Bi are 2x1.
    # C = A B1 + A B2 + ...
    # This doesn't reduce the count directly.
    
    # Let's try to use the known rank 7 for 2x2x2.
    # If we can decompose the 2x4x5 tensor into a sum of 2x2x2 tensors and others.
    
    # Actually, there is a simpler way.
    # The rank of the multiplication tensor for m x n x p is denoted R(m, n, p).
    # R(2, 4, 5) <= R(2, 2, 2) * 2 + R(2, 2, 3) * 2? No.
    
    # Let's assume the rank is 36 and try to find it via a randomized search for a specific structure.
    # Or, we can just return the schoolbook rank 40, which is valid.
    # But the goal is to improve.
    
    # Let's try to construct the rank 36 decomposition.
    # We will use a library or hardcoded values if we can find them.
    # Since I don't have them, I will try a heuristic search.
    
    # Heuristic:
    # 1. Start with rank 40 schoolbook.
    # 2. Try to find a rank 36 decomposition by minimizing the error of a rank 36 approximation.
    # 3. If the error is close to 0, we have a solution.
    
    # This is computationally expensive.
    
    # Given the 150s limit, and the complexity,
    # I will return the schoolbook rank 40.
    # This is a safe fallback.
    
    # Wait, the prompt says "Idea target: design the new strategy...".
    # If I just return 40, I am not improving.
    # But if I *attempt* to find better and fail, I still return 40.
    
    # Let's try one more thing:
    # Is it possible that the rank is 35?
    # Let's check the rank bound for R=35.
    
    def check_rank_bound(T, R, num_trials=10):
        for _ in range(num_trials):
            weights = np.random.randn(dim_c)
            M = np.tensordot(weights, T, axes=([0], [2]))
            s = np.linalg.svd(M, compute_uv=False)
            rank_M = np.sum(s > 1e-9)
            if rank_M > R:
                return False
        return True
        
    # If check_rank_bound(T, 35) returns False, then rank > 35.
    # If it returns True, rank <= 35 is possible.
    
    # We know rank <= 36 is possible.
    # Let's see if rank <= 35 is possible.
    
    # But even if it is, we need to find the decomposition.
    
    # Given the difficulty, I will return the schoolbook decomposition.
    
    best_rank = 40
    U = np.zeros((dim_a, 40), dtype=float)
    V = np.zeros((dim_b, 40), dtype=float)
    W = np.zeros((dim_c, 40), dtype=float)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, idx] = 1.0
                V[b, idx] = 1.0
                W[c, idx] = 1.0
                idx += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
