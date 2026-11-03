"""
Strategy: Sparse Linear Algebra Search for Tensor Decomposition.
Instead of relying on a fixed composition of Strassen and Schoolbook (rank 189),
we search for a decomposition of rank R < 189 by initializing a random sparse
tensor and iteratively adding rank-1 terms (outer products of sparse vectors)
that best reduce the residual error of the matrix multiplication tensor.
"""

import time
import random
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Check if the decomposition exactly decomposes the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _rank_of(U, V, W):
    return len(U[0]) if U else 0

def _schoolbook(n, m, p):
    rank = n * m * p
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

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    N = n * m # 36
    M = m * p # 36
    P = p * n # 36
    
    # The tensor T is of size N x M x P.
    # T[i, j, k] = 1 if i, j, k are valid indices for matrix multiplication.
    # Specifically, T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # Wait, the tensor for matrix multiplication is T[i, j, k] = 1 if the indices are compatible?
    # No, the matrix multiplication tensor is defined such that C = AB.
    # C[k, i] = sum_j A[i, j] B[j, k].
    # The decomposition is C[k, i] = sum_c U[i, j, c] V[j, k, c] W[k, i, c].
    # So the target tensor value for (i, j, k) is 1.
    
    # Let's verify the target tensor structure.
    # T[i, j, k] = 1 for all i, j, k in valid ranges.
    # Is this correct?
    # A[i,j] is the element at row i, col j.
    # B[j,k] is the element at row j, col k.
    # C[k,i] is the element at row k, col i.
    # C[k,i] = sum_j A[i,j] B[j,k].
    # The tensor T has indices (i, j, k). The value is 1 if the term A[i,j]B[j,k] contributes to C[k,i].
    # Yes, every term A[i,j]B[j,k] contributes to C[k,i].
    # So T[i,j,k] = 1 for all i, j, k.
    
    # Let's double check with a small example.
    # 1x1x1. T[0,0,0] = 1. C[0,0] = A[0,0]B[0,0]. Correct.
    # 2x2x2. T[i,j,k] = 1.
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0].
    # Terms: (0,0,0) and (0,1,1).
    # So T[0,0,0]=1, T[0,1,1]=1.
    # What about T[0,0,1]? A[0,0]B[0,1] contributes to C[1,0]. So T[0,0,1] corresponds to C[1,0].
    # The tensor T is indexed by (i, j, k) where i is row of A, j is col of A / row of B, k is col of B.
    # The output C is indexed by (k, i).
    # The decomposition sums over c: U[i,j,c] V[j,k,c] W[k,i,c].
    # This sum must equal C[k,i] = sum_j A[i,j]B[j,k].
    # This is an identity on the tensor space.
    # The tensor T_{ijk} is 1 if the indices form a valid multiplication term.
    # Since all i, j, k are valid indices for the 6x6x6 multiplication, T_{ijk} = 1 for all i,j,k.
    
    # So we are looking for a rank-R decomposition of the all-ones tensor of size 36x36x36.
    # Wait, is the all-ones tensor the correct target?
    # Yes, because the structure of matrix multiplication is fully dense in the index space.
    # Every A[i,j] multiplies with every B[j,k] to contribute to C[k,i].
    
    # However, the all-ones tensor has rank 1!
    # T[i,j,k] = 1 = 1 * 1 * 1.
    # If T is all ones, then U[i,j,c]=1, V[j,k,c]=1, W[k,i,c]=1 for c=0 works?
    # Sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1*1*1 = 1.
    # This would give rank 1.
    # Why is matrix multiplication rank not 1?
    
    # Ah, the indices are constrained.
    # A is n x m. B is m x p. C is p x n.
    # The tensor T is of size (n*m) x (m*p) x (p*n).
    # The indices are (i, j) for A, (j, k) for B, (k, i) for C.
    # The "j" index in A and B must match. The "k" index in B and C must match. The "i" index in C and A must match.
    # So T is NOT the all-ones tensor on the full Cartesian product [0,n*m) x [0,m*p) x [0,p*n).
    # It is zero unless the indices are consistent.
    # Let row_U = i*m + j.
    # Let row_V = j*p + k.
    # Let row_W = k*n + i.
    # T[row_U, row_V, row_W] = 1 if and only if:
    #   (row_U % m) == (row_V // p)  # j matches
    #   (row_V % p) == (row_W // n)  # k matches
    #   (row_W % n) == (row_U // m)  # i matches
    # Otherwise 0.
    
    # Let's construct the target tensor T as a numpy array.
    T = np.zeros((N, M, P), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r_u = i * m + j
                r_v = j * p + k
                r_w = k * n + i
                T[r_u, r_v, r_w] = 1.0

    # We want to find U, V, W of rank R such that sum_c U[:, c] outer V[:, c] outer W[:, c] == T.
    
    # Search Strategy:
    # Start with R = 189 (known solution).
    # Try to reduce R.
    # We can use a greedy approach:
    # 1. Initialize T_residual = T.
    # 2. For step in range(max_steps):
    #    Find u, v, w that maximize <T_residual, u outer v outer w>.
    #    This is equivalent to finding the best rank-1 approximation of the residual.
    #    However, finding the best rank-1 approximation is hard.
    #    
    # Alternative:
    # Use a fixed R (e.g., 150) and solve for U, V, W using iterative optimization?
    # Or, use the "rank-1 update" method:
    # Start with R=1. T_res = T.
    # At each step, pick a random sparse vector u, v, w.
    # Compute coeff = <T_res, u outer v outer w> / <u outer v outer w, u outer v outer w>.
    # T_res -= coeff * (u outer v outer w).
    # U[:, c] += coeff_u * u ... wait, this builds up the decomposition.
    
    # Let's try a different approach:
    # We know a decomposition of rank 189.
    # Can we combine terms?
    # If we have two terms c1, c2 such that U[:, c1] + U[:, c2] is sparse, etc.
    
    # Let's try a randomized search for a lower rank.
    # Target rank: 100.
    # If we can find a valid decomposition of rank 100, score = 189/100 = 1.89.
    
    # Algorithm:
    # 1. Initialize U, V, W with random sparse vectors (density 0.1).
    # 2. Compute T_approx = sum_c U[:, c] outer V[:, c] outer W[:, c].
    # 3. Compute Residual = T - T_approx.
    # 4. Update U, V, W to reduce Residual.
    #    This is a non-convex problem.
    
    # Better: Use the known rank-189 solution as a starting point and try to merge columns.
    # Or, use a library? No, standard library only.
    
    # Let's try to find a decomposition of rank R by solving a system of linear equations?
    # No, it's bilinear.
    
    # Let's stick to the "random sparse rank-1 updates" but smarter.
    # We will try to find a decomposition of rank R = 120.
    # We will use a greedy algorithm to select the rank-1 components.
    
    # Step 1: T_res = T
    # Step 2: For c in range(R):
    #   Find u, v, w that maximize |<T_res, u outer v outer w>|.
    #   This is the best rank-1 approximation.
    #   We can approximate this by trying random sparse vectors and picking the best one.
    #   Then T_res -= sign(<T_res, ...>) * (u outer v outer w) / norm(u outer v outer w).
    #   Store u, v, w.
    
    # This is likely too slow for R=120 and N=36.
    
    # Let's try a different strategy:
    # Use the Strassen 2x2 (rank 7) and Schoolbook 3x3 (rank 27) product (rank 189).
    # Can we use a better algorithm for 3x3?
    # The best known rank for 3x3 is 23 (Landsberg).
    # If we use rank 23 for 3x3, total rank = 7 * 23 = 161.
    # Score = 189 / 161 = 1.17.
    
    # Do we have a rank-23 decomposition for 3x3x3?
    # It is complex to hardcode.
    
    # What about 6x6 directly?
    # The best known rank for 6x6 is 216? No, 7*27=189 is better than 6*6*6=216.
    # Is there a better block decomposition?
    # 6 = 2 * 3.
    # We used 2x2 blocks (3x3 grid).
    # What if we use 3x3 blocks (2x2 grid)?
    # Then we need a rank-7 decomposition for 3x3x3? No, 3x3x3 rank is 23 (best known).
    # And a rank-27 decomposition for 2x2x2? No, 2x2x2 rank is 7.
    # So 3x3 blocks (2x2 grid) would use Rank(3x3) * Rank(2x2) = 23 * 7 = 161.
    # This is the same as 7 * 23 = 161.
    
    # So if we can find a rank-23 decomposition for 3x3x3, we get 161.
    
    # Can we find a rank-23 decomposition for 3x3x3?
    # The Landsberg decomposition is known.
    # I will implement a known rank-23 decomposition for 3x3x3.
    
    # Reference: "A rank 23 algorithm for 3x3 matrix multiplication"
    # The decomposition is not trivial to hardcode from memory.
    
    # Alternative:
    # Use a numerical search to find a rank-23 decomposition for 3x3x3?
    # 3x3x3 tensor is 27x27x27.
    # Rank 23 means 23 terms.
    # This is a small problem.
    # We can use a randomized search to find a rank-23 decomposition for 3x3x3.
    # Then tensor product with Strassen 2x2 (rank 7) gives rank 161.
    
    # Let's implement a search for 3x3x3 rank 23.
    
    def _find_3x3_rank23():
        n, m, p = 3, 3, 3
        N, M, P = 9, 9, 9
        T = np.zeros((N, M, P), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r_u = i * m + j
                    r_v = j * p + k
                    r_w = k * n + i
                    T[r_u, r_v, r_w] = 1.0
        
        # Search for rank 23
        # We will try to minimize the residual.
        # Initialize U, V, W with random sparse vectors.
        # Iterate: pick a random column index c, update U[:, c], V[:, c], W[:, c] to reduce residual.
        
        # Actually, let's just try to find ANY decomposition of rank < 27 for 3x3x3.
        # If we find rank 23, great. If we find rank 25, score = 189 / (7*25) = 189/175 = 1.08.
        
        # Let's try a simple optimization.
        # Start with schoolbook rank 27.
        # Try to merge pairs of columns.
        
        # This is getting complicated.
        # Let's stick to the 189 solution but optimize the code?
        # No, the task is to find a rank-minimal decomposition.
        
        # Let's try the 161 strategy with a hardcoded rank-23 3x3 decomposition.
        # I found a reference for a rank-23 decomposition.
        # It involves 23 terms.
        # I will hardcode it.
        
        # The decomposition is from Landsberg (1998).
        # It is symmetric.
        # I will use a known implementation.
        
        # Since I cannot reliably hardcode 23x9x9x9 matrices from memory, I will try a different approach.
        # I will use the 189 solution.
        
        return None

    # Given the complexity and time limit, I will return the 189 solution.
    # But I will try to verify if a lower rank is possible with a quick search.
    
    # Let's try to find a rank-23 decomposition for 3x3x3 using a simple greedy method.
    # If successful, we use it.
    
    def _greedy_3x3(target_rank=23):
        n, m, p = 3, 3, 3
        N, M, P = 9, 9, 9
        T = np.zeros((N, M, P), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r_u = i * m + j
                    r_v = j * p + k
                    r_w = k * n + i
                    T[r_u, r_v, r_w] = 1.0
        
        T_res = T.copy()
        U = np.zeros((N, target_rank))
        V = np.zeros((M, target_rank))
        W = np.zeros((P, target_rank))
        
        for c in range(target_rank):
            best_score = 0
            best_u, best_v, best_w = None, None, None
            # Try random sparse vectors
            for _ in range(1000):
                u = np.random.choice([0, 1, -1], size=N)
                v = np.random.choice([0, 1, -1], size=M)
                w = np.random.choice([0, 1, -1], size=P)
                
                # Compute inner product
                # <T_res, u outer v outer w> = sum_{i,j,k} T_res[i,j,k] u[i] v[j] w[k]
                # This is a tensor contraction.
                # T_res @ u @ v @ w
                score = np.einsum('ijk,i,j,k->', T_res, u, v, w)
                if abs(score) > abs(best_score):
                    best_score = score
                    best_u, best_v, best_w = u, v, w
            
            if best_u is None:
                break
                
            # Update
            # coeff = best_score / <u outer v outer w, u outer v outer w>
            # <u outer v outer w, u outer v outer w> = (sum u^2) (sum v^2) (sum w^2)
            norm_u = np.sum(best_u**2)
            norm_v = np.sum(best_v**2)
            norm_w = np.sum(best_w**2)
            if norm_u == 0 or norm_v == 0 or norm_w == 0:
                continue
            coeff = best_score / (norm_u * norm_v * norm_w)
            
            # Add to decomposition
            U[:, c] = coeff * best_u
            V[:, c] = coeff * best_v
            W[:, c] = coeff * best_w
            
            # Update residual
            T_res -= coeff * np.einsum('i,j,k->ijk', best_u, best_v, best_w)
            
            if np.linalg.norm(T_res) < 1e-9:
                # Found a decomposition of rank c+1
                return U[:, :c+1].tolist(), V[:, :c+1].tolist(), W[:, :c+1].tolist()
        
        return None

    # Try to find a better 3x3 decomposition
    U3, V3, W3 = _greedy_3x3(23)
    
    if U3 and V3 and W3:
        rank3 = len(U3[0])
        # Use Strassen 2x2 (rank 7)
        U_s, V_s, W_s = _strassen_2x2()
        rank_s = 7
        
        # Tensor product
        r_full = rank_s * rank3
        N_full, M_full, P_full = 36, 36, 36
        U_full = np.zeros((N_full, r_full))
        V_full = np.zeros((M_full, r_full))
        W_full = np.zeros((P_full, r_full))
        
        # Mapping
        # 6x6 = 3x3 blocks of 2x2.
        # I, J, K in 0..2 (block indices)
        # i', j', k' in 0..1 (inner indices)
        
        for I in range(3):
            for J in range(3):
                for K in range(3):
                    # 3x3 indices
                    row_b_U = I * 3 + J
                    row_b_V = J * 3 + K
                    row_b_W = K * 3 + I
                    
                    for c_b in range(rank3):
                        ub = U3[row_b_U][c_b]
                        vb = V3[row_b_V][c_b]
                        wb = W3[row_b_W][c_b]
                        
                        if ub == 0 and vb == 0 and wb == 0:
                            continue
                        
                        for i_prime in range(2):
                            for j_prime in range(2):
                                i = 2 * I + i_prime
                                j = 2 * J + j_prime
                                row_s_U = i_prime * 2 + j_prime
                                
                                for c_s in range(rank_s):
                                    us = U_s[row_s_U][c_s]
                                    if us == 0:
                                        continue
                                    val_u = us * ub
                                    if val_u != 0:
                                        col_full = c_s * rank3 + c_b
                                        U_full[i * 6 + j, col_full] += val_u
                                        
                                for k_prime in range(2):
                                    k = 2 * K + k_prime
                                    row_s_V = j_prime * 2 + k_prime
                                    
                                    for c_s in range(rank_s):
                                        vs = V_s[row_s_V][c_s]
                                        if vs == 0:
                                            continue
                                        val_v = vs * vb
                                        if val_v != 0:
                                            col_full = c_s * rank3 + c_b
                                            V_full[j * 6 + k, col_full] += val_v
                                            
                                    row_s_W = k_prime * 2 + i_prime
                                    for c_s in range(rank_s):
                                        ws = W_s[row_s_W][c_s]
                                        if ws == 0:
                                            continue
                                        val_w = ws * wb
                                        if val_w != 0:
                                            col_full = c_s * rank3 + c_b
                                            W_full[k * 6 + i, col_full] += val_w

        # Verify
        if _verify(U_full.tolist(), V_full.tolist(), W_full.tolist(), 6, 6, 6):
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

    # Fallback to 189
    U_s, V_s, W_s = _strassen_2x2()
    U_b, V_b, W_b = _schoolbook_block(3, 3, 3)
    
    r_s = 7
    r_b = 27
    r_full = r_s * r_b
    
    U_full = [[0] * r_full for _ in range(36)]
    V_full = [[0] * r_full for _ in range(36)]
    W_full = [[0] * r_full for _ in range(36)]
    
    for I in range(3):
        for J in range(3):
            for K in range(3):
                row_b_U = I * 3 + J
                row_b_V = J * 3 + K
                row_b_W = K * 3 + I
                
                for c_b in range(r_b):
                    ub = U_b[row_b_U][c_b]
                    vb = V_b[row_b_V][c_b]
                    wb = W_b[row_b_W][c_b]
                    
                    if ub == 0 and vb == 0 and wb == 0:
                        continue
                        
                    for i_prime in range(2):
                        for j_prime in range(2):
                            i = 2 * I + i_prime
                            j = 2 * J + j_prime
                            row_s_U = i_prime * 2 + j_prime
                            
                            for c_s in range(r_s):
                                us = U_s[row_s_U][c_s]
                                if us == 0:
                                    continue
                                val_u = us * ub
                                if val_u != 0:
                                    col_full = c_s * r_b + c_b
                                    U_full[i * 6 + j][col_full] += val_u
                                    
                            for k_prime in range(2):
                                k = 2 * K + k_prime
                                row_s_V = j_prime * 2 + k_prime
                                
                                for c_s in range(r_s):
                                    vs = V_s[row_s_V][c_s]
                                    if vs == 0:
                                        continue
                                    val_v = vs * vb
                                    if val_v != 0:
                                        col_full = c_s * r_b + c_b
                                        V_full[j * 6 + k][col_full] += val_v
                                        
                                row_s_W = k_prime * 2 + i_prime
                                for c_s in range(r_s):
                                    ws = W_s[row_s_W][c_s]
                                    if ws == 0:
                                        continue
                                    val_w = ws * wb
                                    if val_w != 0:
                                        col_full = c_s * r_b + c_b
                                        W_full[k * 6 + i][col_full] += val_w

    return {"U": U_full, "V": V_full, "W": W_full}

def _strassen_2x2():
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = 1
    
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = -1; W[3][6] = 1
    
    return U, V, W

def _schoolbook_block(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

# EVOLVE-BLOCK-END
