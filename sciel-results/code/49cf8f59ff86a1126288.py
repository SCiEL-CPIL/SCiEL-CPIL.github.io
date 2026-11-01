"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]
import time
import sys
import random
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the tensor decomposition for matrix multiplication of dimensions (2, 3, 9).
    Returns a dict with keys 'U', 'V', 'W' whose shapes are (n*m, rank), (m*p, rank), (p*n, rank)
    and whose entries are integers or half-integers.
    The decomposition is exact: for all i, j, k,
      sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    and for all other (i,j,k) combinations (which don't exist in this tensor) the equation is vacuous.
    Actually, the matrix multiplication tensor has entries T[i,j,k] for i in [0,n), j in [0,m), k in [0,p).
    The decomposition must satisfy: for each (i,j,k) in the tensor,
      sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s limit

    n, m, p = DIMENSIONS
    n_rows = n * m  # 2*3 = 6
    m_rows = m * p  # 3*9 = 27
    p_rows = p * n  # 9*2 = 18

    # Schoolbook decomposition as fallback (rank = n*m*p = 54)
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n_rows)]
        V = [[0] * rank for _ in range(m_rows)]
        W = [[0] * rank for _ in range(p_rows)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Compute error using Fraction for exactness
    def compute_error_frac(U, V, W):
        """
        U, V, W are lists of lists of Fractions (or ints)
        Returns a Fraction representing the sum of squared errors.
        """
        rank = len(U[0]) if U else 0
        total_error = Fraction(0)
        for i in range(n):
            for j in range(m):
                u_row = U[i * m + j]
                for k in range(p):
                    v_row = V[j * p + k]
                    w_row = W[k * n + i]
                    s = Fraction(0)
                    for c in range(rank):
                        s += u_row[c] * v_row[c] * w_row[c]
                    diff = s - 1
                    total_error += diff * diff
        return total_error

    # Compute error using float for speed (used during search)
    def compute_error_float(U, V, W):
        """
        U, V, W are numpy arrays of int64
        Returns float error
        """
        rank = U.shape[1]
        total_error = 0.0
        for i in range(n):
            for j in range(m):
                u_row = U[i * m + j]
                for k in range(p):
                    v_row = V[j * p + k]
                    w_row = W[k * n + i]
                    s = 0.0
                    for c in range(rank):
                        s += u_row[c] * v_row[c] * w_row[c]
                    diff = s - 1.0
                    total_error += diff * diff
        return total_error

    # Vectorized error computation for speed
    def compute_error_vec(U, V, W):
        """
        U: (n_rows, rank), V: (m_rows, rank), W: (p_rows, rank) as numpy int64
        Returns float error, computed efficiently
        """
        rank = U.shape[1]
        total_error = 0.0
        for i in range(n):
            for j in range(m):
                u_row = U[i * m + j]  # shape (rank,)
                for k in range(p):
                    v_row = V[j * p + k]
                    w_row = W[k * n + i]
                    s = np.dot(u_row, v_row * w_row)
                    diff = s - 1.0
                    total_error += diff * diff
        return total_error

    def local_search(rank, max_iters, time_limit, seed=None):
        """
        Local search starting from random initialization with values in {-1, 0, 1}.
        Tries to reduce error to zero.
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        U = np.zeros((n_rows, rank), dtype=np.int64)
        V = np.zeros((m_rows, rank), dtype=np.int64)
        W = np.zeros((p_rows, rank), dtype=np.int64)
        
        for c in range(rank):
            U[:, c] = np.random.choice([-1, 0, 1], size=n_rows)
            V[:, c] = np.random.choice([-1, 0, 1], size=m_rows)
            W[:, c] = np.random.choice([-1, 0, 1], size=p_rows)
            
        current_error = compute_error_vec(U, V, W)
        
        if current_error == 0:
            return U.tolist(), V.tolist(), W.tolist()

        best_error = current_error
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Hill climbing with occasional restarts
        no_improve = 0
        max_no_improve = 200
        
        for it in range(max_iters):
            if time.time() > time_limit:
                break
            
            if no_improve > max_no_improve:
                # Restart with new random initialization
                U = np.zeros((n_rows, rank), dtype=np.int64)
                V = np.zeros((m_rows, rank), dtype=np.int64)
                W = np.zeros((p_rows, rank), dtype=np.int64)
                for c in range(rank):
                    U[:, c] = np.random.choice([-1, 0, 1], size=n_rows)
                    V[:, c] = np.random.choice([-1, 0, 1], size=m_rows)
                    W[:, c] = np.random.choice([-1, 0, 1], size=p_rows)
                current_error = compute_error_vec(U, V, W)
                no_improve = 0
                continue
            
            # Pick a random variable to modify
            mat_idx = random.choice([0, 1, 2])
            if mat_idx == 0:
                mat = U
                rows = n_rows
            elif mat_idx == 1:
                mat = V
                rows = m_rows
            else:
                mat = W
                rows = p_rows
                
            r_idx = random.randint(0, rows - 1)
            c_idx = random.randint(0, rank - 1)
            
            old_val = mat[r_idx, c_idx]
            new_val = old_val + random.choice([-1, 1])
            
            mat[r_idx, c_idx] = new_val
            
            new_error = compute_error_vec(U, V, W)
            
            if new_error < current_error - 1e-12:
                current_error = new_error
                no_improve = 0
                if new_error < best_error:
                    best_error = new_error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if best_error < 1e-9:
                        # Verify with exact arithmetic
                        U_l, V_l, W_l = U.tolist(), V.tolist(), W.tolist()
                        U_f = [[Fraction(x) for x in row] for row in U_l]
                        V_f = [[Fraction(x) for x in row] for row in V_l]
                        W_f = [[Fraction(x) for x in row] for row in W_l]
                        if compute_error_frac(U_f, V_f, W_f) == 0:
                            return U_l, V_l, W_l
            else:
                mat[r_idx, c_idx] = old_val
                no_improve += 1
        
        if best_error < 1e-9:
            U_l, V_l, W_l = best_U.tolist(), best_V.tolist(), best_W.tolist()
            U_f = [[Fraction(x) for x in row] for row in U_l]
            V_f = [[Fraction(x) for x in row] for row in V_l]
            W_f = [[Fraction(x) for x in row] for row in W_l]
            if compute_error_frac(U_f, V_f, W_f) == 0:
                return U_l, V_l, W_l
        return None

    # Strategy: The task is 2x3 * 3x9.
    # We can exploit structure. The tensor is the product of a (2x3) and (3x9) matrix.
    # Consider decomposing by the middle dimension m=3.
    # For each column j of the 2x3 matrix, we have a 2x1 vector times a 1x9 row vector.
    # The rank of a 2x1 * 1x9 multiplication is 2*9 = 18? No, it's min(2*9, ...) 
    # Actually, multiplying a 2x1 by 1x9 gives a 2x9 matrix. The rank of this bilinear map is 2*9=18
    # in the naive sense, but it can be done with rank 9 if we allow the structure.
    # Actually, (a1,a2) * (b1,...,b9) = (a1*b1, a1*b2, ..., a2*b9).
    # This is a bilinear map from R^2 x R^9 -> R^18.
    # The rank of this tensor is 2*9 = 18? No, it's actually 18 because you need to compute a1*bk and a2*bk for each k.
    # Wait, you can compute a1*bk and a2*bk separately, so 18 scalar multiplications.
    # Can we do better? The tensor is sum_{k=1}^9 (e1 * ek) * (e2 * ek) type structure.
    # Actually, for fixed j, the sub-tensor for column j is:
    #   For i in {0,1}, k in {0,...,8}: T[i, j, k] = 1
    # This is a 2x1x9 tensor. Its rank is 2*9 = 18? 
    # No, the rank of a 2x1x9 tensor (with all ones) is 2*9 = 18? Let's think.
    # A 2x1x9 tensor with all entries 1 can be written as sum of rank-1 tensors.
    # Each rank-1 term is u * v * w where u is 2-dim, v is 1-dim, w is 9-dim.
    # v is just a scalar. So each term is (scalar) * u * w.
    # We need sum_c (v_c) * u_c * w_c = ones(2,9).
    # This is equivalent to: the 2x9 matrix of ones = sum_c v_c * (u_c outer w_c).
    # This is a rank-1 decomposition of a 2x9 matrix, which has rank 1, so we can do it with 1 term if v_c != 0.
    # But wait, v_c is a scalar, u_c is 2-dim, w_c is 9-dim. v_c * (u_c outer w_c) is a 2x9 matrix of rank 1.
    # Sum of rank-1 matrices can give a rank-1 matrix with 1 term.
    # So for each column j, we can use 1 rank-1 term? That would give total rank 3.
    # But that can't be right because the output has 18 dimensions and we need to produce specific values.
    
    # Let me reconsider. The matrix multiplication tensor for A (2x3) * B (3x9) = C (2x9).
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has dimensions 2 x 3 x 9, and T[i,j,k] = 1 for all i,j,k.
    # We want to decompose T as sum_c U_c * V_c * W_c where:
    #   U_c is a vector in R^{2*3} = R^6 (flattened A)
    #   V_c is a vector in R^{3*9} = R^27 (flattened B)
    #   W_c is a vector in R^{9*2} = R^18 (flattened C)
    # And T[i,j,k] = sum_c U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i]
    
    # For a rank-1 term, U_c has nonzeros only at positions (i, j_c) for some i, j_c,
    # V_c has nonzeros only at positions (j_c, k) for some k,
    # W_c has nonzeros only at positions (k, i).
    # So each rank-1 term computes one product A[i,j_c] * B[j_c,k] and adds to C[k*n+i].
    # To get all 54 products, we need at least 54 terms? No, that's the schoolbook.
    # Can we share computation?
    
    # The minimum rank of the 2x3 x 3x9 multiplication tensor is unknown in general,
    # but for small matrices, the rank is often equal to n*m*p or slightly less.
    # Known: rank of 2x2 x 2x2 is 7 (Strassen), not 8.
    # For 2x3 x 3x9, the rank might be less than 54.
    
    # Let's try to find a better decomposition using the structure.
    # Since p=9 is large, and n=2, m=3 are small, maybe we can use Strassen-like ideas.
    
    # For now, let's just do a more thorough search with the local search,
    # trying ranks from 40 to 53.
    
    # Use the vectorized error computation for speed
    candidate_ranks = list(range(40, 54))
    random.shuffle(candidate_ranks)
    
    best_result = None
    best_rank_found = 54
    
    for rank in candidate_ranks:
        if time.time() > deadline - 10:
            break
            
        if rank >= best_rank_found:
            continue
            
        found = False
        num_trials = 8 if rank >= 45 else 4
        for trial in range(num_trials):
            if time.time() > deadline - 5:
                break
                
            res = local_search(rank, max_iters=300, time_limit=deadline - 2, seed=trial * 7 + rank)
            if res is not None:
                best_result = {"U": res[0], "V": res[1], "W": res[2]}
                best_rank_found = rank
                found = True
                break
        
        if found:
            pass

    if best_result:
        return best_result
    else:
        return schoolbook()
# EVOLVE-BLOCK-END
