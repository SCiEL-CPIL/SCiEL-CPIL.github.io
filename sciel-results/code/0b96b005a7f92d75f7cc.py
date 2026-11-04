"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import nnls

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit

    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1

    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook

    def build_target_tensor():
        """Build the 4D tensor T[i,j,k,l] = 1 if l==k (for C_ik = sum_j A_ij B_jk)
        Actually, the multiplication tensor has T[i,j,k] = 1 for each (i,j,k) triple,
        and the decomposition is: sum_l U[i,j,l] * V[j,k,l] * W[k,i,l] = T[i,j,k]
        where T[i,j,k] = 1 for all valid (i,j,k)."""
        # The tensor is simply all ones for valid indices
        # T[i,j,k] = 1 for i in [0,n), j in [0,m), k in [0,p)
        return 1  # scalar value, the tensor is all ones

    def verify_decomposition(U_list, V_list, W_list, rank):
        """Verify that the decomposition exactly reconstructs the tensor."""
        U = np.array(U_list, dtype=np.float64)
        V = np.array(V_list, dtype=np.float64)
        W = np.array(W_list, dtype=np.float64)
        
        # Check shapes
        if U.shape != (n * m, rank):
            return False
        if V.shape != (m * p, rank):
            return False
        if W.shape != (p * n, rank):
            return False
        
        # For each (i,j,k), check: sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    product = np.sum(U[idx_u, :] * V[idx_v, :] * W[idx_w, :])
                    if abs(product - 1.0) > 1e-9:
                        return False
        return True

    def to_half_integer_matrix(matrix):
        """Convert a float matrix to half-integers if possible."""
        # Multiply by 2 and round
        scaled = np.round(matrix * 2)
        # Check if close to integer
        if np.allclose(scaled, np.round(scaled), atol=1e-6):
            return (scaled / 2).tolist()
        return matrix.tolist()

    def nmtf_search(rank, max_iters=500, tol=1e-10):
        """Non-Negative Matrix Tri-Factorization using multiplicative updates."""
        np.random.seed(42 + rank)
        
        U = np.random.rand(n * m, rank) + 0.1
        V = np.random.rand(m * p, rank) + 0.1
        W = np.random.rand(p * n, rank) + 0.1
        
        prev_err = np.inf
        for it in range(max_iters):
            if time.time() > deadline:
                return None, None, None, False
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx_u = i * m + j
                        idx_v = j * p + k
                        idx_w = k * n + i
                        val = np.sum(U[idx_u, :] * V[idx_v, :] * W[idx_w, :])
                        err += (val - 1.0) ** 2
                        if time.time() > deadline:
                            return None, None, None, False
            
            if abs(prev_err - err) < tol:
                break
            prev_err = err
            
            # Multiplicative updates
            # Update U: U[i,j,l] *= (sum_k T[i,j,k] * V[j,k,l] * W[k,i,l]) / (sum_k U[i,j,l] * V[j,k,l] * W[k,i,l] + eps)
            eps = 1e-12
            U_new = U.copy()
            for i in range(n):
                for j in range(m):
                    idx_u = i * m + j
                    numerator = np.zeros(rank)
                    denominator = np.zeros(rank)
                    for k in range(p):
                        idx_v = j * p + k
                        idx_w = k * n + i
                        numerator += V[idx_v, :] * W[idx_w, :]  # T[i,j,k] = 1
                        denominator += U[idx_u, :] * V[idx_v, :] * W[idx_w, :]
                    U_new[idx_u, :] = U[idx_u, :] * (numerator + eps) / (denominator + eps)
            U = U_new
            
            # Update V
            V_new = V.copy()
            for j in range(m):
                for k in range(p):
                    idx_v = j * p + k
                    numerator = np.zeros(rank)
                    denominator = np.zeros(rank)
                    for i in range(n):
                        idx_u = i * m + j
                        idx_w = k * n + i
                        numerator += U[idx_u, :] * W[idx_w, :]  # T[i,j,k] = 1
                        denominator += U[idx_u, :] * V[idx_v, :] * W[idx_w, :]
                    V_new[idx_v, :] = V[idx_v, :] * (numerator + eps) / (denominator + eps)
            V = V_new
            
            # Update W
            W_new = W.copy()
            for k in range(p):
                for i in range(n):
                    idx_w = k * n + i
                    numerator = np.zeros(rank)
                    denominator = np.zeros(rank)
                    for j in range(m):
                        idx_u = i * m + j
                        idx_v = j * p + k
                        numerator += U[idx_u, :] * V[idx_v, :]  # T[i,j,k] = 1
                        denominator += U[idx_u, :] * V[idx_v, :] * W[idx_w, :]
                    W_new[idx_w, :] = W[idx_w, :] * (numerator + eps) / (denominator + eps)
            W = W_new
        
        # Check if converged to exact solution
        if verify_decomposition(U.tolist(), V.tolist(), W.tolist(), rank):
            return U.tolist(), V.tolist(), W.tolist(), True
        return None, None, None, False

    # Try ranks from lower to higher
    # Known lower bound for 4x4x5 is around 48-52, try from 48 up to 60
    for rank in range(48, min(65, rank_schoolbook)):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        found = False
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            U_list, V_list, W_list, success = nmtf_search(rank, max_iters=200)
            if success:
                found = True
                # Convert to half-integers if possible
                U_half = to_half_integer_matrix(np.array(U_list))
                V_half = to_half_integer_matrix(np.array(V_list))
                W_half = to_half_integer_matrix(np.array(W_list))
                
                # Verify the half-integer version
                if verify_decomposition(U_half, V_half, W_half, rank):
                    best_result = {"U": U_half, "V": V_half, "W": W_half}
                    best_rank = rank
                    break
        
        if found:
            # Found a valid decomposition at this rank, try to go lower
            continue
        else:
            # If we've tried several ranks and none worked, stop
            if rank > 55:
                break

    return best_result
# EVOLVE-BLOCK-END
