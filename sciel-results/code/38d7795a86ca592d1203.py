"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import logsumexp

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s safety margin
    
    def soft_objective_and_grad(P, beta=50.0):
        # P is (N, D)
        # Compute pairwise squared distances
        # d^2_ij = |Pi|^2 + |Pj|^2 - 2 Pi.Pj
        # Using broadcasting
        sq_norms = np.sum(P * P, axis=1)
        dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2 * (P @ P.T)
        # Ensure non-negative due to numerical errors
        dist_sq = np.maximum(dist_sq, 0)
        
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        d_sq_flat = dist_sq[mask]
        
        # Soft max and min of d_sq
        # max_d_sq ~ logsumexp(beta * d_sq) / beta
        # min_d_sq ~ -logsumexp(-beta * d_sq) / beta
        # Ratio = (logsumexp(beta * d_sq)) / (-logsumexp(-beta * d_sq))
        
        x = beta * d_sq_flat
        log_A = logsumexp(x)
        log_B = logsumexp(-x)
        
        # Objective
        val = log_A / (-log_B)
        
        # Gradient
        # J = f / g where f = log_A, g = -log_B
        # grad J = (g * grad f - f * grad g) / g^2
        
        # grad f = beta * sum( p_ij * grad d_sq_ij )
        # p_ij = exp(x_ij) / A = exp(x_ij - log_A)
        p = np.exp(x - log_A)
        
        # grad g = beta * sum( q_ij * grad d_sq_ij )  (note sign change in derivation above)
        # g = -log_B. d(log_B)/dP = (1/B) sum (-beta exp(-beta d_sq) grad d_sq) = -beta sum q_ij grad d_sq
        # So grad g = - ( -beta sum q_ij grad d_sq ) = beta sum q_ij grad d_sq ?
        # Let's re-verify.
        # B = sum exp(-beta d_sq).
        # d B / dP = sum -beta exp(-beta d_sq) grad d_sq.
        # d log_B / dP = (1/B) d B / dP = -beta sum (exp(-beta d_sq)/B) grad d_sq = -beta sum q_ij grad d_sq.
        # g = -log_B.
        # grad g = - d log_B / dP = beta sum q_ij grad d_sq.
        
        q = np.exp(-x - log_B) # exp(-x) / B
        
        # We need grad d_sq with respect to P.
        # d_sq_ij = |Pi - Pj|^2.
        # grad_Pk d_sq_ij = 2(Pi - Pj) if k=i, -2(Pi - Pj) if k=j, 0 otherwise.
        # We can compute the full gradient matrix G of shape (N, N, D) where G[i,j] = 2(Pi - Pj).
        # Then grad_Pk (sum w_ij G[i,j]_k) ...
        # Actually, sum over all pairs (i,j) of w_ij * 2(Pi - Pj) * (delta_{ki} - delta_{kj})
        # = 2 * sum_j w_ij (Pi - Pj) - 2 * sum_i w_ij (Pi - Pj)
        # Let W be the weight matrix (N, N).
        # grad_Pk = 2 * ( (W @ P)_k - (W * P)_k ) ? No.
        # Let's use the matrix form.
        # grad_P = 2 * (W @ P - W.T @ P) ?
        # Check: (W @ P)_k = sum_j W[k,j] P_j.
        # (W.T @ P)_k = sum_j W[j,k] P_j.
        # Term for pair (i,j) with weight w_ij:
        # Contribution to grad_P_i: 2 w_ij (P_i - P_j).
        # Contribution to grad_P_j: -2 w_ij (P_i - P_j).
        # Sum over all i,j.
        # grad_P_k = sum_{i,j} w_ij (2 delta_{ki} - 2 delta_{kj}) (P_i - P_j)
        # = 2 sum_j w_kj (P_k - P_j) - 2 sum_i w_ik (P_i - P_k)
        # = 2 (sum_j w_kj P_k - sum_j w_kj P_j) - 2 (sum_i w_ik P_i - sum_i w_ik P_k)
        # = 2 (P_k sum_j w_kj - (W P)_k) - 2 ((W.T P)_k - P_k sum_i w_ik)
        # Note sum_j w_kj is row sum, sum_i w_ik is col sum.
        # If W is symmetric (which it is, distances are symmetric), row sum = col sum.
        # Let R_k = sum_j w_kj.
        # grad_P_k = 2 (R_k P_k - (W P)_k) - 2 ((W P)_k - R_k P_k) = 4 R_k P_k - 4 (W P)_k.
        
        # We have two weight matrices: p and q.
        # p is flat, need to reshape to (N, N).
        p_mat = np.zeros((N, N))
        q_mat = np.zeros((N, N))
        p_mat[mask] = p
        q_mat[mask] = q
        
        # Symmetrize just in case (p and q should be symmetric if d_sq is symmetric)
        p_mat = (p_mat + p_mat.T) / 2
        q_mat = (q_mat + q_mat.T) / 2
        
        row_sum_p = np.sum(p_mat, axis=1)
        row_sum_q = np.sum(q_mat, axis=1)
        
        grad_d_sq_p = 4 * (row_sum_p[:, None] * P - p_mat @ P)
        grad_d_sq_q = 4 * (row_sum_q[:, None] * P - q_mat @ P)
        
        # grad f = beta * grad_d_sq_p
        # grad g = beta * grad_d_sq_q
        
        g_val = -log_B
        f_val = log_A
        
        grad_J = (g_val * (beta * grad_d_sq_p) - f_val * (beta * grad_d_sq_q)) / (g_val * g_val)
        
        return val, grad_J

    best_score = np.inf
    best_P = None
    
    # Number of restarts
    n_restarts = 20
    
    for i in range(n_restarts):
        if time.time() > deadline:
            break
            
        # Initialize on unit sphere
        P_init = np.random.randn(N, D)
        norms = np.linalg.norm(P_init, axis=1, keepdims=True)
        P_init = P_init / norms
        
        # Optimize
        try:
            res = minimize(
                lambda x: soft_objective_and_grad(x)[0],
                P_init.flatten(),
                jac=lambda x: soft_objective_and_grad(x)[1].flatten(),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            P_opt = res.x.reshape(N, D)
            
            # Evaluate actual hard score
            # We need to compute the real ratio for the final score
            sq_norms = np.sum(P_opt * P_opt, axis=1)
            dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2 * (P_opt @ P_opt.T)
            dist_sq = np.maximum(dist_sq, 0)
            mask = ~np.eye(N, dtype=bool)
            d_sq_flat = dist_sq[mask]
            
            min_d = np.min(d_sq_flat)
            max_d = np.max(d_sq_flat)
            
            if min_d > 1e-9:
                score = (max_d / min_d)
                # The task asks for squared ratio: (max/min)^2
                # Wait, the prompt says "squared ratio of maximum to minimum pairwise distance".
                # Distance is sqrt(d_sq). Ratio is max_dist / min_dist. Squared is (max_dist/min_dist)^2 = max_d_sq / min_d_sq.
                # So score = max_d_sq / min_d_sq.
                if score < best_score:
                    best_score = score
                    best_P = P_opt
        except Exception:
            pass
            
    if best_P is None:
        # Fallback to seed
        side = int(np.ceil(N ** (1 / D)))
        import itertools
        grid = list(itertools.product(range(side), repeat=D))[:N]
        return np.array(grid, dtype=float)
        
    return best_P
# EVOLVE-BLOCK-END
