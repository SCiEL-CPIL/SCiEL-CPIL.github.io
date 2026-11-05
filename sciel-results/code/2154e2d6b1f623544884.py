import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}, rank

    schoolbook_sol, schoolbook_rank = get_schoolbook()
    best_solution = schoolbook_sol
    best_rank = schoolbook_rank

    def compute_reconstruction(U, V, W, rank):
        Rec = np.zeros((n, m, p))
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            Rec += np.einsum('ij,jk,ki->ijk', u, v, w, optimize=True)
        return Rec

    def is_valid(U, V, W, rank, atol=1e-9):
        Rec = compute_reconstruction(U, V, W, rank)
        return np.allclose(Rec, 1.0, atol=atol)

    def compute_error(U, V, W, rank):
        Rec = compute_reconstruction(U, V, W, rank)
        return np.sum((Rec - 1.0)**2)

    def flatten_params(U, V, W, rank):
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])

    def unflatten_params(params, rank):
        size_U = n * m * rank
        size_V = m * p * rank
        size_W = p * n * rank
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U+size_V]
        W_flat = params[size_U+size_V:size_U+size_V+size_W]
        
        U = U_flat.reshape(n, m, rank).transpose(2, 0, 1) # (rank, n, m) -> (n, m, rank) is easier to keep as (n*m, rank)
        # Let's keep the shape (n*m, rank) for consistency with input
        U = U_flat.reshape(n, m, rank).reshape(n*m, rank)
        V = V_flat.reshape(m, p, rank).reshape(m*p, rank)
        W = W_flat.reshape(p, n, rank).reshape(p*n, rank)
        
        return U, V, W

    def objective(params, rank):
        U, V, W = unflatten_params(params, rank)
        Rec = compute_reconstruction(U, V, W, rank)
        return np.sum((Rec - 1.0)**2)

    def optimize_continuous(rank, seed, max_time_per_restart=5.0):
        np.random.seed(seed)
        # Initialize with small random values
        U = np.random.uniform(-0.5, 0.5, (n*m, rank))
        V = np.random.uniform(-0.5, 0.5, (m*p, rank))
        W = np.random.uniform(-0.5, 0.5, (p*n, rank))
        
        params = flatten_params(U, V, W, rank)
        
        # Use L-BFGS-B for continuous optimization
        # We don't have strict bounds, but small values are better for convergence
        try:
            res = opt.minimize(objective, params, args=(rank,), method='L-BFGS-B', 
                               options={'maxiter': 500, 'maxfun': 5000, 'disp': False})
        except Exception:
            return None, None, None, None

        U_opt, V_opt, W_opt = unflatten_params(res.x, rank)
        
        # Round to nearest half-integer
        U_round = np.round(U_opt * 2) / 2
        V_round = np.round(V_opt * 2) / 2
        W_round = np.round(W_opt * 2) / 2
        
        err = compute_error(U_round, V_round, W_round, rank)
        
        if err < 1e-9:
            return U_round, V_round, W_round, True
        else:
            # Also check if the continuous solution itself is very close
            # and maybe rounding was the issue, but the problem requires half-integers.
            return U_round, V_round, W_round, False

    def search_for_solution():
        nonlocal best_solution, best_rank
        
        # Try ranks from best_rank - 1 down to a lower bound
        # Theoretical lower bound is max(n*m, m*p, p*n) = max(9, 15, 15) = 15.
        # But for matrix multiplication, it's usually higher.
        # Let's try down to 30 first.
        
        for rank in range(best_rank - 1, 29, -1):
            if time.time() > deadline - 10.0:
                break
            
            # Number of restarts depends on time left
            attempts = 10
            for attempt in range(attempts):
                if time.time() > deadline - 5.0:
                    break
                
                seed = 42 + rank * 1000 + attempt * 13
                U_opt, V_opt, W_opt, success = optimize_continuous(rank, seed)
                
                if success:
                    best_solution = {"U": U_opt, "V": V_opt, "W": W_opt}
                    best_rank = rank
                    return True
        
        return False

    search_for_solution()

    def to_lists(mat):
        return mat.tolist()

    return {
        "U": to_lists(best_solution["U"]),
        "V": to_lists(best_solution["V"]),
        "W": to_lists(best_solution["W"])
    }
