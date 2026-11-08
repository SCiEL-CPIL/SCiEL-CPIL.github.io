import numpy as np
import time
import itertools
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 4]

def _build_tensor():
    """Build the matrix multiplication tensor for 3x3 * 3x4."""
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify_decomposition(U, V, W, tol=1e-9):
    """Verify that U, V, W exactly decompose the tensor."""
    T = _build_tensor()
    rank = U.shape[1]
    T_recon = np.zeros_like(T)
    for r in range(rank):
        T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return np.allclose(T_recon, T, atol=tol)

def _verify_list_decomp(U_list, V_list, W_list, tol=1e-9):
    """Verify list-based decomposition."""
    U = np.array(U_list, dtype=float)
    V = np.array(V_list, dtype=float)
    W = np.array(W_list, dtype=float)
    return _verify_decomposition(U, V, W, tol)

def _try_merge_terms(U_list, V_list, W_list, tol=1e-9):
    """
    Try to merge pairs of terms that have similar structure.
    Returns a new decomposition with potentially lower rank, or None if no improvement.
    """
    U = np.array(U_list, dtype=float)
    V = np.array(V_list, dtype=float)
    W = np.array(W_list, dtype=float)
    rank = U.shape[1]
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try all pairs of terms to see if they can be merged
    for i in range(rank):
        for j in range(i+1, rank):
            # Check if term i and term j can be combined
            # This is complex, so let's try a simpler heuristic:
            # If U[:,i] is proportional to U[:,j], etc.
            pass  # This approach is too complex for now
    
    return None

def _random_search_with_structure(rank, max_time, seed=42):
    """
    Search for a decomposition using random initialization with structure-aware perturbations.
    """
    T = _build_tensor()
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    np.random.seed(seed)
    start_time = time.time()
    
    for attempt in range(200):
        if time.time() - start_time > max_time:
            break
        
        # Initialize with sparse random vectors (mostly zeros with some 1s and -1s)
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        for r in range(rank):
            # Each term should have a few non-zero entries
            n_nonzero_U = np.random.randint(1, 4)
            n_nonzero_V = np.random.randint(1, 4)
            n_nonzero_W = np.random.randint(1, 4)
            
            idx_U = np.random.choice(nm, n_nonzero_U, replace=False)
            idx_V = np.random.choice(mp, n_nonzero_V, replace=False)
            idx_W = np.random.choice(pn, n_nonzero_W, replace=False)
            
            U[idx_U, r] = np.random.choice([-1, 1], n_nonzero_U)
            V[idx_V, r] = np.random.choice([-1, 1], n_nonzero_V)
            W[idx_W, r] = np.random.choice([-1, 1], n_nonzero_W)
        
        # Try to optimize
        def objective(params):
            U_f = params[:nm*rank].reshape(nm, rank)
            V_f = params[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
            W_f = params[nm*rank+mp*rank:].reshape(pn, rank)
            T_recon = np.zeros_like(T)
            for r in range(rank):
                T_recon += np.einsum('a,b,c->abc', U_f[:, r], V_f[:, r], W_f[:, r])
            return np.sum((T_recon - T)**2)
        
        def gradient(params):
            U_f = params[:nm*rank].reshape(nm, rank)
            V_f = params[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
            W_f = params[nm*rank+mp*rank:].reshape(pn, rank)
            
            T_recon = np.zeros_like(T)
            for r in range(rank):
                T_recon += np.einsum('a,b,c->abc', U_f[:, r], V_f[:, r], W_f[:, r])
            err = T_recon - T
            
            dU = np.zeros_like(U_f)
            dV = np.zeros_like(V_f)
            dW = np.zeros_like(W_f)
            for r in range(rank):
                dU[:, r] = 2 * np.einsum('abc,b,c->a', err, V_f[:, r], W_f[:, r])
                dV[:, r] = 2 * np.einsum('abc,a,c->b', err, U_f[:, r], W_f[:, r])
                dW[:, r] = 2 * np.einsum('abc,a,b->c', err, U_f[:, r], V_f[:, r])
            
            return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            res = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                          options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10})
        except:
            continue
        
        U_opt = res.x[:nm*rank].reshape(nm, rank)
        V_opt = res.x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
        W_opt = res.x[nm*rank+mp*rank:].reshape(pn, rank)
        
        # Try rounding to half-integers
        for mult in [1, 2, 4]:
            U_round = np.round(U_opt * mult) / mult
            V_round = np.round(V_opt * mult) / mult
            W_round = np.round(W_opt * mult) / mult
            
            if _verify_decomposition(U_round, V_round, W_round):
                return (U_round.tolist(), V_round.tolist(), W_round.tolist())
        
        # Check if close to a valid solution
        if _verify_decomposition(U_opt, V_opt, W_opt, tol=1e-6):
            # Try to find exact values
            for mult in [1, 2, 4, 8]:
                U_round = np.round(U_opt * mult) / mult
                V_round = np.round(V_opt * mult) / mult
                W_round = np.round(W_opt * mult) / mult
                if _verify_decomposition(U_round, V_round, W_round):
                    return (U_round.tolist(), V_round.tolist(), W_round.tolist())
    
    return None

def _get_rank_32_construction():
    """
    Try to construct a rank-32 solution by modifying the rank-34 baseline.
    This is a heuristic approach - may not always work.
    """
    n, m, p = DIMENSIONS
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Start with the rank-34 construction and try to merge 2 terms
    # This is a placeholder - actual construction needs more work
    # For now, return None to indicate we don't have a known rank-32
    return None

def construct():
    deadline = time.time() + 130  # Leave 20s safety margin
    
    # Baseline: rank-34 solution
    def get_baseline():
        n, m, p = DIMENSIONS
        rank = 34
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        # Strassen Left (7 terms for 2x2 block)
        U[0][col]=1; U[4][col]=1; V[0][col]=1; V[5][col]=1; W[0][col]=1; W[4][col]=1; col+=1
        U[3][col]=1; U[4][col]=1; V[0][col]=1; W[1][col]=1; W[4][col]=-1; col+=1
        U[0][col]=1; V[1][col]=1; V[5][col]=-1; W[3][col]=1; W[4][col]=1; col+=1
        U[4][col]=1; V[4][col]=1; V[0][col]=-1; W[0][col]=1; W[1][col]=1; col+=1
        U[0][col]=1; U[1][col]=1; V[5][col]=1; W[0][col]=-1; W[3][col]=1; col+=1
        U[3][col]=1; U[0][col]=-1; V[0][col]=1; V[1][col]=1; W[4][col]=1; col+=1
        U[1][col]=1; U[4][col]=-1; V[4][col]=1; V[5][col]=1; W[0][col]=1; col+=1
        # A12*B2L (4 terms)
        U[2][col]=1; V[8][col]=1; W[0][col]=1; col+=1
        U[2][col]=1; V[9][col]=1; W[3][col]=1; col+=1
        U[5][col]=1; V[8][col]=1; W[1][col]=1; col+=1
        U[5][col]=1; V[9][col]=1; W[4][col]=1; col+=1
        # A21*B1L (4 terms)
        U[6][col]=1; V[0][col]=1; W[2][col]=1; col+=1
        U[6][col]=1; V[1][col]=1; W[5][col]=1; col+=1
        U[7][col]=1; V[4][col]=1; W[2][col]=1; col+=1
        U[7][col]=1; V[5][col]=1; W[5][col]=1; col+=1
        # A22*B2L (2 terms)
        U[8][col]=1; V[8][col]=1; W[2][col]=1; col+=1
        U[8][col]=1; V[9][col]=1; W[5][col]=1; col+=1
        # Strassen Right (7 terms for 2x2 block)
        U[0][col]=1; U[4][col]=1; V[2][col]=1; V[7][col]=1; W[6][col]=1; W[10][col]=1; col+=1
        U[3][col]=1; U[4][col]=1; V[2][col]=1; W[7][col]=1; W[10][col]=-1; col+=1
        U[0][col]=1; V[3][col]=1; V[7][col]=-1; W[9][col]=1; W[10][col]=1; col+=1
        U[4][col]=1; V[6][col]=1; V[2][col]=-1; W[6][col]=1; W[7][col]=1; col+=1
        U[0][col]=1; U[1][col]=1; V[7][col]=1; W[6][col]=-1; W[9][col]=1; col+=1
        U[3][col]=1; U[0][col]=-1; V[2][col]=1; V[3][col]=1; W[10][col]=1; col+=1
        U[1][col]=1; U[4][col]=-1; V[6][col]=1; V[7][col]=1; W[6][col]=1; col+=1
        # A12*B2R (4 terms)
        U[2][col]=1; V[10][col]=1; W[6][col]=1; col+=1
        U[2][col]=1; V[11][col]=1; W[9][col]=1; col+=1
        U[5][col]=1; V[10][col]=1; W[7][col]=1; col+=1
        U[5][col]=1; V[11][col]=1; W[10][col]=1; col+=1
        # A21*B1R (4 terms)
        U[6][col]=1; V[2][col]=1; W[8][col]=1; col+=1
        U[6][col]=1; V[3][col]=1; W[11][col]=1; col+=1
        U[7][col]=1; V[6][col]=1; W[8][col]=1; col+=1
        U[7][col]=1; V[7][col]=1; W[11][col]=1; col+=1
        # A22*B2R (2 terms)
        U[8][col]=1; V[10][col]=1; W[8][col]=1; col+=1
        U[8][col]=1; V[11][col]=1; W[11][col]=1; col+=1
        return {"U": U, "V": V, "W": W}
    
    baseline = get_baseline()
    best_result = baseline
    best_rank = 34
    
    # Verify baseline
    if not _verify_list_decomp(baseline["U"], baseline["V"], baseline["W"]):
        raise ValueError("Baseline construction is invalid")
    
    # Try random search for better ranks
    for rank in range(33, 23, -1):
        if time.time() > deadline - 25:
            break
        
        remaining_time = min(10, max(2, deadline - time.time() - 20))
        result = _random_search_with_structure(rank, remaining_time, seed=rank*1000)
        
        if result is not None:
            U_list, V_list, W_list = result
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            best_rank = rank
            # Continue searching for even better
    
    return best_result
