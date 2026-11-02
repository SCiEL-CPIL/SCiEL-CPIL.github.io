"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 5]
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin for the 150s limit

# EVOLVE-BLOCK-START

def _verify(U, V, W, n=2, m=2, p=5):
    """
    Verify if the given U, V, W correctly decompose the matrix multiplication tensor.
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = 1.0
                computed = 0.0
                for r in range(rank):
                    computed += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if abs(computed - target) > 1e-9:
                    return False
    return True

def _construct_rank18(n=2, m=2, p=5):
    """
    Construct a rank-18 decomposition using Strassen for 2x2x2 (cols 0-1),
    Strassen for 2x2x2 (cols 2-3), and Schoolbook for 2x2x1 (col 4).
    """
    rank = 18
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    def add_strassen_block(col0, col1, base):
        r = base
        U[0][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1; W[col1*n+1][r] = 1
        
        r = base + 1
        U[2][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1
        W[col0*n+1][r] = 1; W[col1*n+1][r] = -1
        
        r = base + 2
        U[0][r] = 1
        V[0*p+col1][r] = 1; V[1*p+col1][r] = -1
        W[col1*n+0][r] = 1; W[col1*n+1][r] = 1
        
        r = base + 3
        U[3][r] = 1
        V[1*p+col0][r] = 1; V[0*p+col0][r] = -1
        W[col0*n+0][r] = 1; W[col0*n+1][r] = 1
        
        r = base + 4
        U[0][r] = 1; U[1][r] = 1
        V[1*p+col1][r] = 1
        W[col0*n+0][r] = -1; W[col1*n+0][r] = 1
        
        r = base + 5
        U[2][r] = 1; U[0][r] = -1
        V[0*p+col0][r] = 1; V[0*p+col1][r] = 1
        W[col1*n+1][r] = 1
        
        r = base + 6
        U[1][r] = 1; U[3][r] = -1
        V[1*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1
    
    add_strassen_block(0, 1, 0)
    add_strassen_block(2, 3, 7)
    
    # Schoolbook for column 4
    r = 14
    U[0][r] = 1; V[0*p+4][r] = 1; W[4*n+0][r] = 1
    r = 15
    U[1][r] = 1; V[1*p+4][r] = 1; W[4*n+0][r] = 1
    r = 16
    U[2][r] = 1; V[0*p+4][r] = 1; W[4*n+1][r] = 1
    r = 17
    U[3][r] = 1; V[1*p+4][r] = 1; W[4*n+1][r] = 1
    
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    return None

def _construct_schoolbook(n=2, m=2, p=5):
    """
    Construct the standard schoolbook decomposition of rank n*m*p.
    """
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

def _als_search(T, R, n, m, p, n_trials=10, max_iter=100, timeout=30, seed=None):
    """
    ALS search for rank-R decomposition.
    """
    start_time = time.time()
    if seed is not None:
        np.random.seed(seed)
    
    best_err = np.inf
    best_result = None
    
    for trial in range(n_trials):
        if time.time() - start_time > timeout - 2:
            break
        
        U = np.random.randn(n*m, R) * 0.1
        V = np.random.randn(m*p, R) * 0.1
        W = np.random.randn(p*n, R) * 0.1
        
        prev_err = np.inf
        for it in range(max_iter):
            # Solve for U
            VvW = np.einsum('vr,wr->vwr', V, W)
            VvW_flat = VvW.transpose(0, 2, 1).reshape(-1, R)
            for u in range(n*m):
                b = T[u].reshape(-1)
                sol, res, rank_s, sv = np.linalg.lstsq(VvW_flat, b, rcond=None)
                U[u] = sol
            
            # Solve for V
            Uw = np.einsum('ur,wr->uwr', U, W)
            Uw_flat = Uw.transpose(0, 2, 1).reshape(-1, R)
            for v in range(m*p):
                b = T[:, v, :].reshape(-1)
                sol, res, rank_s, sv = np.linalg.lstsq(Uw_flat, b, rcond=None)
                V[v] = sol
            
            # Solve for W
            UV = np.einsum('ur,vr->uvr', U, V)
            UV_flat = UV.reshape(-1, R)
            for w in range(p*n):
                b = T[:, :, w].reshape(-1)
                sol, res, rank_s, sv = np.linalg.lstsq(UV_flat, b, rcond=None)
                W[w] = sol
            
            # Compute error
            result = np.zeros((n*m, m*p, p*n), dtype=np.float64)
            for r in range(R):
                result += np.einsum('u,v,w->uvw', U[:,r], V[:,r], W[:,r])
            err = np.sqrt(np.sum((result - T)**2) / (n*m * m*p * p*n))
            
            if err < best_err:
                best_err = err
                best_result = (U.copy(), V.copy(), W.copy())
            
            if abs(prev_err - err) < 1e-14:
                break
            prev_err = err
    
    return best_result, best_err

def _try_rationalize(U, V, W, T, n, m, p, denoms=[1, 2, 4, 8]):
    """
    Try to rationalize the solution to integer/half-integer values.
    """
    for denom in denoms:
        U_s = np.round(U * denom) / denom
        V_s = np.round(V * denom) / denom
        W_s = np.round(W * denom) / denom
        
        result = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for r in range(len(U[0])):
            result += np.einsum('u,v,w->uvw', U_s[:,r], V_s[:,r], W_s[:,r])
        
        if np.allclose(result, T, atol=1e-9):
            U_list = [[float(x) for x in row] for row in U_s]
            V_list = [[float(x) for x in row] for row in V_s]
            W_list = [[float(x) for x in row] for row in W_s]
            
            for mat in [U_list, V_list, W_list]:
                for i in range(len(mat)):
                    for j in range(len(mat[i])):
                        if abs(mat[i][j] - round(mat[i][j])) < 1e-10:
                            mat[i][j] = int(round(mat[i][j]))
                        elif abs(mat[i][j]*2 - round(mat[i][j]*2)) < 1e-10:
                            mat[i][j] = mat[i][j]  # keep as half-integer
                        else:
                            mat[i][j] = mat[i][j]
            
            if _verify(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Even if not rationalized, return if valid
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    if _verify(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _search_rank(R, n=2, m=2, p=5, timeout=30, seed=None):
    """
    Search for a rank-R decomposition using ALS.
    """
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1.0
    
    best_result, best_err = _als_search(T, R, n, m, p, n_trials=15, max_iter=150, timeout=timeout, seed=seed)
    
    if best_result is not None:
        U, V, W = best_result
        if best_err < 1e-9:
            result = _try_rationalize(U, V, W, T, n, m, p)
            if result is not None:
                return result
    
    return None

def _search_rank17_perturbed(n=2, m=2, p=5, timeout=60):
    """
    Search for rank-17 by perturbing the rank-18 structure.
    """
    base = _construct_rank18(n, m, p)
    if base is None:
        return None
    
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1.0
    
    U0 = np.array(base["U"], dtype=np.float64)
    V0 = np.array(base["V"], dtype=np.float64)
    W0 = np.array(base["W"], dtype=np.float64)
    
    start_time = time.time()
    R = 17
    
    for trial in range(20):
        if time.time() - start_time > timeout - 2:
            break
        
        # Start from rank-18 and remove one column, then re-optimize
        # Try removing each column and see if we can still fit
        for col_to_remove in range(18):
            if time.time() - start_time > timeout - 2:
                break
            
            U = np.delete(U0, col_to_remove, axis=1)
            V = np.delete(V0, col_to_remove, axis=1)
            W = np.delete(W0, col_to_remove, axis=1)
            
            # Add small perturbation
            U += np.random.randn(n*m, R) * 0.01
            V += np.random.randn(m*p, R) * 0.01
            W += np.random.randn(p*n, R) * 0.01
            
            prev_err = np.inf
            for it in range(100):
                # Solve for U
                VvW = np.einsum('vr,wr->vwr', V, W)
                VvW_flat = VvW.transpose(0, 2, 1).reshape(-1, R)
                for u in range(n*m):
                    b = T[u].reshape(-1)
                    sol, res, rank_s, sv = np.linalg.lstsq(VvW_flat, b, rcond=None)
                    U[u] = sol
                
                # Solve for V
                Uw = np.einsum('ur,wr->uwr', U, W)
                Uw_flat = Uw.transpose(0, 2, 1).reshape(-1, R)
                for v in range(m*p):
                    b = T[:, v, :].reshape(-1)
                    sol, res, rank_s, sv = np.linalg.lstsq(Uw_flat, b, rcond=None)
                    V[v] = sol
                
                # Solve for W
                UV = np.einsum('ur,vr->uvr', U, V)
                UV_flat = UV.reshape(-1, R)
                for w in range(p*n):
                    b = T[:, :, w].reshape(-1)
                    sol, res, rank_s, sv = np.linalg.lstsq(UV_flat, b, rcond=None)
                    W[w] = sol
                
                result = np.zeros((n*m, m*p, p*n), dtype=np.float64)
                for r in range(R):
                    result += np.einsum('u,v,w->uvw', U[:,r], V[:,r], W[:,r])
                err = np.sqrt(np.sum((result - T)**2) / (n*m * m*p * p*n))
                
                if abs(prev_err - err) < 1e-14:
                    break
                prev_err = err
            
            if err < 1e-9:
                result_dict = _try_rationalize(U, V, W, T, n, m, p)
                if result_dict is not None:
                    return result_dict
    
    return None

def _gf2_syndrome_search(n=2, m=2, p=5, R=15, timeout=30):
    """
    Combinatorial coding theory approach: model as linear code over GF(2).
    Search for parity-check matrix satisfying bilinear constraints via syndrome decoding.
    """
    start_time = time.time()
    random.seed(42)
    
    # The tensor has n*m*p = 20 nonzero entries
    # Each rank-1 term can cover multiple entries
    # Over GF(2), we look for structured solutions
    
    # Build the support structure
    # For each (i,j,k), the entry is 1
    # A rank-1 term (u,v,w) covers entries where u_i = 1, v_j = 1, w_k = 1
    
    # Strategy: try to find R rank-1 terms whose sum equals the tensor
    # Over GF(2), addition is XOR
    
    # We'll do a greedy + local search over GF(2)
    
    # Initialize with random rank-1 patterns
    best_solution = None
    best_score = np.inf
    
    for trial in range(50):
        if time.time() - start_time > timeout - 2:
            break
        
        # Random initialization
        U = np.zeros((n*m, R), dtype=np.int8)
        V = np.zeros((m*p, R), dtype=np.int8)
        W = np.zeros((p*n, R), dtype=np.int8)
        
        for r in range(R):
            # Random sparse patterns
            U[np.random.choice(n*m, size=random.randint(1,3), replace=False), r] = 1
            V[np.random.choice(m*p, size=random.randint(1,3), replace=False), r] = 1
            W[np.random.choice(p*n, size=random.randint(1,3), replace=False), r] = 1
        
        # Local search: flip bits to improve
        for it in range(200):
            if time.time() - start_time > timeout - 2:
                break
            
            improved = False
            # Compute current tensor
            T_current = np.zeros((n*m, m*p, p*n), dtype=np.int8)
            for r in range(R):
                T_current += np.einsum('u,v,w->uvw', U[:,r], V[:,r], W[:,r])
                T_current %= 2
            
            # Target tensor
            T_target = np.zeros((n*m, m*p, p*n), dtype=np.int8)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_target[i*m+j, j*p+k, k*n+i] = 1
            
            # Count mismatches
            mismatches = np.sum(T_current != T_target)
            
            if mismatches == 0:
                # Found a GF(2) solution - try to lift to integers
                U_int = U.astype(np.float64)
                V_int = V.astype(np.float64)
                W_int = W.astype(np.float64)
                
                # Verify over reals
                if _verify(U_int.tolist(), V_int.tolist(), W_int.tolist(), n, m, p):
                    return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                
                # Try signed variants
                for sign_u in [1, -1]:
                    for sign_v in [1, -1]:
                        for sign_w in [1, -1]:
                            U_s = (U_int * sign_u).tolist()
                            V_s = (V_int * sign_v).tolist()
                            W_s = (W_int * sign_w).tolist()
                            if _verify(U_s, V_s, W_s, n, m, p):
                                return {"U": U_s, "V": V_s, "W": W_s}
            
            # Try flipping random bits
            for _ in range(10):
                mat = random.choice(['U', 'V', 'W'])
                row = random.randint(0, {'U': n*m-1, 'V': m*p-1, 'W': p*n-1}[mat])
                col = random.randint(0, R-1)
                
                if mat == 'U':
                    U[row, col] ^= 1
                elif mat == 'V':
                    V[row, col] ^= 1
                else:
                    W[row, col] ^= 1
                
                # Recompute
                T_new = np.zeros((n*m, m*p, p*n), dtype=np.int8)
                for r in range(R):
                    T_new += np.einsum('u,v,w->uvw', U[:,r], V[:,r], W[:,r])
                    T_new %= 2
                
                new_mismatches = np.sum(T_new != T_target)
                
                if new_mismatches < mismatches:
                    improved = True
                    break
                else:
                    # Revert
                    if mat == 'U':
                        U[row, col] ^= 1
                    elif mat == 'V':
                        V[row, col] ^= 1
                    else:
                        W[row, col] ^= 1
            
            if not improved:
                break
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = START_TIME + TIME_LIMIT
    
    # Try rank 15 search with GF(2) approach
    if time.time() < deadline - 100:
        result = _gf2_syndrome_search(n, m, p, R=15, timeout=min(20, deadline - time.time() - 100))
        if result is not None:
            return result
    
    # Try rank 15 ALS search
    if time.time() < deadline - 80:
        result = _search_rank(15, n, m, p, timeout=min(30, deadline - time.time() - 80), seed=12345)
        if result is not None:
            return result
    
    # Try rank 16 ALS search
    if time.time() < deadline - 60:
        result = _search_rank(16, n, m, p, timeout=min(25, deadline - time.time() - 60), seed=67890)
        if result is not None:
            return result
    
    # Try rank 17 perturbed search
    if time.time() < deadline - 40:
        result = _search_rank17_perturbed(n, m, p, timeout=min(35, deadline - time.time() - 40))
        if result is not None:
            return result
    
    # Try rank 17 ALS search
    if time.time() < deadline - 20:
        result = _search_rank(17, n, m, p, timeout=min(15, deadline - time.time() - 20), seed=11111)
        if result is not None:
            return result
    
    # Fallback: rank 18
    result = _construct_rank18(n, m, p)
    if result is not None:
        return result
    
    # Fallback: Schoolbook (rank 20)
    return _construct_schoolbook(n, m, p)

# EVOLVE-BLOCK-END
