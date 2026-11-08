import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10

    # Build target tensor
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def validate(U, V, W, atol=1e-8):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(prod, T, atol=atol)

    def snap_and_validate(U, V, W):
        # Try multiple snap granularities
        for denom in [1, 2, 3, 4, 5, 6, 10, 12]:
            scale = float(denom)
            U_s = np.round(U * scale) / scale
            V_s = np.round(V * scale) / scale
            W_s = np.round(W * scale) / scale
            if validate(U_s, V_s, W_s):
                return True, U_s, V_s, W_s
        return False, None, None

    def get_trivial_solution():
        rank = n * m * p
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def compute_error(U, V, W):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.sum((prod - T) ** 2)

    def greedy_merge(U, V, W):
        """Greedily try to merge columns to reduce rank"""
        U_curr, V_curr, W_curr = U.copy(), V.copy(), W.copy()
        rank = U_curr.shape[1]
        
        for _ in range(rank - 1):
            if time.time() > deadline - 10:
                break
            if U_curr.shape[1] <= 1:
                break
            
            best_merge = None
            best_error = compute_error(U_curr, V_curr, W_curr)
            best_alpha = 0
            best_pair = None
            
            for a in range(U_curr.shape[1]):
                for b in range(a + 1, U_curr.shape[1]):
                    # Try several alpha values
                    for alpha in [-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2]:
                        U_new = U_curr.copy()
                        V_new = V_curr.copy()
                        W_new = W_curr.copy()
                        
                        U_new[:, a] = U_new[:, a] + alpha * U_new[:, b]
                        V_new[:, a] = V_new[:, a] + alpha * V_new[:, b]
                        W_new[:, a] = W_new[:, a] + alpha * W_new[:, b]
                        
                        # Remove column b
                        U_new = np.delete(U_new, b, axis=1)
                        V_new = np.delete(V_new, b, axis=1)
                        W_new = np.delete(W_new, b, axis=1)
                        
                        error = compute_error(U_new, V_new, W_new)
                        if error < best_error:
                            best_error = error
                            best_alpha = alpha
                            best_pair = (a, b)
            
            if best_pair is not None and best_error < 1e-6:
                a, b = best_pair
                U_new = U_curr.copy()
                V_new = V_curr.copy()
                W_new = W_curr.copy()
                
                U_new[:, a] = U_new[:, a] + best_alpha * U_new[:, b]
                V_new[:, a] = V_new[:, a] + best_alpha * V_new[:, b]
                W_new[:, a] = W_new[:, a] + best_alpha * W_new[:, b]
                
                U_curr = np.delete(U_new, b, axis=1)
                V_curr = np.delete(V_new, b, axis=1)
                W_curr = np.delete(W_new, b, axis=1)
            else:
                break
        
        return U_curr, V_curr, W_curr

    def structured_init(rank, seed):
        """Create structured initialization based on matrix multiplication structure"""
        np.random.seed(seed)
        
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        
        # Assign each (i,j,k) triple to a rank component
        assigned = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    
                    # Try to find an existing rank component that can cover this
                    found = False
                    for r in range(rank):
                        # Check if this rank component already has entries for i,j or j,k or k,i
                        if U[u_idx, r] != 0 and V[v_idx, r] == 0 and W[w_idx, r] == 0:
                            # Can add V and W entries
                            V[v_idx, r] = 1
                            W[w_idx, r] = 1
                            assigned[(i, j, k)] = r
                            found = True
                            break
                        elif U[u_idx, r] == 0 and V[v_idx, r] != 0 and W[w_idx, r] == 0:
                            U[u_idx, r] = 1
                            W[w_idx, r] = 1
                            assigned[(i, j, k)] = r
                            found = True
                            break
                        elif U[u_idx, r] == 0 and V[v_idx, r] == 0 and W[w_idx, r] != 0:
                            U[u_idx, r] = 1
                            V[v_idx, r] = 1
                            assigned[(i, j, k)] = r
                            found = True
                            break
                    
                    if not found:
                        # Assign to a new rank component if available
                        for r in range(rank):
                            if U[:, r].sum() == 0:
                                U[u_idx, r] = 1
                                V[v_idx, r] = 1
                                W[w_idx, r] = 1
                                assigned[(i, j, k)] = r
                                break
        
        return U, V, W

    def sparse_sign_init(rank, seed):
        """
        Exploit small n=p=2 structure: fix U and W to sparse sign patterns {-1,0,1}
        that satisfy local constraints, then optimize V.
        
        Key insight: For minimal rank decompositions of matrix multiplication,
        the U and W matrices often have a structured sparse pattern.
        """
        np.random.seed(seed)
        
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        
        # Strategy: Each rank component should correspond to a "path" in the computation
        # For 2x5 * 5x5, we can think of each (i,j,k) triple as needing coverage
        
        # Assign (i,j,k) triples to rank components with sign patterns
        triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        
        # Shuffle for different initializations
        np.random.shuffle(triples)
        
        col_assignments = {}
        for t_idx, (i, j, k) in enumerate(triples):
            r = t_idx % rank
            u_idx = i * m + j
            v_idx = j * p + k
            w_idx = k * n + i
            
            # Use sign patterns to help with cancellations
            sign_u = 1 if (i + j) % 2 == 0 else -1
            sign_v = 1 if (j + k) % 2 == 0 else -1
            sign_w = 1 if (k + i) % 2 == 0 else -1
            
            U[u_idx, r] += sign_u
            V[v_idx, r] += sign_v
            W[w_idx, r] += sign_w
        
        # Normalize to unit magnitude where possible
        for r in range(rank):
            for idx in range(U_dim):
                if U[idx, r] != 0:
                    U[idx, r] = np.sign(U[idx, r])
            for idx in range(V_dim):
                if V[idx, r] != 0:
                    V[idx, r] = np.sign(V[idx, r])
            for idx in range(W_dim):
                if W[idx, r] != 0:
                    W[idx, r] = np.sign(W[idx, r])
        
        return U, V, W

    def optimize_with_fixed_UW(U_fixed, W_fixed, rank, seed):
        """
        Fix U and W, optimize only V using least squares.
        This exploits the linear structure in V.
        """
        np.random.seed(seed)
        
        # For each (u_idx, w_idx) pair, we need:
        # sum_r U[u_idx, r] * V[:, r] * W[w_idx, r] = T[u_idx, :, w_idx]
        #
        # Let's reformulate: For fixed U and W, V appears linearly in each output.
        # We can set up a linear system.
        
        # Actually, the tensor decomposition is:
        # T[u, v, w] = sum_r U[u, r] * V[v, r] * W[w, r]
        #
        # For fixed U and W, this is linear in V.
        # Let's build the linear system.
        
        # Number of unknowns: V_dim * rank
        # Number of equations: U_dim * V_dim * W_dim (but many are zero)
        
        # Only non-zero entries of T matter:
        # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
        
        # Let's build the linear system A * v_flat = b
        # where v_flat is V flattened
        
        # For each (u, v, w) with T[u,v,w] = 1:
        # sum_r U[u,r] * V[v,r] * W[w,r] = 1
        #
        # This gives us equations. Let's collect them.
        
        equations = []
        rhs = []
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    
                    # Equation: sum_r U[u_idx, r] * V[v_idx, r] * W[w_idx, r] = 1
                    row = np.zeros(V_dim * rank)
                    for r in range(rank):
                        coeff = U_fixed[u_idx, r] * W_fixed[w_idx, r]
                        row[v_idx * rank + r] = coeff
                    
                    equations.append(row)
                    rhs.append(1.0)
        
        A = np.array(equations)
        b = np.array(rhs)
        
        # Solve least squares
        try:
            V_flat, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
            V_opt = V_flat.reshape(V_dim, rank)
            return V_opt
        except Exception:
            return None

    def polish_solution(U, V, W, max_iters=100):
        """
        Polish a solution by alternating optimization.
        """
        U_curr, V_curr, W_curr = U.copy(), V.copy(), W.copy()
        rank = U_curr.shape[1]
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Optimize U given V, W (linear)
            equations_u = []
            rhs_u = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        
                        row = np.zeros(rank)
                        for r in range(rank):
                            row[r] = V_curr[v_idx, r] * W_curr[w_idx, r]
                        equations_u.append(row)
                        rhs_u.append(1.0)
            
            A_u = np.array(equations_u)
            b_u = np.array(rhs_u)
            try:
                U_flat, _, _, _ = np.linalg.lstsq(A_u, b_u, rcond=None)
                U_new = U_flat.reshape(U_dim, rank)
                if compute_error(U_new, V_curr, W_curr) < compute_error(U_curr, V_curr, W_curr):
                    U_curr = U_new
            except:
                pass
            
            # Optimize V given U, W (linear)
            equations_v = []
            rhs_v = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        
                        row = np.zeros(rank)
                        for r in range(rank):
                            row[r] = U_curr[u_idx, r] * W_curr[w_idx, r]
                        equations_v.append(row)
                        rhs_v.append(1.0)
            
            A_v = np.array(equations_v)
            b_v = np.array(rhs_v)
            try:
                V_flat, _, _, _ = np.linalg.lstsq(A_v, b_v, rcond=None)
                V_new = V_flat.reshape(V_dim, rank)
                if compute_error(U_curr, V_new, W_curr) < compute_error(U_curr, V_curr, W_curr):
                    V_curr = V_new
            except:
                pass
            
            # Optimize W given U, V (linear)
            equations_w = []
            rhs_w = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        
                        row = np.zeros(rank)
                        for r in range(rank):
                            row[r] = U_curr[u_idx, r] * V_curr[v_idx, r]
                        equations_w.append(row)
                        rhs_w.append(1.0)
            
            A_w = np.array(equations_w)
            b_w = np.array(rhs_w)
            try:
                W_flat, _, _, _ = np.linalg.lstsq(A_w, b_w, rcond=None)
                W_new = W_flat.reshape(W_dim, rank)
                if compute_error(U_curr, V_curr, W_new) < compute_error(U_curr, V_curr, W_curr):
                    W_curr = W_new
            except:
                pass
        
        return U_curr, V_curr, W_curr

    # Start with trivial solution
    U50, V50, W50 = get_trivial_solution()
    
    best_rank = 50
    best_U, best_V, best_W = U50, V50, W50

    # First, try greedy merging from the trivial solution
    U_merged, V_merged, W_merged = greedy_merge(U50, V50, W50)
    if validate(U_merged, V_merged, W_merged):
        valid, U_s, V_s, W_s = snap_and_validate(U_merged, V_merged, W_merged)
        if valid:
            best_rank = U_s.shape[1]
            best_U, best_V, best_W = U_s, V_s, W_s

    # Try the new approach: fix U and W to sparse sign patterns, optimize V
    for target_rank in range(48, 20, -1):
        if time.time() > deadline - 30:
            break
        if target_rank >= best_rank:
            continue
        
        found = False
        for seed in range(15):
            if time.time() > deadline - 15:
                break
            
            # Try sparse sign initialization
            U_init, V_init, W_init = sparse_sign_init(target_rank, seed)
            
            # First, try optimizing V with fixed U, W
            V_opt = optimize_with_fixed_UW(U_init, W_init, target_rank, seed)
            if V_opt is not None:
                U_curr, V_curr, W_curr = U_init, V_opt, W_init
            else:
                U_curr, V_curr, W_curr = U_init, V_init, W_init
            
            # Polish with alternating optimization
            U_curr, V_curr, W_curr = polish_solution(U_curr, V_curr, W_curr, max_iters=50)
            
            # Try to snap and validate
            valid, U_s, V_s, W_s = snap_and_validate(U_curr, V_curr, W_curr)
            if valid:
                best_rank = target_rank
                best_U, best_V, best_W = U_s, V_s, W_s
                found = True
                break
            
            # Also try structured init + full optimization
            U_struct, V_struct, W_struct = structured_init(target_rank, seed)
            U_pol, V_pol, W_pol = polish_solution(U_struct, V_struct, W_struct, max_iters=50)
            valid, U_s, V_s, W_s = snap_and_validate(U_pol, V_pol, W_pol)
            if valid:
                best_rank = target_rank
                best_U, best_V, best_W = U_s, V_s, W_s
                found = True
                break
        
        if found:
            continue

    return {
        "U": [[float(x) for x in row] for row in best_U],
        "V": [[float(x) for x in row] for row in best_V],
        "W": [[float(x) for x in row] for row in best_W]
    }
