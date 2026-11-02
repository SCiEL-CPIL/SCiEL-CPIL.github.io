import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 10
    dim_v = m * p  # 25
    dim_w = p * n  # 10
    
    # Build target tensor: T[i*m+j, j*p+k, k*n+i] = 1
    target = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                target[u_idx, v_idx, w_idx] = 1.0
    
    def compute_tensor(U, V, W):
        return np.einsum('ar,br,cr->abc', U, V, W)
    
    def loss_and_grad(params, rank):
        u_flat = params[:dim_u * rank]
        v_flat = params[dim_u * rank : dim_u * rank + dim_v * rank]
        w_flat = params[dim_u * rank + dim_v * rank : ]
        
        U = u_flat.reshape(dim_u, rank)
        V = v_flat.reshape(dim_v, rank)
        W = w_flat.reshape(dim_w, rank)
        
        T = compute_tensor(U, V, W)
        diff = T - target
        loss = np.sum(diff * diff)
        
        grad_u = 2.0 * np.einsum('abc,br,cr->ar', diff, V, W)
        grad_v = 2.0 * np.einsum('abc,ar,cr->br', diff, U, W)
        grad_w = 2.0 * np.einsum('abc,ar,br->cr', diff, U, V)
        
        grad = np.concatenate([grad_u.flatten(), grad_v.flatten(), grad_w.flatten()])
        return loss, grad
    
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_result = None
    
    def get_schoolbook():
        rank = n * m * p  # 50
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    def verify_solution(U, V, W):
        T = compute_tensor(U, V, W)
        return np.max(np.abs(T - target)) < 1e-10
    
    # Smart initialization exploiting n=2 structure
    # For each rank component r, we can try to assign it to a "pattern"
    # that covers a subset of the (i,j,k) triples
    def smart_init(rank, seed):
        np.random.seed(seed)
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        
        # For n=2, we have i in {0,1}
        # Each rank component can be assigned to handle certain (i,j,k) combos
        # Strategy: distribute the 50 (i,j,k) triples across rank components
        # with some overlap to allow for rank reduction
        
        triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        
        if rank >= len(triples):
            # Schoolbook-like but with random grouping
            for r in range(rank):
                # Assign a random subset of triples to this rank component
                if r < len(triples):
                    i, j, k = triples[r]
                    U[i*m+j, r] = 1.0
                    V[j*p+k, r] = 1.0
                    W[k*n+i, r] = 1.0
                else:
                    # Extra components: small random values
                    U[:, r] = np.random.randn(dim_u) * 0.1
                    V[:, r] = np.random.randn(dim_v) * 0.1
                    W[:, r] = np.random.randn(dim_w) * 0.1
        else:
            # Need to share rank components across multiple triples
            # Use a greedy approach: assign each triple to the component
            # that best matches its pattern
            for r in range(rank):
                # Randomly select a "center" triple
                i0, j0, k0 = triples[np.random.randint(len(triples))]
                
                # This component handles triples close to (i0, j0, k0)
                # For n=2, i is binary, so we can split by i
                for i in range(n):
                    if i != i0 and np.random.rand() > 0.3:
                        continue  # Skip some i's for rank reduction
                    
                    for j in range(m):
                        for k in range(p):
                            # Assign with probability based on distance
                            dist = abs(i - i0) + abs(j - j0)/m + abs(k - k0)/p
                            prob = max(0, 1.0 - dist)
                            if np.random.rand() < prob:
                                U[i*m+j, r] += 1.0
                                V[j*p+k, r] += 1.0
                                W[k*n+i, r] += 1.0
                
                # Add small random perturbation
                U[:, r] += np.random.randn(dim_u) * 0.05
                V[:, r] += np.random.randn(dim_v) * 0.05
                W[:, r] += np.random.randn(dim_w) * 0.05
        
        # Normalize to avoid numerical issues
        for r in range(rank):
            norm_u = np.linalg.norm(U[:, r])
            norm_v = np.linalg.norm(V[:, r])
            norm_w = np.linalg.norm(W[:, r])
            if norm_u > 1e-10 and norm_v > 1e-10 and norm_w > 1e-10:
                # Keep relative scaling
                avg_norm = (norm_u + norm_v + norm_w) / 3
                U[:, r] *= avg_norm / norm_u
                V[:, r] *= avg_norm / norm_v
                W[:, r] *= avg_norm / norm_w
        
        return U, V, W
    
    # Try ranks from 35 down to 25
    # Focus on lower ranks first since they're more likely to be the minimum
    for target_rank in range(35, 24, -1):
        if time.time() > deadline:
            break
        
        found = False
        n_restarts = 30 if target_rank <= 30 else 20
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            seed = 100000 + target_rank * 1000 + restart
            U0, V0, W0 = smart_init(target_rank, seed)
            
            params = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            def fun(params):
                loss, grad = loss_and_grad(params, target_rank)
                return loss, grad
            
            try:
                res = minimize(fun, params, method='L-BFGS-B', jac=True,
                              options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
                
                if res.fun < 1e-8:
                    u_final = res.x[:dim_u * target_rank].reshape(dim_u, target_rank)
                    v_final = res.x[dim_u * target_rank : dim_u * target_rank + dim_v * target_rank].reshape(dim_v, target_rank)
                    w_final = res.x[dim_u * target_rank + dim_v * target_rank : ].reshape(dim_w, target_rank)
                    
                    # Snap to nearest 0.5
                    u_snap = np.round(u_final * 2.0) / 2.0
                    v_snap = np.round(v_final * 2.0) / 2.0
                    w_snap = np.round(w_final * 2.0) / 2.0
                    
                    if verify_solution(u_snap, v_snap, w_snap):
                        found = True
                        best_result = (u_snap, v_snap, w_snap, target_rank)
                        break
                        
            except Exception:
                continue
        
        if found:
            break
    
    if best_result is not None:
        U, V, W, rank = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        U, V, W, rank = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
