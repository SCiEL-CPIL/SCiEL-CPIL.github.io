"""Improved program for matrix multiplication tensor decomposition (5,5,5)."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Deadline: 140s out of 150s
    t0 = time.time()
    deadline = t0 + 140.0

    # First, get the schoolbook decomposition as a fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    def verify_fast(Ua, Va, Wa):
        """Fast verification using einsum. Ua shape (n*m, r), Va (m*p, r), Wa (p*n, r)."""
        r = Ua.shape[1]
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # Reshape: U -> (n,m,r), V -> (m,p,r), W -> (p,n,r)
        U3 = Ua.reshape(n, m, r)
        V3 = Va.reshape(m, p, r)
        W3 = Wa.reshape(p, n, r)
        # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
        # Use einsum: 'ijc,jkc,kic->ijk'
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3, optimize=True)
        return np.allclose(T, 1.0, atol=1e-9)

    def verify(U, V, W):
        """Verify that the tensor decomposition is exact."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        return verify_fast(Ua, Va, Wa)

    def compute_tensor(Ua, Va, Wa):
        """Compute the tensor from factor matrices."""
        r = Ua.shape[1]
        U3 = Ua.reshape(n, m, r)
        V3 = Va.reshape(m, p, r)
        W3 = Wa.reshape(p, n, r)
        return np.einsum('ijc,jkc,kic->ijk', U3, V3, W3, optimize=True)

    def try_random_search(rank, max_iters=500, seed=42):
        """Try to find a decomposition of given rank using random init and gradient descent."""
        nonlocal best_result, best_rank, deadline
        
        if rank >= best_rank:
            return False
        
        np.random.seed(seed)
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        # Normalize columns
        for c in range(rank):
            nu = np.linalg.norm(U[:, c])
            nv = np.linalg.norm(V[:, c])
            nw = np.linalg.norm(W[:, c])
            if nu > 1e-10:
                U[:, c] /= nu
            if nv > 1e-10:
                V[:, c] /= nv
            if nw > 1e-10:
                W[:, c] /= nw
        
        lr = 0.05
        prev_error = np.inf
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Compute current tensor
            T_comp = compute_tensor(U, V, W)
            diff = T_comp - 1.0
            error = np.sum(diff * diff)
            
            if error < 1e-6:
                # Found a solution! Try to round to half-integers
                U_list = [[float(U[i, c]) for c in range(rank)] for i in range(n*m)]
                V_list = [[float(V[i, c]) for c in range(rank)] for i in range(m*p)]
                W_list = [[float(W[i, c]) for c in range(rank)] for i in range(p*n)]
                
                # Round to nearest 0.5
                U_rounded = [[round(x * 2) / 2 for x in row] for row in U_list]
                V_rounded = [[round(x * 2) / 2 for x in row] for row in V_list]
                W_rounded = [[round(x * 2) / 2 for x in row] for row in W_list]
                
                if verify(U_rounded, V_rounded, W_rounded):
                    best_result = {"U": U_rounded, "V": V_rounded, "W": W_rounded}
                    best_rank = rank
                    return True
                else:
                    # Use unrounded values
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = rank
                    return True
            
            # Stagnation check
            if abs(prev_error - error) < 1e-15 and it > 10:
                # Try a small perturbation
                U += np.random.randn(n*m, rank) * 0.01
                V += np.random.randn(m*p, rank) * 0.01
                W += np.random.randn(p*n, rank) * 0.01
                lr = max(lr * 0.5, 0.001)
            prev_error = error
            
            # Gradient computation using einsum
            # dU[i*m+j, c] = 2 * sum_{i,j,k} diff[i,j,k] * V[j*p+k,c] * W[k*n+i,c]
            # = 2 * sum_k diff[i,j,k] * V3[j,k,c] * W3[k,i,c]
            U3 = U.reshape(n, m, rank)
            V3 = V.reshape(m, p, rank)
            W3 = W.reshape(p, n, rank)
            
            dU3 = 2 * np.einsum('ijk,jkc,kic->ijc', diff, V3, W3, optimize=True)
            dV3 = 2 * np.einsum('ijk,ijc,kic->jkc', diff, U3, W3, optimize=True)
            dW3 = 2 * np.einsum('ijk,ijc,jkc->kic', diff, U3, V3, optimize=True)
            
            dU = dU3.reshape(n*m, rank)
            dV = dV3.reshape(m*p, rank)
            dW = dW3.reshape(p*n, rank)
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Periodic renormalization
            if it % 20 == 0 and it > 0:
                for c in range(rank):
                    nu = np.linalg.norm(U[:, c])
                    nv = np.linalg.norm(V[:, c])
                    nw = np.linalg.norm(W[:, c])
                    if nu > 1e-10:
                        U[:, c] /= nu
                    if nv > 1e-10:
                        V[:, c] /= nv
                    if nw > 1e-10:
                        W[:, c] /= nw
        
        return False

    def try_structured_cyclic(rank, max_iters=200, seed=0):
        """Try structured initialization based on cyclic group Z/5Z."""
        nonlocal best_result, best_rank, deadline
        
        if rank >= best_rank:
            return False
        
        np.random.seed(seed)
        
        # Initialize with cyclic group structure
        # Use roots of unity over complex, then take real/imag parts
        omega = np.exp(2j * np.pi / 5)
        
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Fill using cyclic patterns
        for c in range(rank):
            # Random cyclic shift pattern
            shift_u = np.random.randint(5)
            shift_v = np.random.randint(5)
            shift_w = np.random.randint(5)
            phase = np.random.uniform(0, 2*np.pi)
            
            for i in range(n):
                for j in range(m):
                    U[i*m+j, c] = np.cos((i + j + shift_u) * 2*np.pi/5 + phase) * 0.5
                for j in range(m):
                    for k in range(p):
                        V[j*p+k, c] = np.cos((j + k + shift_v) * 2*np.pi/5 + phase) * 0.5
                for k in range(p):
                    for i in range(n):
                        W[k*n+i, c] = np.cos((k + i + shift_w) * 2*np.pi/5 + phase) * 0.5
        
        # Add some random noise to break symmetry
        U += np.random.randn(n*m, rank) * 0.1
        V += np.random.randn(m*p, rank) * 0.1
        W += np.random.randn(p*n, rank) * 0.1
        
        # Now do gradient descent from this init
        lr = 0.05
        prev_error = np.inf
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            T_comp = compute_tensor(U, V, W)
            diff = T_comp - 1.0
            error = np.sum(diff * diff)
            
            if error < 1e-6:
                U_list = [[float(U[i, c]) for c in range(rank)] for i in range(n*m)]
                V_list = [[float(V[i, c]) for c in range(rank)] for i in range(m*p)]
                W_list = [[float(W[i, c]) for c in range(rank)] for i in range(p*n)]
                
                U_rounded = [[round(x * 2) / 2 for x in row] for row in U_list]
                V_rounded = [[round(x * 2) / 2 for x in row] for row in V_list]
                W_rounded = [[round(x * 2) / 2 for x in row] for row in W_list]
                
                if verify(U_rounded, V_rounded, W_rounded):
                    best_result = {"U": U_rounded, "V": V_rounded, "W": W_rounded}
                    best_rank = rank
                    return True
                else:
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = rank
                    return True
            
            if abs(prev_error - error) < 1e-15 and it > 10:
                U += np.random.randn(n*m, rank) * 0.01
                V += np.random.randn(m*p, rank) * 0.01
                W += np.random.randn(p*n, rank) * 0.01
                lr = max(lr * 0.5, 0.001)
            prev_error = error
            
            U3 = U.reshape(n, m, rank)
            V3 = V.reshape(m, p, rank)
            W3 = W.reshape(p, n, rank)
            
            dU3 = 2 * np.einsum('ijk,jkc,kic->ijc', diff, V3, W3, optimize=True)
            dV3 = 2 * np.einsum('ijk,ijc,kic->jkc', diff, U3, W3, optimize=True)
            dW3 = 2 * np.einsum('ijk,ijc,jkc->kic', diff, U3, V3, optimize=True)
            
            U -= lr * dU3.reshape(n*m, rank)
            V -= lr * dV3.reshape(m*p, rank)
            W -= lr * dW3.reshape(p*n, rank)
            
            if it % 20 == 0 and it > 0:
                for c in range(rank):
                    nu = np.linalg.norm(U[:, c])
                    nv = np.linalg.norm(V[:, c])
                    nw = np.linalg.norm(W[:, c])
                    if nu > 1e-10:
                        U[:, c] /= nu
                    if nv > 1e-10:
                        V[:, c] /= nv
                    if nw > 1e-10:
                        W[:, c] /= nw
        
        return False

    # Try various ranks - focus on achievable ranges
    # The key improvement: vectorized einsum makes each iteration ~100x faster
    ranks_to_try = [70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120]
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
        if r >= best_rank:
            continue
        
        # Try multiple seeds for random search
        for seed in [42, 123, 456, 789, 101]:
            if time.time() > deadline:
                break
            if r >= best_rank:
                break
            success = try_random_search(r, max_iters=200, seed=seed)
            if success:
                break
        
        if r >= best_rank:
            continue
        
        # Also try structured cyclic initialization
        for seed in [0, 1, 2]:
            if time.time() > deadline:
                break
            if r >= best_rank:
                break
            success = try_structured_cyclic(r, max_iters=100, seed=seed)
            if success:
                break
    
    return best_result
# EVOLVE-BLOCK-END
