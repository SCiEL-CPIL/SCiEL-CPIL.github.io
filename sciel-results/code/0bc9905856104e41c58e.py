"""Rewritten program for matrix multiplication 3x3x7 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 21
    pn = p * n  # 21
    rank_school = n * m * p  # 63

    # Build schoolbook decomposition
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    best_rank = rank_school

    def verify_decomposition(U, V, W, tol=1e-9):
        """Check if U, V, W form a valid decomposition."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def tensor_error(U, V, W):
        """Compute the Frobenius norm of the tensor error."""
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                    err += (val - 1.0) ** 2
        return np.sqrt(err)

    # Strategy 1: Hybrid Strassen + schoolbook
    # Use Strassen for the 3x3x3 block (first 3 columns of B), schoolbook for remaining 4 columns
    # Strassen gives rank 7 for 3x3x3, schoolbook gives 3*3*4=36 for 3x3x4
    # Total: 7 + 36 = 43
    
    # Strassen's 7 rank-1 tensors for 3x3x3
    # Using 0-indexed: i,j,k in {0,1,2}
    # M1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1])B[0,0]
    # M3 = A[0,0](B[1,0]-B[1,1])
    # M4 = A[1,1](B[0,1]-B[0,0])
    # M5 = (A[0,0]+A[0,1])B[1,1]
    # M6 = (A[1,0]-A[0,0])(B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1])(B[1,0]+B[1,1])
    
    strassen_U = []
    strassen_V = []
    strassen_W = []
    
    # M1: A[0,0]+A[1,1] times B[0,0]+B[1,1]
    # U: (0,0)=1, (1,1)=1
    # V: (0,0)=1, (1,1)=1
    # W: contributes to C[0,0] and C[1,1]
    strassen_U.append([(0,0,1), (1,1,1)])
    strassen_V.append([(0,0,1), (1,1,1)])
    strassen_W.append([(0,0,1), (1,1,1)])  # C[0,0] += M1, C[1,1] += M1
    
    # M2: (A[1,0]+A[1,1]) * B[0,0]
    strassen_U.append([(1,0,1), (1,1,1)])
    strassen_V.append([(0,0,1)])
    strassen_W.append([(1,0,1)])  # C[1,0] += M2
    
    # M3: A[0,0] * (B[1,0]-B[1,1])
    strassen_U.append([(0,0,1)])
    strassen_V.append([(1,0,1), (1,1,-1)])
    strassen_W.append([(0,1,1)])  # C[0,1] += M3
    
    # M4: A[1,1] * (B[0,1]-B[0,0])
    strassen_U.append([(1,1,1)])
    strassen_V.append([(0,1,1), (0,0,-1)])
    strassen_W.append([(0,0,-1), (1,1,-1)])  # C[0,0] -= M4, C[1,1] -= M4
    
    # M5: (A[0,0]+A[0,1]) * B[1,1]
    strassen_U.append([(0,0,1), (0,1,1)])
    strassen_V.append([(1,1,1)])
    strassen_W.append([(0,0,1)])  # C[0,0] += M5
    
    # M6: (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    strassen_U.append([(1,0,1), (0,0,-1)])
    strassen_V.append([(0,0,1), (0,1,1)])
    strassen_W.append([(1,0,1)])  # C[1,0] += M6
    
    # M7: (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    strassen_U.append([(0,1,1), (1,1,-1)])
    strassen_V.append([(1,0,1), (1,1,1)])
    strassen_W.append([(0,1,1)])  # C[0,1] += M7
    
    # Now build the full 3x3x7 decomposition
    # For k=0,1,2: use Strassen
    # For k=3,4,5,6: use schoolbook
    
    hybrid_rank = 7 + n * m * (p - 3)  # 7 + 36 = 43
    U_hybrid = np.zeros((nm, hybrid_rank))
    V_hybrid = np.zeros((mp, hybrid_rank))
    W_hybrid = np.zeros((pn, hybrid_rank))
    
    col = 0
    # Add Strassen components for k=0,1,2
    for r in range(7):
        for (i, j, val) in strassen_U[r]:
            U_hybrid[i * m + j, col] += val
        for (j, k, val) in strassen_V[r]:
            V_hybrid[j * p + k, col] += val
        for (k, i, val) in strassen_W[r]:
            W_hybrid[k * n + i, col] += val
        col += 1
    
    # Add schoolbook components for k=3,4,5,6
    for i in range(n):
        for j in range(m):
            for k in range(3, p):
                U_hybrid[i * m + j, col] = 1
                V_hybrid[j * p + k, col] = 1
                W_hybrid[k * n + i, col] = 1
                col += 1
    
    if verify_decomposition(U_hybrid, V_hybrid, W_hybrid):
        best_U = U_hybrid.copy()
        best_V = V_hybrid.copy()
        best_W = W_hybrid.copy()
        best_rank = hybrid_rank
    
    # Strategy 2: Try to improve with optimization
    # Use the hybrid as starting point and try to reduce rank
    
    def optimize_decomposition(U_init, V_init, W_init, target_rank, max_iters=200, time_limit=15.0):
        """Try to optimize a decomposition to reach target_rank."""
        nonlocal best_U, best_V, best_W, best_rank
        start = time.time()
        
        # Initialize with random perturbations
        np.random.seed(123)
        U = U_init[:, :target_rank].copy() if U_init.shape[1] >= target_rank else np.zeros((nm, target_rank))
        V = V_init[:, :target_rank].copy() if V_init.shape[1] >= target_rank else np.zeros((mp, target_rank))
        W = W_init[:, :target_rank].copy() if W_init.shape[1] >= target_rank else np.zeros((pn, target_rank))
        
        # Add small random perturbations to escape local minima
        U += np.random.randn(nm, target_rank) * 0.1
        V += np.random.randn(mp, target_rank) * 0.1
        W += np.random.randn(pn, target_rank) * 0.1
        
        best_err = tensor_error(U, V, W)
        best_U_opt = U.copy()
        best_V_opt = V.copy()
        best_W_opt = W.copy()
        
        lr = 0.01
        for it in range(max_iters):
            if time.time() - start > time_limit:
                break
            if best_err < 1e-9:
                break
            
            # Compute gradients
            U_grad = np.zeros_like(U)
            V_grad = np.zeros_like(V)
            W_grad = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        
                        val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                        err = val - 1.0
                        
                        # d(err^2)/d(U[u_idx,r]) = 2*err * V[v_idx,r] * W[w_idx,r]
                        U_grad[u_idx, :] += 2 * err * V[v_idx, :] * W[w_idx, :]
                        V_grad[v_idx, :] += 2 * err * U[u_idx, :] * W[w_idx, :]
                        W_grad[w_idx, :] += 2 * err * U[u_idx, :] * V[v_idx, :]
            
            # Update
            U -= lr * U_grad
            V -= lr * V_grad
            W -= lr * W_grad
            
            err = tensor_error(U, V, W)
            if err < best_err:
                best_err = err
                best_U_opt = U.copy()
                best_V_opt = V.copy()
                best_W_opt = W.copy()
                lr = max(lr * 0.95, 0.001)
            else:
                lr = max(lr * 0.5, 0.0001)
        
        if best_err < 1e-8:
            # Round to nearest half-integer
            U_rounded = np.round(best_U_opt * 2) / 2
            V_rounded = np.round(best_V_opt * 2) / 2
            W_rounded = np.round(best_W_opt * 2) / 2
            
            if verify_decomposition(U_rounded, V_rounded, W_rounded):
                if target_rank < best_rank:
                    best_U = U_rounded.copy()
                    best_V = V_rounded.copy()
                    best_W = W_rounded.copy()
                    best_rank = target_rank
                    return True
        
        return False

    # Try to improve from the hybrid decomposition
    if time.time() < deadline - 30:
        for target_rank in range(30, min(43, best_rank)):
            if time.time() > deadline - 20:
                break
            if optimize_decomposition(U_hybrid, V_hybrid, W_hybrid, target_rank, max_iters=100, time_limit=10.0):
                break
    
    # Strategy 3: Random search with structured initialization
    def try_structured_random(rank, max_attempts=50, time_limit=8.0):
        """Try structured random initialization."""
        nonlocal best_U, best_V, best_W, best_rank
        start = time.time()
        
        for attempt in range(max_attempts):
            if time.time() - start > time_limit:
                break
            
            np.random.seed(42 + attempt)
            
            # Initialize with a mix of structured and random
            U = np.zeros((nm, rank))
            V = np.zeros((mp, rank))
            W = np.zeros((pn, rank))
            
            # First few components: structured (schoolbook-like)
            num_structured = min(rank, n * m)
            for r in range(num_structured):
                i = r // m
                j = r % m
                k = r % p
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
            
            # Remaining components: random
            for r in range(num_structured, rank):
                U[:, r] = np.random.choice([-1, -0.5, 0, 0.5, 1], size=nm)
                V[:, r] = np.random.choice([-1, -0.5, 0, 0.5, 1], size=mp)
                W[:, r] = np.random.choice([-1, -0.5, 0, 0.5, 1], size=pn)
            
            if verify_decomposition(U, V, W):
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                best_rank = rank
                return True
        
        return False

    if time.time() < deadline - 15:
        for rank in [25, 28, 30, 32, 35, 38, 40]:
            if time.time() > deadline - 10:
                break
            if rank >= best_rank:
                break
            if try_structured_random(rank, max_attempts=30, time_limit=5.0):
                break

    # Final verification
    if not verify_decomposition(best_U, best_V, best_W):
        # Fall back to schoolbook
        best_U = U_school.copy()
        best_V = V_school.copy()
        best_W = W_school.copy()
        best_rank = rank_school

    # Convert to lists for return
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
