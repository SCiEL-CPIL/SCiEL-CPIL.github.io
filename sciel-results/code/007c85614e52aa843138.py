"""Program for matrix multiplication tensor decomposition task."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x6 matrix.
    Strategy: The 3x3 by 3x6 multiplication splits into two independent 3x3 by 3x3 multiplications.
    We search for a low-rank decomposition of the 3x3x3 multiplication tensor.
    If a rank-R decomposition is found for 3x3x3, the total rank is 2R.
    We try ranks from 20 upwards (lower bound for 3x3x3 is 19).
    """
    n, m, p = DIMENSIONS  # 3, 3, 6
    deadline = time.time() + 110  # Safety margin

    def snap_to_half(x):
        """Snap value to nearest 0.5 multiple."""
        return round(x * 2) / 2.0

    def snap_matrix(M):
        """Snap all entries in matrix to nearest 0.5 multiple."""
        return np.array([[snap_to_half(x) for x in row] for row in M])

    def verify_3x3x3(U, V, W, rank, tol=1e-9):
        """
        Verify if U, V, W (snapped) form an exact decomposition of 3x3x3 multiplication.
        U: (9, rank), V: (9, rank), W: (9, rank)
        """
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def find_3x3x3_decomposition(rank, max_time=10):
        """
        Attempt to find a rank-R decomposition for 3x3x3 multiplication.
        Returns (U, V, W) if found, else None.
        """
        start_time = time.time()
        rng = np.random.RandomState(rank * 100 + 42)
        
        # Target tensor T[i*3+j, j*3+k, k*3+i] = 1
        # We want sum_r U[i*3+j, r] V[j*3+k, r] W[k*3+i, r] = 1
        
        for attempt in range(50):
            if time.time() - start_time > max_time:
                break
            
            # Initialize
            U = rng.randn(9, rank) * 0.5
            V = rng.randn(9, rank) * 0.5
            W = rng.randn(9, rank) * 0.5
            
            # Simple gradient descent / ALS-like refinement
            # We want to minimize sum_{i,j,k} (sum_r U[i,j,r]V[j,k,r]W[k,i,r] - 1)^2
            # Note: only 27 terms are 1, others are 0?
            # No, the tensor is sparse. The condition is ONLY for the 27 valid (i,j,k).
            # For invalid indices, the product sum can be anything?
            # Wait, the decomposition defines a bilinear map.
            # The map is defined on the whole space.
            # The "multiplication tensor" is the tensor of the bilinear form.
            # The entries of the tensor T are 1 at (i,j,k) positions corresponding to the multiplication.
            # Specifically, $T_{u,v,w} = 1$ if $u=i \cdot 3 + j, v=j \cdot 3 + k, w=k \cdot 3 + i$.
            # All other entries of T are 0.
            # So we need:
            # 1. For valid (i,j,k): sum_r ... = 1
            # 2. For invalid (i,j,k): sum_r ... = 0
            
            # This is a system of 27 * 27 * 27 = 19683 equations?
            # No, the tensor is defined on indices $u \in [0,9), v \in [0,9), w \in [0,9)$.
            # The condition is that the reconstructed tensor matches the multiplication tensor.
            
            # Let's define the target tensor T_target
            T_target = np.zeros((9, 9, 9))
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        T_target[i*3+j, j*3+k, k*3+i] = 1.0
            
            # Optimize U, V, W to match T_target
            for step in range(100):
                if time.time() - start_time > max_time:
                    break
                
                # Compute current tensor T_current
                # T_current[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
                # This is a 3-way outer product sum.
                # We can compute it efficiently.
                T_current = np.einsum('ur,vr,wr->uvw', U, V, W)
                
                # Residual
                Residual = T_current - T_target
                
                # Check convergence
                if np.max(np.abs(Residual)) < 1e-8:
                    break
                
                # Gradient Descent
                # dL/dU[u, r] = sum_{v,w} Residual[u, v, w] V[v, r] W[w, r]
                # dL/dV[v, r] = sum_{u,w} Residual[u, v, w] U[u, r] W[w, r]
                # dL/dW[w, r] = sum_{u,v} Residual[u, v, w] U[u, r] V[v, r]
                
                grad_U = np.einsum('uvw,vr,wr->ur', Residual, V, W)
                grad_V = np.einsum('uvw,ur,wr->vr', Residual, U, W)
                grad_W = np.einsum('uvw,ur,vr->wr', Residual, U, V)
                
                # Learning rate
                lr = 0.01
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Snap to half-integers periodically to encourage discrete solutions
                if step % 10 == 9:
                    U = snap_matrix(U)
                    V = snap_matrix(V)
                    W = snap_matrix(W)
                    
                    # Check if snapped solution is exact
                    if verify_3x3x3(U, V, W, rank):
                        return U.tolist(), V.tolist(), W.tolist()

            # Final check after optimization
            U_snapped = snap_matrix(U)
            V_snapped = snap_matrix(V)
            W_snapped = snap_matrix(W)
            
            if verify_3x3x3(U_snapped, V_snapped, W_snapped, rank):
                return U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist()
                
        return None

    # Try ranks from low to high for 3x3x3
    # Lower bound is 19. Try 19, 20, 21, 22, 23.
    best_decomp = None
    best_rank = None
    
    for rank in range(19, 24):
        if time.time() > deadline - 20:
            break
        decomp = find_3x3x3_decomposition(rank, max_time=5)
        if decomp is not None:
            best_decomp = decomp
            best_rank = rank
            break # Found the lowest rank, stop
            
    if best_decomp is not None:
        U3, V3, W3 = best_decomp
        rank_3x3 = best_rank
        total_rank = 2 * rank_3x3
        
        # Construct full 3x3x6 decomposition
        # U_full: (9, total_rank)
        # V_full: (18, total_rank)
        # W_full: (18, total_rank)
        
        U_full = [[0] * total_rank for _ in range(9)]
        V_full = [[0] * total_rank for _ in range(18)]
        W_full = [[0] * total_rank for _ in range(18)]
        
        # First 3x3 block (k=0,1,2)
        for r in range(rank_3x3):
            for i in range(3):
                for j in range(3):
                    U_full[i*3+j][r] = U3[i*3+j][r]
                for j in range(3):
                    for k in range(3):
                        V_full[j*3+k][r] = V3[j*3+k][r]
                    for i in range(3):
                        W_full[k*3+i][r] = W3[k*3+i][r]
                        
        # Second 3x3 block (k=3,4,5)
        for r in range(rank_3x3):
            for i in range(3):
                for j in range(3):
                    U_full[i*3+j][r+rank_3x3] = U3[i*3+j][r]
                for j in range(3):
                    for k in range(3):
                        V_full[j*3+(k+3)][r+rank_3x3] = V3[j*3+k][r]
                    for i in range(3):
                        W_full[(k+3)*3+i][r+rank_3x3] = W3[k*3+i][r]
                        
        return {"U": U_full, "V": V_full, "W": W_full}
    else:
        # Fallback: Schoolbook
        rank = 54
        U = [[0] * rank for _ in range(9)]
        V = [[0] * rank for _ in range(18)]
        W = [[0] * rank for _ in range(18)]
        column = 0
        for i in range(3):
            for j in range(3):
                for k in range(6):
                    U[i * 3 + j][column] = 1
                    V[j * 6 + k][column] = 1
                    W[k * 3 + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
