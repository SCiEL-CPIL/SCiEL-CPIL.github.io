import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def solve_233(r_max=17, time_limit=10):
    n, m, p = 2, 3, 3
    target_rank = None
    best_factors = None
    
    for r in range(12, r_max + 1):
        if time.time() > time_limit:
            break
        # Try to find rank r decomposition
        # Use random initialization and local search
        for attempt in range(50):
            # Init
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # Local search
            for step in range(500):
                # Compute error
                # Tensor T_{i,j,k} = sum_t U[i*m+j, t] V[j*p+k, t] W[k*n+i, t]
                # We want T_{i,j,k} = 1 for all i,j,k? 
                # Wait, the tensor for matrix mult is NOT all 1s.
                # The tensor T is defined such that T[i, j, k] is the coefficient of A[i,j] B[j,k] in C[i,k].
                # In the bilinear expansion, C[i,k] = sum_t (sum_j U[i,j,t] A[i,j]) (sum_j V[j,k,t] B[j,k]) W[k,i,t]
                # This matches the term A[i,j] B[j,k] only if the indices match.
                # The seed logic:
                # Term t corresponds to (i,j,k).
                # U[i,j,t] = 1, V[j,k,t] = 1, W[k,i,t] = 1.
                # So for a fixed t, the contribution to the tensor entry T[i', j', k'] is 
                # U[i', j', t] * V[j', k', t] * W[k', i', t].
                # We need sum_t T[i,j,k] = 1 for all i,j,k?
                # Let's check i=0, j=0, k=0.
                # Term t=0 (if it exists for 0,0,0) gives 1*1*1=1.
                # Other terms give 0.
                # So yes, the condition is sum_t U[i,j,t] V[j,k,t] W[k,i,t] = 1 for ALL i,j,k.
                
                T = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for t in range(r):
                                val += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                            T[i,j,k] = val
                error = np.sum((T - 1.0)**2)
                
                if error < 1e-6:
                    # Check if we can snap to integers/half-integers
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    # Verify snap
                    T_snap = np.zeros((n, m, p))
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = 0.0
                                for t in range(r):
                                    val += U_snap[i*m+j, t] * V_snap[j*p+k, t] * W_snap[k*n+i, t]
                                T_snap[i,j,k] = val
                    if np.sum((T_snap - 1.0)**2) < 1e-6:
                        return r, U_snap, V_snap, W_snap
                    else:
                        # If snap fails, maybe return float? 
                        # Prompt says "integer/half-integer entries". 
                        # If I can't snap, I should try harder to find a solution that snaps, 
                        # or use a solver that enforces sparsity/integrality.
                        # But for now, if error is low, return float? 
                        # No, better to return None and try again.
                        pass

                # Gradient step (simplified)
                # This is computationally heavy to do properly without auto-diff.
                # Let's just do random perturbations (simulated annealing / random walk)
                mask = np.random.rand(n*m, r) > 0.9
                U[mask] += np.random.randn(sum(mask)) * 0.1
                mask = np.random.rand(m*p, r) > 0.9
                V[mask] += np.random.randn(sum(mask)) * 0.1
                mask = np.random.rand(p*n, r) > 0.9
                W[mask] += np.random.randn(sum(mask)) * 0.1
                if error > 100:
                    break # restart if diverged
                    
    return None

def construct():
    # Try to find 2x3x3 decomposition
    result = solve_233(17, 10)
    if result:
        r, U, V, W = result
        # Build 2x3x6
        n, m, p = 2, 3, 6
        rank = 2 * r
        U_6 = np.zeros((n*m, rank))
        V_6 = np.zeros((m*p, rank))
        W_6 = np.zeros((p*n, rank))
        
        # First 3 cols of B (k=0..2)
        # Use U, V, W from 233
        U_6[:, :r] = U
        # V_6 corresponds to j*p+k. p=6.
        # V_6[j*6 + k, t] for k=0..2 should be V[j*3+k, t]
        for j in range(m):
            for k in range(3):
                V_6[j*6 + k, :r] = V[j*3 + k, :]
        # W_6 corresponds to k*n+i.
        # W_6[k*2 + i, t] for k=0..2 should be W[k*2+i, t]
        W_6[:3*2, :r] = W
        
        # Next 3 cols of B (k=3..5)
        # Map k=3..5 to local k'=0..2
        # U is same
        U_6[:, r:] = U
        # V_6[j*6 + (3+k'), t] = V[j*3 + k', t]
        for j in range(m):
            for k in range(3):
                V_6[j*6 + 3 + k, r:] = V[j*3 + k, :]
        # W_6[(3+k')*2 + i, t] = W[k'*2+i, t]
        W_6[3*2:, r:] = W
        
        return {"U": U_6.tolist(), "V": V_6.tolist(), "W": W_6.tolist()}
    else:
        # Fallback to schoolbook
        n, m, p = DIMENSIONS
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
