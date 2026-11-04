"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build sparse representation of T
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_entries.append((a, b, c, 1.0))
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    for idx, (a, b, c, val) in enumerate(T_entries):
        U_school[a, idx] = 1
        V_school[b, idx] = 1
        W_school[c, idx] = 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Try to find lower rank using ALS
    for rank in range(30, best_rank):
        if time.time() > deadline:
            break
        
        found = False
        for seed in range(5):
            if time.time() > deadline:
                break
            
            np.random.seed(seed * 1000 + rank)
            U = np.random.randn(nm, rank) * 0.1
            V = np.random.randn(mp, rank) * 0.1
            W = np.random.randn(pn, rank) * 0.1
            
            for it in range(30):
                if time.time() > deadline:
                    break
                
                # Update U: for each a, minimize sum_{b,c} (sum_r U[a,r]V[b,r]W[c,r] - T[a,b,c])^2
                # = sum_{b,c} (U[a,:] @ (V[b,:] * W[c,:]) - T[a,b,c])^2
                # Gradient wrt U[a,:]: 2 * sum_{b,c} (U[a,:] @ (V[b,:] * W[c,:]) - T[a,b,c]) * (V[b,:] * W[c,:])
                # Set to 0: U[a,:] @ sum_{b,c} (V[b,:] * W[c,:])^T * (V[b,:] * W[c,:]) = sum_{b,c} T[a,b,c] * (V[b,:] * W[c,:])
                
                # Precompute VW_outer: (V[b,:] * W[c,:]) for all b,c
                # This is expensive, so let's use a different approach
                
                # Use normal equations with aggregation
                # For U: A_U = sum_{b,c} outer(V[b]*W[c], V[b]*W[c]), b_U = sum_{b,c} T[a,b,c] * (V[b]*W[c])
                # But this is O(nm * mp * pn * rank^2) which is too slow
                
                # Instead, use a simpler iterative update
                # U[a,r] -= lr * sum_{b,c} (sum_r' U[a,r']V[b,r']W[c,r'] - T[a,b,c]) * V[b,r] * W[c,r]
                
                lr = 0.01
                for a in range(nm):
                    grad = np.zeros(rank)
                    for b, c, val in [(b, c, T_val) for (aa, b, c, T_val) in T_entries if aa == a]:
                        # Only non-zero T entries
                        pred = np.dot(U[a, :], V[b, :] * W[c, :])
                        grad += (pred - val) * V[b, :] * W[c, :]
                    U[a, :] -= lr * grad
                
                for b in range(mp):
                    grad = np.zeros(rank)
                    for aa, c, val in [(aa, c, T_val) for (aa, bb, c, T_val) in T_entries if bb == b]:
                        pred = np.dot(U[aa, :], V[b, :] * W[c, :])
                        grad += (pred - val) * U[aa, :] * W[c, :]
                    V[b, :] -= lr * grad
                
                for c in range(pn):
                    grad = np.zeros(rank)
                    for aa, bb, val in [(aa, bb, T_val) for (aa, bb, cc, T_val) in T_entries if cc == c]:
                        pred = np.dot(U[aa, :], V[bb, :] * W[c, :])
                        grad += (pred - val) * U[aa, :] * V[bb, :]
                    W[c, :] -= lr * grad
                
                # Check residual
                residual = 0
                for a, b, c, val in T_entries:
                    pred = np.dot(U[a, :], V[b, :] * W[c, :])
                    residual += (pred - val) ** 2
                # Also check zero entries (sample)
                if residual < 1e-8:
                    # Verify all entries
                    all_zero = True
                    for a in range(nm):
                        for b in range(mp):
                            for c in range(pn):
                                pred = np.dot(U[a, :], V[b, :] * W[c, :])
                                expected = 1.0 if (a, b, c) in [(i*m+j, j*p+k, k*n+i) for i in range(n) for j in range(m) for k in range(p)] else 0.0
                                if abs(pred - expected) > 1e-6:
                                    all_zero = False
                                    break
                            if not all_zero:
                                break
                        if not all_zero:
                            break
                    if all_zero:
                        if rank < best_rank:
                            best_rank = rank
                            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        found = True
                        break
            
            if found:
                break
        
        if found:
            break
    
    def to_list(arr):
        return arr.tolist()
    
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
# EVOLVE-BLOCK-END
