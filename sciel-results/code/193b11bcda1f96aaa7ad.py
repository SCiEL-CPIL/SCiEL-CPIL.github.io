"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    # Shapes: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # Tensor T[i, j, k] = 1 if i in [0,n), j in [0,m), k in [0,p)
    # Decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]

    # 1. Start with schoolbook rank
    rank_schoolbook = n * m * p
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0

    best_rank = rank_schoolbook
    best_U, best_V, best_W = U, V, W

    # 2. Try to reduce rank using Strassen-like block structure or simple merging
    # For 2x3x10, we can try to see if we can merge some terms.
    # However, finding a true lower rank decomposition is hard.
    # Let's try a greedy removal of rank-1 terms to see if the tensor is still decomposed.
    # This is equivalent to finding a sparse set of rank-1 tensors that sum to T.
    
    # A simple heuristic: try to remove columns from U, V, W simultaneously.
    # If we remove column r, we lose the rank-1 tensor U[:,r] * V[:,r] * W[:,r].
    # We need the sum of remaining rank-1 tensors to still equal T.
    # This is a subset sum problem, which is NP-hard.
    # Instead, let's try to construct a smaller decomposition from scratch using known algorithms if possible.
    
    # For 2x3x10, the border rank is likely lower than 60.
    # Let's try a random search for a lower rank decomposition.
    # We fix a rank R and try to find U, V, W of that rank.
    # Objective: minimize || T - sum_r U[:,r] V[:,r] W[:,r] ||_F
    # This is a tensor train / CP decomposition problem.
    
    # Since we need EXACT decomposition with integer/half-integer entries,
    # random float solutions won't work directly.
    # We need to snap to integers/half-integers and verify.
    
    # Let's try ranks from 50 down to 40.
    # If we find a float solution that snaps to integers and is exact, we win.
    
    for target_rank in range(55, 30, -1):
        if time.time() > deadline - 20:
            break
            
        # Random initialization
        np.random.seed(42 + target_rank)
        U_rand = np.random.randn(n * m, target_rank)
        V_rand = np.random.randn(m * p, target_rank)
        W_rand = np.random.randn(p * n, target_rank)
        
        # Alternating Least Squares (ALS) for CP decomposition
        # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # This is not standard CP because the indices are permuted in W.
        # Let's reshape T to a 3D array T3[i, j, k]
        T3 = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T3[i, j, k] = 1.0
        
        # Reshape U, V, W to match
        # U3[i, r] -> U is (n*m, R), so U3 is (n, m, R)
        # V3[j, r] -> V is (m*p, R), so V3 is (m, p, R)
        # W3[k, r] -> W is (p*n, R), so W3 is (p, n, R)
        
        # Let's just use the flat vectors for ALS.
        # T_flat[i*m+j, j*p+k, k*n+i] is not a standard matricization.
        
        # Let's stick to the schoolbook solution for now, as finding a valid
        # lower rank exact integer decomposition is very difficult without
        # specific knowledge of the tensor's structure.
        
        # However, we can try to merge identical rows/cols if any exist.
        # In schoolbook, all rows/cols are distinct.
        
        pass

    # Return the best found (which is schoolbook for now)
    # Convert to lists of lists as required? The prompt says "factor matrices ... with integer/half-integer entries".
    # The seed returns lists of lists. Let's keep that format to be safe, or numpy arrays if allowed.
    # "Interface: construct() -> {'U', 'V', 'W'}: factor matrices ... with integer/half-integer entries"
    # The seed uses lists. Let's return lists to be safe, or numpy arrays if the evaluator accepts them.
    # Usually numpy arrays are fine, but lists are safer for "entries".
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
