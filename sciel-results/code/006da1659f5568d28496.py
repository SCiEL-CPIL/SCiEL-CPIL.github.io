import numpy as np
import time
import random

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the schoolbook decomposition for 2x5 * 5x5 matrix multiplication.
    Rank is n*m*p = 50.
    
    This implementation is optimized for speed and reliability.
    It includes a brief, structured search for potential rank reductions,
    but falls back to the guaranteed schoolbook rank if no improvement is found.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10
    
    # Build the target tensor T
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i else 0
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def snap_to_half_integers(arr):
        # Round to nearest 0.5
        return np.round(arr * 2) / 2

    def validate(U, V, W, atol=1e-6):
        # Check if entries are integer or half-integer
        U_snap = snap_to_half_integers(U)
        V_snap = snap_to_half_integers(V)
        W_snap = snap_to_half_integers(W)
        
        # Check if snapped matrices are close to original
        if not (np.allclose(U, U_snap, atol=atol) and 
                np.allclose(V, V_snap, atol=atol) and 
                np.allclose(W, W_snap, atol=atol)):
            return False, None, None, None
            
        # Compute tensor with snapped values
        prod = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap, optimize=True)
        if np.allclose(prod, T, atol=atol):
            return True, U_snap, V_snap, W_snap
        return False, None, None, None

    def compute_loss(U, V, W):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        diff = prod - T
        return np.sum(diff * diff)

    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1.0
                    V[j * p + k][column] = 1.0
                    W[k * n + i][column] = 1.0
                    column += 1
        return {"U": U, "V": V, "W": W}

    # Attempt a brief search for a lower rank
    # Based on the structure, we try ranks slightly below 50.
    # However, given the complexity and the likelihood of the schoolbook rank being minimal 
    # for this specific size without a known identity, we limit the search time strictly.
    
    best_solution = None
    best_rank = n * m * p
    
    # Try rank 45 and 48 with very few restarts to see if a simple pattern exists
    # This is a "refinement" of the search strategy: less time, more targeted.
    ranks_to_try = [45, 48]
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        R = target_rank
        found = False
        
        # Limited restarts
        num_restarts = 2
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            random.seed(42 + restart * 1000 + R)
            
            U = np.zeros((U_dim, R))
            V = np.zeros((V_dim, R))
            W = np.zeros((W_dim, R))
            
            # Structured initialization: Distribute schoolbook terms
            all_terms = list(range(n*m*p))
            assignments = [random.choice(range(R)) for _ in range(n*m*p)]
            
            for term_idx, rank_pos in enumerate(assignments):
                i = term_idx // (m*p)
                j = (term_idx // p) % m
                k = term_idx % p
                
                U[i*m+j, rank_pos] += 1.0
                V[j*p+k, rank_pos] += 1.0
                W[k*n+i, rank_pos] += 1.0
            
            # Short gradient descent
            lr = 0.01
            momentum = 0.9
            vU = np.zeros_like(U)
            vV = np.zeros_like(V)
            vW = np.zeros_like(W)
            
            max_iter = 100  # Very short
            prev_loss = float('inf')
            
            for it in range(max_iter):
                if time.time() > deadline - 2:
                    break
                
                prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
                diff = prod - T
                
                gradU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
                gradV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
                gradW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
                
                vU = momentum * vU + lr * gradU
                vV = momentum * vV + lr * gradV
                vW = momentum * vW + lr * gradW
                
                U -= vU
                V -= vV
                W -= vW
                
                loss = compute_loss(U, V, W)
                
                if loss < 1e-8:
                    break
                
                if loss < prev_loss - 1e-10:
                    prev_loss = loss
                else:
                    # If stuck, break early
                    if it > 50:
                        break
            
            valid, U_snap, V_snap, W_snap = validate(U, V, W)
            if valid:
                best_solution = (U_snap, V_snap, W_snap)
                best_rank = R
                found = True
                break
        
        if found:
            break
    
    # If still not found, return schoolbook
    if best_solution:
        U, V, W = best_solution
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        return schoolbook()

# EVOLVE-BLOCK-END
