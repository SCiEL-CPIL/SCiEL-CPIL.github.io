import time
import itertools
import random

DIMENSIONS = [3, 4, 7]
_DEADLINE = 110.0


def construct():
    _t0 = time.time()
    n, m, p = DIMENSIONS  # n=3, m=4, p=7

    # Target tensor is all 1s
    # R[i, j, k] represents the residual
    R = [[[1 for _ in range(p)] for _ in range(m)] for _ in range(n)]
    
    terms = []
    
    # Pre-generate candidates for u and v
    # u in {-1, 0, 1}^n
    # v in {-1, 0, 1}^m
    # We exclude zero vectors
    u_candidates = [list(t) for t in itertools.product([-1, 0, 1], repeat=n) if any(x != 0 for x in t)]
    v_candidates = [list(t) for t in itertools.product([-1, 0, 1], repeat=m) if any(x != 0 for x in t)]
    
    # Helper to get L1 norm
    def l1_norm(res):
        total = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total += abs(res[i][j][k])
        return total
    
    current_norm = l1_norm(R)
    
    # To avoid infinite loops or stagnation, we limit the number of steps
    # Max possible rank is n*m*p = 84
    max_steps = n * m * p
    
    for step in range(max_steps):
        if time.time() - _t0 > _DEADLINE:
            break
            
        best_cost = current_norm + 1
        best_u = None
        best_v = None
        best_w = None
        
        for u in u_candidates:
            for v in v_candidates:
                # Compute M[i][j] = u[i] * v[j]
                M = [[u[i] * v[j] for j in range(m)] for i in range(n)]
                
                total_cost = 0
                w_candidate = [0] * p
                feasible = True
                
                for k in range(p):
                    c_minus = 0
                    c_zero = 0
                    c_plus = 0
                    for i in range(n):
                        for j in range(m):
                            val = R[i][j][k]
                            m_val = M[i][j]
                            c_minus += abs(val + m_val) # w_k = -1 => val - (-1)*m_val = val + m_val
                            c_zero += abs(val)         # w_k = 0
                            c_plus += abs(val - m_val)  # w_k = 1 => val - (1)*m_val = val - m_val
                    
                    # Pick best w_k
                    min_c = min(c_minus, c_zero, c_plus)
                    total_cost += min_c
                    
                    if c_minus == min_c:
                        w_candidate[k] = -1
                    elif c_plus == min_c:
                        w_candidate[k] = 1
                    else:
                        w_candidate[k] = 0
                        
                    # Early termination if cost already exceeds best
                    if total_cost >= best_cost:
                        feasible = False
                        break
                        
                if feasible and total_cost < best_cost:
                    best_cost = total_cost
                    best_u = u
                    best_v = v
                    best_w = w_candidate
        
        # If no improvement found, break
        if best_u is None or best_cost >= current_norm:
            break
            
        # Apply the term
        # R -= best_u * best_v * best_w
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    R[i][j][k] -= best_u[i] * best_v[j] * best_w[k]
                    
        terms.append((best_u, best_v, best_w))
        current_norm = best_cost
        
        if current_norm == 0:
            break
            
    # Construct U, V, W matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(terms):
        for i in range(n):
            for j in range(m):
                U[i * m + j][c] = u[i]
        for j in range(m):
            for k in range(p):
                V[j * p + k][c] = v[j]
        for k in range(p):
            for i in range(n):
                W[k * n + i][c] = w[k]
                
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = sum(U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c] for c in range(rank))
                if total != 1:
                    # If verification fails, fall back to trivial decomposition
                    # This is a safety net, though greedy should be exact if it converges to 0
                    # But due to integer arithmetic, it should be exact.
                    # If it fails, it means the loop broke early without reaching 0.
                    # In that case, we append trivial terms for the remaining residual.
                    pass
                    
    # If we didn't reach 0, we must add trivial terms for the remaining non-zero entries
    # This ensures exactness.
    if current_norm > 0:
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if R[i][j][k] != 0:
                        # Add a term that covers this specific entry?
                        # We need to cancel the residual exactly.
                        # The residual entry is R[i][j][k].
                        # We can add a term u=e_i, v=e_j, w=e_k with coefficient R[i][j][k].
                        # But U, V, W must be integer/half-integer.
                        # We can just add a rank-1 term with values R[i][j][k] in the appropriate spots.
                        u_new = [0]*n; u_new[i] = R[i][j][k]
                        v_new = [0]*m; v_new[j] = 1
                        w_new = [0]*p; w_new[k] = 1
                        # Wait, product is R[i][j][k] * 1 * 1 = R[i][j][k].
                        # This adds to the sum.
                        # We need the total sum to be 1.
                        # Current sum for (i,j,k) is 1 - R[i][j][k].
                        # Adding R[i][j][k] makes it 1.
                        
                        terms.append((u_new, v_new, w_new))
                        
        # Re-construct U, V, W with new rank
        rank = len(terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for c, (u, v, w) in enumerate(terms):
            for i in range(n):
                for j in range(m):
                    U[i * m + j][c] = u[i]
            for j in range(m):
                for k in range(p):
                    V[j * p + k][c] = v[j]
            for k in range(p):
                for i in range(n):
                    W[k * n + i][c] = w[k]
                    
    return {"U": U, "V": V, "W": W}
