import time
import numpy as np

DIMENSIONS = [3, 3, 7]
_START_TIME = time.time()
_DEADLINE = 140

def construct():
    n, m, p = DIMENSIONS
    
    # Try to find a valid decomposition for 3x3x3 with rank < 27
    # using a greedy search, with a time budget
    search_start = time.time()
    search_budget = 60  # seconds
    
    # The 3x3x3 tensor: T[i,j,k] = 1 for all i,j,k in {0,1,2}
    # We need: sum_t u_t[i,j] * v_t[j,k] * w_t[k,i] = 1
    
    # Greedy search: build decomposition term by term
    # Each term: (u, v, w) where u,v,w are 9-dim vectors
    # u indexed by (i,j), v by (j,k), w by (k,i)
    
    def idx_u(i,j): return i*3+j
    def idx_v(j,k): return j*3+k
    def idx_w(k,i): return k*3+i
    
    # Residual tensor (starts at all 1s)
    residual = np.ones((3,3,3), dtype=float)
    
    terms = []
    
    max_terms = 26  # Try to find rank <= 26
    np.random.seed(42)
    
    for t in range(max_terms):
        if time.time() - search_start > search_budget:
            break
        
        # Find best rank-1 term for current residual
        # Use a simple heuristic: try random directions
        best_score = -1
        best_term = None
        
        for attempt in range(200):
            if time.time() - search_start > search_budget:
                break
            
            u = np.random.randn(9)
            v = np.random.randn(9)
            w = np.random.randn(9)
            
            # Compute contribution tensor
            contrib = np.zeros((3,3,3))
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        contrib[i,j,k] = u[idx_u(i,j)] * v[idx_v(j,k)] * w[idx_w(k,i)]
            
            # Score: how well does this term align with residual?
            score = np.sum(contrib * residual)
            
            if score > best_score:
                best_score = score
                best_term = (u.copy(), v.copy(), w.copy())
        
        if best_term is None or best_score <= 0:
            break
        
        u, v, w = best_term
        terms.append((u, v, w))
        
        # Update residual
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    residual[i,j,k] -= u[idx_u(i,j)] * v[idx_v(j,k)] * w[idx_w(k,i)]
        
        # Check if residual is (approximately) zero
        if np.max(np.abs(residual)) < 1e-6:
            break
    
    # Verify the decomposition
    valid = len(terms) > 0
    if valid:
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    total = sum(u[idx_u(i,j)] * v[idx_v(j,k)] * w[idx_w(k,i)] 
                               for u,v,w in terms)
                    if abs(total - 1) > 1e-6:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
    
    if valid and len(terms) < 27:
        # Use this decomposition for 3x3x3, schoolbook for 3x3x4
        rank_333 = len(terms)
        rank_334 = 36
        total_rank = rank_333 + rank_334
        
        U = [[0] * total_rank for _ in range(n*m)]
        V = [[0] * total_rank for _ in range(m*p)]
        W = [[0] * total_rank for _ in range(p*n)]
        
        for t_idx, (u, v, w) in enumerate(terms):
            for i in range(3):
                for j in range(3):
                    U[i*3+j][t_idx] = u[idx_u(i,j)]
            for j in range(3):
                for k in range(3):
                    V[j*3+k][t_idx] = v[idx_v(j,k)]
            for k in range(3):
                for i in range(3):
                    W[k*3+i][t_idx] = w[idx_w(k,i)]
        
        col_offset = rank_333
        for i in range(n):
            for j in range(m):
                for k in range(3, p):
                    U[i*m+j][col_offset] = 1
                    V[j*p+k][col_offset] = 1
                    W[k*n+i][col_offset] = 1
                    col_offset += 1
        
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback: full schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n*m)]
        V = [[0] * rank for _ in range(m*p)]
        W = [[0] * rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
