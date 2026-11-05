"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 20
    pn = p * n  # 10
    
    # Build the target tensor T[a, b, c]
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Structured rank-1 peeling algorithm
    # For each output slice c, the matrix T[:,:,c] has a specific structure.
    # We'll peel rank-1 terms that are linearly independent.
    
    U_list = []
    V_list = []
    W_list = []
    residual = T.copy()
    
    # Try to find a decomposition with rank = 20
    target_rank = 20
    
    for step in range(target_rank):
        # Find the best rank-1 approximation of the residual
        # We'll try structured candidates based on the tensor structure
        
        best_u = None
        best_v = None
        best_w = None
        best_score = -np.inf
        
        # Generate structured candidates for each (i, j, k) combination
        # and also try linear combinations
        
        candidates = []
        
        # Basic structured candidates: for each (i, j, k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    u = np.zeros(nm)
                    v = np.zeros(mp)
                    w = np.zeros(pn)
                    u[a] = 1.0
                    v[b] = 1.0
                    w[c] = 1.0
                    candidates.append((u, v, w))
        
        # Also try to create linear combinations that cover multiple terms
        # For fixed i, try to cover all (j, k) with a few rank-1 terms
        for i in range(n):
            for j in range(m):
                # For fixed i, j: terms are (i, j, k) for k in [0, p)
                # We can create a rank-1 term that covers all k for this (i, j)
                u = np.zeros(nm)
                v = np.zeros(mp)
                w = np.zeros(pn)
                a = i * m + j
                u[a] = 1.0
                for k in range(p):
                    b = j * p + k
                    c = k * n + i
                    v[b] = 1.0
                    w[c] = 1.0
                candidates.append((u, v, w))
            
            # For fixed i: try to cover all (j, k)
            for j in range(m):
                pass  # Already covered above
        
        # For fixed k, try to cover all (i, j)
        for k in range(p):
            for i in range(n):
                u = np.zeros(nm)
                v = np.zeros(mp)
                w = np.zeros(pn)
                c = k * n + i
                w[c] = 1.0
                for j in range(m):
                    a = i * m + j
                    b = j * p + k
                    u[a] = 1.0
                    v[b] = 1.0
                candidates.append((u, v, w))
        
        # Try to find the best rank-1 term by checking which candidate
        # best matches the residual structure
        
        # For each candidate, compute how well it fits the residual
        # A rank-1 term u⊗v⊗w contributes u[a]*v[b]*w[c] to T[a,b,c]
        # We want to maximize the inner product with the residual
        
        for u, v, w in candidates:
            # Compute the contribution of this rank-1 term
            contrib = np.einsum('a,b,c->abc', u, v, w)
            # Inner product with residual
            score = np.sum(residual * contrib)
            
            # Check if this term is "new" (not already in our decomposition)
            # by checking if it's linearly independent from existing terms
            if score > best_score:
                best_score = score
                best_u, best_v, best_w = u, v, w
        
        if best_u is None or best_score <= 0:
            break
        
        # Add this rank-1 term to our decomposition
        U_list.append(best_u)
        V_list.append(best_v)
        W_list.append(best_w)
        
        # Subtract from residual
        contrib = np.einsum('a,b,c->abc', best_u, best_v, best_w)
        residual -= contrib
    
    # If we didn't get enough terms, fall back to schoolbook
    if len(U_list) < target_rank:
        # Use schoolbook decomposition
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
    
    # Convert to lists of lists
    U = [list(map(float, u)) for u in np.array(U_list).T]
    V = [list(map(float, v)) for v in np.array(V_list).T]
    W = [list(map(float, w)) for w in np.array(W_list).T]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
