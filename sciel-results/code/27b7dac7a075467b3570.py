"""Program for rank-minimal tensor decomposition of 2x3x7 matrix multiplication."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2x3 by 3x7 matrix.
    
    The tensor T[i,j,k] has T[i*m+j, j*p+k, k*n+i] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    We seek U (n*m x r), V (m*p x r), W (p*n x r) such that
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    
    The schoolbook decomposition uses rank = n*m*p = 42.
    We attempt to find a lower-rank decomposition through structured search.
    """
    import time
    import random
    import numpy as np
    
    start_time = time.time()
    deadline = 140  # seconds, leaving margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the target tensor as a numpy array
    # T[row, col, kidx] = 1 if the triple corresponds to a valid (i,j,k), else 0
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                row = i * m + j
                col = j * p + k
                kidx = k * n + i
                T[row, col, kidx] = 1.0
    
    # Schoolbook baseline
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W, tol=1e-9):
        """Verify that U, V, W exactly decompose the tensor."""
        r = len(U[0])
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # Compute the tensor decomposition: sum_c U[row,c] * V[col,c] * W[kidx,c]
        # This is equivalent to: for each c, outer product U[:,c] x V[:,c] x W[:,c]
        # We can compute this efficiently using einsum
        result = np.einsum('rc,sc,tc->rst', U_arr, V_arr, W_arr)
        
        # Check against target
        if np.max(np.abs(result - T)) > tol:
            return False
        return True
    
    def try_rank_structured(r, max_iters=2000):
        """
        Try to find a rank-r decomposition using structured random search
        with local refinement.
        """
        # Candidate entry values: 0, ±0.5, ±1, ±1.5, ±2
        vals = [0, 0.5, 1.0, -0.5, -1.0, 1.5, -1.5, 2.0, -2.0]
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Randomly initialize U, V, W with sparse structure
            U = [[random.choice(vals) for _ in range(r)] for _ in range(nm)]
            V = [[random.choice(vals) for _ in range(r)] for _ in range(mp)]
            W = [[random.choice(vals) for _ in range(r)] for _ in range(pn)]
            
            # Local refinement: iterate Gauss-Seidel style
            # For each rank term c, try to optimize it
            improved = True
            max_inner = 50
            inner_count = 0
            while improved and inner_count < max_inner:
                improved = False
                inner_count += 1
                
                # Try to improve each rank term
                for c in range(r):
                    # Compute current contribution of term c
                    # For each (row, col, kidx), the contribution is U[row,c]*V[col,c]*W[kidx,c]
                    
                    # Try perturbing U[:,c]
                    for row in range(nm):
                        old_val = U[row][c]
                        if old_val == 0:
                            continue
                        
                        # Compute the residual if we remove this term's contribution
                        # and see if we can improve by changing U[row][c]
                        # This is complex, so let's just try random perturbations
                        
                        # Try setting to 0
                        U[row][c] = 0
                        if verify(U, V, W, tol=1e-9):
                            return {"U": U, "V": V, "W": W}
                        
                        # Try other values
                        for v in vals:
                            if v != old_val:
                                U[row][c] = v
                                if verify(U, V, W, tol=1e-9):
                                    return {"U": U, "V": V, "W": W}
                        
                        U[row][c] = old_val
                    
                    # Similar for V and W would be expensive, so let's just do a few
                    if it % 3 == 0:
                        for col in range(mp):
                            old_val = V[col][c]
                            if old_val == 0:
                                continue
                            
                            V[col][c] = 0
                            if verify(U, V, W, tol=1e-9):
                                return {"U": U, "V": V, "W": W}
                            
                            for v in vals:
                                if v != old_val:
                                    V[col][c] = v
                                    if verify(U, V, W, tol=1e-9):
                                        return {"U": U, "V": V, "W": W}
                            
                            V[col][c] = old_val
                    elif it % 3 == 1:
                        for kidx in range(pn):
                            old_val = W[kidx][c]
                            if old_val == 0:
                                continue
                            
                            W[kidx][c] = 0
                            if verify(U, V, W, tol=1e-9):
                                return {"U": U, "V": V, "W": W}
                            
                            for v in vals:
                                if v != old_val:
                                    W[kidx][c] = v
                                    if verify(U, V, W, tol=1e-9):
                                        return {"U": U, "V": V, "W": W}
                            
                            W[kidx][c] = old_val
            
            # Full verification
            if verify(U, V, W, tol=1e-9):
                return {"U": U, "V": V, "W": W}
        
        return None
    
    # Try a structured approach: start from schoolbook and try to merge terms
    def try_merge_schoolbook():
        """
        Start from the schoolbook decomposition and try to merge/eliminate terms
        to reduce the rank.
        """
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        
        # Try to eliminate terms by finding linear dependencies
        # For each pair of terms, check if one can be expressed as a combination
        # of others (this is complex, so we'll use a simpler heuristic)
        
        # Try to find if any term is redundant by checking if removing it
        # still gives a valid decomposition (with other terms adjusted)
        # This is a hard problem, so let's just try random perturbations
        
        best_rank = rank
        best_result = {"U": U, "V": V, "W": W}
        
        # Try to reduce rank by checking if we can set some entries to 0
        # and still have a valid decomposition
        for c in range(rank - 1, -1, -1):
            if time.time() > deadline:
                break
            
            # Try setting term c to 0
            U_save = [row[:] for row in U]
            V_save = [row[:] for row in V]
            W_save = [row[:] for row in W]
            
            U = [row[:] for row in U]
            V = [row[:] for row in V]
            W = [row[:] for row in W]
            
            # Set term c to 0
            for row in range(nm):
                U[row][c] = 0
            for col in range(mp):
                V[col][c] = 0
            for kidx in range(pn):
                W[kidx][c] = 0
            
            # Check if we can still satisfy the tensor equation
            # by adjusting other terms
            # This is a linear system problem, which is complex
            # For now, just check if the decomposition is still valid
            if verify(U, V, W, tol=1e-9):
                best_rank = c
                best_result = {"U": U, "V": V, "W": W}
                # Remove the zero column
                U = [row[:c] for row in U]
                V = [row[:c] for row in V]
                W = [row[:c] for row in W]
            else:
                # Restore
                U = U_save
                V = V_save
                W = W_save
        
        if best_rank < rank:
            return best_result
        return None
    
    # Try various ranks from lower to higher
    # Start with structured search for ranks below 42
    for r in range(25, 42):
        if time.time() > deadline:
            break
        result = try_rank_structured(r, max_iters=1000)
        if result is not None:
            return result
    
    # Try the merge approach
    if time.time() < deadline - 10:
        result = try_merge_schoolbook()
        if result is not None:
            return result
    
    # If no better solution found, return schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
