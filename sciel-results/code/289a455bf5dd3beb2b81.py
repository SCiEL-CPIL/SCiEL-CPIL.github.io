import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Precompute the target tensor T
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    T_flat = T.flatten()

    start_time = time.time()
    deadline = start_time + 145  # Leave 5s margin for 150s limit

    # Try ranks from 17 down to 14
    # Schoolbook is 18. We hope to beat it.
    for rank in [17, 16, 15, 14]:
        if time.time() > deadline - 10:
            break
            
        best_err = float('inf')
        best_sol = None
        
        # Number of restarts allowed based on time
        # Each restart/local search might take ~1-2s depending on convergence
        max_restarts = 20
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            # Use a grid of values. Sparse is often a good heuristic.
            # Values: -1, -0.5, 0, 0.5, 1
            vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
            probs = np.array([0.1, 0.2, 0.5, 0.2, 0.1])
            
            U = np.random.choice(vals, size=(nm, rank), p=probs)
            V = np.random.choice(vals, size=(mp, rank), p=probs)
            W = np.random.choice(vals, size=(pn, rank), p=probs)
            
            # Local Search (Coordinate Descent / Hill Climbing)
            # We will try to reduce the error by tweaking individual elements.
            no_improve_count = 0
            max_no_improve = 500
            
            current_err = np.sum((np.einsum('as,bs,cs->abc', U, V, W) - T)**2)
            
            while no_improve_count < max_no_improve and time.time() < deadline - 5:
                # Pick a random matrix and entry
                mat_idx = np.random.randint(3)
                if mat_idx == 0:
                    r_idx = np.random.randint(nm)
                    c_idx = np.random.randint(rank)
                    mat = U
                elif mat_idx == 1:
                    r_idx = np.random.randint(mp)
                    c_idx = np.random.randint(rank)
                    mat = V
                else:
                    r_idx = np.random.randint(pn)
                    c_idx = np.random.randint(rank)
                    mat = W
                
                old_val = mat[r_idx, c_idx]
                
                # Try changing to a nearby value on the grid
                # Neighbors: -0.5, 0, +0.5 shift
                neighbors = np.array([-0.5, 0.0, 0.5])
                new_vals = old_val + neighbors
                # Filter to reasonable range [-2, 2] and valid grid points
                valid_new_vals = []
                for nv in new_vals:
                    # Check if it's close to a half-integer
                    rounded = round(nv * 2) / 2
                    if abs(nv - rounded) < 1e-9 and -2 <= rounded <= 2:
                        if rounded != old_val:
                            valid_new_vals.append(rounded)
                
                if not valid_new_vals:
                    no_improve_count += 1
                    continue
                    
                # Try the best neighbor
                best_neighbor_val = old_val
                best_neighbor_err = current_err
                
                for nv in valid_new_vals:
                    mat[r_idx, c_idx] = nv
                    err = np.sum((np.einsum('as,bs,cs->abc', U, V, W) - T)**2)
                    if err < best_neighbor_err:
                        best_neighbor_err = err
                        best_neighbor_val = nv
                    mat[r_idx, c_idx] = old_val # restore for next test
                
                # Apply best neighbor if it improved
                if best_neighbor_val != old_val:
                    mat[r_idx, c_idx] = best_neighbor_val
                    current_err = best_neighbor_err
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    
            if current_err < best_err:
                best_err = current_err
                best_sol = (U.copy(), V.copy(), W.copy())
                
            if best_err < 1e-9:
                return {"U": best_sol[0].tolist(), "V": best_sol[1].tolist(), "W": best_sol[2].tolist()}

    # Fallback: Schoolbook decomposition of rank n*m*p
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
