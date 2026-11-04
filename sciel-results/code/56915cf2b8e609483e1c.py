import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Precompute indices
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    
    # We search for rank R
    # Start from a lower bound, e.g., 30
    # If not found, increase R
    
    # Time limit guard
    start_time = time.time()
    time_limit = 100  # seconds
    
    # Try ranks from 25 to 72
    for R in range(25, rank_schoolbook + 1):
        if time.time() - start_time > time_limit:
            break
            
        # Try a few random initializations for this rank
        for attempt in range(50):
            if time.time() - start_time > time_limit:
                break
            
            # Initialize U, V randomly
            # Use small integers
            U = np.random.randint(-2, 3, size=(n*m, R))
            V = np.random.randint(-2, 3, size=(m*p, R))
            
            # Iterate: Solve for W given U, V
            # Then maybe update U, V? 
            # For now, just solve W and check error.
            # If error is 0, we are done.
            
            W = np.zeros((p*n, R))
            
            # For each w (which is k*n + i), solve for W[w, :]
            for w in range(p*n):
                # Extract k, i from w
                k = w // n
                i = w % n
                
                # Find all (u, v) such that the triple (u, v, w) is in the tensor domain
                # u = i*m + j
                # v = j*p + k
                # The constraint is on S_{u,v,w}
                # For a fixed w (i, k), the valid u are (i, j) for j in 0..m-1
                # The valid v are (j, k) for j in 0..m-1
                # So the pairs (u, v) are:
                # for j in 0..m-1:
                #   u = i*m + j
                #   v = j*p + k
                
                # Targets:
                # The tensor T has 1 at (u, v, w) if the indices match the multiplication.
                # Here, for fixed i, k, and varying j:
                # u = (i, j), v = (j, k), w = (i, k).
                # This is ALWAYS a valid multiplication term for the specific i, k.
                # So T_{u,v,w} = 1 for all j.
                # Wait, are there other (u', v') that map to the same w?
                # w is (i, k).
                # u' must be (i, j') and v' must be (j', k).
                # So yes, for each j', we have a constraint.
                # Are there constraints where T=0?
                # The tensor T is defined on the whole space U x V x W.
                # For a fixed w=(i,k), consider any u=(i', j') and v=(j'', k'').
                # If i' != i or k'' != k, then T=0.
                # Also if j' != j'', then T=0.
                # So for fixed w=(i,k):
                # Non-zero constraints:
                #   for j in 0..m-1: u=(i,j), v=(j,k). Target 1.
                # Zero constraints:
                #   All other (u, v) pairs.
                
                # Total pairs (u, v) is (n*m) * (m*p).
                # We only need to include the non-zero ones and a subset of zeros?
                # To ensure exactness, we must include ALL zero constraints?
                # That's 9*24 = 216 constraints per w.
                # 24 w's -> 5184 constraints.
                # This is small enough for least squares.
                
                # Construct A_w and b_w
                # Rows correspond to all (u, v) in U x V
                # But we only need to enforce the constraints that are "active" for this w?
                # No, the decomposition must be exact for ALL entries.
                # So we must enforce S_{u,v,w} = 0 for all (u,v) where T_{u,v,w}=0.
                
                # Let's build the system for this w.
                # Number of (u,v) pairs: n*m * m*p = 9 * 24 = 216.
                # We create a matrix A of shape (216, R).
                # And a vector b of shape (216,).
                
                A_w = np.zeros((n*m * m*p, R))
                b_w = np.zeros(n*m * m*p)
                
                row = 0
                for u in range(n*m):
                    for v in range(m*p):
                        # Calculate z_{u,v} = U[u, :] * V[v, :]
                        A_w[row, :] = U[u, :] * V[v, :]
                        
                        # Determine target
                        # u = i_u * m + j_u
                        # v = j_v * p + k_v
                        # w = k_w * n + i_w (fixed)
                        
                        i_u = u // m
                        j_u = u % m
                        j_v = v // p
                        k_v = v % p
                        k_w = w // n
                        i_w = w % n
                        
                        if i_u == i_w and k_v == k_w and j_u == j_v:
                            b_w[row] = 1
                        else:
                            b_w[row] = 0
                        row += 1
                
                # Solve A_w W[w, :]^T = b_w
                # Least squares
                W_w, residuals, rank_w, s = np.linalg.lstsq(A_w, b_w, rcond=None)
                W[w, :] = W_w
            
            # Check total error
            # Compute S_{u,v,w} for all u,v,w and compare to T
            # This is O(n*m*m*p*p*n*R) which is small.
            error = 0
            for u in range(n*m):
                for v in range(m*p):
                    for w in range(p*n):
                        s_val = np.dot(U[u, :] * V[v, :], W[w, :])
                        # Target
                        i_u = u // m
                        j_u = u % m
                        j_v = v // p
                        k_v = v % p
                        k_w = w // n
                        i_w = w % n
                        
                        if i_u == i_w and k_v == k_w and j_u == j_v:
                            target = 1
                        else:
                            target = 0
                        
                        error += (s_val - target) ** 2
            
            if error < 1e-9:
                # Found a solution
                # Convert to list of lists with integer/half-integer values
                # The values are from lstsq, so they are floats.
                # We need to snap them to nearest 0.5?
                # The problem allows integer/half-integer.
                # If the solution is exact, the floats should be very close to such values.
                
                def snap(x):
                    # Snap to nearest 0.5
                    return round(x * 2) / 2.0
                
                U_out = [[snap(val) for val in row] for row in U]
                V_out = [[snap(val) for val in row] for row in V]
                W_out = [[snap(val) for val in row] for row in W]
                
                # Verify with snapped values
                error_snapped = 0
                for u in range(n*m):
                    for v in range(m*p):
                        for w in range(p*n):
                            s_val = sum(U_out[u][r] * V_out[v][r] * W_out[w][r] for r in range(R))
                            i_u = u // m
                            j_u = u % m
                            j_v = v // p
                            k_v = v % p
                            k_w = w // n
                            i_w = w % n
                            
                            if i_u == i_w and k_v == k_w and j_u == j_v:
                                target = 1
                            else:
                                target = 0
                            
                            error_snapped += (s_val - target) ** 2
                
                if error_snapped < 1e-9:
                    return {"U": U_out, "V": V_out, "W": W_out}
    
    # If no better solution found, return schoolbook
    rank = rank_schoolbook
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
