import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Create dense tensor T
    # Shape: (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
                
    # Matching Pursuit
    R = T.copy()
    U_list = []
    V_list = []
    W_list = []
    
    max_terms = 105
    found_solution = None
    
    for t in range(max_terms):
        if time.time() > deadline:
            break
            
        if np.linalg.norm(R) < 1e-9:
            break
            
        # Find best rank-1 approximation of R
        # Use power iteration
        u = np.random.randn(n*m)
        v = np.random.randn(m*p)
        w = np.random.randn(p*n)
        
        for _ in range(50):
            # u = R * (v, w)
            # R is 3D.
            # u[i] = sum_{j,k} R[i,j,k] v[j] w[k]
            u_new = np.tensordot(R, (v, w), axes=([1, 2], [0, 0]))
            # v = R * (u, w)
            v_new = np.tensordot(R, (u, w), axes=([0, 2], [0, 0]))
            # w = R * (u, v)
            w_new = np.tensordot(R, (u, v), axes=([0, 1], [0, 0]))
            
            u = u_new / (np.linalg.norm(u_new) + 1e-12)
            v = v_new / (np.linalg.norm(v_new) + 1e-12)
            w = w_new / (np.linalg.norm(w_new) + 1e-12)
            
        # Add to lists
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)
        
        # Update R
        R -= np.outer(np.outer(u, v), w)
        
        # Check if we have a solution
        if np.linalg.norm(R) < 1e-9:
            # We have a decomposition
            # Check if we can make it integer/half-integer
            U_mat = np.array(U_list).T
            V_mat = np.array(V_list).T
            W_mat = np.array(W_list).T
            
            # Try to scale and round
            # The values are likely not integers.
            # We can try to find a scaling factor.
            # Or just return the real values if allowed?
            # "integer/half-integer entries"
            
            # Let's try to snap to half-integers
            U_round = np.round(U_mat * 2) / 2
            V_round = np.round(V_mat * 2) / 2
            W_round = np.round(W_mat * 2) / 2
            
            # Check error
            T_check = np.tensordot(U_round, V_round, axes=([1], [1]))
            # This is not right. tensordot with two matrices gives a 2D matrix?
            # We need the full tensor product.
            # T_check[i,j,k] = sum_c U[i,c] V[j,c] W[k,c]
            
            # We can compute the error directly
            error = 0
            for i in range(n*m):
                for j in range(m*p):
                    for k in range(p*n):
                        val = np.dot(U_round[i], V_round[j]) # No, this is not right.
                        # val = sum_c U[i,c] V[j,c] W[k,c]
                        # This is not a simple dot product.
                        
            # Let's use the tensor product
            T_check = np.einsum('ic,jc,kc->ijk', U_round, V_round, W_round)
            err = np.linalg.norm(T_check - T)
            
            if err < 1e-9:
                found_solution = (U_round, V_round, W_round)
                break
            else:
                # If rounding fails, we might need to search for a better solution
                # or accept a higher rank
                pass
                
    if found_solution is None:
        # Return schoolbook
        rank_school = n * m * p
        U_school = np.zeros((n*m, rank_school), dtype=float)
        V_school = np.zeros((m*p, rank_school), dtype=float)
        W_school = np.zeros((p*n, rank_school), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m+j, col] = 1.0
                    V_school[j*p+k, col] = 1.0
                    W_school[k*n+i, col] = 1.0
                    col += 1
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }
    else:
        return {
            "U": found_solution[0].tolist(),
            "V": found_solution[1].tolist(),
            "W": found_solution[2].tolist()
        }
