import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100.0 # 20s margin
    
    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_school, V_school, W_school

    # Try to find lower rank solutions
    # Known lower bound for 2x3x4 is 19.
    # We will try ranks from 19 up to 23.
    
    def verify(U, V, W):
        # Check if U, V, W are integers or half-integers
        # And if they satisfy the condition
        # Condition: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1 for all i,j,k
        # U, V, W shapes: (n*m, R), (m*p, R), (p*n, R)
        
        # Check integrality
        for mat in [U, V, W]:
            if not np.allclose(mat * 2, np.round(mat * 2)):
                return False
            # Check range? Not strictly required, but good for stability
            if np.any(np.abs(mat) > 100):
                return False

        # Check decomposition
        # Construct T_target of shape (n, m, p) with all 1s
        # Construct T_calc
        # T_calc[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Efficient check using tensor contraction
        # Reshape U to (n, m, R)
        U_reshaped = U.reshape(n, m, -1)
        V_reshaped = V.reshape(m, p, -1)
        W_reshaped = W.reshape(p, n, -1)
        
        # We want sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # This is not a standard dot product because indices cycle i->j->k->i.
        # We can iterate over r and accumulate?
        # Or use einsum?
        # np.einsum('ijr,jkr,kr i->ijk', U_reshaped, V_reshaped, W_reshaped) 
        # Note: W index is k*n+i, so W_reshaped is (p, n, R) -> indices (k, i, r)
        
        try:
            T_calc = np.einsum('ijr,jkr,kr i->ijk', U_reshaped, V_reshaped, W_reshaped)
            # Wait, W_reshaped is (p, n, R). Index 0 is k, index 1 is i, index 2 is r.
            # So 'kri' is correct for W?
            # Let's check shapes:
            # U_reshaped: (n, m, R) -> i, j, r
            # V_reshaped: (m, p, R) -> j, k, r
            # W_reshaped: (p, n, R) -> k, i, r
            
            # einsum subscripts: 'ijr,jkr,kr i->ijk'
            # i from U, j from U/V, k from V/W, i from W.
            # This sums over r.
            # Result shape: (n, m, p) i, j, k.
            
            if not np.allclose(T_calc, 1.0, atol=1e-6):
                return False
        except Exception:
            return False
            
        return True

    def round_to_half(x):
        return np.round(x * 2) / 2.0

    def search_rank(R):
        # Random restarts
        num_restarts = 50
        for restart in range(num_restarts):
            if time.time() > deadline:
                return None
            
            # Initialize with small random integers
            # Scale matters. Let's try integers in {-1, 0, 1} or {-2, -1, 0, 1, 2}
            U = np.random.randint(-2, 3, size=(n*m, R))
            V = np.random.randint(-2, 3, size=(m*p, R))
            W = np.random.randint(-2, 3, size=(p*n, R))
            
            # Simple local search: perturb one element at a time
            # Loss function: sum of squared errors
            def loss(U, V, W):
                U_r = U.reshape(n, m, R)
                V_r = V.reshape(m, p, R)
                W_r = W.reshape(p, n, R)
                try:
                    T_calc = np.einsum('ijr,jkr,kr i->ijk', U_r, V_r, W_r)
                    return np.sum((T_calc - 1.0)**2)
                except:
                    return float('inf')
            
            current_loss = loss(U, V, W)
            
            # If already close, verify
            if current_loss < 1e-6:
                U_int = round_to_half(U)
                V_int = round_to_half(V)
                W_int = round_to_half(W)
                if verify(U_int, V_int, W_int):
                    return U_int, V_int, W_int
            
            # Local search
            for step in range(200):
                if time.time() > deadline:
                    return None
                
                # Pick random matrix and element
                mat_idx = np.random.randint(3)
                if mat_idx == 0:
                    idx = np.random.randint(n*m)
                    r = np.random.randint(R)
                    delta = np.random.choice([-1, 1])
                    U[idx, r] += delta
                elif mat_idx == 1:
                    idx = np.random.randint(m*p)
                    r = np.random.randint(R)
                    delta = np.random.choice([-1, 1])
                    V[idx, r] += delta
                else:
                    idx = np.random.randint(p*n)
                    r = np.random.randint(R)
                    delta = np.random.choice([-1, 1])
                    W[idx, r] += delta
                
                new_loss = loss(U, V, W)
                
                if new_loss < current_loss:
                    current_loss = new_loss
                    if current_loss < 1e-6:
                        U_int = round_to_half(U)
                        V_int = round_to_half(V)
                        W_int = round_to_half(W)
                        if verify(U_int, V_int, W_int):
                            return U_int, V_int, W_int
                else:
                    # Revert
                    if mat_idx == 0:
                        U[idx, r] -= delta
                    elif mat_idx == 1:
                        V[idx, r] -= delta
                    else:
                        W[idx, r] -= delta
            
            # If we got stuck, maybe try a different initialization
            # But local search on integers is a discrete problem.
            # Gradient descent on reals might be better, then round.
            
            # Try gradient descent
            U_f = U.astype(float)
            V_f = V.astype(float)
            W_f = W.astype(float)
            
            lr = 0.01
            for step in range(100):
                if time.time() > deadline:
                    return None
                
                # Compute gradients
                # L = sum (1 - sum_r U_ijk V_jkr W_kri)^2
                # This is computationally heavy to do with automatic differentiation in pure Python/NumPy without JAX/Torch.
                # We can use numerical gradients or just stick to discrete search.
                # Discrete search is often better for small integer solutions.
                pass
            
            # If discrete search failed, try random restart with different scale
            pass
            
        return None

    # Try ranks 19 to 23
    for R in range(19, 24):
        if time.time() > deadline:
            break
        result = search_rank(R)
        if result:
            U, V, W = result
            best_rank = R
            best_U, best_V, best_W = U, V, W
            break # Found a solution, stop searching for higher ranks? 
            # Actually, we want minimal rank, so if we find 19, we are done.
            # If we find 20, we should continue to see if 19 is possible? 
            # The loop goes 19, 20, 21... so if 19 fails, it tries 20.
            # If 20 succeeds, it breaks and returns 20.
            # This is correct because we iterate from low to high.

    # Convert to lists for the output interface
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
