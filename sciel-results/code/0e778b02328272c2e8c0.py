import time
import numpy as np
from scipy.linalg import solve

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook rank
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Try to find a lower rank decomposition
    # We parameterize U and V with random entries from {-1, 0, 1}
    # and solve for W. If W has entries close to integers/half-integers, we snap them.
    # We verify the decomposition exactly.
    
    # Since n=2 is small, maybe we can exploit structure?
    # But for now, let's try random search for rank < 42.
    
    # We will try ranks from 41 down to 35.
    # For each rank, we do a few random trials.
    
    def verify_decomposition(U, V, W):
        # Check if U, V, W exactly decompose the tensor
        # Tensor T[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
        # We need T[i, j, k] = 1 if k corresponds to the multiplication, else 0.
        # Wait, the tensor is for matrix multiplication C = A * B.
        # C[i, k] = sum_j A[i, j] B[j, k].
        # The tensor T has indices (i, j, k) where i is row of A, j is col of A/row of B, k is col of B.
        # T[i, j, k] = 1 for all i, j, k? No.
        # The standard tensor for matrix multiplication has T[i, j, k] = 1 if the term A[i, j]B[j, k] contributes to C[i, k].
        # Yes, T[i, j, k] = 1 for all valid i, j, k.
        
        # Let's check the condition:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i, j, k.
        
        # We can compute the full tensor and compare.
        # Shape of T is (n, m, p).
        # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
        
        # Flatten indices:
        # U_idx = i*m + j
        # V_idx = j*p + k
        # W_idx = k*n + i
        
        # We want to check if sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1
        
        # Efficient check:
        # For each i, k, we want sum_j (sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]) == 1?
        # No, the tensor entry is 1 for each (i,j,k).
        # So for each i, j, k, the sum must be 1.
        
        # Let's just compute it for all i, j, k.
        # n=2, m=3, p=7. 42 entries.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def snap_to_half_integer(arr, tol=1e-6):
        # Snap entries to nearest half-integer if close
        snapped = np.round(arr * 2) / 2
        if np.allclose(arr, snapped, atol=tol):
            return snapped
        return None

    # Search for lower ranks
    # We'll try random U, V with entries in {-1, 0, 1}
    # and solve for W.
    # W[k*n+i, r] is determined by:
    # sum_j sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
    # This is not a linear system in W directly because W is multiplied by V and U.
    # Wait, for fixed i, k, we have:
    # sum_j (sum_r U[i*m+j, r] V[j*p+k, r]) W[k*n+i, r] = 1
    # This is a linear equation in W[k*n+i, :].
    # Let M_ik be a matrix where M_ik[j, r] = U[i*m+j, r] V[j*p+k, r].
    # Then M_ik @ W[k*n+i, :] = 1 (vector of ones of length m? No, the sum is over j).
    # Actually, the condition is:
    # For each i, k: sum_j (M_ik[j, :] * W[k*n+i, :]) = 1?
    # No.
    # T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1.
    # This must hold for ALL j.
    # So for fixed i, k, we have m equations (one for each j):
    # sum_r (U[i*m+j, r] V[j*p+k, r]) W[k*n+i, r] = 1 for j=0..m-1.
    
    # So for each i, k, we can set up a linear system:
    # A_ik @ w_ik = b
    # where A_ik is m x rank, A_ik[j, r] = U[i*m+j, r] V[j*p+k, r]
    # b is a vector of ones of length m.
    # w_ik is W[k*n+i, :].
    
    # If the system is consistent, we have a candidate W.
    
    for rank in range(rank_sb - 1, max(30, rank_sb - 10), -1):
        if time.time() > deadline:
            break
            
        max_trials = 50
        for trial in range(max_trials):
            if time.time() > deadline:
                break
                
            # Generate random U, V
            U = np.random.randint(-1, 2, size=(nm, rank))
            V = np.random.randint(-1, 2, size=(mp, rank))
            
            W = np.zeros((pn, rank))
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Build A_ik
                    A_ik = np.zeros((m, rank))
                    for j in range(m):
                        A_ik[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    
                    # Solve A_ik @ w = ones
                    # Use least squares or exact solve if square/invertible
                    # Since m=3 and rank can be > 3, it's underdetermined if rank > 3.
                    # We want a solution.
                    # If rank < m, it might be overdetermined.
                    # Let's use np.linalg.lstsq to find the minimum norm solution, then snap.
                    # But we need EXACT solution.
                    # If the system is consistent, lstsq will find it.
                    
                    try:
                        w, residuals, rank_A, s = np.linalg.lstsq(A_ik, np.ones(m), rcond=None)
                        # Check if it's a solution
                        if not np.allclose(A_ik @ w, np.ones(m), atol=1e-9):
                            valid = False
                            break
                        W[k * n + i, :] = w
                    except:
                        valid = False
                        break
            
            if not valid:
                continue
                
            # Snap W to half-integers
            W_snapped = snap_to_half_integer(W)
            if W_snapped is None:
                continue
                
            # Verify the whole decomposition with snapped W
            if verify_decomposition(U, V, W_snapped):
                best_U, best_V, best_W = U, V, W_snapped
                best_rank = rank
                break # Found a valid one for this rank, move to lower rank
                
    # Convert to lists of lists for return
    U_ret = best_U.tolist()
    V_ret = best_V.tolist()
    W_ret = best_W.tolist()
    
    return {"U": U_ret, "V": V_ret, "W": W_ret}
