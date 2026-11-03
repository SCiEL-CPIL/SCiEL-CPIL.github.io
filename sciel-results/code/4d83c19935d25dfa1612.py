"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 4, 5]
DEADLINE = 150.0

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook as fallback
    rank_schoolbook = n * m * p
    U_school = [[0]*rank_schoolbook for _ in range(nm)]
    V_school = [[0]*rank_schoolbook for _ in range(mp)]
    W_school = [[0]*rank_schoolbook for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions using iterative refinement
    # Start with various initial ranks below schoolbook
    for init_rank in [30, 28, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10]:
        if time.time() > DEADLINE - 15.0:
            break
        
        # Multiple random restarts for this rank
        for restart in range(3):
            if time.time() > DEADLINE - 15.0:
                break
            
            # Initialize with random values
            U = np.random.randn(nm, init_rank) * 0.1
            V = np.random.randn(mp, init_rank) * 0.1
            W = np.random.randn(pn, init_rank) * 0.1
            
            # Iterative refinement
            for iteration in range(200):
                if time.time() > DEADLINE - 15.0:
                    break
                
                # For each (i,j,k) entry, compute current contribution
                # and try to minimize error
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx_u = i*m + j
                            idx_v = j*p + k
                            idx_w = k*n + i
                            # Current value from decomposition
                            current = np.dot(U[idx_u], V[idx_v]) * W[idx_w]
                            # Target is 1 if this is a valid multiplication entry
                            # Actually, the tensor T[i,j,k; i',j',k'] = delta_{j,j'} * delta_{k,k'}
                            # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k'*n+i', c] = delta_{j,j'} * delta_{k,k'}
                            # For the diagonal: j=j', k=k', should be 1
                            # For off-diagonal: should be 0
                            target = 1.0  # This is for the (i,j,k) -> (i,j,k) case
                            # But we need to handle the full tensor structure
                            pass
                
                # Simplified approach: use ALS (Alternating Least Squares)
                # Fix V and W, solve for U
                for c in range(init_rank):
                    # Update U[:, c]
                    for idx_u in range(nm):
                        i = idx_u // m
                        j = idx_u % m
                        # Sum over k of V[j*p+k, c] * W[k*n+i, c]
                        denom = 0.0
                        numer = 0.0
                        for k in range(p):
                            v_val = V[j*p+k, c]
                            w_val = W[k*n+i, c]
                            denom += v_val * w_val
                            # Target: sum over all (i',j',k') where j'=j, k'=k
                            # This is complex, so use a simpler heuristic
                        U[idx_u, c] = 1.0 if denom > 0.5 else 0.0
                
                # Check convergence
                max_diff = 0.0
                for c in range(init_rank):
                    for idx in range(nm):
                        pass
                
                # Verify if this is a valid decomposition
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # Check if T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                            # The tensor should have 1s on the "diagonal" and 0s elsewhere
                            # For matrix multiplication: (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
                            # Tensor: T[i,j,k] where we multiply A (n x m) by B (m x p)
                            # The structure is: for each (i,j,k), the coefficient should be 1 if this 
                            # contributes to the product, which it always does with value 1
                            # Actually the tensor is: T_{i,j,k} = 1 for all i,j,k in the standard basis
                            # No, the matrix multiplication tensor is:
                            # T[i,j,k; i',j',k'] = delta_{j,j'} * delta_{k,k'}
                            # So summing over j',k': T[i,j,k] = 1 for all i,j,k
                            val = np.dot(U[i*m+j, :], V[j*p+k, :]) * W[k*n+i, :]
                            # This should equal 1 for the "diagonal" term
                            if abs(np.sum(val) - 1.0) > 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid:
                    # Convert to lists
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    # Round to nearest half-integer if close
                    for row in U_list:
                        for idx in range(len(row)):
                            row[idx] = round(row[idx] * 2) / 2
                    for row in V_list:
                        for idx in range(len(row)):
                            row[idx] = round(row[idx] * 2) / 2
                    for row in W_list:
                        for idx in range(len(row)):
                            row[idx] = round(row[idx] * 2) / 2
                    if init_rank < best_rank:
                        best_rank = init_rank
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                    break  # Found a valid decomposition for this rank, try lower
            
            if time.time() > DEADLINE - 10.0:
                break
    
    return best_result
# EVOLVE-BLOCK-END
