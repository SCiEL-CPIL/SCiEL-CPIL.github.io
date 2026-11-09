import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try to find a lower rank decomposition
    # Search space is small enough for random sampling with small integers
    # We look for rank r < best_rank
    for r in range(30, best_rank):
        if time.time() > deadline:
            break
            
        # Try a few random initializations
        for attempt in range(100):
            if time.time() > deadline:
                break
                
            # Initialize U, V, W with small random integers
            # Using -1, 0, 1
            U = np.zeros((n * m, r), dtype=int)
            V = np.zeros((m * p, r), dtype=int)
            W = np.zeros((p * n, r), dtype=int)
            
            # Randomly populate with 1s and -1s
            for i in range(n * m):
                for c in range(r):
                    if np.random.random() < 0.5:
                        U[i, c] = 1
                    else:
                        U[i, c] = 0
            for j in range(m * p):
                for c in range(r):
                    if np.random.random() < 0.5:
                        V[j, c] = 1
                    else:
                        V[j, c] = 0
            for k in range(p * n):
                for c in range(r):
                    if np.random.random() < 0.5:
                        W[k, c] = 1
                    else:
                        W[k, c] = 0
            
            # Check if this decomposition is exact
            # The condition is: sum_c U[i*m+j, c] * V[j*p+k, c] == 1 if k*n+i is the correct output?
            # No, the tensor T_{i,j,k} = 1.
            # The decomposition is: T_{i,j,k} = sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}?
            # Wait, the standard definition:
            # C_{i,k} = sum_j A_{i,j} B_{j,k}
            # Decomposition: C_{i,k} = sum_c (sum_j U_{i,j,c} A_{i,j}) (sum_k V_{j,k,c} B_{j,k}) ? No.
            # The tensor T is defined by T_{i,j,k} = 1.
            # A decomposition of rank r is a set of r tensors T^{(c)} such that T = sum_c T^{(c)}.
            # Each T^{(c)} is a rank-1 tensor, which means it can be written as u^{(c)} \otimes v^{(c)} \otimes w^{(c)}?
            # No, for matrix multiplication, the rank-1 tensor corresponds to a single multiplication A_{i,j} B_{j,k}.
            # So T^{(c)}_{i,j,k} = U_{i,j,c} V_{j,k,c} W_{k,i,c}?
            # The convention in the prompt: W[k*n+i, r].
            # So T_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
            # We need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k.
            
            # This is a constraint satisfaction problem.
            # Random search with binary entries is unlikely to find a solution for r < 40 easily.
            # However, if a solution exists with small integers, it might be found.
            
            # Efficient check:
            # For each c, compute the term.
            # We can vectorize this check.
            # T_reconstructed[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            
            # Reshape for numpy
            U_mat = U # (nm, r)
            V_mat = V # (mp, r)
            W_mat = W # (pn, r)
            
            # We need to check if sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] == 1
            # This is equivalent to checking the tensor product.
            
            # Let's just do a simple loop for the check, it's small (2*4*5=40 points)
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        idx_u = i * m + j
                        idx_v = j * p + k
                        idx_w = k * n + i
                        for c in range(r):
                            s += U_mat[idx_u, c] * V_mat[idx_v, c] * W_mat[idx_w, c]
                        if s != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # If valid, we found a decomposition of rank r
            best_rank = r
            best_U = U_mat.tolist()
            best_V = V_mat.tolist()
            best_W = W_mat.tolist()
            break # Found one for this r, try smaller r? Or is r the min?
            # We are iterating r from 30 upwards. So first found is the smallest in this range.
            # But we should break the r loop too?
            # The loop is "for r in range(30, best_rank)".
            # If we find one, we update best_rank and break inner loop.
            # The outer loop continues to next r? No, we want the minimum.
            # Since we go 30, 31, ..., if we find 35, we should check 30-34?
            # My loop goes 30 to 39. So if I find 35, I should stop?
            # No, I should check if there is a smaller one?
            # I am iterating r in increasing order. So if I find one at r=35,
            # it means 30-34 were not found (or I skipped them?).
            # I should iterate r in increasing order.
            # If I find a solution at r, it is the minimum found so far.
            # I should break the outer loop?
            # Yes, because I am looking for the minimum.
            break

    # Convert to lists if needed
    return {"U": best_U, "V": best_V, "W": best_W}
