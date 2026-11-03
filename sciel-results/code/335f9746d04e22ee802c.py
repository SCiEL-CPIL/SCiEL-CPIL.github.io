import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def _verify_exact(U, V, W, n, m, p):
    """
    Verify that the decomposition exactly computes matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Condition: sum_r U[i*m+j', r] * V[j'*p+k', r] * W[k*n+i, r] = delta_{j',j} * delta_{k',k}
    for all i, j, k, j', k'.
    """
    rank = len(U[0]) if U else 0
    # Build the tensor T[i, j, k, j', k'] = sum_r U[i*m+j', r] * V[j'*p+k', r] * W[k*n+i, r]
    # We need T[i, j, k, j', k'] = 1 if j==j' and k==k', else 0.
    # Actually, the condition is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Using decomposition: C[i,k] = sum_r W[k*n+i, r] * (sum_{j'} U[i*m+j', r] A[i,j']) * (sum_{k'} V[j'*p+k', r] B[j',k'])
    # This is not quite right. Let's re-derive.
    # Standard decomposition:
    # M_r = (sum_{i,j} U_{i,j}^{(r)} A_{i,j}) * (sum_{j,k} V_{j,k}^{(r)} B_{j,k})
    # C_{i,k} = sum_r W_{i,k}^{(r)} M_r
    # So C_{i,k} = sum_r W_{i,k}^{(r)} (sum_{i',j'} U_{i',j'}^{(r)} A_{i',j'}) (sum_{j',k'} V_{j',k'}^{(r)} B_{j',k'})
    # = sum_{i',j',j',k'} A_{i',j'} B_{j',k'} (sum_r W_{i,k}^{(r)} U_{i',j'}^{(r)} V_{j',k'}^{(r)})
    # We need C_{i,k} = sum_j A_{i,j} B_{j,k}
    # So we need: sum_r W_{i,k}^{(r)} U_{i',j'}^{(r)} V_{j',k'}^{(r)} = delta_{i,i'} delta_{j,j'} delta_{k,k'}? No.
    # The coefficient of A_{i',j'} B_{j',k'} in C_{i,k} must be 1 if i'=i, j'=j, k'=k, else 0.
    # But j' is the index of B. In C_{i,k} = sum_j A_{i,j} B_{j,k}, the term A_{i',j'} B_{j',k'} appears only if i'=i, j'=j, k'=k.
    # So the condition is:
    # sum_r W_{k*n+i, r} U_{i'*m+j', r} V_{j'*p+k', r} = delta_{i,i'} delta_{k,k'} delta_{j',j'}? No, j' is free in the sum.
    # Wait, the middle index of U is j' (row of A, col of A). The middle index of V is j' (row of B, col of B).
    # In the product M_r, the sum over j' in U and the sum over j' in V are independent? No.
    # M_r = (sum_{i,j} U_{i,j}^{(r)} A_{i,j}) * (sum_{j,k} V_{j,k}^{(r)} B_{j,k})
    # The index j in U and the index j in V are dummy indices, but they are summed independently.
    # So M_r = sum_{i,j,j',k} U_{i,j}^{(r)} V_{j',k}^{(r)} A_{i,j} B_{j',k}
    # Then C_{i,k} = sum_r W_{i,k}^{(r)} M_r = sum_{j,j'} A_{i,j} B_{j',k} (sum_r W_{i,k}^{(r)} U_{i,j}^{(r)} V_{j',k}^{(r)})
    # We need this to equal sum_j A_{i,j} B_{j,k}.
    # So we need: sum_r W_{i,k}^{(r)} U_{i,j}^{(r)} V_{j',k}^{(r)} = delta_{j,j'} for all i,k,j,j'.
    # Note: The condition does NOT depend on i and k except through W.
    # So for fixed i,k, the matrix S_{j,j'} = sum_r W_{i,k}^{(r)} U_{i,j}^{(r)} V_{j',k}^{(r)} must be the identity matrix.
    # This must hold for ALL i,k.
    # So the verification is:
    # For each i in [0,n), k in [0,p):
    #   S = sum_r W[k*n+i, r] * outer(U[i*m:(i+1)*m, r], V[:, k])  # V[:, k] is not right, V is (m*p, rank)
    #   V is indexed by j'*p+k'. So V[j'*p+k', r].
    #   For fixed k, V[:, k] would be V[0*p+k, r], V[1*p+k, r], ..., V[(p-1)*p+k, r]? No, V has m*p rows.
    #   Row index of V is j'*p + k'? No, V is (m*p, rank). The index is j'*p + k' where j' in [0,m), k' in [0,p).
    #   So for fixed k', the rows are j'*p + k' for j' in [0,m).
    #   So we need: S_{j,j'} = sum_r W[k*n+i, r] U[i*m+j, r] V[j'*p+k, r] = delta_{j,j'}
    #   for all j in [0,m), j' in [0,m).
    
    for i in range(n):
        for k in range(p):
            S = np.zeros((m, m))
            for r in range(rank):
                w_val = W[k*n+i][r]
                if w_val == 0:
                    continue
                for j in range(m):
                    u_val = U[i*m+j][r]
                    if u_val == 0:
                        continue
                    for jp in range(m):
                        v_val = V[jp*p+k][r]
                        S[j, jp] += w_val * u_val * v_val
            # Check S == I
            for j in range(m):
                for jp in range(m):
                    expected = 1 if j == jp else 0
                    if abs(S[j, jp] - expected) > 1e-9:
                        return False
    return True

def _compute_error(U, V, W, n, m, p):
    """Compute the max absolute error of the decomposition."""
    rank = len(U[0]) if U else 0
    max_err = 0.0
    for i in range(n):
        for k in range(p):
            S = np.zeros((m, m))
            for r in range(rank):
                w_val = W[k*n+i][r]
                if w_val == 0:
                    continue
                for j in range(m):
                    u_val = U[i*m+j][r]
                    if u_val == 0:
                        continue
                    for jp in range(m):
                        v_val = V[jp*p+k][r]
                        S[j, jp] += w_val * u_val * v_val
            for j in range(m):
                for jp in range(m):
                    expected = 1 if j == jp else 0
                    err = abs(S[j, jp] - expected)
                    if err > max_err:
                        max_err = err
    return max_err

def _snap_to_half_int(arr, tol=1e-6):
    """Snap values close to half-integers to exact half-integers."""
    arr = np.array(arr, dtype=float)
    snapped = np.round(arr * 2) / 2
    # Only snap if close
    mask = np.abs(arr - snapped) < tol
    result = np.where(mask, snapped, arr)
    return result

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Schoolbook solution as baseline
    rank_school = n * m * p  # 36
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    
    # Try to find lower rank decompositions
    # Known upper bound for R(3,3,4) is likely around 27-30
    # Let's try ranks from 25 up to 35
    
    random.seed(42)
    
    for target_rank in range(25, min(36, rank_school)):
        if time.time() > deadline:
            break
        
        found = False
        # Try multiple random initializations
        for init in range(20):
            if time.time() > deadline:
                break
            
            # Random half-integer initialization
            Ur = np.random.randint(-2, 3, size=(n*m, target_rank)) * 0.5
            Vr = np.random.randint(-2, 3, size=(m*p, target_rank)) * 0.5
            Wr = np.random.randint(-2, 3, size=(p*n, target_rank)) * 0.5
            
            # Local search: perturb one entry at a time
            for it in range(200):
                if time.time() > deadline:
                    break
                
                err = _compute_error(Ur.tolist(), Vr.tolist(), Wr.tolist(), n, m, p)
                
                if err < 1e-8:
                    if _verify_exact(Ur.tolist(), Vr.tolist(), Wr.tolist(), n, m, p):
                        best_rank = target_rank
                        best_U = Ur.copy()
                        best_V = Vr.copy()
                        best_W = Wr.copy()
                        found = True
                        break
                    break
                
                # Perturb one entry
                choice = random.random()
                if choice < 0.33:
                    idx = random.randint(0, n*m-1)
                    col = random.randint(0, target_rank-1)
                    Ur[idx, col] = round((Ur[idx, col] + random.choice([-0.5, 0.5])) * 2) / 2
                elif choice < 0.66:
                    idx = random.randint(0, m*p-1)
                    col = random.randint(0, target_rank-1)
                    Vr[idx, col] = round((Vr[idx, col] + random.choice([-0.5, 0.5])) * 2) / 2
                else:
                    idx = random.randint(0, p*n-1)
                    col = random.randint(0, target_rank-1)
                    Wr[idx, col] = round((Wr[idx, col] + random.choice([-0.5, 0.5])) * 2) / 2
        
        if found:
            break
    
    # Final verification
    if not _verify_exact(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p):
        # Fallback to schoolbook
        best_U = U_school
        best_V = V_school
        best_W = W_school
        best_rank = rank_school
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
