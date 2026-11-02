import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    n, m, p = DIMENSIONS
    
    # 1. Generate the baseline Rank 50 Strassen decomposition
    rank = 50
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    def add_strassen_222(i0, j0, k0):
        nonlocal col
        if col >= rank:
            return False
        
        # Indices for A (2x2 block at i0, j0)
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        
        # Indices for B (2x2 block at j0, k0)
        e = j0 * p + k0
        f = j0 * p + (k0 + 1)
        g = (j0 + 1) * p + k0
        h = (j0 + 1) * p + (k0 + 1)
        
        # Indices for C (2x2 block at i0, k0) - Trace convention W[k*n+i, r]
        # C indices: row = i, col = k. Flat index = k*n + i.
        c00 = k0 * n + i0
        c01 = (k0 + 1) * n + i0
        c10 = k0 * n + (i0 + 1)
        c11 = (k0 + 1) * n + (i0 + 1)
        
        # 1
        U[a][col] += 1; U[d][col] += 1
        V[e][col] += 1; V[h][col] += 1
        W[c00][col] += 1; W[c11][col] += 1
        col += 1
        # 2
        U[c][col] += 1; U[d][col] += 1
        V[e][col] += 1
        W[c10][col] += 1; W[c11][col] -= 1
        col += 1
        # 3
        U[a][col] += 1
        V[f][col] += 1; V[h][col] -= 1
        W[c01][col] += 1; W[c11][col] += 1
        col += 1
        # 4
        U[d][col] += 1
        V[g][col] += 1; V[e][col] -= 1
        W[c00][col] += 1; W[c10][col] += 1
        col += 1
        # 5
        U[a][col] += 1; U[b][col] += 1
        V[h][col] += 1
        W[c00][col] -= 1; W[c01][col] += 1
        col += 1
        # 6
        U[c][col] += 1; U[a][col] -= 1
        V[e][col] += 1; V[f][col] += 1
        W[c11][col] += 1
        col += 1
        # 7
        U[b][col] += 1; U[d][col] -= 1
        V[g][col] += 1; V[h][col] += 1
        W[c00][col] += 1
        col += 1
        
        return True

    def add_schoolbook_221(i0, j0, k0):
        nonlocal col
        if col >= rank:
            return False
        
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        
        e = j0 * p + k0
        g = (j0 + 1) * p + k0
        
        c00 = k0 * n + i0
        c10 = k0 * n + (i0 + 1)
        
        U[a][col] += 1; V[e][col] += 1; W[c00][col] += 1
        col += 1
        U[b][col] += 1; V[g][col] += 1; W[c00][col] += 1
        col += 1
        U[c][col] += 1; V[e][col] += 1; W[c10][col] += 1
        col += 1
        U[d][col] += 1; V[g][col] += 1; W[c10][col] += 1
        col += 1
        return True

    # Build baseline
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    # Strassen blocks for k=0,2,4
    for k0 in [0, 2, 4]:
        add_strassen_222(0, 0, k0)
        
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    # Schoolbook for k=6
    add_schoolbook_221(0, 0, 6)
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    # Strassen blocks for j=2
    for k0 in [0, 2, 4]:
        add_strassen_222(0, 2, k0)
        
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    # Schoolbook for j=2, k=6
    add_schoolbook_221(0, 2, 6)

    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # 2. Attempt Rank Reduction
    # Construct the target tensor T_flattened (size n*m*p)
    # T[i, k, j] = 1 if A[i,k] * B[k,j] -> C[i,j]
    # In our indices: U[i*m+k], V[k*p+j], W[j*n+i]
    # Equation: C[j*n+i] = sum_k U[i*m+k, r] V[k*p+j, r] W[j*n+i, r] ??
    # No, the standard form is:
    # C = sum_r U_r (as vector for A?) ...
    # Actually, the decomposition is:
    # C_{ij} = sum_k A_{ik} B_{kj} = sum_r U_{ik, r} V_{kj, r} W_{ij, r}
    # Here W is indexed by (i,j) or (j,i).
    # Contract: W[k*n+i, r] is given. So W index is flat (j*n+i).
    # The term is U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r].
    # We want this to equal 1 for all i,j,k (in the sense of the tensor).
    
    # Flatten U, V, W to solve a linear system.
    # Let U, V, W be the variables.
    # We have constraints: For each (i, j, k), sum_r U_{i*m+k, r} V_{k*p+j, r} W_{j*n+i, r} = delta?
    # No, the tensor T has entries T_{i,k,j} = 1.
    # The reconstruction is C_{ij} = sum_k A_{ik} B_{kj}.
    # So for any i, j, sum_k (A_{ik} B_{kj}) must be reproduced.
    # This is satisfied if:
    # sum_r U_{i*m+k, r} V_{k*p+j, r} W_{j*n+i, r} = A_{ik} B_{kj} (symbolically)
    # This implies that for each (i,k,j), the coefficient of the monomial A_{ik} B_{kj} in the sum must be 1.
    # Since U, V, W are linear in A, B, C coefficients, this is a 3-way equality.
    # It is easier to check the final matrices.
    
    # We will try to drop one column (rank -1).
    # Fix the other columns, solve for the new column?
    # No, we want to DROP a column.
    # We fix U[:, :R-1], V[:, :R-1], W[:, :R-1].
    # And we want to see if we can adjust them (or a subset) to satisfy the tensor with R-1 terms.
    # This is under-determined.
    
    # Alternative: 
    # Take the current R=50 solution.
    # Pick a column index 'drop_idx'.
    # Set U[:, drop_idx], V[:, drop_idx], W[:, drop_idx] to 0.
    # Now we have a system of equations for the remaining 49 vectors?
    # No, the remaining 49 vectors are fixed from the baseline.
    # We can allow the remaining vectors to CHANGE.
    # We have 49 * (nm + mp + pn) variables.
    # Constraints: nm * mp * pn ? No.
    # The number of independent constraints is nm * mp? 
    # Actually, the tensor has size n*m*p = 56 constraints? No.
    # The identity C_{ij} = sum_k A_{ik} B_{kj} has n*p equations.
    # Each equation involves m terms.
    # Total terms: n*p*m = 56.
    # We have 49 vectors of size nm+mp+pn = 8 + 28 + 14 = 50.
    # Total variables: 49 * 50 = 2450.
    # Constraints: 56 equations? No, the structure is bilinear/trilinear.
    
    # This is too complex to solve with simple lstsq in 120s.
    # I will stick to the Rank 50 baseline but formatted cleanly.
    
    # Convert to list of lists for interface
    U_list = [[U[i][j] for j in range(rank)] for i in range(len(U))]
    V_list = [[V[i][j] for j in range(rank)] for i in range(len(V))]
    W_list = [[W[i][j] for j in range(rank)] for i in range(len(W))]

    return {"U": U_list, "V": V_list, "W": W_list}
