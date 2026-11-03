import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s deadline, 20s safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, np_ = n*m, m*p, p*n
    
    # Target tensor: T[(i,j), (j,k), (k,i)] = 1 for all valid i,j,k
    # Flatten: U index = i*m+j, V index = j*p+k, W index = k*n+i
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_indices.append((i*m+j, j*p+k, k*n+i))
    
    # Try ranks from low to high
    for rank in [10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120]:
        if time.time() > deadline - 30:
            break
        
        found = _try_rank(n, m, p, rank, target_indices, deadline)
        if found is not None:
            U, V, W = found
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_rank(n, m, p, rank, target_indices, deadline):
    nm, mp, np_ = n*m, m*p, p*n
    target_indices = np.array(target_indices)
    
    # We need U (nm x rank), V (mp x rank), W (np_ x rank)
    # such that for each (a,b,c) in target_indices: sum_r U[a,r]*V[b,r]*W[c,r] = 1
    
    # Strategy: fix W, solve for U,V or use alternating minimization
    # Try random initializations with half-integer entries
    
    np.random.seed(42)
    
    for attempt in range(200):
        if time.time() > deadline - 10:
            break
        
        # Initialize with random half-integer values
        def rand_halfint(shape):
            vals = np.random.choice([-1, -0.5, 0, 0.5, 1], size=shape)
            return vals
        
        U = rand_halfint((nm, rank))
        V = rand_halfint((mp, rank))
        W = rand_halfint((np_, rank))
        
        # Iterate: fix two, solve for third
        for it in range(200):
            if time.time() > deadline - 10:
                break
            
            # Check if solution is exact
            errors = []
            for a, b, c in target_indices:
                val = np.dot(U[a], V[b] * W[c])
                errors.append(abs(val - 1))
            max_err = max(errors) if errors else 0
            
            if max_err < 1e-10:
                # Verify all entries are half-integers
                if _all_halfint(U) and _all_halfint(V) and _all_halfint(W):
                    return (U.tolist(), V.tolist(), W.tolist())
            
            # Alternate: fix V,W, solve for U (least squares)
            # For each row a of U: sum_r U[a,r] * (V[b,r]*W[c,r]) = 1 for all (a,b,c)
            # This is underdetermined in general, so use least squares
            
            # Build system for U: for each (a,b,c), we have linear eq in U[a,:]
            # But it's bilinear, so fix V,W and it's linear in U
            # Group by a: for each a, we have equations sum_r U[a,r] * V[b,r] * W[c,r] = 1
            # for all b,c such that (a,b,c) is in target
            
            # Actually, for fixed V,W, the problem for U is:
            # For each a: find U[a,:] such that for all (a,b,c) in target: U[a,:] dot (V[b,:] * W[c,:]) = 1
            # This is a system of linear equations in U[a,:]
            
            # Similarly for V and W
            
            # Solve for U
            U_new = np.zeros((nm, rank))
            for a in range(nm):
                # Find all (b,c) such that (a,b,c) in target
                mask = target_indices[:, 0] == a
                if not mask.any():
                    continue
                bc = target_indices[mask, 1:]
                # Build matrix: rows are bc pairs, columns are r, entry = V[b,r]*W[c,r]
                A_mat = np.zeros((len(bc), rank))
                for idx, (b, c) in enumerate(bc):
                    A_mat[idx] = V[b] * W[c]
                # Solve A_mat @ U[a,:] = 1
                try:
                    sol = np.linalg.lstsq(A_mat, np.ones(len(bc)), rcond=None)[0]
                    # Round to nearest half-integer
                    U_new[a] = np.round(sol * 2) / 2
                except:
                    U_new[a] = U[a]
            U = U_new
            
            # Solve for V
            V_new = np.zeros((mp, rank))
            for b in range(mp):
                mask = target_indices[:, 1] == b
                if not mask.any():
                    continue
                ac = target_indices[mask, [0, 2]]
                A_mat = np.zeros((len(ac), rank))
                for idx, (a, c) in enumerate(ac):
                    A_mat[idx] = U[a] * W[c]
                try:
                    sol = np.linalg.lstsq(A_mat, np.ones(len(ac)), rcond=None)[0]
                    V_new[b] = np.round(sol * 2) / 2
                except:
                    V_new[b] = V[b]
            V = V_new
            
            # Solve for W
            W_new = np.zeros((np_, rank))
            for c in range(np_):
                mask = target_indices[:, 2] == c
                if not mask.any():
                    continue
                ab = target_indices[mask, [0, 1]]
                A_mat = np.zeros((len(ab), rank))
                for idx, (a, b) in enumerate(ab):
                    A_mat[idx] = U[a] * V[b]
                try:
                    sol = np.linalg.lstsq(A_mat, np.ones(len(ab)), rcond=None)[0]
                    W_new[c] = np.round(sol * 2) / 2
                except:
                    W_new[c] = W[c]
            W = W_new
        
        # Final check
        errors = []
        for a, b, c in target_indices:
            val = np.dot(U[a], V[b] * W[c])
            errors.append(abs(val - 1))
        if errors and max(errors) < 1e-10:
            if _all_halfint(U) and _all_halfint(V) and _all_halfint(W):
                return (U.tolist(), V.tolist(), W.tolist())
    
    return None

def _all_halfint(arr):
    return np.allclose(arr * 2, np.round(arr * 2))
