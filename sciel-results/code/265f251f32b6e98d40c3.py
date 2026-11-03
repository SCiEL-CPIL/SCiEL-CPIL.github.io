"""Matrix multiplication 2x4x6 rank-minimal tensor decomposition.

Uses structured block decomposition and optimization to find low-rank
exact decompositions.
"""

import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = 145.0

    n, m, p = DIMENSIONS  # 2, 4, 6
    nm, mp, pn = n * m, m * p, p * n  # 8, 24, 12

    # Build target tensor as sparse dict
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[(a, b, c)] = 1.0

    def verify(U, V, W, tol=1e-8):
        """Verify that sum_r U[a,r]*V[b,r]*W[c,r] == T[a,b,c] for all entries."""
        rank = len(U[0]) if U else 0
        # Check all non-zero entries
        for (a, b, c), val in T.items():
            s = 0.0
            for r in range(rank):
                s += U[a][r] * V[b][r] * W[c][r]
            if abs(s - val) > tol:
                return False
        # Check all zero entries
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    if (a, b, c) in T:
                        continue
                    s = 0.0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if abs(s) > tol:
                        return False
        return True

    # Try to find a low-rank decomposition using alternating optimization
    def find_decomposition(target_rank, max_iter=200, seed=42):
        """Try to find a rank=target_rank exact decomposition via ALS."""
        rng = np.random.RandomState(seed)
        
        # Initialize
        U = rng.randn(nm, target_rank) * 0.1
        V = rng.randn(mp, target_rank) * 0.1
        W = rng.randn(pn, target_rank) * 0.1
        
        for it in range(max_iter):
            if time.time() - start_time > deadline - 5:
                break
            
            # Update U: for each a, solve for U[a,:] given V, W
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For fixed a: T[a,b,c] = sum_r U[a,r] * (V[b,r] * W[c,r])
            # This is a linear system: for each (b,c), T[a,b,c] = sum_r U[a,r] * Z[b,c,r]
            # where Z[b,c,r] = V[b,r]*W[c,r]
            # So we have a linear system in U[a,:]
            
            # Build the system for U
            # For each a, we need: for all (b,c), sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
            # This is: M_a @ U[a,:] = t_a where M_a[b,c,r] = V[b,r]*W[c,r]
            # Flatten: for each (b,c), the equation is sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
            
            for a in range(nm):
                # Build linear system: X @ u = y where X has shape (num_eq, rank)
                # Each equation: for (b,c), sum_r u[r]*V[b,r]*W[c,r] = T[a,b,c]
                rows = []
                rhs = []
                for b in range(mp):
                    for c in range(pn):
                        target = T.get((a, b, c), 0.0)
                        row = [V[b][r] * W[c][r] for r in range(target_rank)]
                        rows.append(row)
                        rhs.append(target)
                X = np.array(rows)
                y = np.array(rhs)
                # Solve least squares
                u, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                U[a] = u
            
            # Similarly update V
            for b in range(mp):
                rows = []
                rhs = []
                for a in range(nm):
                    for c in range(pn):
                        target = T.get((a, b, c), 0.0)
                        row = [U[a][r] * W[c][r] for r in range(target_rank)]
                        rows.append(row)
                        rhs.append(target)
                X = np.array(rows)
                y = np.array(rhs)
                v, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                V[b] = v
            
            # Update W
            for c in range(pn):
                rows = []
                rhs = []
                for a in range(nm):
                    for b in range(mp):
                        target = T.get((a, b, c), 0.0)
                        row = [U[a][r] * V[b][r] for r in range(target_rank)]
                        rows.append(row)
                        rhs.append(target)
                X = np.array(rows)
                y = np.array(rhs)
                w, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                W[c] = w
            
            # Check convergence
            if verify(U.tolist(), V.tolist(), W.tolist()):
                return U.tolist(), V.tolist(), W.tolist()
        
        return None

    # Try ranks from low to high
    best_result = None
    for target_rank in range(8, 25):
        if time.time() - start_time > deadline - 10:
            break
        
        result = find_decomposition(target_rank, max_iter=50, seed=target_rank)
        if result is not None:
            U, V, W = result
            if verify(U, V, W):
                best_result = (U, V, W, target_rank)
                break
    
    if best_result is not None:
        U, V, W, rank = best_result
        print(f"Found rank {rank} decomposition", file=sys.stderr)
        return {"U": U, "V": V, "W": W}
    
    # Fallback: Strassen-based rank 42
    strassen_2x2x2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [0, 0, 1, 1]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 1, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 1, 0]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, 1, 0, 1]),
    ]
    
    total_rank = 42
    U = [[0.0] * total_rank for _ in range(nm)]
    V = [[0.0] * total_rank for _ in range(mp)]
    W = [[0.0] * total_rank for _ in range(pn)]
    
    col_offset = 0
    for t in range(3):
        for (U_r, V_r, W_r) in strassen_2x2x2:
            for local_idx in range(4):
                val = U_r[local_idx]
                if val == 0:
                    continue
                li, lj = divmod(local_idx, 2)
                global_a = li * 4 + lj
                U[global_a][col_offset] += val
            
            for local_idx in range(4):
                val = V_r[local_idx]
                if val == 0:
                    continue
                lj, lk = divmod(local_idx, 2)
                global_b = lj * 6 + (2 * t + lk)
                V[global_b][col_offset] += val
            
            for local_idx in range(4):
                val = W_r[local_idx]
                if val == 0:
                    continue
                lk, li = divmod(local_idx, 2)
                global_c = (2 * t + lk) * 2 + li
                W[global_c][col_offset] += val
            
            col_offset += 1
        
        for (U_r, V_r, W_r) in strassen_2x2x2:
            for local_idx in range(4):
                val = U_r[local_idx]
                if val == 0:
                    continue
                li, lj = divmod(local_idx, 2)
                global_a = li * 4 + (2 + lj)
                U[global_a][col_offset] += val
            
            for local_idx in range(4):
                val = V_r[local_idx]
                if val == 0:
                    continue
                lj, lk = divmod(local_idx, 2)
                global_b = (2 + lj) * 6 + (2 * t + lk)
                V[global_b][col_offset] += val
            
            for local_idx in range(4):
                val = W_r[local_idx]
                if val == 0:
                    continue
                lk, li = divmod(local_idx, 2)
                global_c = (2 * t + lk) * 2 + li
                W[global_c][col_offset] += val
            
            col_offset += 1
    
    if not verify(U, V, W):
        # Schoolbook fallback
        rank0 = n * m * p
        U = [[0.0] * rank0 for _ in range(nm)]
        V = [[0.0] * rank0 for _ in range(mp)]
        W = [[0.0] * rank0 for _ in range(pn)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i * (m * p) + j * p + k
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a][col] = 1.0
                    V[b][col] = 1.0
                    W[c][col] = 1.0
    
    print(f"Using fallback rank {len(U[0])}", file=sys.stderr)
    return {"U": U, "V": V, "W": W}
