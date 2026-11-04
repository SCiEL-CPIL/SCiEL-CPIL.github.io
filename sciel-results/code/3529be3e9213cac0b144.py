import numpy as np
import random
import time
import sys
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Build the target tensor T
    # T[u, v, w] where u in [0, nm), v in [0, mp), w in [0, pn)
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    # T[u, v, w] = 1 if j_u == j_v and k_v == k_w and i_w == i_u else 0
    T = np.zeros((nm, mp, pn), dtype=np.int32)
    # Precompute non-zero indices for each w
    T_nonzero = {w: [] for w in range(pn)}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1
                T_nonzero[w].append((u, v))

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best = schoolbook()
    best_rank = n * m * p

    def verify_full(U, V, W, rank):
        """Full tensor verification."""
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)
        
        for w in range(pn):
            M = np.zeros((nm, mp), dtype=np.int64)
            for c in range(rank):
                if W_np[w, c] != 0:
                    M += U_np[:, c] * V_np[:, c] * W_np[w, c]
            if not np.array_equal(M, T[:, :, w]):
                return False
        return True

    # Search for lower rank
    min_rank = 30 # Try down to 30
    
    # Precompute a set of zero-indices for each w to pick random rows
    all_indices = np.arange(nm * mp)
    
    for target_rank in range(min_rank, best_rank):
        if time.time() > deadline - 10:
            break
            
        # Try multiple random candidates for this rank
        # Increase attempts since each is fast
        max_attempts = 5000
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Generate random U, V with small integers
            # Wider range might help find integer solutions
            U = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(target_rank)] for _ in range(nm)]
            V = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(target_rank)] for _ in range(mp)]
            
            U_np = np.array(U, dtype=np.float64)
            V_np = np.array(V, dtype=np.float64)
            
            W = [[0] * target_rank for _ in range(pn)]
            valid = True
            
            # Solve for each w using subset of rows
            for w in range(pn):
                # Construct A_w (nm*mp x target_rank)
                # A_w[u*mp + v, c] = U[u, c] * V[v, c]
                # We only need to construct it partially or on the fly?
                # Let's construct the full A_w once per attempt? No, per w.
                # A_w is 126 x r.
                A_w = np.zeros((nm * mp, target_rank), dtype=np.float64)
                for u in range(nm):
                    for v in range(mp):
                        A_w[u * mp + v, :] = U_np[u, :] * V_np[v, :]
                
                b_w = T[:, :, w].flatten().astype(np.float64)
                
                # Select a subset of rows to solve
                # We need target_rank rows.
                # Prefer rows where b_w is 1 (non-zero) to constrain the solution strongly
                nonzero_idx = [u * mp + v for u, v in T_nonzero[w]]
                if len(nonzero_idx) < target_rank:
                    # Fill with random zero rows
                    remaining = target_rank - len(nonzero_idx)
                    zero_indices = list(set(all_indices.tolist()) - set(nonzero_idx))
                    random.shuffle(zero_indices)
                    selected_indices = nonzero_idx + zero_indices[:remaining]
                else:
                    random.shuffle(nonzero_idx)
                    selected_indices = nonzero_idx[:target_rank]
                
                A_sub = A_w[selected_indices, :]
                b_sub = b_w[selected_indices]
                
                # Solve A_sub W_w = b_sub
                try:
                    # Use inverse if square and non-singular? lstsq is safer.
                    # But for square, solve is faster.
                    # Check condition number or just try solve.
                    W_w = np.linalg.solve(A_sub, b_sub)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                # Check if solution is close to integer/half-integer
                # Snap to nearest half-integer
                W_w_snapped = np.round(W_w * 2) / 2.0
                
                # Verify snapped solution against full A_w
                residual = A_w.dot(W_w_snapped) - b_w
                if np.linalg.norm(residual) > 1e-6:
                    valid = False
                    break
                
                # Store W_w
                for c in range(target_rank):
                    W[w][c] = W_w_snapped[c]
            
            if valid:
                # Final verification with full check (integer exactness)
                if verify_full(U, V, W, target_rank):
                    best = {"U": U, "V": V, "W": W}
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            continue
    
    return best
# EVOLVE-BLOCK-END
