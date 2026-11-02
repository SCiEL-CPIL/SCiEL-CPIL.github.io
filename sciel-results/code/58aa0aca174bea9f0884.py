"""Matrix multiplication 2x3x4 rank-minimal exact decomposition."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    n, m, p = DIMENSIONS  # 2, 3, 4
    schoolbook_rank = n * m * p  # 24

    # Build schoolbook solution (rank 24)
    U_sb, V_sb, W_sb = _build_schoolbook(n, m, p)
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank

    # Try to find rank 23
    if time.time() < deadline - 10:
        result = _search_rank_optimized(n, m, p, 23, deadline)
        if result is not None:
            best_result = result
            best_rank = 23

    # If rank 23 not found, try rank 22 with less time
    if best_rank == schoolbook_rank and time.time() < deadline - 5:
        result = _search_rank_optimized(n, m, p, 22, deadline)
        if result is not None:
            best_result = result
            best_rank = 22

    return best_result

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _verify_exact(U, V, W, n, m, p):
    """Exact verification using Fraction."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                u_row = U[i * m + j]
                v_row = V[j * p + k]
                w_row = W[k * n + i]
                for r in range(rank):
                    s += Fraction(u_row[r]) * Fraction(v_row[r]) * Fraction(w_row[r])
                if s != Fraction(1):
                    return False
    return True

def _snap_to_rational(U, V, W, n, m, p, max_denom=16):
    """Snap to rationals with small denominators and verify exactly."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for denom in [1, 2, 4, 8, 16]:
        # Check if values are close to multiples of 1/denom
        U_scaled = U_arr * denom
        V_scaled = V_arr * denom
        W_scaled = W_arr * denom
        
        if np.allclose(U_scaled, np.round(U_scaled), atol=1e-5) and \
           np.allclose(V_scaled, np.round(V_scaled), atol=1e-5) and \
           np.allclose(W_scaled, np.round(W_scaled), atol=1e-5):
            
            U_int = np.round(U_scaled).astype(int)
            V_int = np.round(V_scaled).astype(int)
            W_int = np.round(W_scaled).astype(int)
            
            # Convert to Fraction lists for exact verification
            U_frac = [[Fraction(a, denom) for a in row] for row in U_int]
            V_frac = [[Fraction(a, denom) for a in row] for row in V_int]
            W_frac = [[Fraction(a, denom) for a in row] for row in W_int]
            
            if _verify_exact(U_frac, V_frac, W_frac, n, m, p):
                return U_frac, V_frac, W_frac
    return None

def _search_rank_optimized(n, m, p, target_rank, deadline):
    """
    Optimized local search for a specific rank.
    Uses float arithmetic for speed, snaps to rationals for verification.
    """
    um, vm, wm = n * m, m * p, p * n
    np.random.seed(42)

    # Precompute index mappings for faster residual access
    # Residuals are indexed by (i, j, k) -> idx
    # idx = i * (m * p) + j * p + k
    
    max_attempts = 50
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break

        # Initialize U, V, W
        # Strategy: Start with random small integers/half-integers
        # Or start from schoolbook and perturb?
        # For rank < schoolbook, random initialization is often better to escape local minima
        
        # Use a mix of 0, 1, -1, 0.5, -0.5
        choices = np.array([0, 1, -1, 0.5, -0.5])
        
        U = np.random.choice(choices, size=(um, target_rank))
        V = np.random.choice(choices, size=(vm, target_rank))
        W = np.random.choice(choices, size=(wm, target_rank))
        
        # Ensure some structure: maybe force some 1s to help convergence?
        # Let's try a "sparse" initialization
        for r in range(target_rank):
            U[np.random.randint(um), r] = 1.0
            V[np.random.randint(vm), r] = 1.0
            W[np.random.randint(wm), r] = 1.0

        # Local Search
        # We want to minimize max |sum(U*V*W) - 1|
        
        # Compute initial residuals
        residuals = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * (m * p) + j * p + k
                    u_row = U[i * m + j]
                    v_row = V[j * p + k]
                    w_row = W[k * n + i]
                    # Vectorized dot product
                    residuals[idx] = np.dot(u_row, v_row * w_row) - 1.0
        
        max_res = np.max(np.abs(residuals))
        if max_res < 1e-5:
            snapped = _snap_to_rational(U, V, W, n, m, p)
            if snapped:
                return {"U": snapped[0], "V": snapped[1], "W": snapped[2]}

        # Coordinate Descent
        # To speed up, we only update if it reduces the max residual significantly
        steps = 200
        for step in range(steps):
            if time.time() > deadline:
                break
            
            # Find the worst residual
            worst_idx = np.argmax(np.abs(residuals))
            wi = worst_idx // (m * p)
            wj = (worst_idx // p) % m
            wk = worst_idx % p
            
            current_worst_val = residuals[worst_idx]
            
            # Try adjusting U
            improved = False
            for r in range(target_rank):
                old_val = U[wi * m + wj, r]
                # Try deltas
                for delta in [-1.0, -0.5, 0.5, 1.0]:
                    new_val = old_val + delta
                    # Skip if same
                    if new_val == old_val:
                        continue
                    U[wi * m + wj, r] = new_val
                    
                    # Update residual for this (i, j, k)
                    # res = sum U V W
                    # Delta res = delta * V[j,k,r] * W[k,i,r]
                    v_val = V[wj * p + wk, r]
                    w_val = W[wk * n + wi, r]
                    new_res = current_worst_val + delta * v_val * w_val
                    
                    # Check if this improves the global max?
                    # Approximation: just check if this specific residual improves
                    if abs(new_res) < abs(current_worst_val):
                        residuals[worst_idx] = new_res
                        improved = True
                        break
                    else:
                        U[wi * m + wj, r] = old_val
                if improved:
                    break
            
            if improved:
                # Recompute max residual occasionally or track it
                # For speed, we just continue. The next iteration finds the new worst.
                continue

            # Try adjusting V
            for r in range(target_rank):
                old_val = V[wj * p + wk, r]
                for delta in [-1.0, -0.5, 0.5, 1.0]:
                    new_val = old_val + delta
                    if new_val == old_val:
                        continue
                    V[wj * p + wk, r] = new_val
                    u_val = U[wi * m + wj, r]
                    w_val = W[wk * n + wi, r]
                    new_res = current_worst_val + delta * u_val * w_val
                    if abs(new_res) < abs(current_worst_val):
                        residuals[worst_idx] = new_res
                        improved = True
                        break
                    else:
                        V[wj * p + wk, r] = old_val
                if improved:
                    break
            
            if improved:
                continue

            # Try adjusting W
            for r in range(target_rank):
                old_val = W[wk * n + wi, r]
                for delta in [-1.0, -0.5, 0.5, 1.0]:
                    new_val = old_val + delta
                    if new_val == old_val:
                        continue
                    W[wk * n + wi, r] = new_val
                    u_val = U[wi * m + wj, r]
                    v_val = V[wj * p + wk, r]
                    new_res = current_worst_val + delta * u_val * v_val
                    if abs(new_res) < abs(current_worst_val):
                        residuals[worst_idx] = new_res
                        improved = True
                        break
                    else:
                        W[wk * n + wi, r] = old_val
                if improved:
                    break
            
            if not improved:
                # Stuck, try a random perturbation to escape local minimum
                if step % 10 == 0:
                    idx_u = np.random.randint(um)
                    idx_v = np.random.randint(vm)
                    idx_w = np.random.randint(wm)
                    r = np.random.randint(target_rank)
                    U[idx_u, r] += np.random.choice([-1, -0.5, 0.5, 1])
                    V[idx_v, r] += np.random.choice([-1, -0.5, 0.5, 1])
                    W[idx_w, r] += np.random.choice([-1, -0.5, 0.5, 1])
                    # Recompute residuals roughly? Or just continue?
                    # Let's do a full recompute every 50 steps if stuck
                    if step % 50 == 0:
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    idx = i * (m * p) + j * p + k
                                    u_row = U[i * m + j]
                                    v_row = V[j * p + k]
                                    w_row = W[k * n + i]
                                    residuals[idx] = np.dot(u_row, v_row * w_row) - 1.0
                        max_res = np.max(np.abs(residuals))
                        if max_res < 1e-5:
                            snapped = _snap_to_rational(U, V, W, n, m, p)
                            if snapped:
                                return {"U": snapped[0], "V": snapped[1], "W": snapped[2]}
                else:
                    break

        # Final check after search
        max_res = np.max(np.abs(residuals))
        if max_res < 1e-5:
            snapped = _snap_to_rational(U, V, W, n, m, p)
            if snapped:
                return {"U": snapped[0], "V": snapped[1], "W": snapped[2]}

    return None
# EVOLVE-BLOCK-END
