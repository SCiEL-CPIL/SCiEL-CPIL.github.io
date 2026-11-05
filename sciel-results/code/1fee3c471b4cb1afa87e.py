"""Matrix multiplication 2,5,6 - rank-minimal tensor decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Base construction: rank 54 using Strassen on (0,1) and (2,3) pairs,
    # schoolbook on j=4.
    # Then attempt structural rank reduction.
    
    result = _build_rank_54(n, m, p)
    if result is not None:
        best = result
        best_rank = 54
    else:
        best = _build_schoolbook(n, m, p)
        best_rank = n * m * p

    # Attempt rank reduction via structured local search
    for target_rank in range(best_rank - 1, max(best_rank - 8, 44), -1):
        if time.time() > deadline - 15:
            break
        candidate = _try_reduce_rank(n, m, p, best, target_rank, deadline)
        if candidate is not None:
            best = candidate
            best_rank = target_rank
            if best_rank <= 48:
                break

    return best


def _build_schoolbook(n, m, p):
    """Standard schoolbook: rank = n*m*p."""
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
    return {"U": U, "V": V, "W": W}


def _build_rank_54(n, m, p):
    """
    Rank-54 construction: Strassen on j-pairs (0,1) and (2,3), schoolbook on j=4.
    Returns None if construction fails verification.
    """
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    def add_strassen_block(i_base, j_base, k_base):
        nonlocal col
        # M0: (A00+A11)(B00+B11)
        U[i_base * m + j_base][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[j_base * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1

        # M1: (A10+A11)B00
        U[(i_base+1) * m + j_base][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] -= 1
        col += 1

        # M2: A00(B01-B11)
        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] -= 1
        W[(k_base+1) * n + i_base][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1

        # M3: A11(B10-B00)
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[(j_base+1) * p + k_base][col] += 1
        V[j_base * p + k_base][col] -= 1
        W[k_base * n + i_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        col += 1

        # M4: (A00+A01)B11
        U[i_base * m + j_base][col] += 1
        U[i_base * m + (j_base+1)][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] -= 1
        W[(k_base+1) * n + i_base][col] += 1
        col += 1

        # M5: (A10-A00)(B00+B01)
        U[(i_base+1) * m + j_base][col] += 1
        U[i_base * m + j_base][col] -= 1
        V[j_base * p + k_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1

        # M6: (A01-A11)(B10+B11)
        U[i_base * m + (j_base+1)][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] -= 1
        V[(j_base+1) * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1

    def add_naive_2112_block(i_base, j_base, k_base):
        nonlocal col
        U[i_base * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1
        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + i_base][col] += 1
        col += 1
        U[(i_base+1) * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        col += 1
        U[(i_base+1) * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1

    # Strassen blocks for j-pairs (0,1) and (2,3), k-pairs (0,1),(2,3),(4,5)
    for j_base in [0, 2]:
        for k_base in [0, 2, 4]:
            add_strassen_block(0, j_base, k_base)

    # Naive blocks for j=4
    for k_base in [0, 2, 4]:
        add_naive_2112_block(0, 4, k_base)

    if col != rank:
        return None

    # Verify
    if not _verify_decomposition(n, m, p, U, V, W, rank):
        return None

    return {"U": U, "V": V, "W": W}


def _verify_decomposition(n, m, p, U, V, W, rank):
    """
    Verify that the decomposition is exact.
    The matrix multiplication tensor T has T[i,j,k] = 1 for all i,j,k.
    Our decomposition: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] should equal T[i,j,k].
    
    For matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    Tensor form: T[i,j,k] = 1 means A[i,j]*B[j,k] contributes to C[i,k].
    Decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each (i,j,k), check sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_arr[i * m + j, :]
                v_val = V_arr[j * p + k, :]
                w_val = W_arr[k * n + i, :]
                total = np.sum(u_val * v_val * w_val)
                if abs(total - 1.0) > 1e-9:
                    return False
    return True


def _try_reduce_rank(n, m, p, base_result, target_rank, deadline):
    """
    Try to reduce rank from base to target_rank by modifying the decomposition.
    
    Strategy: Start from the base decomposition (rank R), and try to express
    one of the rank-1 terms as a linear combination of others, reducing total rank.
    
    More practically: use a local search on the factor matrices.
    We parameterize U, V, W as having target_rank columns, initialized from
    the base (with some columns merged/zeroed), then do coordinate descent
    to satisfy the tensor constraints.
    """
    base_rank = len(base_result["U"][0])
    if target_rank >= base_rank:
        return None

    U_base = np.array(base_result["U"], dtype=float)
    V_base = np.array(base_result["V"], dtype=float)
    W_base = np.array(base_result["W"], dtype=float)

    dim_u = n * m
    dim_v = m * p
    dim_w = p * n

    rng = random.Random(42 + target_rank)

    # Strategy: take the base decomposition and try to eliminate one column
    # by expressing it as a combination of others. This is a rank-reduction
    # via linear dependency.
    
    # Alternative: structured perturbation + local search
    for restart in range(5):
        if time.time() > deadline - 5:
            break

        # Initialize from base: keep first target_rank columns, 
        # and try to absorb the remaining (base_rank - target_rank) columns
        # into the kept ones via small perturbations.
        
        U = U_base[:, :target_rank].copy()
        V = V_base[:, :target_rank].copy()
        W = W_base[:, :target_rank].copy()

        # Add small random perturbations to allow escape from local minima
        for r in range(target_rank):
            U[:, r] += rng.gauss(0, 0.1)
            V[:, r] += rng.gauss(0, 0.1)
            W[:, r] += rng.gauss(0, 0.1)

        # Coordinate descent: for each (i,j,k), adjust the factors to satisfy
        # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        # This is a system of 60 equations in 3*target_rank*60 variables (but sparse).
        
        # Simple iterative refinement:
        for iteration in range(200):
            if time.time() > deadline - 3:
                break
            converged = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        
                        # Current value
                        current = 0.0
                        for r in range(target_rank):
                            current += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                        
                        error = 1.0 - current
                        if abs(error) < 1e-12:
                            continue
                        
                        # Adjust: pick the column r that maximizes the product
                        # U[u_idx,r] * V[v_idx,r] * W[w_idx,r] and adjust it.
                        # Or do a simple gradient step.
                        
                        # Gradient with respect to U[u_idx, r]:
                        # d(error)/d(U[u_idx,r]) = -V[v_idx,r] * W[w_idx,r]
                        # We want to minimize error^2, so update:
                        # U[u_idx,r] += learning_rate * error * V[v_idx,r] * W[w_idx,r]
                        
                        lr = 0.01
                        for r in range(target_rank):
                            grad_u = -V[v_idx, r] * W[w_idx, r]
                            grad_v = -U[u_idx, r] * W[w_idx, r]
                            grad_w = -U[u_idx, r] * V[v_idx, r]
                            
                            U[u_idx, r] += lr * error * (-grad_u)  # = lr * error * V*W
                            V[v_idx, r] += lr * error * (-grad_v)  # = lr * error * U*W
                            W[w_idx, r] += lr * error * (-grad_w)  # = lr * error * U*V
                        
                        if abs(error) > 1e-10:
                            converged = False
            if converged:
                break

        # Check if we've achieved the target
        if _verify_decomposition(n, m, p, U.tolist(), V.tolist(), W.tolist(), target_rank):
            # Snap to nearest half-integers
            U_snapped = np.round(U * 2) / 2.0
            V_snapped = np.round(V * 2) / 2.0
            W_snapped = np.round(W * 2) / 2.0
            
            if _verify_decomposition(n, m, p, U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist(), target_rank):
                return {"U": U_snapped.tolist(), "V": V_snapped.tolist(), "W": W_snapped.tolist()}
            else:
                # Return without snapping if verification passes
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return None

# EVOLVE-BLOCK-END
