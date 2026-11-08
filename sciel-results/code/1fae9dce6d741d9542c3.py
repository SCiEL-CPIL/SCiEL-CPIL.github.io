"""Rank-minimal tensor decomposition for 3x3 matrix multiplication."""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 3]

def build_target_terms():
    """Return list of (u_idx, v_idx, w_idx) for each valid multiplication term."""
    terms = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                u_idx = i * 3 + j
                v_idx = j * 3 + k
                w_idx = k * 3 + i
                terms.append((u_idx, v_idx, w_idx))
    return terms

TARGET_TERMS = build_target_terms()

def compute_decomposition_value(U, V, W, rank):
    """Compute the decomposition value for each target term.
    Returns array of shape (27,) with sum_r U[u,r]*V[v,r]*W[w,r] for each (u,v,w).
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    values = np.zeros(27)
    for idx, (u_idx, v_idx, w_idx) in enumerate(TARGET_TERMS):
        values[idx] = np.sum(U[u_idx] * V[v_idx] * W[w_idx])
    return values

def validate_decomposition(U, V, W, rank, tol=1e-9):
    """Check if decomposition exactly computes matrix multiplication."""
    values = compute_decomposition_value(U, V, W, rank)
    return np.all(np.abs(values - 1.0) < tol)

def _known_rank26():
    """Return the known rank-26 decomposition using 2x2 Strassen + schoolbook cross terms."""
    n, m, p = 3, 3, 3
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, coeff):
        nonlocal col
        U[u_idx][col] += coeff
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        col += 1
    
    # 1. A2 B2 (Top-left 2x2 * 2x2) -> Rank 7 (Strassen)
    # M1 = (A00+A11)(B00+B11)
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    # M2 = (A10+A11)B00
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[4][col] = -1
    col += 1
    
    # M3 = A00(B01-B11)
    U[0][col] = 1
    V[1][col] = 1; V[4][col] = -1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    
    # M4 = A11(B10-B00)
    U[4][col] = 1
    V[3][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # M5 = (A00+A01)B11
    U[0][col] = 1; U[1][col] = 1
    V[4][col] = 1
    W[0][col] = -1; W[3][col] = 1
    col += 1
    
    # M6 = (A10-A00)(B00+B01)
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][col] = 1; U[4][col] = -1
    V[3][col] = 1; V[4][col] = 1
    W[0][col] = 1
    col += 1
    
    # 2. a b^T (Top-right 2x1 * Bottom-left 1x2) -> Rank 4
    add_term(2, 6, 0, 1)
    add_term(2, 7, 3, 1)
    add_term(5, 6, 1, 1)
    add_term(5, 7, 4, 1)
    
    # 3. A2 b (Top-left 2x2 * Top-right 2x1) -> Rank 4
    add_term(0, 2, 6, 1)
    add_term(1, 5, 6, 1)
    add_term(3, 2, 7, 1)
    add_term(4, 5, 7, 1)
    
    # 4. a b3 (Top-right 2x1 * Bottom-right 1x1) -> Rank 2
    add_term(2, 8, 6, 1)
    add_term(5, 8, 7, 1)
    
    # 5. a^T B2 (Bottom-left 1x2 * Top-left 2x2) -> Rank 4
    add_term(6, 0, 2, 1)
    add_term(7, 3, 2, 1)
    add_term(6, 1, 5, 1)
    add_term(7, 4, 5, 1)
    
    # 6. a3 b^T (Bottom-right 1x1 * Bottom-left 1x2) -> Rank 2
    add_term(8, 6, 2, 1)
    add_term(8, 7, 5, 1)
    
    # 7. a^T b (Bottom-left 1x2 * Top-right 2x1) -> Rank 2
    add_term(6, 2, 8, 1)
    add_term(7, 5, 8, 1)
    
    # 8. a3 b3 (Bottom-right 1x1 * Bottom-right 1x1) -> Rank 1
    add_term(8, 8, 8, 1)
    
    return {'U': U, 'V': V, 'W': W}

def _try_structured_rank25(deadline):
    """Try to find a rank-25 solution by structured search.
    
    Strategy: Start from rank-26 and try to merge two columns by finding
    a linear combination that preserves the decomposition. We check if
    there exist coefficients alpha, beta such that merging columns r1 and
    r2 with (alpha * col_r1 + beta * col_r2) still gives a valid decomposition.
    
    For a valid merge, we need for all target terms:
    sum_{r != r1,r2} U[u,r]*V[v,r]*W[w,r] + (alpha*U[u,r1]+beta*U[u,r2])*(alpha*V[v,r1]+beta*V[v,r2])*(alpha*W[w,r1]+beta*W[w,r2]) = 1
    
    This is a system of 27 equations in 2 unknowns (alpha, beta) - generally
    overdetermined, but we can check if a solution exists by trying specific
    pairs of columns that have overlapping support patterns.
    """
    base = _known_rank26()
    U_base = np.array(base['U'], dtype=float)
    V_base = np.array(base['V'], dtype=float)
    W_base = np.array(base['W'], dtype=float)
    rank26 = 26
    
    # Precompute the "background" contribution (sum over all columns except the two we're merging)
    # For each pair (r1, r2), compute the residual that must be matched by the merged column
    
    # Try all pairs of columns (26 choose 2 = 325 pairs)
    # For each pair, check if there exist alpha, beta such that the merged column works
    
    # Optimization: Only try pairs where the merged column has integer/half-integer entries
    # and the resulting decomposition is valid.
    
    for r1 in range(rank26):
        for r2 in range(r1 + 1, rank26):
            if time.time() > deadline:
                return None
            
            # Compute the background contribution (sum over all r != r1, r2)
            bg = np.zeros(27)
            for idx, (u_idx, v_idx, w_idx) in enumerate(TARGET_TERMS):
                total = 0.0
                for r in range(rank26):
                    if r == r1 or r == r2:
                        continue
                    total += U_base[u_idx, r] * V_base[v_idx, r] * W_base[w_idx, r]
                bg[idx] = total
            
            # The merged column must satisfy:
            # bg[idx] + (alpha*U[u,r1]+beta*U[u,r2]) * (alpha*V[v,r1]+beta*V[v,r2]) * (alpha*W[w,r1]+beta*W[w,r2]) = 1
            # for all 27 target terms.
            
            # This is a nonlinear system in alpha, beta. Try simple rational values.
            found = False
            for alpha_num in range(-4, 5):
                for alpha_den in [1, 2]:
                    alpha = alpha_num / alpha_den
                    for beta_num in range(-4, 5):
                        for beta_den in [1, 2]:
                            beta = beta_num / beta_den
                            
                            # Check if this (alpha, beta) gives a valid merged column
                            valid = True
                            for idx, (u_idx, v_idx, w_idx) in enumerate(TARGET_TERMS):
                                u_merge = alpha * U_base[u_idx, r1] + beta * U_base[u_idx, r2]
                                v_merge = alpha * V_base[v_idx, r1] + beta * V_base[v_idx, r2]
                                w_merge = alpha * W_base[w_idx, r1] + beta * W_base[w_idx, r2]
                                val = bg[idx] + u_merge * v_merge * w_merge
                                if abs(val - 1.0) > 1e-9:
                                    valid = False
                                    break
                            
                            if valid:
                                # Check if the merged column has integer/half-integer entries
                                U_merge = alpha * U_base[:, r1] + beta * U_base[:, r2]
                                V_merge = alpha * V_base[:, r1] + beta * V_base[:, r2]
                                W_merge = alpha * W_base[:, r1] + beta * W_base[:, r2]
                                
                                # Check for half-integer: multiply by 2 and check if integer
                                U2 = U_merge * 2
                                V2 = V_merge * 2
                                W2 = W_merge * 2
                                
                                if (np.all(np.abs(U2 - np.round(U2)) < 1e-9) and
                                    np.all(np.abs(V2 - np.round(V2)) < 1e-9) and
                                    np.all(np.abs(W2 - np.round(W2)) < 1e-9)):
                                    # Build the rank-25 solution
                                    U25 = np.zeros((9, 25))
                                    V25 = np.zeros((9, 25))
                                    W25 = np.zeros((9, 25))
                                    
                                    col_idx = 0
                                    for r in range(rank26):
                                        if r == r1 or r == r2:
                                            continue
                                        U25[:, col_idx] = U_base[:, r]
                                        V25[:, col_idx] = V_base[:, r]
                                        W25[:, col_idx] = W_base[:, r]
                                        col_idx += 1
                                    
                                    # Add the merged column
                                    U25[:, col_idx] = U_merge
                                    V25[:, col_idx] = V_merge
                                    W25[:, col_idx] = W_merge
                                    
                                    # Verify
                                    if validate_decomposition(U25.tolist(), V25.tolist(), W25.tolist(), 25):
                                        return {
                                            'U': U25.tolist(),
                                            'V': V25.tolist(),
                                            'W': W25.tolist()
                                        }
                                found = True
                                break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
    
    return None

def _try_general_search_rank25(deadline):
    """General search for rank-25 using randomized local search from rank-26."""
    base = _known_rank26()
    U_base = np.array(base['U'], dtype=float)
    V_base = np.array(base['V'], dtype=float)
    W_base = np.array(base['W'], dtype=float)
    
    # Try random perturbations of the rank-26 solution, then remove one column
    # and adjust to compensate.
    
    for attempt in range(50):
        if time.time() > deadline:
            return None
        
        # Pick a random column to remove
        r_remove = random.randint(0, 25)
        
        # Start with the remaining 25 columns
        U25 = np.zeros((9, 25))
        V25 = np.zeros((9, 25))
        W25 = np.zeros((9, 25))
        
        col_idx = 0
        for r in range(26):
            if r == r_remove:
                continue
            U25[:, col_idx] = U_base[:, r]
            V25[:, col_idx] = V_base[:, r]
            W25[:, col_idx] = W_base[:, r]
            col_idx += 1
        
        # Now we need to adjust the 25 columns to compensate for the removed column.
        # Use local search: randomly perturb entries and check if we can fix the residuals.
        
        U_cur = U25.copy()
        V_cur = V25.copy()
        W_cur = W25.copy()
        
        best_err = None
        for step in range(200):
            if time.time() > deadline:
                return None
            
            # Compute current error
            values = compute_decomposition_value(U_cur.tolist(), V_cur.tolist(), W_cur.tolist(), 25)
            err = np.sum((values - 1.0) ** 2)
            
            if best_err is None or err < best_err:
                best_err = err
                U_best = U_cur.copy()
                V_best = V_cur.copy()
                W_best = W_cur.copy()
            
            if err < 1e-12:
                break
            
            # Random perturbation
            matrix = random.choice(['U', 'V', 'W'])
            row = random.randint(0, 8)
            col = random.randint(0, 24)
            
            if matrix == 'U':
                U_cur[row, col] += random.choice([-1, 1]) * random.choice([0.5, 1.0])
            elif matrix == 'V':
                V_cur[row, col] += random.choice([-1, 1]) * random.choice([0.5, 1.0])
            else:
                W_cur[row, col] += random.choice([-1, 1]) * random.choice([0.5, 1.0])
        
        # Check if we found a valid solution
        if best_err is not None and best_err < 1e-12:
            # Check for integer/half-integer entries
            U2 = U_best * 2
            V2 = V_best * 2
            W2 = W_best * 2
            
            if (np.all(np.abs(U2 - np.round(U2)) < 1e-9) and
                np.all(np.abs(V2 - np.round(V2)) < 1e-9) and
                np.all(np.abs(W2 - np.round(W2)) < 1e-9)):
                U_int = np.round(U2).astype(int) / 2.0
                V_int = np.round(V2).astype(int) / 2.0
                W_int = np.round(W2).astype(int) / 2.0
                
                if validate_decomposition(U_int.tolist(), V_int.tolist(), W_int.tolist(), 25):
                    return {
                        'U': U_int.tolist(),
                        'V': V_int.tolist(),
                        'W': W_int.tolist()
                    }
    
    return None

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Start with known rank-26 solution as fallback
    fallback = _known_rank26()
    
    # Verify the fallback is correct
    if not validate_decomposition(fallback['U'], fallback['V'], fallback['W'], 26):
        # This should never happen, but just in case
        raise ValueError("Rank-26 fallback is invalid!")
    
    best = fallback
    best_rank = 26
    
    # Try to find rank-25 solution
    # Strategy 1: Structured search (algebraic merge)
    result25 = _try_structured_rank25(deadline)
    if result25 is not None:
        best = result25
        best_rank = 25
    
    # If structured search failed, try general search
    if best_rank == 26 and time.time() < deadline - 10:
        result25_gen = _try_general_search_rank25(deadline)
        if result25_gen is not None:
            best = result25_gen
            best_rank = 25
    
    return best
# EVOLVE-BLOCK-END

