"""Program for rank-minimal exact tensor decomposition of 3x5x6 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction
import itertools

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _verify_exact(U, V, W, n, m, p):
    """
    Verify exact decomposition using integer/rational arithmetic.
    U, V, W are lists of lists with integer or half-integer entries.
    Returns True if the decomposition exactly equals the multiplication tensor.
    """
    if not U or not V or not W:
        return False
    
    rank = len(U[0])
    
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    for row in U:
        if len(row) != rank:
            return False
    for row in V:
        if len(row) != rank:
            return False
    for row in W:
        if len(row) != rank:
            return False
    
    # Convert to Fractions for exact arithmetic
    Uf = [[Fraction(x) for x in row] for row in U]
    Vf = [[Fraction(x) for x in row] for row in V]
    Wf = [[Fraction(x) for x in row] for row in W]
    
    # Build tensor and check each entry equals 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += Uf[i * m + j][c] * Vf[j * p + k][c] * Wf[k * n + i][c]
                if total != 1:
                    return False
    return True

def _verify_float_fast(U, V, W, n, m, p, atol=1e-8):
    """Fast float verification using numpy operations."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    # Build tensor efficiently
    tensor = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        Uc = U[:, c].reshape(n, m)
        Vc = V[:, c].reshape(m, p)
        Wc = W[:, c].reshape(p, n)
        # For each (i,j), tensor[i,j,k] += Uc[i,j] * Vc[j,k] * Wc[k,i]
        for i in range(n):
            for j in range(m):
                if Uc[i, j] != 0:
                    tensor[i, j, :] += Uc[i, j] * (Vc[j, :] * Wc[:, i])
    
    return np.allclose(tensor, 1.0, atol=atol)

def _schoolbook(n, m, p):
    """Return the schoolbook (rank = n*m*p) decomposition."""
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

def _try_remove_term_greedy(U, V, W, n, m, p, c_remove, deadline):
    """
    Try to remove term c_remove by adjusting other terms.
    Returns (success, new_U, new_V, new_W) or (False, None, None, None).
    """
    if time.time() > deadline:
        return False, None, None, None
    
    rank = len(U[0])
    
    # Make copies
    U_copy = [row[:] for row in U]
    V_copy = [row[:] for row in V]
    W_copy = [row[:] for row in W]
    
    # Zero out term c_remove
    for idx in range(n * m):
        U_copy[idx][c_remove] = 0
    for idx in range(m * p):
        V_copy[idx][c_remove] = 0
    for idx in range(p * n):
        W_copy[idx][c_remove] = 0
    
    # Check if it still works
    if _verify_float_fast(U_copy, V_copy, W_copy, n, m, p):
        if _verify_exact(U_copy, V_copy, W_copy, n, m, p):
            return True, U_copy, V_copy, W_copy
    
    # Try adjusting neighboring terms to compensate
    # Find terms that share indices with c_remove
    i0 = c_remove // (m * p)
    j0 = (c_remove // p) % m
    k0 = c_remove % p
    
    # Terms that share the same (i,j) pair - can adjust V and W
    # Terms that share the same (j,k) pair - can adjust U and W
    # Terms that share the same (k,i) pair - can adjust U and V
    
    candidate_terms = set()
    
    # Same (i0, j0): terms (i0, j0, k) for all k
    for k in range(p):
        c = i0 * m * p + j0 * p + k
        if c != c_remove:
            candidate_terms.add(c)
    
    # Same (j0, k0): terms (i, j0, k0) for all i
    for i in range(n):
        c = i * m * p + j0 * p + k0
        if c != c_remove:
            candidate_terms.add(c)
    
    # Same (k0, i0): terms (i0, j, k0) for all j
    for j in range(m):
        c = i0 * m * p + j * p + k0
        if c != c_remove:
            candidate_terms.add(c)
    
    # Try small adjustments to candidate terms
    # Adjust by ±1 or ±0.5
    adjustments = [1, -1, 0.5, -0.5]
    
    for c_adj in sorted(candidate_terms):
        if time.time() > deadline:
            return False, None, None, None
        
        # Determine which indices this term affects
        i_adj = c_adj // (m * p)
        j_adj = (c_adj // p) % m
        k_adj = c_adj % p
        
        # Try adjusting U[i_adj*m+j_adj][c_adj]
        for adj in adjustments:
            if time.time() > deadline:
                return False, None, None, None
            
            orig_val = U_copy[i_adj * m + j_adj][c_adj]
            new_val = orig_val + adj
            # Only allow integer or half-integer values
            if abs(new_val * 2 - round(new_val * 2)) > 1e-9:
                continue
            
            U_copy[i_adj * m + j_adj][c_adj] = new_val
            
            if _verify_float_fast(U_copy, V_copy, W_copy, n, m, p):
                if _verify_exact(U_copy, V_copy, W_copy, n, m, p):
                    return True, U_copy, V_copy, W_copy
            
            U_copy[i_adj * m + j_adj][c_adj] = orig_val
        
        # Try adjusting V[j_adj*p+k_adj][c_adj]
        for adj in adjustments:
            if time.time() > deadline:
                return False, None, None, None
            
            orig_val = V_copy[j_adj * p + k_adj][c_adj]
            new_val = orig_val + adj
            if abs(new_val * 2 - round(new_val * 2)) > 1e-9:
                continue
            
            V_copy[j_adj * p + k_adj][c_adj] = new_val
            
            if _verify_float_fast(U_copy, V_copy, W_copy, n, m, p):
                if _verify_exact(U_copy, V_copy, W_copy, n, m, p):
                    return True, U_copy, V_copy, W_copy
            
            V_copy[j_adj * p + k_adj][c_adj] = orig_val
        
        # Try adjusting W[k_adj*n+i_adj][c_adj]
        for adj in adjustments:
            if time.time() > deadline:
                return False, None, None, None
            
            orig_val = W_copy[k_adj * n + i_adj][c_adj]
            new_val = orig_val + adj
            if abs(new_val * 2 - round(new_val * 2)) > 1e-9:
                continue
            
            W_copy[k_adj * n + i_adj][c_adj] = new_val
            
            if _verify_float_fast(U_copy, V_copy, W_copy, n, m, p):
                if _verify_exact(U_copy, V_copy, W_copy, n, m, p):
                    return True, U_copy, V_copy, W_copy
            
            W_copy[k_adj * n + i_adj][c_adj] = orig_val
    
    return False, None, None, None

def _try_structured_reduction(n, m, p, deadline):
    """
    Try structured rank reductions by greedily removing terms.
    """
    start_time = time.time()
    
    # Start with schoolbook
    U = [[0] * (n * m * p) for _ in range(n * m)]
    V = [[0] * (n * m * p) for _ in range(m * p)]
    W = [[0] * (n * m * p) for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_rank = n * m * p
    best_solution = (U, V, W)
    
    # Try removing terms in order of "removability"
    # Terms with more shared indices are easier to remove
    term_order = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                # Count how many other terms share at least one index
                shared = 0
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            c2 = i2 * m * p + j2 * p + k2
                            if c2 == c:
                                continue
                            if i2 == i or j2 == j or k2 == k:
                                shared += 1
                term_order.append((shared, c))
    
    # Sort by most shared first (easier to remove)
    term_order.sort(reverse=True)
    
    removed_count = 0
    for shared, c_remove in term_order:
        if time.time() > deadline:
            break
        
        # Check if this term is still active
        active = False
        for idx in range(n * m):
            if U[idx][c_remove] != 0:
                active = True
                break
        if not active:
            continue
        
        success, U_new, V_new, W_new = _try_remove_term_greedy(
            U, V, W, n, m, p, c_remove, deadline
        )
        
        if success:
            U, V, W = U_new, V_new, W_new
            removed_count += 1
            best_rank = n * m * p - removed_count
            best_solution = (U, V, W)
    
    return best_solution, best_rank

def _local_search_polish(U, V, W, n, m, p, deadline):
    """
    Local search to try to reduce rank further by finding zero columns
    and attempting to merge terms.
    """
    start_time = time.time()
    rank = len(U[0])
    
    # Find columns that are all zeros (already removed)
    zero_cols = []
    for c in range(rank):
        is_zero = True
        for idx in range(n * m):
            if U[idx][c] != 0:
                is_zero = False
                break
        if is_zero:
            for idx in range(m * p):
                if V[idx][c] != 0:
                    is_zero = False
                    break
        if is_zero:
            for idx in range(p * n):
                if W[idx][c] != 0:
                    is_zero = False
                    break
        if is_zero:
            zero_cols.append(c)
    
    # Remove zero columns
    if zero_cols:
        non_zero_cols = [c for c in range(rank) if c not in zero_cols]
        U = [row[c] for row in U for c in non_zero_cols]
        # Reshape back
        new_rank = len(non_zero_cols)
        U = [U[i*new_rank:(i+1)*new_rank] for i in range(n*m)]
        V = [V[i*new_rank:(i+1)*new_rank] for i in range(m*p)]
        W = [W[i*new_rank:(i+1)*new_rank] for i in range(p*n)]
    
    # Try pairwise term merging
    # Two terms c1 and c2 can potentially be merged if they have similar patterns
    # This is complex, so we'll do a simple check: try setting one to zero
    # and see if the other can be adjusted
    
    rank = len(U[0])
    for c1 in range(rank):
        if time.time() > deadline:
            break
        
        for c2 in range(c1 + 1, rank):
            if time.time() > deadline:
                break
            
            # Check if terms c1 and c2 have overlapping support
            # i.e., they affect the same (i,j,k) entries
            overlap = False
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        prod1 = U[i*m+j][c1] * V[j*p+k][c1] * W[k*n+i][c1]
                        prod2 = U[i*m+j][c2] * V[j*p+k][c2] * W[k*n+i][c2]
                        if prod1 != 0 and prod2 != 0:
                            overlap = True
                            break
                    if overlap:
                        break
                if overlap:
                    break
            if overlap:
                # Try removing c2 and adjusting c1
                U_copy = [row[:] for row in U]
                V_copy = [row[:] for row in V]
                W_copy = [row[:] for row in W]
                
                # Zero out c2
                for idx in range(n * m):
                    U_copy[idx][c2] = 0
                for idx in range(m * p):
                    V_copy[idx][c2] = 0
                for idx in range(p * n):
                    W_copy[idx][c2] = 0
                
                # Try adjusting c1 by various amounts
                adjustments = [1, -1, 0.5, -0.5, 2, -2]
                for adj in adjustments:
                    if time.time() > deadline:
                        break
                    
                    # Try adjusting U
                    for i in range(n):
                        for j in range(m):
                            orig = U_copy[i*m+j][c1]
                            new_val = orig + adj
                            if abs(new_val * 2 - round(new_val * 2)) > 1e-9:
                                continue
                            U_copy[i*m+j][c1] = new_val
                            if _verify_float_fast(U_copy, V_copy, W_copy, n, m, p):
                                if _verify_exact(U_copy, V_copy, W_copy, n, m, p):
                                    # Remove column c2
                                    non_zero = [c for c in range(len(U_copy[0])) if c != c2]
                                    U_copy = [U_copy[i][c] for i in range(n*m) for c in non_zero]
                                    U_copy = [U_copy[i*len(non_zero):(i+1)*len(non_zero)] for i in range(n*m)]
                                    V_copy = [V_copy[i][c] for i in range(m*p) for c in non_zero]
                                    V_copy = [V_copy[i*len(non_zero):(i+1)*len(non_zero)] for i in range(m*p)]
                                    W_copy = [W_copy[i][c] for i in range(p*n) for c in non_zero]
                                    W_copy = [W_copy[i*len(non_zero):(i+1)*len(non_zero)] for i in range(p*n)]
                                    return U_copy, V_copy, W_copy
                            U_copy[i*m+j][c1] = orig
                    
                    # Try adjusting V
                    for j in range(m):
                        for k in range(p):
                            orig = V_copy[j*p+k][c1]
                            new_val = orig + adj
                            if abs(new_val * 2 - round(new_val * 2)) > 1e-9:
                                continue
                            V_copy[j*p+k][c1] = new_val
                            if _verify_float_fast(U_copy, V_copy, W_copy, n, m, p):
                                if _verify_exact(U_copy, V_copy, W_copy, n, m, p):
                                    non_zero = [c for c in range(len(U_copy[0])) if c != c2]
                                    U_copy = [U_copy[i][c] for i in range(n*m) for c in non_zero]
                                    U_copy = [U_copy[i*len(non_zero):(i+1)*len(non_zero)] for i in range(n*m)]
                                    V_copy = [V_copy[i][c] for i in range(m*p) for c in non_zero]
                                    V_copy = [V_copy[i*len(non_zero):(i+1)*len(non_zero)] for i in range(m*p)]
                                    W_copy = [W_copy[i][c] for i in range(p*n) for c in non_zero]
                                    W_copy = [W_copy[i*len(non_zero):(i+1)*len(non_zero)] for i in range(p*n)]
                                    return U_copy, V_copy, W_copy
                            V_copy[j*p+k][c1] = orig
                    
                    # Try adjusting W
                    for k in range(p):
                        for i in range(n):
                            orig = W_copy[k*n+i][c1]
                            new_val = orig + adj
                            if abs(new_val * 2 - round(new_val * 2)) > 1e-9:
                                continue
                            W_copy[k*n+i][c1] = new_val
                            if _verify_float_fast(U_copy, V_copy, W_copy, n, m, p):
                                if _verify_exact(U_copy, V_copy, W_copy, n, m, p):
                                    non_zero = [c for c in range(len(U_copy[0])) if c != c2]
                                    U_copy = [U_copy[i][c] for i in range(n*m) for c in non_zero]
                                    U_copy = [U_copy[i*len(non_zero):(i+1)*len(non_zero)] for i in range(n*m)]
                                    V_copy = [V_copy[i][c] for i in range(m*p) for c in non_zero]
                                    V_copy = [V_copy[i*len(non_zero):(i+1)*len(non_zero)] for i in range(m*p)]
                                    W_copy = [W_copy[i][c] for i in range(p*n) for c in non_zero]
                                    W_copy = [W_copy[i*len(non_zero):(i+1)*len(non_zero)] for i in range(p*n)]
                                    return U_copy, V_copy, W_copy
                            W_copy[k*n+i][c1] = orig
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0  # 110s budget, 40s safety margin
    
    # Try structured reduction
    best_solution, best_rank = _try_structured_reduction(n, m, p, deadline - 20)
    
    U, V, W = best_solution
    
    # Try local search polish
    U, V, W = _local_search_polish(U, V, W, n, m, p, deadline)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
