import numpy as np
import time
import random
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 4, 4]

def _verify_full(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if len(U) > 0 else 0
    for u in range(n * m):
        i = u // m
        j = u % m
        for v in range(m * p):
            j_v = v // p
            k = v % p
            for w in range(p * n):
                k_w = w // n
                i_w = w % n
                target = 1.0 if (i == i_w and j == j_v and k == k_w) else 0.0
                s = 0.0
                for c in range(rank):
                    s += U[u][c] * V[v][c] * W[w][c]
                if abs(s - target) > 1e-9:
                    return False
    return True

def _schoolbook():
    """Return the trivial schoolbook decomposition with rank = n*m*p."""
    n, m, p = DIMENSIONS
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

def _randomized_algebraic_decompose(deadline=None):
    """
    Randomized algebraic approach to find low-rank tensor decomposition.
    
    Strategy: Generate random rank-3 slices and use Gaussian elimination
    to iteratively build a basis for the tensor space, discovering a
    low-rank decomposition from linear dependencies.
    
    The multiplication tensor T has entries T[i,j,k] = 1 if (i,j,k) is valid,
    else 0. We seek U, V, W such that T = sum_c u_c ⊗ v_c ⊗ w_c.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Target tensor entries
    target_entries = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                target_entries[(u, v, w)] = 1.0
    
    # We'll build the decomposition incrementally
    # U_cols: list of column vectors (each of length nm)
    # V_cols: list of column vectors (each of length mp)
    # W_cols: list of column vectors (each of length pn)
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Residual tensor
    residual = dict(target_entries)
    
    max_rank = 40  # Upper bound to try
    max_iter = 2000  # Maximum iterations
    
    start_time = time.time()
    
    for rank_step in range(max_rank):
        if deadline and time.time() > deadline:
            break
        if time.time() - start_time > 140:  # Safety margin
            break
        
        if not residual:
            break
        
        # Find a good rank-1 term to add
        # Use random sampling to find a term that covers many residual entries
        
        best_u = None
        best_v = None
        best_w = None
        best_coverage = 0
        best_residual_reduction = 0
        
        # Try multiple random strategies
        for attempt in range(50):
            if deadline and time.time() > deadline:
                break
            if time.time() - start_time > 140:
                break
            
            # Randomly sample a subset of residual entries to guide the search
            sample_size = min(10, len(residual))
            sample_keys = random.sample(list(residual.keys()), sample_size)
            
            # Try to find u, v, w that match these entries
            # Initialize with small random values
            u_vec = np.zeros(nm)
            v_vec = np.zeros(mp)
            w_vec = np.zeros(pn)
            
            # For each sampled entry, try to set values
            for (su, sv, sw) in sample_keys:
                if random.random() < 0.5:
                    u_vec[su] = random.choice([-1, 1])
                    v_vec[sv] = random.choice([-1, 1])
                    w_vec[sw] = random.choice([-1, 1])
            
            # Compute how well this term matches the residual
            coverage = 0
            reduction = 0
            for (u, v, w), val in residual.items():
                term_val = u_vec[u] * v_vec[v] * w_vec[w]
                if abs(term_val) > 1e-10:
                    coverage += 1
                    # Estimate reduction: how much of val we can cancel
                    if val * term_val > 0:
                        reduction += min(abs(val), abs(term_val))
            
            if reduction > best_residual_reduction:
                best_residual_reduction = reduction
                best_u = u_vec
                best_v = v_vec
                best_w = w_vec
                best_coverage = coverage
        
        if best_u is None or best_residual_reduction < 1e-10:
            break
        
        # Add this term to the decomposition
        U_cols.append(best_u)
        V_cols.append(best_v)
        W_cols.append(best_w)
        
        # Update residual
        for (u, v, w) in list(residual.keys()):
            term_val = best_u[u] * best_v[v] * best_w[w]
            residual[(u, v, w)] -= term_val
            if abs(residual[(u, v, w)]) < 1e-10:
                del residual[(u, v, w)]
        
        # Check if we've covered all entries
        if not residual:
            break
    
    # Check if we have a complete decomposition
    if not residual and U_cols:
        # Convert to the required format
        rank = len(U_cols)
        U = [[float(U_cols[c][u]) for c in range(rank)] for u in range(nm)]
        V = [[float(V_cols[c][v]) for c in range(rank)] for v in range(mp)]
        W = [[float(W_cols[c][w]) for c in range(rank)] for w in range(pn)]
        
        # Verify the decomposition
        if _verify_full(U, V, W):
            return U, V, W, rank
    
    return None

def _structured_block_decompose(deadline=None):
    """
    Try a structured block decomposition approach.
    Split the 4x4 matrix into 2x2 blocks and use Strassen-like ideas.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Target tensor entries
    target_entries = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                target_entries[(u, v, w)] = 1.0
    
    # Try to find a decomposition using block structure
    # This is a heuristic approach - try various block patterns
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    residual = dict(target_entries)
    
    start_time = time.time()
    max_iter = 500
    
    for rank_step in range(40):
        if deadline and time.time() > deadline:
            break
        if time.time() - start_time > 120:
            break
        
        if not residual:
            break
        
        # Try to find a term that covers a structured pattern
        best_u = None
        best_v = None
        best_w = None
        best_reduction = 0
        
        # Try different structured patterns
        for pattern_type in range(10):
            if deadline and time.time() > deadline:
                break
            if time.time() - start_time > 120:
                break
            
            u_vec = np.zeros(nm)
            v_vec = np.zeros(mp)
            w_vec = np.zeros(pn)
            
            # Pattern 1: Single entry
            if pattern_type == 0:
                if residual:
                    su, sv, sw = random.choice(list(residual.keys()))
                    u_vec[su] = 1
                    v_vec[sv] = 1
                    w_vec[sw] = 1
            
            # Pattern 2: Two entries sharing u
            elif pattern_type == 1:
                if len(residual) >= 2:
                    keys = list(residual.keys())
                    # Find two entries with same u
                    for idx1 in range(min(20, len(keys))):
                        for idx2 in range(idx1 + 1, min(20, len(keys))):
                            if keys[idx1][0] == keys[idx2][0]:
                                su, sv1, sw1 = keys[idx1]
                                _, sv2, sw2 = keys[idx2]
                                u_vec[su] = 1
                                v_vec[sv1] = 1
                                v_vec[sv2] = 1
                                w_vec[sw1] = 1
                                w_vec[sw2] = 1
                                break
                        else:
                            continue
                        break
            
            # Pattern 3: Random sparse
            else:
                num_nonzero = random.randint(1, 3)
                for _ in range(num_nonzero):
                    if residual:
                        su, sv, sw = random.choice(list(residual.keys()))
                        u_vec[su] = random.choice([-1, 1])
                        v_vec[sv] = random.choice([-1, 1])
                        w_vec[sw] = random.choice([-1, 1])
            
            # Compute reduction
            reduction = 0
            for (u, v, w), val in residual.items():
                term_val = u_vec[u] * v_vec[v] * w_vec[w]
                if abs(term_val) > 1e-10:
                    if val * term_val > 0:
                        reduction += min(abs(val), abs(term_val))
            
            if reduction > best_reduction:
                best_reduction = reduction
                best_u = u_vec
                best_v = v_vec
                best_w = w_vec
        
        if best_u is None or best_reduction < 1e-10:
            break
        
        U_cols.append(best_u)
        V_cols.append(best_v)
        W_cols.append(best_w)
        
        # Update residual
        for (u, v, w) in list(residual.keys()):
            term_val = best_u[u] * best_v[v] * best_w[w]
            residual[(u, v, w)] -= term_val
            if abs(residual[(u, v, w)]) < 1e-10:
                del residual[(u, v, w)]
    
    if not residual and U_cols:
        rank = len(U_cols)
        U = [[float(U_cols[c][u]) for c in range(rank)] for u in range(nm)]
        V = [[float(V_cols[c][v]) for c in range(rank)] for v in range(mp)]
        W = [[float(W_cols[c][w]) for c in range(rank)] for w in range(pn)]
        
        if _verify_full(U, V, W):
            return U, V, W, rank
    
    return None

def _iterative_refinement(U, V, W, deadline=None):
    """
    Try to refine an existing decomposition by finding linear dependencies
    and reducing the rank.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank = len(U[0]) if len(U) > 0 else 0
    
    if rank <= 1:
        return U, V, W, rank
    
    # Build the tensor and check for linear dependencies
    # T = sum_c u_c ⊗ v_c ⊗ w_c
    
    # Try to find a subset of columns that spans the same space
    # This is a heuristic - try removing columns one by one
    
    U_cols = [np.array(U[u][c] for u in range(nm)) for c in range(rank)]
    V_cols = [np.array(V[v][c] for v in range(mp)) for c in range(rank)]
    W_cols = [np.array(W[w][c] for w in range(pn)) for c in range(rank)]
    
    start_time = time.time()
    
    for _ in range(20):  # Try up to 20 removal attempts
        if deadline and time.time() > deadline:
            break
        if time.time() - start_time > 100:
            break
        
        if len(U_cols) <= 1:
            break
        
        # Try removing a random column
        idx_to_remove = random.randint(0, len(U_cols) - 1)
        
        # Check if the remaining columns still span the tensor
        new_U_cols = [U_cols[i] for i in range(len(U_cols)) if i != idx_to_remove]
        new_V_cols = [V_cols[i] for i in range(len(V_cols)) if i != idx_to_remove]
        new_W_cols = [W_cols[i] for i in range(len(W_cols)) if i != idx_to_remove]
        
        # Verify
        new_rank = len(new_U_cols)
        new_U = [[float(new_U_cols[c][u]) for c in range(new_rank)] for u in range(nm)]
        new_V = [[float(new_V_cols[c][v]) for c in range(new_rank)] for v in range(mp)]
        new_W = [[float(new_W_cols[c][w]) for c in range(new_rank)] for w in range(pn)]
        
        if _verify_full(new_U, new_V, new_W):
            U, V, W = new_U, new_V, new_W
            U_cols, V_cols, W_cols = new_U_cols, new_V_cols, new_W_cols
    
    return U, V, W, len(U_cols)

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave 10s margin
    
    # Schoolbook fallback
    U_school, V_school, W_school = _schoolbook()
    best_rank = n * m * p
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try randomized algebraic approach
    result = _randomized_algebraic_decompose(deadline)
    if result is not None:
        U, V, W, rank = result
        if rank < best_rank:
            best_rank = rank
            best_U = U
            best_V = V
            best_W = W
    
    # Try structured block decomposition
    result = _structured_block_decompose(deadline)
    if result is not None:
        U, V, W, rank = result
        if rank < best_rank:
            best_rank = rank
            best_U = U
            best_V = V
            best_W = W
    
    # Try iterative refinement on the best result
    if best_rank < n * m * p and time.time() < deadline:
        U, V, W, rank = _iterative_refinement(best_U, best_V, best_W, deadline)
        if rank < best_rank:
            best_rank = rank
            best_U = U
            best_V = V
            best_W = W
    
    return {"U": best_U, "V": best_V, "W": best_W}
