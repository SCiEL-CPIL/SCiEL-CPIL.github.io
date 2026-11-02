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
    if rank == 0:
        # Empty decomposition only valid for zero tensor
        return all(
            not (i == iw and j == jv and k == kw)
            for i in range(n) for j in range(m) for k in range(p)
            for iw in range(n) for jv in range(m) for kw in range(p)
            if (i*m+j, j*p+k, k*n+iw) == (i*m+j, j*p+k, k*n+i)
        )
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

def _gf2_solve(A, b):
    """
    Solve A x = b over GF(2). A is a list of lists (0/1), b is a list (0/1).
    Returns solution x (list of 0/1) or None if no unique solution.
    """
    n_rows = len(A)
    n_cols = len(A[0]) if n_rows > 0 else 0
    # Augmented matrix
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    
    # Gaussian elimination over GF(2)
    pivot_rows = []
    col = 0
    for row in range(n_rows):
        if col >= n_cols:
            break
        # Find pivot
        pivot = None
        for r in range(row, n_rows):
            if M[r][col] == 1:
                pivot = r
                break
        if pivot is None:
            continue
        M[row], M[pivot] = M[pivot], M[row]
        # Eliminate
        for r in range(n_rows):
            if r != row and M[r][col] == 1:
                for c in range(n_cols + 1):
                    M[r][c] ^= M[row][c]
        pivot_rows.append(col)
        col += 1
    
    # Check for inconsistency
    for row in range(n_rows):
        if all(M[row][c] == 0 for c in range(n_cols)) and M[row][n_cols] == 1:
            return None
    
    # Build solution
    x = [0] * n_cols
    for i, pcol in enumerate(pivot_rows):
        x[pcol] = M[i][n_cols]
    
    # Verify
    for row in range(n_rows):
        s = 0
        for c in range(n_cols):
            s ^= (A[row][c] & x[c])
        if s != b[row]:
            return None
    
    return x

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

def _find_gf2_support(rank, deadline, max_restarts=1000):
    """
    Find a binary support structure over GF(2) for rank `rank`.
    Returns (U_support, V_support, W_support) as lists of binary vectors, or None.
    
    Strategy: 
    1. Generate random binary supports for each rank-1 term
    2. Check if the GF(2) tensor equation is satisfiable
    3. If yes, lift to integer coefficients
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Target tensor: T[u,v,w] = 1 iff u=i*m+j, v=j*p+k, w=k*n+i for some i,j,k
    target = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[(i*m+j, j*p+k, k*n+i)] = 1
    
    start_time = time.time()
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        if time.time() - start_time > 100:
            break
        
        # Generate random binary supports
        U_sup = []
        V_sup = []
        W_sup = []
        
        for c in range(rank):
            u_vec = [0] * nm
            v_vec = [0] * mp
            w_vec = [0] * pn
            
            # Random sparse supports
            nnz_u = random.randint(1, min(4, nm))
            nnz_v = random.randint(1, min(4, mp))
            nnz_w = random.randint(1, min(4, pn))
            
            for idx in random.sample(range(nm), nnz_u):
                u_vec[idx] = 1
            for idx in random.sample(range(mp), nnz_v):
                v_vec[idx] = 1
            for idx in random.sample(range(pn), nnz_w):
                w_vec[idx] = 1
            
            U_sup.append(u_vec)
            V_sup.append(v_vec)
            W_sup.append(w_vec)
        
        # Build GF(2) linear system
        # For each (u,v,w) in target: sum_c U[u][c]*V[v][c]*W[w][c] = 1 (mod 2)
        # For each (u,v,w) NOT in target but potentially hit: sum_c ... = 0 (mod 2)
        
        # First, identify which (u,v,w) triples are "hit" by at least one term
        hit_triples = set()
        for c in range(rank):
            u_nonzero = [i for i in range(nm) if U_sup[c][i] == 1]
            v_nonzero = [i for i in range(mp) if V_sup[c][i] == 1]
            w_nonzero = [i for i in range(pn) if W_sup[c][i] == 1]
            for u in u_nonzero:
                for v in v_nonzero:
                    for w in w_nonzero:
                        hit_triples.add((u, v, w))
        
        # Build equations
        # Variables: coefficients a_c for each term c (over integers, but we first check GF(2))
        # Actually, over GF(2), the coefficients are just 0 or 1, and the equation is:
        # sum_{c: U[u][c]=1, V[v][c]=1, W[w][c]=1} a_c = target[u,v,w] (mod 2)
        
        # But wait - the coefficients a_c are what we're solving for. Let me reconsider.
        # We have fixed binary supports. We need to find integer coefficients such that:
        # sum_c a_c * U_sup[c][u] * V_sup[c][v] * W_sup[c][w] = target[u,v,w]
        # 
        # Over GF(2), this becomes a linear system in the a_c's (which are 0 or 1).
        # But the a_c's can be any integer, not just 0/1. However, if we find a GF(2) solution,
        # we can try to lift it.
        
        # Let's build the system over GF(2) first
        # Equation for each (u,v,w) in hit_triples:
        # sum_{c: U_sup[c][u]=1 AND V_sup[c][v]=1 AND W_sup[c][w]=1} a_c ≡ target.get((u,v,w), 0) (mod 2)
        
        equations_A = []
        equations_b = []
        
        for (u, v, w) in sorted(hit_triples):
            row = [0] * rank
            for c in range(rank):
                if U_sup[c][u] == 1 and V_sup[c][v] == 1 and W_sup[c][w] == 1:
                    row[c] = 1
            b_val = target.get((u, v, w), 0)
            equations_A.append(row)
            equations_b.append(b_val)
        
        # Also add equations for target entries that should be 1 but might not be hit
        for (u, v, w) in target.keys():
            if (u, v, w) not in hit_triples:
                # This entry needs to be 1 but no term hits it - infeasible
                break
        else:
            # Solve over GF(2)
            if equations_A:
                sol = _gf2_solve(equations_A, equations_b)
                if sol is not None:
                    # Found a GF(2) solution - lift to integers
                    # The GF(2) solution gives a_c ∈ {0,1}
                    # Try using these as integer coefficients
                    U_int = [[U_sup[c][u] * sol[c] for c in range(rank)] for u in range(nm)]
                    V_int = [[V_sup[c][v] * sol[c] for c in range(rank)] for v in range(mp)]
                    W_int = [[W_sup[c][w] * sol[c] for c in range(rank)] for w in range(pn)]
                    
                    if _verify_full(U_int, V_int, W_int):
                        return U_int, V_int, W_int
                    
                    # Try small integer coefficients around the GF(2) solution
                    # Scale or perturb
                    for scale in [1, 2, -1, -2]:
                        U_try = [[U_sup[c][u] * sol[c] * scale for c in range(rank)] for u in range(nm)]
                        V_try = [[V_sup[c][v] * sol[c] * scale for c in range(rank)] for v in range(mp)]
                        W_try = [[W_sup[c][w] * sol[c] * scale for c in range(rank)] for w in range(pn)]
                        if _verify_full(U_try, V_try, W_try):
                            return U_try, V_try, W_try
                    
                    # Try adding small perturbations to some coefficients
                    for attempt in range(20):
                        U_try = [row[:] for row in U_int]
                        V_try = [row[:] for row in V_int]
                        W_try = [row[:] for row in W_int]
                        
                        # Perturb some coefficients
                        for _ in range(random.randint(1, 3)):
                            c = random.randint(0, rank - 1)
                            if random.random() < 0.5:
                                U_try[0][c] += random.choice([-1, 1])
                            else:
                                V_try[0][c] += random.choice([-1, 1])
                        
                        if _verify_full(U_try, V_try, W_try):
                            return U_try, V_try, W_try
        
        # If we got here, this restart failed
        continue
    
    return None

def _randomized_algebraic_decompose(deadline=None):
    # DEPRECATED: Replaced by GF(2) support search
    return None
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
    # DEPRECATED: Replaced by GF(2) support search
    return None
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
    # Try to reduce rank by finding linear dependencies
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank = len(U[0]) if len(U) > 0 else 0
    
    if rank <= 1:
        return U, V, W, rank
    
    U_cols = [np.array([U[u][c] for u in range(nm)]) for c in range(rank)]
    V_cols = [np.array([V[v][c] for v in range(mp)]) for c in range(rank)]
    W_cols = [np.array([W[w][c] for w in range(pn)]) for c in range(rank)]
    
    start_time = time.time()
    
    for _ in range(20):
        if deadline and time.time() > deadline:
            break
        if time.time() - start_time > 100:
            break
        
        if len(U_cols) <= 1:
            break
        
        idx_to_remove = random.randint(0, len(U_cols) - 1)
        
        new_U_cols = [U_cols[i] for i in range(len(U_cols)) if i != idx_to_remove]
        new_V_cols = [V_cols[i] for i in range(len(V_cols)) if i != idx_to_remove]
        new_W_cols = [W_cols[i] for i in range(len(W_cols)) if i != idx_to_remove]
        
        new_rank = len(new_U_cols)
        new_U = [[float(new_U_cols[c][u]) for c in range(new_rank)] for u in range(nm)]
        new_V = [[float(new_V_cols[c][v]) for c in range(new_rank)] for v in range(mp)]
        new_W = [[float(new_W_cols[c][w]) for c in range(new_rank)] for w in range(pn)]
        
        if _verify_full(new_U, new_V, new_W):
            U, V, W = new_U, new_V, new_W
            U_cols, V_cols, W_cols = new_U_cols, new_V_cols, new_W_cols
    
    return U, V, W, len(U_cols)
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
    
    # Try GF(2) support search for various ranks
    # Known lower bounds suggest rank >= 24 for 3x4x4
    # Try ranks from 24 upward
    for rank in range(24, 40):
        if time.time() > deadline - 30:  # Leave 30s for refinement
            break
        
        result = _find_gf2_support(rank, deadline - 10)
        if result is not None:
            U, V, W = result
            if rank < best_rank:
                best_rank = rank
                best_U = U
                best_V = V
                best_W = W
                break  # Found the best possible, stop
    
    # Try iterative refinement on the best result
    if best_rank < n * m * p and time.time() < deadline - 5:
        U, V, W, rank = _iterative_refinement(best_U, best_V, best_W, deadline - 5)
        if rank < best_rank:
            best_rank = rank
            best_U = U
            best_V = V
            best_W = W
    
    return {"U": best_U, "V": best_V, "W": best_W}
