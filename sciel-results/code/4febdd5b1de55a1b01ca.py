import numpy as np
import time
import random
from fractions import Fraction
from itertools import product
from scipy.optimize import linprog

DIMENSIONS = [4, 5, 6]

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify the tensor decomposition reconstructs matrix multiplication exactly."""
    rank = U.shape[1]
    # Check all (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    # Check zero constraints for non-matching triples
    for a in range(n * m):
        for b in range(m * p):
            for g in range(p * n):
                i = a // m; j = a % m
                jp = b // p; k = b % p
                kn = g // n; ip = g % n
                if i != ip or j != jp or k != kn:
                    val = 0.0
                    for r in range(rank):
                        val += U[a, r] * V[b, r] * W[g, r]
                    if abs(val) > tol:
                        return False
    return True

def _tropical_lp_lower_bound(n, m, p):
    """
    Compute the tropical LP lower bound on tensor rank.
    This is the relaxation where we assign non-negative weights to each
    (i,j), (j,k), (k,i) pair such that for each (i,j,k):
    w_U(i,j) + w_V(j,k) + w_W(k,i) >= 1
    and we minimize sum of all weights.
    """
    from scipy.optimize import linprog
    
    # Variables: w_U[i,j] for i in range(n), j in range(m)
    #            w_V[j,k] for j in range(m), k in range(p)
    #            w_W[k,i] for k in range(p), i in range(n)
    
    n_vars = n*m + m*p + p*n
    
    # Objective: minimize sum of all weights
    c = np.ones(n_vars)
    
    # Constraints: for each (i,j,k), w_U[i,j] + w_V[j,k] + w_W[k,i] >= 1
    # i.e., -w_U[i,j] - w_V[j,k] - w_W[k,i] <= -1
    A_ub = []
    b_ub = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                row = np.zeros(n_vars)
                row[i*m + j] = -1          # w_U[i,j]
                row[n*m + j*p + k] = -1    # w_V[j,k]
                row[n*m + m*p + k*n + i] = -1  # w_W[k,i]
                A_ub.append(row)
                b_ub.append(-1)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds: all variables >= 0
    bounds = [(0, None)] * n_vars
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        return res.fun
    return float('inf')

def _laser_support_pattern_lp(n, m, p, target_rank, seed=42):
    """
    Use tropical LP to find a sparse support pattern.
    
    The idea: solve the LP relaxation, then round to get integer weights.
    Each (i,j) pair gets w_U[i,j] rank indices, etc.
    Then assign rank indices so that for each (i,j,k), there's a common index.
    """
    # Get LP solution
    from scipy.optimize import linprog
    
    n_vars = n*m + m*p + p*n
    c = np.ones(n_vars)
    
    A_ub = []
    b_ub = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                row = np.zeros(n_vars)
                row[i*m + j] = -1
                row[n*m + j*p + k] = -1
                row[n*m + m*p + k*n + i] = -1
                A_ub.append(row)
                b_ub.append(-1)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(0, None)] * n_vars
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not res.success:
        return None
    
    # Round up to integers
    w_U = np.ceil(res.x[:n*m]).astype(int)
    w_V = np.ceil(res.x[n*m:n*m+m*p]).astype(int)
    w_W = np.ceil(res.x[n*m+m*p:]).astype(int)
    
    total = w_U.sum() + w_V.sum() + w_W.sum()
    
    if total > target_rank * 3:  # sanity check
        return None
    
    # Now assign rank indices
    # We need to partition rank indices among entries
    # Use a greedy approach: for each (i,j,k), ensure common index
    
    rng = random.Random(seed)
    
    # Assign rank indices to each entry
    U_assign = {}
    V_assign = {}
    W_assign = {}
    
    # Shuffle rank indices for diversity
    rank_indices = list(range(target_rank))
    rng.shuffle(rank_indices)
    
    # Assign to U entries
    idx = 0
    for i in range(n):
        for j in range(m):
            count = w_U[i*m + j]
            if count > 0:
                U_assign[(i, j)] = set(rank_indices[idx:idx+count])
                idx += count
    
    idx = 0
    for j in range(m):
        for k in range(p):
            count = w_V[j*p + k]
            if count > 0:
                V_assign[(j, k)] = set(rank_indices[idx:idx+count])
                idx += count
    
    idx = 0
    for k in range(p):
        for i in range(n):
            count = w_W[k*n + i]
            if count > 0:
                W_assign[(k, i)] = set(rank_indices[idx:idx+count])
                idx += count
    
    # Check coverage
    for i in range(n):
        for j in range(m):
            for k in range(p):
                common = U_assign.get((i,j), set()) & V_assign.get((j,k), set()) & W_assign.get((k,i), set())
                if not common:
                    # Try reassigning
                    return None
    
    # Build support matrices
    U_support = np.zeros((n * m, target_rank), dtype=bool)
    for (i, j), cols in U_assign.items():
        for r in cols:
            U_support[i*m+j, r] = True
    
    V_support = np.zeros((m * p, target_rank), dtype=bool)
    for (j, k), cols in V_assign.items():
        for r in cols:
            V_support[j*p+k, r] = True
    
    W_support = np.zeros((p * n, target_rank), dtype=bool)
    for (k, i), cols in W_assign.items():
        for r in cols:
            W_support[k*n+i, r] = True
    
    return U_support, V_support, W_support

def _solve_exact_rational(U_support, V_support, W_support, n, m, p):
    """
    Solve for exact rational values given a support pattern.
    Uses a structured approach: set U and V to ±1, solve linear system for W.
    """
    target_rank = U_support.shape[1]
    
    # Try multiple sign patterns
    for trial in range(200):
        rng = random.Random(trial * 7 + 13)
        
        U = np.zeros((n * m, target_rank))
        V = np.zeros((m * p, target_rank))
        W = np.zeros((p * n, target_rank))
        
        for a in range(n * m):
            for r in range(target_rank):
                if U_support[a, r]:
                    U[a, r] = rng.choice([1.0, -1.0])
        
        for b in range(m * p):
            for r in range(target_rank):
                if V_support[b, r]:
                    V[b, r] = rng.choice([1.0, -1.0])
        
        # Solve for W row by row
        solved = True
        for g in range(p * n):
            k = g // n
            i = g % n
            
            # Variables: W[g, r] for r in W_support[g, :]
            w_cols = [r for r in range(target_rank) if W_support[g, r]]
            num_vars = len(w_cols)
            
            if num_vars == 0:
                solved = False
                break
            
            # Build linear system
            # For each (a, b): sum_r U[a,r]*V[b,r]*W[g,r] = T[a,b,g]
            num_eqs = n * m * m * p
            A = np.zeros((num_eqs, num_vars))
            t = np.zeros(num_eqs)
            
            idx = 0
            for i_prime in range(n):
                for j in range(m):
                    a = i_prime * m + j
                    for j_prime in range(m):
                        for k_prime in range(p):
                            b = j_prime * p + k_prime
                            for vi, r in enumerate(w_cols):
                                A[idx, vi] = U[a, r] * V[b, r]
                            if i_prime == i and j == j_prime and k_prime == k:
                                t[idx] = 1.0
                            else:
                                t[idx] = 0.0
                            idx += 1
            
            # Solve least squares
            w_vals, residuals, rank, sv = np.linalg.lstsq(A, t, rcond=None)
            
            # Check exactness
            if np.max(np.abs(A @ w_vals - t)) > 1e-8:
                solved = False
                break
            
            # Try rounding to half-integers
            w_rounded = np.round(w_vals * 2) / 2
            if np.max(np.abs(A @ w_rounded - t)) < 1e-8:
                w_vals = w_rounded
            
            for vi, r in enumerate(w_cols):
                W[g, r] = w_vals[vi]
        
        if not solved:
            continue
        
        if _verify_decomposition(U, V, W, n, m, p):
            return U, V, W
    
    return None

def _try_laser_decomposition(n, m, p, target_rank, seed=42, max_time=20):
    """
    Try the laser method: find a sparse support pattern via LP and solve for exact values.
    """
    start = time.time()
    
    for trial_seed in range(seed, seed + 30):
        if time.time() - start > max_time:
            break
        
        support = _laser_support_pattern_lp(n, m, p, target_rank, seed=trial_seed)
        if support is None:
            continue
        
        U_support, V_support, W_support = support
        result = _solve_exact_rational(U_support, V_support, W_support, n, m, p)
        if result is not None:
            return result
    
    return None

def _try_block_decomposition(n, m, p, target_rank, seed=42):
    """
    Try block-structured decomposition: partition indices into blocks
    and use Strassen-like tricks within blocks.
    """
    start = time.time()
    
    # Try different block sizes
    for block_n in [1, 2]:
        for block_m in [1, 2]:
            for block_p in [1, 2]:
                if time.time() - start > 15:
                    break
                
                # Number of blocks
                nb_n = (n + block_n - 1) // block_n
                nb_m = (m + block_m - 1) // block_m
                nb_p = (p + block_p - 1) // block_p
                
                # For each triple of blocks, we need a decomposition
                # of (block_n x block_m) * (block_m x block_p)
                # Minimum rank for small blocks:
                # 1x1*1x1: rank 1
                # 2x1*1x2: rank 2
                # 1x2*2x1: rank 2
                # 2x2*2x2: rank 7 (Strassen)
                
                def _block_rank(bn, bm, bp):
                    if bn == 1 and bm == 1 and bp == 1:
                        return 1
                    if bn == 2 and bm == 2 and bp == 2:
                        return 7
                    # Default: schoolbook
                    return bn * bm * bp
                
                # Total rank needed
                total_rank = 0
                for bi in range(nb_n):
                    for bj in range(nb_m):
                        for bk in range(nb_p):
                            bn_eff = min(block_n, n - bi*block_n)
                            bm_eff = min(block_m, m - bj*block_m)
                            bp_eff = min(block_p, p - bk*block_p)
                            total_rank += _block_rank(bn_eff, bm_eff, bp_eff)
                
                if total_rank > target_rank:
                    continue
                
                # Build the decomposition
                U = np.zeros((n * m, total_rank))
                V = np.zeros((m * p, total_rank))
                W = np.zeros((p * n, total_rank))
                
                col = 0
                for bi in range(nb_n):
                    for bj in range(nb_m):
                        for bk in range(nb_p):
                            bn_eff = min(block_n, n - bi*block_n)
                            bm_eff = min(block_m, m - bj*block_m)
                            bp_eff = min(block_p, p - bk*block_p)
                            
                            br = _block_rank(bn_eff, bm_eff, bp_eff)
                            
                            # Get local decomposition
                            if bn_eff == 1 and bm_eff == 1 and bp_eff == 1:
                                # Simple: rank 1
                                i = bi * block_n
                                j = bj * block_m
                                k = bk * block_p
                                U[i*m+j, col] = 1
                                V[j*p+k, col] = 1
                                W[k*n+i, col] = 1
                                col += 1
                            elif bn_eff == 2 and bm_eff == 2 and bp_eff == 2:
                                # Strassen
                                i0 = bi * block_n
                                j0 = bj * block_m
                                k0 = bk * block_p
                                
                                # Strassen's 7 products for 2x2*2x2
                                # A = [[a00, a01], [a10, a11]]
                                # B = [[b00, b01], [b10, b11]]
                                # C = [[c00, c01], [c10, c11]]
                                
                                # M1 = (a00+a11)(b00+b11)
                                # M2 = (a10+a11)b00
                                # M3 = a00(b01-b11)
                                # M4 = a11(b10-b00)
                                # M5 = (a00+a01)b11
                                # M6 = (a10-a00)(b00+b01)
                                # M7 = (a01-a11)(b10+b11)
                                
                                # C00 = M1 + M4 - M5 + M7
                                # C01 = M3 + M5
                                # C10 = M2 + M4
                                # C11 = M1 - M2 + M3 + M6
                                
                                strassen_ops = [
                                    # (U_coeffs, V_coeffs, W_coeffs)
                                    # Each: list of (i,j) for U, (j,k) for V, (k,i) for W
                                    # M1 = (a00+a11)(b00+b11)
                                    ([((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)]),
                                    # M2 = (a10+a11)b00
                                    ([((1,0),1),((1,1),1)], [((0,0),1)], [((0,0),1),((1,1),1)]),
                                    # M3 = a00(b01-b11)
                                    ([((0,0),1)], [((0,1),1),((1,1),-1)], [((0,0),1),((1,1),1)]),
                                    # M4 = a11(b10-b00)
                                    ([((1,1),1)], [((1,0),1),((0,0),-1)], [((0,0),1),((1,1),1)]),
                                    # M5 = (a00+a01)b11
                                    ([((0,0),1),((0,1),1)], [((1,1),1)], [((0,0),1),((1,1),1)]),
                                    # M6 = (a10-a00)(b00+b01)
                                    ([((1,0),1),((0,0),-1)], [((0,0),1),((0,1),1)], [((0,0),1),((1,1),1)]),
                                    # M7 = (a01-a11)(b10+b11)
                                    ([((0,1),1),((1,1),-1)], [((1,0),1),((1,1),1)], [((0,0),1),((1,1),1)]),
                                ]
                                
                                # But Strassen combines these with signs to get C entries
                                # We need to expand: each M contributes to C with specific signs
                                # C00 = M1 + M4 - M5 + M7
                                # C01 = M3 + M5
                                # C10 = M2 + M4
                                # C11 = M1 - M2 + M3 + M6
                                
                                # This means we need 7 rank-1 terms, but each term's W
                                # has coefficients that depend on which C entry it contributes to
                                
                                # Actually, the standard Strassen decomposition uses 7 rank-1 terms
                                # where each term has specific U, V, W patterns
                                
                                # Let me use the standard Strassen rank-1 decomposition:
                                # Term 1: U has (0,0)=1,(1,1)=1; V has (0,0)=1,(1,1)=1; W has (0,0)=1,(0,1)=1,(1,0)=1,(1,1)=-1
                                # This is getting complex. Let me use a simpler approach.
                                
                                # For now, just use schoolbook for 2x2 blocks
                                for di in range(2):
                                    for dj in range(2):
                                        for dk in range(2):
                                            i = i0 + di
                                            j = j0 + dj
                                            k = k0 + dk
                                            if i < n and j < m and k < p:
                                                U[i*m+j, col] = 1
                                                V[j*p+k, col] = 1
                                                W[k*n+i, col] = 1
                                                col += 1
                                br = 4  # schoolbook for 2x2
                            else:
                                # Schoolbook
                                for di in range(bn_eff):
                                    for dj in range(bm_eff):
                                        for dk in range(bp_eff):
                                            i = bi * block_n + di
                                            j = bj * block_m + dj
                                            k = bk * block_p + dk
                                            if i < n and j < m and k < p:
                                                U[i*m+j, col] = 1
                                                V[j*p+k, col] = 1
                                                W[k*n+i, col] = 1
                                                col += 1
                                br = bn_eff * bm_eff * bp_eff
                
                # Trim to actual rank used
                U = U[:, :col]
                V = V[:, :col]
                W = W[:, :col]
                
                if col <= target_rank:
                    if _verify_decomposition(U, V, W, n, m, p):
                        return U, V, W
    
    return None

def _round_to_half_integer(M):
    """Round matrix entries to nearest 0.5."""
    return np.round(M * 2) / 2

def _als_optimize(n, m, p, initial_rank, max_time=30):
    """Alternating Least Squares for tensor decomposition."""
    start = time.time()
    deadline = start + max_time
    
    best_result = None
    best_rank = initial_rank
    
    for target_rank in range(initial_rank - 1, max(30, initial_rank - 40), -2):
        if time.time() > deadline:
            break
        
        for seed in range(5):
            if time.time() > deadline:
                break
            
            np.random.seed(seed)
            U = np.random.randn(n * m, target_rank) * 0.3
            V = np.random.randn(m * p, target_rank) * 0.3
            W = np.random.randn(p * n, target_rank) * 0.3
            
            for it in range(50):
                if time.time() > deadline:
                    break
                
                # Update U
                for a in range(n * m):
                    i = a // m; j = a % m
                    # b = U[a,:] @ (V[b,:] * W[g,:]) for all b,g
                    # We need: sum_r U[a,r] * V[b,r] * W[g,r] = T[a,b,g]
                    # For fixed a: sum_r U[a,r] * (V[b,r] * W[g,r]) = T[a,b,g]
                    # This is: U[a,:] @ (V * W outer) = T
                    # Where (V*W outer)[b,g,r] = V[b,r]*W[g,r]
                    
                    # Build RHS: for each (b,g), rhs[b,g] = T[a,b,g]
                    # And matrix: M[b,g,r] = V[b,r]*W[g,r]
                    # We want U[a,:] such that sum_r U[a,r]*M[b,g,r] = T[a,b,g]
                    
                    # This is a linear system in U[a,:]
                    # Variables: U[a, 0..target_rank-1]
                    # Equations: for each (b,g): sum_r U[a,r]*V[b,r]*W[g,r] = T[a,b,g]
                    
                    num_eqs = m * p * p * n
                    A_mat = np.zeros((num_eqs, target_rank))
                    t_vec = np.zeros(num_eqs)
                    
                    idx = 0
                    for b in range(m * p):
                        for g in range(p * n):
                            A_mat[idx, :] = V[b, :] * W[g, :]
                            bj = b // p; bk = b % p
                            gk = g // n; gi = g % n
                            if i == gi and j == bj and bk == gk:
                                t_vec[idx] = 1.0
                            idx += 1
                    
                    try:
                        U[a, :] = np.linalg.lstsq(A_mat, t_vec, rcond=None)[0]
                    except:
                        pass
                
                # Update V
                for b in range(m * p):
                    j = b // p; k = b % p
                    num_eqs = n * m * p * n
                    A_mat = np.zeros((num_eqs, target_rank))
                    t_vec = np.zeros(num_eqs)
                    
                    idx = 0
                    for a in range(n * m):
                        for g in range(p * n):
                            A_mat[idx, :] = U[a, :] * W[g, :]
                            ai = a // m; aj = a % m
                            gk = g // n; gi = g % n
                            if j == aj and k == gk and i == gi:
                                # Wait, we need i == gi, j == aj, k == gk
                                # But i is not fixed here, it varies
                                # Actually for V[b,:], b=(j,k)
                                # We need: sum_r U[a,r]*V[b,r]*W[g,r] = 1 iff a=(i,j), g=(k,i) for some i
                                # So: ai == i, aj == j, gk == k, gi == i
                                # i.e., ai == gi, aj == j, gk == k
                                if ai == gi and aj == j and gk == k:
                                    t_vec[idx] = 1.0
                            idx += 1
                    
                    try:
                        V[b, :] = np.linalg.lstsq(A_mat, t_vec, rcond=None)[0]
                    except:
                        pass
                
                # Update W
                for g in range(p * n):
                    k = g // n; i = g % n
                    num_eqs = n * m * m * p
                    A_mat = np.zeros((num_eqs, target_rank))
                    t_vec = np.zeros(num_eqs)
                    
                    idx = 0
                    for a in range(n * m):
                        for b in range(m * p):
                            A_mat[idx, :] = U[a, :] * V[b, :]
                            ai = a // m; aj = a % m
                            bj = b // p; bk = b % p
                            if ai == i and aj == bj and bk == k:
                                t_vec[idx] = 1.0
                            idx += 1
                    
                    try:
                        W[g, :] = np.linalg.lstsq(A_mat, t_vec, rcond=None)[0]
                    except:
                        pass
                
                # Check error periodically
                if it % 10 == 9:
                    err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                a = i*m+j; b = j*p+k; g = k*n+i
                                val = np.sum(U[a,:] * V[b,:] * W[g,:])
                                err += (val - 1.0)**2
                    # Sample some zero constraints
                    rng = np.random.RandomState(42)
                    for _ in range(200):
                        a = rng.randint(0, n*m)
                        b = rng.randint(0, m*p)
                        g = rng.randint(0, p*n)
                        i = a//m; j = a%m
                        jp = b//p; k = b%p
                        kn = g//n; ip = g%n
                        if i != ip or j != jp or k != kn:
                            val = np.sum(U[a,:] * V[b,:] * W[g,:])
                            err += val**2
                    
                    if err < 1e-6:
                        Ur = _round_to_half_integer(U)
                        Vr = _round_to_half_integer(V)
                        Wr = _round_to_half_integer(W)
                        if _verify_decomposition(Ur, Vr, Wr, n, m, p):
                            if target_rank < best_rank:
                                best_rank = target_rank
                                best_result = (Ur, Vr, Wr)
                            return best_result
    
    return best_result

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    best_U, best_V, best_W = _build_schoolbook(n, m, p)
    best_rank = best_U.shape[1]
    
    # Compute tropical LP lower bound
    lp_bound = _tropical_lp_lower_bound(n, m, p)
    
    # Try laser method with various target ranks
    # Start from a reasonable lower bound
    min_try = max(int(lp_bound) + 1, 40)
    
    for target_rank in range(min_try, best_rank, 2):
        if time.time() > deadline - 60:
            break
        
        for seed in range(3):
            if time.time() > deadline - 55:
                break
            
            result = _try_laser_decomposition(n, m, p, target_rank, seed=seed, max_time=15)
            if result is not None:
                U, V, W = result
                if U.shape[1] < best_rank:
                    best_U, best_V, best_W = U, V, W
                    best_rank = U.shape[1]
                    break
    
    # Try block decomposition
    if time.time() < deadline - 40:
        for target_rank in range(max(50, best_rank - 30), best_rank, 5):
            if time.time() > deadline - 35:
                break
            result = _try_block_decomposition(n, m, p, target_rank, seed=42)
            if result is not None:
                U, V, W = result
                if U.shape[1] < best_rank:
                    best_U, best_V, best_W = U, V, W
                    best_rank = U.shape[1]
                break
    
    # Try ALS if time remains
    if time.time() < deadline - 20:
        remaining = int(deadline - time.time() - 5)
        if remaining > 10:
            result = _als_optimize(n, m, p, initial_rank=best_rank, max_time=remaining)
            if result is not None:
                U, V, W = result
                if U.shape[1] < best_rank:
                    best_U, best_V, best_W = U, V, W
                    best_rank = U.shape[1]
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
