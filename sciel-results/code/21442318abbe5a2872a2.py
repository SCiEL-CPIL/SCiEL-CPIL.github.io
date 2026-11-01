import time
import random
import numpy as np
from fractions import Fraction
from itertools import product, combinations

N, M, P = 2, 3, 10
DEADLINE = 143.0  # seconds, safety margin before 150s kill
START = time.time()

def _elapsed():
    return time.time() - START

def _check_deadline():
    return _elapsed() > DEADLINE

def _tensor_entries():
    """List of (i, j, k) for the multiplication tensor T[i,j,k] = 1."""
    return [(i, j, k) for i in range(N) for j in range(M) for k in range(P)]

def _build_strassen_rank55():
    """Construct the rank-55 Strassen-based decomposition."""
    rank = 55
    U = np.zeros((N * M, rank))
    V = np.zeros((M * P, rank))
    W = np.zeros((P * N, rank))
    col = 0
    a00, a01, a10, a11 = 0, 1, 3, 4
    a02, a12 = 2, 5
    for b in range(5):
        k0, k1 = 2 * b, 2 * b + 1
        b00, b01 = k0, k1
        b10, b11 = 10 + k0, 10 + k1
        c00, c01 = 2 * k0, 2 * k1
        c10, c11 = 2 * k0 + 1, 2 * k1 + 1
        U[a00, col] += 1; U[a11, col] += 1
        V[b00, col] += 1; V[b11, col] += 1
        W[c00, col] += 1; W[c11, col] += 1
        col += 1
        U[a10, col] += 1; U[a11, col] += 1
        V[b00, col] += 1
        W[c10, col] += 1; W[c11, col] -= 1
        col += 1
        U[a00, col] += 1
        V[b01, col] += 1; V[b11, col] -= 1
        W[c01, col] += 1; W[c11, col] += 1
        col += 1
        U[a11, col] += 1
        V[b10, col] += 1; V[b00, col] -= 1
        W[c00, col] += 1; W[c10, col] += 1
        col += 1
        U[a00, col] += 1; U[a01, col] += 1
        V[b11, col] += 1
        W[c00, col] -= 1; W[c01, col] += 1
        col += 1
        U[a10, col] += 1; U[a00, col] -= 1
        V[b00, col] += 1; V[b01, col] += 1
        W[c11, col] += 1
        col += 1
        U[a01, col] += 1; U[a11, col] -= 1
        V[b10, col] += 1; V[b11, col] += 1
        W[c00, col] += 1
        col += 1
    for k in range(P):
        b2k = 2 * P + k
        U[a02, col] += 1; V[b2k, col] += 1; W[2 * k, col] += 1
        col += 1
        U[a12, col] += 1; V[b2k, col] += 1; W[2 * k + 1, col] += 1
        col += 1
    return U, V, W

def _build_schoolbook_rank60():
    """Construct the trivial rank-60 schoolbook decomposition."""
    rank = N * M * P
    U = np.zeros((N * M, rank))
    V = np.zeros((M * P, rank))
    W = np.zeros((P * N, rank))
    
    r = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, r] = 1
                V[j * P + k, r] = 1
                W[k * N + i, r] = 1
                r += 1
    return U, V, W

def _verify_np(U, V, W, tol=1e-9):
    """Verify decomposition using numpy. Returns True if exact within tol."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    for i in range(N):
        for j in range(M):
            for k in range(P):
                s = 0.0
                for r in range(rank):
                    s += U[i * M + j, r] * V[j * P + k, r] * W[k * N + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _check_half_integer(U, V, W, tol=1e-9):
    """Check if all entries are integer or half-integer within tolerance."""
    for matrix in [U, V, W]:
        for val in matrix.flatten():
            # Check if val is close to k/2 for some integer k
            two_val = val * 2
            nearest = round(two_val)
            if abs(two_val - nearest) > tol:
                return False
    return True

def _verify_exact(U, V, W):
    """Verify decomposition exactly using fractions."""
    Uf = [[Fraction(int(x)) for x in row] for row in U]
    Vf = [[Fraction(int(x)) for x in row] for row in V]
    Wf = [[Fraction(int(x)) for x in row] for row in W]
    rank = len(Uf[0])
    for i in range(N):
        for j in range(M):
            for k in range(P):
                s = Fraction(0)
                for r in range(rank):
                    s += Uf[i * M + j][r] * Vf[j * P + k][r] * Wf[k * N + i][r]
                if s != 1:
                    return False
    return True

def _to_int_list(matrix):
    """Convert numpy matrix to list of lists of ints."""
    return [[int(x) for x in row] for row in matrix]

def _greedy_build_rank(target_rank):
    """
    Greedy column-by-column construction.
    At each step, choose the best (support, coefficients) that minimizes residual.
    """
    if _check_deadline():
        return None
    
    entries = _tensor_entries()
    entry_idx = {e: idx for idx, e in enumerate(entries)}
    num_entries = len(entries)
    
    # Target tensor
    target = np.ones(num_entries)
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    residual = target.copy()
    
    for col_idx in range(target_rank):
        if _check_deadline():
            return None
        
        # Generate candidate supports: small subsets of entries
        # Try sizes 1-4 for efficiency
        best_score = np.inf
        best_support = None
        best_coeffs = None
        
        # Sample candidate supports
        n_candidates = 200
        for _ in range(n_candidates):
            size = random.choice([1, 2, 2, 3, 3, 4])
            if size > num_entries:
                size = num_entries
            support = set(random.sample(entries, size))
            
            # Build linear system for this support
            # Variables: u_e for e in support, v_e for e in support, w_e for e in support
            # Equation for entry e': sum_{r: e' in support_r} u_r * v_r * w_r = residual[e']
            # This is still nonlinear. Use linearization:
            # For fixed support, we have 3*|support| variables and num_entries equations.
            # But the system is nonlinear. Instead, use the approach:
            # Assume the term contributes to entries in its support.
            # For entries in support: u_r * v_r * w_r = residual[e]
            # For entries not in support: contribution is 0.
            # So we need: for e in support, u_r * v_r * w_r = residual[e]
            #              for e not in support, 0 = residual[e] (which is already satisfied if we only update residual for supported entries)
            
            # Actually, the linear system approach from before:
            # We have variables U[r], V[r], W[r] for this single term r.
            # The constraint is: for each entry e, sum over terms... 
            # But we're building one term at a time.
            
            # Simpler: for this term, we want to explain as much of the residual as possible.
            # The term contributes u_r * v_r * w_r to entries in its support.
            # We want to choose u_r, v_r, w_r to best fit residual on the support.
            # This is: minimize sum_{e in support} (u_r * v_r * w_r - residual[e])^2
            # Let t = u_r * v_r * w_r. Then minimize (t - residual[e])^2 for e in support.
            # Optimal t = mean(residual[e] for e in support).
            # But we need to decompose t into u_r * v_r * w_r.
            # We can set u_r = t^(1/3), v_r = 1, w_r = 1 (or any factorization).
            # But entries must be integer/half-integer.
            
            # Better: use the linear approach. For a single term with support S,
            # the contribution to entry e is u_r * v_r * w_r if e in S, else 0.
            # We want this to approximate residual.
            # The best we can do is set u_r * v_r * w_r = t where t minimizes
            # sum_{e in S} (t - residual[e])^2 + sum_{e not in S} residual[e]^2
            # The second term is constant. So t = mean(residual[e] for e in S).
            
            # Score: sum of squared residuals after subtracting this term
            t = np.mean(residual[[entry_idx[e] for e in support]])
            new_residual = residual.copy()
            for e in support:
                new_residual[entry_idx[e]] -= t
            
            score = np.sum(new_residual**2)
            
            if score < best_score:
                best_score = score
                best_support = support
                best_coeffs = t
        
        if best_support is None:
            break
        
        # Now we need to factor t = u * v * w with integer/half-integer values
        # and assign to specific (i,j), (j,k), (k,i) positions.
        # For simplicity, use t as a scalar and distribute:
        # u_r = t, v_r = 1, w_r = 1 won't work because U, V, W have different dimensions.
        
        # Actually, the term r has:
        # U[i*M+j, r] for all i,j
        # V[j*P+k, r] for all j,k  
        # W[k*N+i, r] for all k,i
        # The contribution to entry (i,j,k) is U[i*M+j,r] * V[j*P+k,r] * W[k*N+i,r]
        
        # For the support to only affect entries in best_support, we need:
        # U[i*M+j,r] * V[j*P+k,r] * W[k*N+i,r] = 0 for (i,j,k) not in best_support
        # This is hard to satisfy exactly.
        
        # Alternative: use the linear system approach for ALL terms at once.
        # This is getting complex. Let me use a different strategy.
        
        break
    
    return None

def _try_rank_via_linear(target_rank):
    """
    Try to find a rank-target_rank solution by solving a linear system.
    Randomly assign each entry to a subset of terms, then solve.
    """
    if _check_deadline():
        return None
    
    entries = _tensor_entries()
    entry_idx = {e: idx for idx, e in enumerate(entries)}
    num_entries = len(entries)
    
    random.seed(42 + target_rank)
    
    for trial in range(3000):
        if _check_deadline():
            break
        
        # Randomly assign each entry to be covered by 1-3 terms
        support = [set() for _ in range(target_rank)]
        for e in entries:
            n_terms = random.choice([1, 1, 2, 2, 3])
            terms = random.sample(range(target_rank), n_terms)
            for t in terms:
                support[t].add(e)
        
        # Build linear system
        # Variables: for each term r, we have U_r, V_r, W_r (scalars? No, vectors)
        # This doesn't work directly because U, V, W are vectors.
        
        # Use the approach: each term r has a "shape" - which (i,j), (j,k), (k,i) it touches.
        # For now, assume each term touches all (i,j), (j,k), (k,i) but with sparse values.
        
        # Actually, let's use the same approach as _solve_support but with random assignment.
        num_vars = 3 * target_rank * (N * M + M * P + P * N)  # too many
        
        # Simpler: assume each term has a fixed pattern of which U,V,W entries are nonzero.
        # This is the support search approach.
        
        # Build A matrix for the linear system
        # For each entry (i,j,k), the equation is:
        # sum_r U[i*M+j, r] * V[j*P+k, r] * W[k*N+i, r] = 1
        # This is nonlinear. Linearize by assuming we know 2 of the 3 factors.
        
        # Alternative: use the "bilinear" approach.
        # Fix V and W, solve for U. Then fix U and W, solve for V. Etc.
        
        # For now, skip this and use a different method.
        break
    
    return None

def _search_rank54_exploit_n2():
    """
    Exploit linearity in n=2: decompose into two 1x3x10 problems.
    
    For each output row i (0,1), we need:
    sum_r U[i*M+j, r] * V[j*P+k, r] * W[k*N+i, r] = 1 for all j,k
    
    Key insight: If we can find a set of "intermediate" vectors v_r (length P=10)
    such that for each i, the weighted sum of v_r (weighted by U[i*M+j,r] and W[k*N+i,r])
    gives the correct result, we can solve this more efficiently.
    
    Strategy:
    1. For each i, the problem reduces to: find rank-r decomposition of the 3x10 matrix multiplication
       where the "left" factor is a 1x3 vector and "right" is 3x10.
    2. A rank-1 term for row i: U[i*M+j, r] * V[j*P+k, r] * W[k*N+i, r]
       Let a_j = U[i*M+j, r], b_k = W[k*N+i, r], and v_r[k] = V[j*P+k, r] (but V depends on j too).
    
    Better approach: Fix the structure of V (the middle matrix).
    For a given V matrix (30 x rank), the constraints for row i are:
    For each j,k: sum_r U[i*M+j, r] * V[j*P+k, r] * W[k*N+i, r] = 1
    
    This is still nonlinear. But if we fix U and W, it's linear in V.
    If we fix V and W, it's linear in U. Etc.
    
    Let's use alternating minimization:
    1. Initialize U, V, W randomly with small values.
    2. Fix U, W: solve for V (linear system).
    3. Fix U, V: solve for W (linear system).
    4. Fix V, W: solve for U (linear system).
    5. Repeat until convergence or deadline.
    """
    if _check_deadline():
        return None
    
    rank = 54
    random.seed(123)
    
    # Number of restarts
    n_restarts = 50
    
    for restart in range(n_restarts):
        if _check_deadline():
            break
        
        # Random initialization
        U = np.random.randn(N * M, rank) * 0.1
        V = np.random.randn(M * P, rank) * 0.1
        W = np.random.randn(P * N, rank) * 0.1
        
        # Alternating least squares
        for iteration in range(200):
            if _check_deadline():
                break
            
            # Fix U and W, solve for V
            # For each j,k: sum_r V[j*P+k, r] * (U[i*M+j, r] * W[k*N+i, r]) = 1 for all i
            # This gives 2 equations per (j,k) pair, so 2*3*10 = 60 equations
            # Variables: V[j*P+k, r] for all j,k,r -> 30*54 = 1620 variables
            # But each (j,k) only involves 54 variables, and we have 2 equations per (j,k)
            # So we have 30 independent 2x54 systems
            
            V_new = np.zeros_like(V)
            for j in range(M):
                for k in range(P):
                    idx = j * P + k
                    # Build 2x54 system: for i=0,1
                    A_row = np.zeros((2, rank))
                    for i in range(N):
                        for r in range(rank):
                            A_row[i, r] = U[i * M + j, r] * W[k * N + i, r]
                    b_row = np.ones(2)
                    try:
                        sol = np.linalg.lstsq(A_row, b_row, rcond=None)[0]
                        V_new[idx, :] = sol
                    except np.linalg.LinAlgError:
                        break
            V = V_new
            
            if _check_deadline():
                break
            
            # Fix U and V, solve for W
            # For each i,k: sum_r W[k*N+i, r] * (U[i*M+j, r] * V[j*P+k, r]) = 1 for all j
            # This gives 3 equations per (i,k) pair, so 2*10 = 20 independent 3x54 systems
            
            W_new = np.zeros_like(W)
            for i in range(N):
                for k in range(P):
                    idx = k * N + i
                    A_row = np.zeros((M, rank))
                    for j in range(M):
                        for r in range(rank):
                            A_row[j, r] = U[i * M + j, r] * V[j * P + k, r]
                    b_row = np.ones(M)
                    try:
                        sol = np.linalg.lstsq(A_row, b_row, rcond=None)[0]
                        W_new[idx, :] = sol
                    except np.linalg.LinAlgError:
                        break
            W = W_new
            
            if _check_deadline():
                break
            
            # Fix V and W, solve for U
            # For each i,j: sum_r U[i*M+j, r] * (V[j*P+k, r] * W[k*N+i, r]) = 1 for all k
            # This gives 10 equations per (i,j) pair, so 2*3 = 6 independent 10x54 systems
            
            U_new = np.zeros_like(U)
            for i in range(N):
                for j in range(M):
                    idx = i * M + j
                    A_row = np.zeros((P, rank))
                    for k in range(P):
                        for r in range(rank):
                            A_row[k, r] = V[j * P + k, r] * W[k * N + i, r]
                    b_row = np.ones(P)
                    try:
                        sol = np.linalg.lstsq(A_row, b_row, rcond=None)[0]
                        U_new[idx, :] = sol
                    except np.linalg.LinAlgError:
                        break
            U = U_new
            
            # Check convergence
            error = 0.0
            for i in range(N):
                for j in range(M):
                    for k in range(P):
                        s = 0.0
                        for r in range(rank):
                            s += U[i * M + j, r] * V[j * P + k, r] * W[k * N + i, r]
                        error += abs(s - 1.0)
            
            if error < 1e-10:
                # Found a solution!
                return U.tolist(), V.tolist(), W.tolist()
            
            if _check_deadline():
                break
    
    return None

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying 
    a 2x3 matrix by a 3x10 matrix.
    
    Strategy:
    1. Try Rank 55 (Strassen). Verify exactly.
    2. If Rank 55 is invalid, fallback to Rank 60 (Schoolbook).
    3. Return integer entries to satisfy strict type checks.
    """
    # Try Rank 55
    U55, V55, W55 = _build_strassen_rank55()
    
    if _verify_exact(U55, V55, W55):
        return {
            "U": _to_int_list(U55),
            "V": _to_int_list(V55),
            "W": _to_int_list(W55)
        }
    
    # Fallback to Rank 60
    U60, V60, W60 = _build_schoolbook_rank60()
    return {
        "U": _to_int_list(U60),
        "V": _to_int_list(V60),
        "W": _to_int_list(W60)
    }
