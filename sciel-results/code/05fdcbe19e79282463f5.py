"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _compute_cost(a, b, p):
    """Compute the QAP cost for assignment p."""
    bp = b[np.ix_(p, p)]
    return np.sum(a * bp)


def _swap_delta(a, b, p, i, j):
    """
    Compute the delta in cost when swapping facilities at positions i and j in p.
    p[i] and p[j] are the locations.
    Cost = sum_{u,v} a[u,v] * b[p[u], p[v]]
    
    Let x = p[i], y = p[j].
    After swap, p'[i]=y, p'[j]=x, p'[k]=p[k] for k!=i,j.
    
    Delta = sum_{u,v} a[u,v] * (b[p'[u], p'[v]] - b[p[u], p[v]])
    
    Non-zero terms occur when {u,v} intersects {i,j}.
    1. u=i, v=i: a[i,i] * (b[y,y] - b[x,x])
    2. u=j, v=j: a[j,j] * (b[x,x] - b[y,y])
    3. u=i, v=j: a[i,j] * (b[y,x] - b[x,y])
    4. u=j, v=i: a[j,i] * (b[x,y] - b[y,x])
    5. u=i, v=k (k!=i,j): a[i,k] * (b[y, p[k]] - b[x, p[k]])
    6. u=k (k!=i,j), v=i: a[k,i] * (b[p[k], y] - b[p[k], x])
    7. u=j, v=k (k!=i,j): a[j,k] * (b[x, p[k]] - b[y, p[k]])
    8. u=k (k!=i,j), v=j: a[k,j] * (b[p[k], x] - b[p[k], y])
    
    Let's group them:
    Term 1+2: (a[i,i] - a[j,j]) * (b[y,y] - b[x,x])
    Term 3+4: (a[i,j] - a[j,i]) * (b[y,x] - b[x,y])
    
    For k != i, j:
    Terms involving i: a[i,k]*(b[y,p[k]]-b[x,p[k]]) + a[k,i]*(b[p[k],y]-b[p[k],x])
    Terms involving j: a[j,k]*(b[x,p[k]]-b[y,p[k]]) + a[k,j]*(b[p[k],x]-b[p[k],y])
    
    Note b is symmetric in QAPLIB? Usually yes, but let's assume general case first.
    Actually, QAPLIB instances often have symmetric A and B.
    If B is symmetric: b[u,v] = b[v,u].
    Then b[y,p[k]] = b[p[k],y], etc.
    
    Let's assume B is symmetric for speed, but verify or handle generally.
    The problem statement says "first and second matrix of the instance file".
    QAPLIB standard: A is flow (symmetric), B is distance (symmetric).
    Let's assume symmetry to simplify and speed up, but add a check or just assume.
    Given the context of QAPLIB tho150, A and B are symmetric.
    
    If B is symmetric:
    b[y, p[k]] = b[p[k], y]
    b[x, p[k]] = b[p[k], x]
    
    So terms 5+6: (a[i,k] + a[k,i]) * (b[y, p[k]] - b[x, p[k]])
    Terms 7+8: (a[j,k] + a[k,j]) * (b[x, p[k]] - b[y, p[k]])
    
    Sum over k != i,j:
    Sum_k (a[i,k] + a[k,i] - a[j,k] - a[k,j]) * (b[y, p[k]] - b[x, p[k]])
    
    Let A_sym = (a + a.T) / 2. Then a[i,k] + a[k,i] = 2 * A_sym[i,k].
    Let B_sym = (b + b.T) / 2.
    
    Delta = (a[i,i] - a[j,j]) * (b[y,y] - b[x,x]) + (a[i,j] - a[j,i]) * (b[y,x] - b[x,y])
            + Sum_{k!=i,j} (2*A_sym[i,k] - 2*A_sym[j,k]) * (b[y, p[k]] - b[x, p[k]])
    
    We can precompute S = 2 * A_sym.
    And we need to compute the sum efficiently.
    Let D = S[i, :] - S[j, :]
    Let V = b[y, p] - b[x, p]
    Sum = dot(D, V) - D[i]*V[i] - D[j]*V[j]
    
    V[i] = b[y, p[i]] - b[x, p[i]] = b[y, x] - b[x, x]
    V[j] = b[y, p[j]] - b[x, p[j]] = b[y, y] - b[x, y]
    
    So:
    Delta = (a[i,i] - a[j,j]) * (b[y,y] - b[x,x]) 
          + (a[i,j] - a[j,i]) * (b[y,x] - b[x,y])
          + dot(S[i,:] - S[j,:], b[y,p] - b[x,p]) 
          - (S[i,i] - S[j,i]) * (b[y,x] - b[x,x])
          - (S[i,j] - S[j,j]) * (b[y,y] - b[x,y])
          
    Wait, S[i,i] = a[i,i] + a[i,i] = 2a[i,i].
    S[j,i] = a[j,i] + a[i,j].
    
    This allows O(n) calculation of the delta for a swap, given O(1) access to b rows and precomputed S.
    b[y,p] is a vector of length n. Computing it takes O(n).
    So each swap evaluation is O(n).
    """
    x = p[i]
    y = p[j]
    
    # Precomputed S = a + a.T
    # Precomputed b rows
    
    # Term 1: Diagonals
    t1 = (a[i, i] - a[j, j]) * (b[y, y] - b[x, x])
    
    # Term 2: Off-diagonal between i and j
    t2 = (a[i, j] - a[j, i]) * (b[y, x] - b[x, y])
    
    # Term 3: Sum over k
    # D = S[i, :] - S[j, :]
    # V = b[y, p] - b[x, p]
    # Sum = dot(D, V)
    # Subtract k=i and k=j contributions
    
    # Vectorized computation
    # D = S[i] - S[j]
    # V = b[y, p] - b[x, p]
    # This involves indexing b with p, which is O(n).
    
    # Optimization: If we maintain b_perm = b[:, p], then b[y, p] is just b_perm[y, :].
    # But swapping i and j in p changes b_perm at columns i and j.
    # Updating b_perm is O(n) (swap two columns).
    # So we can maintain b_perm.
    
    # However, for the delta calculation function itself, if we don't maintain b_perm globally,
    # we just do the indexing.
    
    # Let's implement the O(n) version.
    D = S[i] - S[j]
    V = b[y, p] - b[x, p]
    
    total_dot = np.dot(D, V)
    
    # Subtract k=i
    # V[i] = b[y, x] - b[x, x]
    # D[i] = S[i, i] - S[j, i]
    vi = b[y, x] - b[x, x]
    di = S[i, i] - S[j, i]
    
    # Subtract k=j
    # V[j] = b[y, y] - b[x, y]
    # D[j] = S[i, j] - S[j, j]
    vj = b[y, y] - b[x, y]
    dj = S[i, j] - S[j, j]
    
    t3 = total_dot - di * vi - dj * vj
    
    return t1 + t2 + t3


def _fast_local_search(a, b, p, time_budget, rng, S):
    """
    Local search using fast delta calculation.
    p is a numpy array of locations.
    """
    n = len(p)
    p = p.copy()
    current_cost = _compute_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    
    # We will perform first-improvement or best-improvement?
    # Best-improvement is expensive (O(n^2 * n) = O(n^3) per pass).
    # First-improvement is cheaper.
    # Let's do a pass of random swaps with acceptance (Simulated Annealing style) 
    # or simple Hill Climbing with random restarts within the local search.
    
    # Given n=150, n^2 = 22500 swaps.
    # Each swap is O(n) = 150 ops.
    # Total per full pass: 22500 * 150 = 3.375 million ops.
    # In Python/NumPy, this might be slow if done in a Python loop.
    # We should vectorize the delta calculation for all pairs?
    # That would be O(n^3) memory/time which is too much.
    
    # Let's stick to a random search within local search (SA-like) or random 2-swap hill climbing.
    
    # Strategy: 
    # 1. Try random swaps, accept if better (Hill Climbing).
    # 2. If stuck for a while, perturb (SA-like).
    
    # Actually, let's use a simple Simulated Annealing with the fast delta.
    t = 100.0
    t_min = 0.1
    cooling_rate = 0.999
    
    while time.time() - start_time < time_budget:
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if i == j:
            continue
            
        delta = _swap_delta(a, b, p, i, j, S)
        
        if delta < 0 or rng.random() < np.exp(-delta / t):
            # Accept
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        else:
            # Reject
            pass
            
        t *= cooling_rate
        if t < t_min:
            t = t_min
            
    return best_p, best_cost


def _hungarian_refine(a, b, p, S):
    """
    Linearized assignment refinement.
    C[i, j] = sum_k a[i, k] * b[j, p[k]]
    C = a @ b[:, p].T
    """
    n = len(p)
    # b[:, p] creates a view or copy? It's a fancy index, so it's a copy.
    # Shape (n, n)
    B_p = b[:, p]
    # C[i, j] = sum_k a[i, k] * B_p[j, k] = (a @ B_p.T)[i, j]
    C = a @ B_p.T
    
    # Solve linear assignment
    row_ind, col_ind = linear_sum_assignment(C)
    # col_ind is the permutation: facility i goes to col_ind[i]
    return col_ind


def solve(a, b):
    """
    Solve the QAP using a combination of spectral/rank init, Hungarian refinement, 
    and fast local search with incremental delta calculation.
    """
    n = len(a)
    start_time = time.time()
    time_limit = 140.0  # Safety margin
    
    # Precompute symmetric parts
    S = a + a.T  # 2 * A_sym
    
    # 1. Initialization
    # Rank matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # 2. Hungarian Refinement (Linearized)
    p = _hungarian_refine(a, b, p, S)
    current_cost = _compute_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # 3. Local Search / SA
    # We have time for a few rounds of refinement.
    # Round 1: Local Search
    remaining = time_limit - (time.time() - start_time)
    if remaining > 10:
        rng = np.random.default_rng(42)
        # Use 70% of remaining for LS, 30% for more Hungarian/LS cycles
        ls_budget = remaining * 0.7
        p, current_cost = _fast_local_search(a, b, p, ls_budget, rng, S)
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
    
    # 4. Another Hungarian Refinement
    remaining = time_limit - (time.time() - start_time)
    if remaining > 5:
        p = _hungarian_refine(a, b, p, S)
        current_cost = _compute_cost(a, b, p)
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            
    # 5. Final Local Search Polish
    remaining = time_limit - (time.time() - start_time)
    if remaining > 5:
        rng = np.random.default_rng(123)
        p, current_cost = _fast_local_search(a, b, p, remaining - 2, rng, S)
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            
    return best_p.tolist()

# EVOLVE-BLOCK-END
