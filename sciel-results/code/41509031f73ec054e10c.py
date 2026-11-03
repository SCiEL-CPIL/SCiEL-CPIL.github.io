"""
Strategy: Simulated Annealing with 2-Opt (Block Rotation) neighborhood.
Rationale: 1-swap local search (current) gets stuck in local optima. 
2-opt moves allow larger structural changes. SA allows escaping local optima 
via probabilistic acceptance of worse moves, guided by temperature.
"""

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _swap_delta(a, b, p, i, j):
    """
    Compute the change in cost when swapping facilities at positions i and j in permutation p.
    Vectorized for speed.
    """
    x = p[i]
    y = p[j]
    
    # Direct pair contribution
    delta = a[i, j] * (b[y, x] - b[x, y]) + a[j, i] * (b[x, y] - b[y, x])
    
    # Vectorized sum over all k
    V1 = b[y, p]
    V2 = b[x, p]
    V3 = b[p, y]
    V4 = b[p, x]
    
    T1 = a[i, :] * (V1 - V2)
    T2 = a[:, i] * (V3 - V4)
    T3 = a[j, :] * (V2 - V1)
    T4 = a[:, j] * (V4 - V3)
    
    S = np.sum(T1 + T2 + T3 + T4)
    
    # Subtract terms for k=i and k=j to correct for double counting in the vectorized sum
    # T(i) terms: 
    # k=i in T1: a[i,i]*(b[y,p[i]] - b[x,p[i]]) -> a[i,i]*(b[y,x]-b[x,x])
    # k=i in T2: a[i,i]*(b[p[i],y] - b[p[i],x]) -> a[i,i]*(b[x,y]-b[x,x])
    # k=i in T3: a[j,i]*(b[x,p[i]] - b[y,p[i]]) -> a[j,i]*(b[x,x]-b[y,x])
    # k=i in T4: a[i,i]*(b[p[i],x] - b[p[i],y]) -> a[i,i]*(b[x,x]-b[x,y])
    # Wait, T4 is a[:,j] * (V4-V3). For k=i, it is a[i,j]*(b[p[i],x] - b[p[i],y]) = a[i,j]*(b[x,x]-b[x,y])
    
    # Let's re-verify the correction terms carefully.
    # The sum S = sum_k [ a[i,k](b[y,p[k]]-b[x,p[k]]) + a[k,i](b[p[k],y]-b[p[k],x]) + a[j,k](b[x,p[k]]-b[y,p[k]]) + a[k,j](b[p[k],x]-b[p[k],y]) ]
    
    # For k=i:
    # Term 1: a[i,i] * (b[y, p[i]] - b[x, p[i]]) = a[i,i] * (b[y,x] - b[x,x])
    # Term 2: a[i,i] * (b[p[i],y] - b[p[i],x]) = a[i,i] * (b[x,y] - b[x,x])
    # Term 3: a[j,i] * (b[x, p[i]] - b[y, p[i]]) = a[j,i] * (b[x,x] - b[y,x])
    # Term 4: a[i,j] * (b[p[i],x] - b[p[i],y]) = a[i,j] * (b[x,x] - b[x,y])
    
    # For k=j:
    # Term 1: a[i,j] * (b[y, p[j]] - b[x, p[j]]) = a[i,j] * (b[y,y] - b[x,y])
    # Term 2: a[j,i] * (b[p[j],y] - b[p[j],x]) = a[j,i] * (b[y,y] - b[y,x])
    # Term 3: a[j,j] * (b[x, p[j]] - b[y, p[j]]) = a[j,j] * (b[x,y] - b[y,y])
    # Term 4: a[j,j] * (b[p[j],x] - b[p[j],y]) = a[j,j] * (b[y,x] - b[y,y])
    
    T_i = (a[i, i] * (b[y, x] - b[x, x]) + 
           a[i, i] * (b[x, y] - b[x, x]) + 
           a[j, i] * (b[x, x] - b[y, x]) + 
           a[i, j] * (b[x, x] - b[x, y]))
           
    T_j = (a[i, j] * (b[y, y] - b[x, y]) + 
           a[j, i] * (b[y, y] - b[y, x]) + 
           a[j, j] * (b[x, y] - b[y, y]) + 
           a[j, j] * (b[y, x] - b[y, y]))
           
    delta += S - T_i - T_j
    
    return int(delta)

def _reverse_delta(a, b, p, i, j):
    """
    Compute the change in cost when reversing the segment p[i..j] (inclusive).
    This is a 2-opt move.
    """
    if i == j:
        return 0
    if i > j:
        i, j = j, i
    
    # The move reverses p[i], p[i+1], ..., p[j].
    # Let S = p[i..j]. The new segment is reversed(S).
    # Cost change = New_Cost - Old_Cost.
    
    # We can compute this by summing the deltas of individual swaps? No, that's O(n^2) swaps.
    # Better: Compute the cost of the sub-permutation change directly.
    
    # Let old_segment = p[i:j+1]
    # Let new_segment = old_segment[::-1]
    
    # The cost terms involving indices in [i, j] and indices outside [i, j] change.
    # Terms involving only indices inside [i, j] also change (because relative order changes).
    # Terms involving only indices outside [i, j] do not change.
    
    # Let In = set of indices in [i, j]
    # Let Out = set of indices not in In
    
    # Cost = Sum_{u,v in In} a[u,v]b[p[u],p[v]] + Sum_{u in In, v in Out} a[u,v]b[p[u],p[v]] + Sum_{u in Out, v in In} a[u,v]b[p[u],p[v]] + Sum_{u,v in Out} a[u,v]b[p[u],p[v]]
    
    # The last term is constant.
    # The first term changes because the mapping p[u] for u in In is reversed.
    # The middle terms change because p[u] for u in In is reversed.
    
    # Let's compute the delta by iterating over the pairs.
    # For n=256, a segment of length L has L^2/4 pairs internally and L*(n-L) pairs externally.
    # If L is small, this is fast. If L is large, it's slow.
    # We will restrict the length of the reversal to be small (e.g., < 20) to keep it fast,
    # or use a heuristic to pick i, j such that the move is likely beneficial.
    
    # Actually, for SA, we need to evaluate random moves.
    # A random reversal of length L has cost O(L*n + L^2).
    # If we pick L ~ 10, cost is ~ 2560 + 100 ~ 2600 ops.
    # 150s / 2600 ops ~ 50,000 iterations.
    # 150s / 1000 ops (swap) ~ 150,000 iterations.
    # SA with 2-opt might be slower per step but potentially more effective per step.
    
    # Let's implement a fast version for small segments.
    
    old_segment = p[i:j+1].copy()
    new_segment = old_segment[::-1]
    
    # Indices in the segment
    idx_in = np.arange(i, j+1)
    idx_out = np.setdiff1d(np.arange(len(p)), idx_in)
    
    # Extract submatrices
    # A_in = a[np.ix_(idx_in, idx_in)]
    # B_old_in = b[np.ix_(old_segment, old_segment)]
    # B_new_in = b[np.ix_(new_segment, new_segment)]
    
    # A_in_out = a[np.ix_(idx_in, idx_out)]
    # B_old_in_out = b[np.ix_(old_segment, p[idx_out])]
    # B_new_in_out = b[np.ix_(new_segment, p[idx_out])]
    
    # A_out_in = a[np.ix_(idx_out, idx_in)]
    # B_old_out_in = b[np.ix_(p[idx_out], old_segment)]
    # B_new_out_in = b[np.ix_(p[idx_out], new_segment)]
    
    # Delta = Sum(A_in * (B_new_in - B_old_in)) + Sum(A_in_out * (B_new_in_out - B_old_in_out)) + Sum(A_out_in * (B_new_out_in - B_old_out_in))
    
    # This involves several large array multiplications.
    # For L=10, n=256:
    # In-In: 10x10
    # In-Out: 10x246
    # Out-In: 246x10
    # Total elements ~ 100 + 2460 + 2460 ~ 5000.
    # NumPy sum is fast.
    
    # Let's try this.
    
    # To avoid allocating huge temporary arrays for Out-Out (which is constant), we only compute deltas for terms involving In.
    
    # Term 1: In-In
    B_old_in = b[np.ix_(old_segment, old_segment)]
    B_new_in = b[np.ix_(new_segment, new_segment)]
    A_in = a[np.ix_(idx_in, idx_in)]
    delta_1 = np.sum(A_in * (B_new_in - B_old_in))
    
    # Term 2: In-Out
    B_old_in_out = b[np.ix_(old_segment, p[idx_out])]
    B_new_in_out = b[np.ix_(new_segment, p[idx_out])]
    A_in_out = a[np.ix_(idx_in, idx_out)]
    delta_2 = np.sum(A_in_out * (B_new_in_out - B_old_in_out))
    
    # Term 3: Out-In
    B_old_out_in = b[np.ix_(p[idx_out], old_segment)]
    B_new_out_in = b[np.ix_(p[idx_out], new_segment)]
    A_out_in = a[np.ix_(idx_out, idx_in)]
    delta_3 = np.sum(A_out_in * (B_new_out_in - B_old_out_in))
    
    delta = delta_1 + delta_2 + delta_3
    
    return int(delta)

def _greedy_init(a, b, n):
    """Greedy initialization based on row sums."""
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p

def _sa_search(a, b, p_init, time_limit, max_segment_len=15):
    """
    Simulated Annealing with 2-Opt (reversal) moves.
    """
    n = len(p_init)
    p = p_init.copy()
    current_cost = _qap_cost(a, b, p)
    best_cost = current_cost
    best_p = p.copy()
    
    start_time = time.time()
    
    # Temperature schedule
    T_start = 1000000.0 # High initial temp
    T_end = 1.0         # Low final temp
    # We will adjust T based on time remaining
    # Or use a fixed number of iterations? 
    # Let's use time-based cooling.
    
    # Estimate iterations per second
    # Let's just run as many iterations as possible and cool down linearly with time.
    
    # Precompute some stuff?
    
    while time.time() - start_time < time_limit:
        # Pick a random reversal
        # Length L between 2 and max_segment_len
        L = random.randint(2, max_segment_len)
        # Start index i such that i+L-1 < n
        max_i = n - L
        if max_i < 0:
            L = 2
            max_i = n - 2
        if max_i < 0:
            break
        i = random.randint(0, max_i)
        j = i + L - 1
        
        delta = _reverse_delta(a, b, p, i, j)
        
        # Acceptance criterion
        if delta < 0:
            # Always accept improvement
            p[i:j+1] = p[i:j+1][::-1]
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        else:
            # Accept with probability exp(-delta / T)
            # T needs to be scaled.
            # Let's use a simple cooling schedule based on elapsed time.
            elapsed = time.time() - start_time
            # T goes from T_start to T_end over time_limit
            progress = elapsed / time_limit
            T = T_start * (1 - progress) + T_end * progress
            if T < T_end:
                T = T_end
            
            # If T is very small, exp(-delta/T) will be 0 for large delta
            # To avoid underflow, check if delta > T * 50
            if delta > T * 50:
                prob = 0.0
            else:
                prob = np.exp(-delta / T)
            
            if random.random() < prob:
                p[i:j+1] = p[i:j+1][::-1]
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
    
    return best_p, best_cost

def _solve_tai256c(a, b):
    n = len(a)
    deadline = time.time() + 140  # Leave 10s safety margin
    
    # Phase 1: Generate diverse initial solutions
    inits = []
    inits.append(_greedy_init(a, b, n))
    for _ in range(3):
        inits.append(np.random.permutation(n))
    
    best_p = None
    best_cost = np.inf
    
    # Phase 2: Quick SA on each initialization
    # Allocate ~10s per init
    quick_time = 10.0
    
    for i, p_init in enumerate(inits):
        if time.time() >= deadline - 20:
            break
        
        p_sa, cost_sa = _sa_search(a, b, p_init, time_limit=quick_time, max_segment_len=10)
        
        if cost_sa < best_cost:
            best_cost = cost_sa
            best_p = p_sa.copy()
    
    # Phase 3: Long SA from the best quick solution
    if best_p is not None and time.time() < deadline - 5:
        polish_time = deadline - time.time() - 5
        p_polished, cost_polished = _sa_search(a, b, best_p, time_limit=polish_time, max_segment_len=20)
        
        if cost_polished < best_cost:
            best_cost = cost_polished
            best_p = p_polished
    
    # Fallback
    if best_p is None:
        p_greedy = _greedy_init(a, b, n)
        best_p, best_cost = _sa_search(a, b, p_greedy, time_limit=min(10.0, deadline - time.time() - 5))
    
    return best_p.tolist()

def solve(a, b):
    return _solve_tai256c(a, b)
# EVOLVE-BLOCK-END
