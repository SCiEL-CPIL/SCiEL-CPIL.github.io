"""Improved QAP solver for wil100: vectorized swap-based Simulated Annealing."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    """Compute exact QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    bp = b[np.ix_(p, p)]
    return int(np.sum(a * bp))


def _swap_cost_delta(a, b, p, i, j):
    """
    Compute the change in cost when swapping positions i and j in permutation p.
    p[i] and p[j] are the facilities at locations i and j.
    We swap them: new_p[i] = old_p[j], new_p[j] = old_p[i].
    
    Cost = sum_{k,l} a[k,l] * b[p[k], p[l]]
    
    Only terms involving position i or j change.
    Delta = sum_{l != i, j} a[i,l]*(b[p[j],p[l]] - b[p[i],p[l]])
          + sum_{l != i, j} a[j,l]*(b[p[i],p[l]] - b[p[j],p[l]])
          + a[i,j]*(b[p[j],p[i]] - b[p[i],p[j]])
          + a[j,i]*(b[p[i],p[j]] - b[p[j],p[i]])
    
    Let fi = p[i], fj = p[j].
    Delta = sum_{l not in {i,j}} (a[i,l] - a[j,l]) * (b[fj, p[l]] - b[fi, p[l]])
          + (a[i,j] - a[j,i]) * (b[fj, fi] - b[fi, fj])
    """
    fi = p[i]
    fj = p[j]
    
    # Vectorized computation over all l except i and j
    # a_row_diff = a[:, fi] - a[:, fj]  -> but we need a[i,l] - a[j,l] for each l
    # Actually: for each l, the coefficient is (a[i,l] - a[j,l])
    # and the distance change is (b[fj, p[l]] - b[fi, p[l]])
    
    a_diff = a[i, :] - a[j, :]  # shape (n,)
    # For each position l, the new b value at (fi, p[l]) vs old b value at (fi, p[l])... 
    # Wait, let me re-derive.
    
    # Original contribution involving position i (as first index in a):
    # sum_l a[i,l] * b[fi, p[l]]
    # After swap, position i has facility fj:
    # sum_l a[i,l] * b[fj, p[l]]
    # Change from i's row: sum_l a[i,l] * (b[fj, p[l]] - b[fi, p[l]])
    
    # Original contribution involving position j (as first index in a):
    # sum_l a[j,l] * b[fj, p[l]]
    # After swap, position j has facility fi:
    # sum_l a[j,l] * b[fi, p[l]]
    # Change from j's row: sum_l a[j,l] * (b[fi, p[l]] - b[fj, p[l]])
    
    # Total change from rows i and j of a:
    # sum_l (a[i,l] - a[j,l]) * (b[fj, p[l]] - b[fi, p[l]])
    # But this includes l=i and l=j which we need to handle carefully.
    
    # When l=i: b[fj, p[i]] = b[fj, fi], b[fi, p[i]] = b[fi, fi]
    # When l=j: b[fj, p[j]] = b[fj, fj], b[fi, p[j]] = b[fi, fj]
    
    # Also, the contributions where i and j appear as the SECOND index in b:
    # b[p[k], fi] and b[p[k], fj] terms.
    
    # Let me just compute it directly with vectors.
    
    # All terms: sum_{k,l} a[k,l] * b[p[k], p[l]]
    # Terms that change are those where k in {i,j} OR l in {i,j}.
    
    # Let's compute the full delta by considering all four cases:
    # (k,l) in {i,j} x {i,j}
    
    # Case (i,i): a[i,i]*(b[fj,fi] - b[fi,fi])
    # Case (i,j): a[i,j]*(b[fj,fj] - b[fi,fj])
    # Case (j,i): a[j,i]*(b[fi,fi] - b[fj,fi])
    # Case (j,j): a[j,j]*(b[fi,fj] - b[fj,fj])
    
    # Plus terms where k=i, l not in {i,j}: a[i,l]*(b[fj,p[l]] - b[fi,p[l]])
    # Plus terms where k=j, l not in {i,j}: a[j,l]*(b[fi,p[l]] - b[fj,p[l]])
    # Plus terms where l=i, k not in {i,j}: a[k,i]*(b[p[k],fj] - b[p[k],fi])
    # Plus terms where l=j, k not in {i,j}: a[k,j]*(b[p[k],fi] - b[p[k],fj])
    
    n = len(p)
    
    # Vector for b[fj, :] and b[fi, :]
    b_fj = b[fj, :]  # b[fj, p[l]] for all l
    b_fi = b[fi, :]
    
    # k=i, l varies: a[i,l]*(b[fj,p[l]] - b[fi,p[l]])
    # = a[i,:] * (b[fj,p] - b[fi,p])
    row_i_diff = a[i, :] * (b_fj[p] - b_fi[p])
    
    # k=j, l varies: a[j,l]*(b[fi,p[l]] - b[fj,p[l]])
    row_j_diff = a[j, :] * (b_fi[p] - b_fj[p])
    
    # l=i, k varies: a[k,i]*(b[p[k],fj] - b[p[k],fi])
    col_i_diff = a[:, i] * (b[p, fj] - b[p, fi])
    
    # l=j, k varies: a[k,j]*(b[p[k],fi] - b[p[k],fj])
    col_j_diff = a[:, j] * (b[p, fi] - b[p, fj])
    
    total = row_i_diff.sum() + row_j_diff.sum() + col_i_diff.sum() + col_j_diff.sum()
    
    # But we've double-counted the (i,i), (i,j), (j,i), (j,j) terms.
    # In row_i_diff, l=i gives a[i,i]*(b[fj,fi]-b[fi,fi]), l=j gives a[i,j]*(b[fj,fj]-b[fi,fj])
    # In col_i_diff, k=i gives a[i,i]*(b[fi,fj]-b[fi,fi]), k=j gives a[j,i]*(b[fj,fj]-b[fj,fi])
    # In row_j_diff, l=i gives a[j,i]*(b[fi,fi]-b[fj,fi]), l=j gives a[j,j]*(b[fi,fj]-b[fj,fj])
    # In col_j_diff, k=i gives a[i,j]*(b[fi,fi]-b[fi,fj]), k=j gives a[j,j]*(b[fi,fj]-b[fj,fj])
    
    # The 2x2 block {i,j}x{i,j} is counted twice. We need to subtract it once.
    # The 2x2 contribution should be:
    # a[i,i]*(b[fj,fi]-b[fi,fi]) + a[i,j]*(b[fj,fj]-b[fi,fj])
    # + a[j,i]*(b[fi,fi]-b[fj,fi]) + a[j,j]*(b[fi,fj]-b[fj,fj])
    
    # Let's compute the 2x2 block separately and subtract the double count.
    block = (a[i,i]*(b[fj,fi]-b[fi,fi]) + 
             a[i,j]*(b[fj,fj]-b[fi,fj]) + 
             a[j,i]*(b[fi,fi]-b[fj,fi]) + 
             a[j,j]*(b[fi,fj]-b[fj,fj]))
    
    return int(total - block)


def _greedy_initial(a, b):
    """Rank-matching greedy initialization."""
    n = len(a)
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Precompute for speed
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    
    # Initial solution
    p = _greedy_initial(a, b).copy()
    best_p = p.copy()
    best_cost = _qap_cost(a, b, p)
    current_cost = best_cost
    
    # Simulated Annealing parameters
    T_start = max(1.0, current_cost * 0.01)
    T_end = max(1.0, current_cost * 1e-5)
    cooling_rate = (T_end / T_start) ** (1.0 / 50000)
    
    T = T_start
    rng = np.random.default_rng(42)
    
    iteration = 0
    while time.time() < deadline:
        # Pick two random distinct positions
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if i == j:
            continue
        
        delta = _swap_cost_delta(a, b, p, i, j)
        
        if delta < 0:
            # Accept
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        else:
            # Accept with probability exp(-delta/T)
            if T > 1e-10:
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
        
        # Cool down
        T *= cooling_rate
        iteration += 1
        
        # Check deadline periodically
        if iteration % 5000 == 0 and time.time() > deadline:
            break
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
