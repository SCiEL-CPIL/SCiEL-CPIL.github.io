"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_cost(a, b, p):
    """Compute the QAP cost sum_{i,j} a[i,j]*b[p[i],p[j]] efficiently."""
    # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # = sum_i sum_j a[i,j] * b[p[i], p[j]]
    # Vectorized: for each i, sum_j a[i,j] * b[p[i], p[j]] = a[i,:] @ b[p[i],:]
    # But b[p[i],:] is a row of b. So cost = sum_i a[i,:] @ b[p[i],:]
    # = sum_i np.dot(a[i], b[p[i]])
    # This is O(n^2) per evaluation if done naively, but we can vectorize.
    # Actually: cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # Let's use: for each pair (i,j), the contribution is a[i,j]*b[p[i],p[j]]
    # We can compute this as: sum over i of (a[i,:] * b[p[i], p[None, :]]).sum()
    # Better: build matrix M where M[i,j] = a[i,j] * b[p[i], p[j]]
    # = a * b[np.ix_(p, p)]
    # cost = np.sum(a * b[np.ix_(p, p)])
    return int(np.sum(a * b[np.ix_(p, p)]))


def _delta_swap(a, b, p, i, j):
    """Compute the change in cost when swapping facilities i and j.
    
    Currently: facility i at location p[i], facility j at location p[j].
    After swap: facility i at location p[j], facility j at location p[i].
    
    Cost = sum_{k,l} a[k,l] * b[p[k], p[l]]
    
    The change comes from all terms involving i or j:
    Old terms with i or j:
      sum_l a[i,l]*b[p[i],p[l]] + sum_k a[k,i]*b[p[k],p[i]] 
      + sum_l a[j,l]*b[p[j],p[l]] + sum_k a[k,j]*b[p[k],p[j]]
      - a[i,i]*b[p[i],p[i]] - a[j,j]*b[p[j],p[j]]  (avoid double count of diagonal if i!=j)
      Wait, let's be more careful.
    
    Cost = sum_{k,l} a[k,l] * b[p[k], p[l]]
    
    Terms involving index i (as k) or j (as k), or i (as l) or j (as l):
    Let's denote the contribution of all (k,l) pairs where k==i or l==i or k==j or l==j.
    
    Actually, a cleaner way:
    delta = new_cost - old_cost
    Only terms where p[k] or p[l] changes matter. p changes only at positions i and j.
    So p'[i] = p[j], p'[j] = p[i], p'[k] = p[k] for k != i,j.
    
    The cost difference is:
    sum_{k,l} a[k,l] * (b[p'[k], p'[l]] - b[p[k], p[l]])
    
    b[p'[k], p'[l]] != b[p[k], p[l]] only when k in {i,j} or l in {i,j}.
    
    Let's compute:
    Old contribution from pairs where k in {i,j} or l in {i,j}:
    C_old = sum_{l} a[i,l]*b[p[i],p[l]] + sum_{k} a[k,i]*b[p[k],p[i]] 
          + sum_{l} a[j,l]*b[p[j],p[l]] + sum_{k} a[k,j]*b[p[k],p[j]]
          - a[i,i]*b[p[i],p[i]] - a[j,j]*b[p[j],p[j]]
          + a[i,j]*b[p[i],p[j]] + a[j,i]*b[p[j],p[i]]
          Wait, this is getting messy with overlaps.
    
    Let me use a different approach. Let's define:
    For a given assignment p, cost = sum_{k,l} a[k,l] * b[p[k], p[l]]
    
    Let S(p) = cost.
    
    When we swap positions i and j, let p' be the new assignment.
    
    S(p') - S(p) = sum_{k,l} a[k,l] * (b[p'[k], p'[l]] - b[p[k], p[l]])
    
    = sum_{k,l: k in {i,j} or l in {i,j}} a[k,l] * (b[p'[k], p'[l]] - b[p[k], p[l]])
    
    Let's break this down:
    Case 1: k=i, l=j: a[i,j]*(b[p[j],p[i]] - b[p[i],p[j]])
    Case 2: k=j, l=i: a[j,i]*(b[p[i],p[j]] - b[p[j],p[i]])
    Case 3: k=i, l not in {i,j}: a[i,l]*(b[p[j],p[l]] - b[p[i],p[l]])
    Case 4: k=j, l not in {i,j}: a[j,l]*(b[p[i],p[l]] - b[p[j],p[l]])
    Case 5: k not in {i,j}, l=i: a[k,i]*(b[p[k],p[j]] - b[p[k],p[i]])
    Case 6: k not in {i,j}, l=j: a[k,j]*(b[p[k],p[i]] - b[p[k],p[j]])
    
    So delta = a[i,j]*(b[p[j],p[i]] - b[p[i],p[j]]) + a[j,i]*(b[p[i],p[j]] - b[p[j],p[i]])
             + sum_{l != i,j} a[i,l]*(b[p[j],p[l]] - b[p[i],p[l]])
             + sum_{l != i,j} a[j,l]*(b[p[i],p[l]] - b[p[j],p[l]])
             + sum_{k != i,j} a[k,i]*(b[p[k],p[j]] - b[p[k],p[i]])
             + sum_{k != i,j} a[k,j]*(b[p[k],p[i]] - b[p[k],p[j]])
    
    Let's simplify:
    Let di = p[i], dj = p[j]
    
    delta = (a[i,j] - a[j,i])*(b[dj,di] - b[di,dj])
           + sum_{l != i,j} (a[i,l] - a[j,l])*(b[dj,p[l]] - b[di,p[l]])
           + sum_{k != i,j} (a[k,i] - a[k,j])*(b[p[k],dj] - b[p[k],di])
    
    This can be computed in O(n) if we precompute some things, but for now let's just compute it directly in O(n).
    """
    di = p[i]
    dj = p[j]
    
    # Term 1: (a[i,j] - a[j,i]) * (b[dj, di] - b[di, dj])
    term1 = (a[i, j] - a[j, i]) * (b[dj, di] - b[di, dj])
    
    # Terms 2 and 3: sums over l != i,j and k != i,j
    # For efficiency, compute the full sums and subtract the i,j terms
    
    # sum_l a[i,l]*b[dj,p[l]] - a[i,l]*b[di,p[l]] for all l
    # = a[i,:] @ (b[dj, p] - b[di, p])
    # Similarly for j
    
    # Let's compute:
    # S_i = sum_l a[i,l] * (b[dj, p[l]] - b[di, p[l]])
    #      = a[i, :] @ (b[dj, p] - b[di, p])
    # S_j = sum_l a[j,l] * (b[di, p[l]] - b[dj, p[l]])
    #      = a[j, :] @ (b[di, p] - b[dj, p])
    
    # S_ki = sum_k a[k,i] * (b[p[k], dj] - b[p[k], di])
    #       = a[:, i] @ (b[p, dj] - b[p, di])
    # S_kj = sum_k a[k,j] * (b[p[k], di] - b[p[k], dj])
    #       = a[:, j] @ (b[p, di] - b[p, dj])
    
    diff1 = b[dj, p] - b[di, p]  # shape (n,)
    diff2 = b[di, p] - b[dj, p]  # = -diff1
    diff3 = b[p, dj] - b[p, di]  # shape (n,)
    diff4 = b[p, di] - b[p, dj]  # = -diff3
    
    S_i = np.dot(a[i, :], diff1)
    S_j = np.dot(a[j, :], diff2)
    S_ki = np.dot(a[:, i], diff3)
    S_kj = np.dot(a[:, j], diff4)
    
    # Subtract the terms where l==i, l==j, k==i, k==j
    # For S_i: l=i gives a[i,i]*(b[dj,di]-b[di,di]), l=j gives a[i,j]*(b[dj,dj]-b[di,dj])
    S_i -= a[i, i] * (b[dj, di] - b[di, di])
    S_i -= a[i, j] * (b[dj, dj] - b[di, dj])
    
    # For S_j: l=i gives a[j,i]*(b[di,di]-b[dj,di]), l=j gives a[j,j]*(b[di,dj]-b[dj,dj])
    S_j -= a[j, i] * (b[di, di] - b[dj, di])
    S_j -= a[j, j] * (b[di, dj] - b[dj, dj])
    
    # For S_ki: k=i gives a[i,i]*(b[di,dj]-b[di,di]), k=j gives a[j,i]*(b[dj,dj]-b[dj,di])
    S_ki -= a[i, i] * (b[di, dj] - b[di, di])
    S_ki -= a[j, i] * (b[dj, dj] - b[dj, di])
    
    # For S_kj: k=i gives a[i,j]*(b[di,di]-b[di,dj]), k=j gives a[j,j]*(b[dj,di]-b[dj,dj])
    S_kj -= a[i, j] * (b[di, di] - b[di, dj])
    S_kj -= a[j, j] * (b[dj, di] - b[dj, dj])
    
    delta = term1 + S_i + S_j + S_ki + S_kj
    return delta


def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    deadline = time.time() + 115  # 5s safety margin
    
    # Initial solution: rank-matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.astype(np.int64)
    
    best_p = p.copy()
    best_cost = _compute_cost(a, b, p)
    
    # Tabu search with best-improvement
    tabu_len = 20
    tabu = np.zeros(n, dtype=np.int64)  # iteration when each position becomes untabu
    tabu_count = 0
    no_improve = 0
    max_no_improve = 100
    
    while time.time() < deadline:
        # Find best improving swap
        best_delta = 0
        best_i, best_j = -1, -1
        
        # Sample a subset of pairs if n is large, or do full search
        # For n=150, full search is 150*149/2 = 11175 pairs, each O(n) = ~1.7M ops
        # That's feasible.
        
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if tabu[i] > tabu_count or tabu[j] > tabu_count:
                    continue
                d = _delta_swap(a, b, p, i, j)
                if d < best_delta:
                    best_delta = d
                    best_i, best_j = i, j
        
        if best_i == -1:
            # No improving non-tabu move; try any improving move or best tabu move
            best_delta = 0
            best_i, best_j = -1, -1
            for i in range(n):
                for j in range(i + 1, n):
                    d = _delta_swap(a, b, p, i, j)
                    if d < best_delta:
                        best_delta = d
                        best_i, best_j = i, j
            
            if best_i == -1:
                # No improving move at all; reset tabu and try random
                tabu[:] = 0
                # Pick a random pair
                idx = np.random.choice(n * (n - 1) // 2, size=1, replace=False)
                # Map linear index to (i,j)
                # This is complex; let's just pick random i,j
                i, j = np.random.choice(n, size=2, replace=False)
                if i > j:
                    i, j = j, i
                best_i, best_j = i, j
                best_delta = _delta_swap(a, b, p, i, j)
        
        # Apply the best move
        p[best_i], p[best_j] = p[best_j], p[best_i]
        tabu[best_i] = tabu_count + tabu_len
        tabu[best_j] = tabu_count + tabu_len
        tabu_count += 1
        
        new_cost = best_cost + int(best_delta)
        if new_cost < best_cost:
            best_cost = new_cost
            best_p = p.copy()
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve > max_no_improve:
            # Diversification: random restart from best
            p = best_p.copy()
            # Apply a few random swaps
            for _ in range(5):
                i, j = np.random.choice(n, size=2, replace=False)
                p[i], p[j] = p[j], p[i]
            no_improve = 0
            tabu[:] = 0
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
