"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    """Compute QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]"""
    return np.sum(a * b[np.ix_(p, p)])


def _compute_C(a, b, p):
    """Compute C[i] = sum_j a[i,j] * b[p[i], p[j]] for all i (row costs)."""
    n = len(p)
    # C[i] = sum_j a[i,j] * b[p[i], p[j]]
    # = sum_j a[i,j] * b[p[i], p[j]]
    # For fixed i: sum_j a[i,j] * b[p[i], p[j]]
    # We can compute this as: for each i, take row a[i,:], multiply element-wise with b[p[i], p], sum
    C = np.zeros(n, dtype=np.float64)
    for i in range(n):
        C[i] = np.dot(a[i, :], b[p[i], p])
    return C


def _swap_delta(C, a, b, p, i, j):
    """
    Compute the change in total cost when swapping facilities at positions i and j.
    p[i] = loc_i, p[j] = loc_j
    
    Current contribution involving i or j:
    - C[i] = sum_k a[i,k]*b[pi,pk]
    - C[j] = sum_k a[j,k]*b[pj,pk]
    
    After swap: p[i]=pj, p[j]=pi
    New C[i] = sum_k a[i,k]*b[pj,pk] (but with pk swapped for k=j)
    New C[j] = sum_k a[j,k]*b[pi,pk] (but with pk swapped for k=i)
    
    Actually, let's use the standard formula:
    delta = 2 * (sum_{k != i,j} a[i,k]*(b[pj,pk] - b[pi,pk]) 
                    + a[i,j]*(b[pj,pi] - b[pi,pj]))
          + 2 * (sum_{k != i,j} a[j,k]*(b[pi,pk] - b[pj,pk]) 
                    + a[j,i]*(b[pi,pj] - b[pj,pi]))
    
    Wait, let me be more careful. The total cost is sum_{i,j} a[i,j]*b[p[i],p[j]].
    
    Terms that change when swapping p[i] and p[j]:
    1. Terms where i is the row: sum_j a[i,j]*b[p[i],p[j]] -> sum_j a[i,j]*b[p[j],p[j]] (no, that's wrong)
    
    Let me reconsider. After swap, p'[i]=p[j], p'[j]=p[i], p'[k]=p[k] for k!=i,j.
    
    Changed terms:
    - (i,i): a[i,i]*b[p[i],p[i]] -> a[i,i]*b[p[j],p[j]]
    - (i,j): a[i,j]*b[p[i],p[j]] -> a[i,j]*b[p[j],p[i]]
    - (i,k) for k!=i,j: a[i,k]*b[p[i],p[k]] -> a[i,k]*b[p[j],p[k]]
    - (j,i): a[j,i]*b[p[j],p[i]] -> a[j,i]*b[p[i],p[j]]
    - (j,j): a[j,j]*b[p[j],p[j]] -> a[j,j]*b[p[i],p[i]]
    - (j,k) for k!=i,j: a[j,k]*b[p[j],p[k]] -> a[j,k]*b[p[i],p[k]]
    - (k,i) for k!=i,j: a[k,i]*b[p[k],p[i]] -> a[k,i]*b[p[k],p[j]]
    - (k,j) for k!=i,j: a[k,j]*b[p[k],p[j]] -> a[k,j]*b[p[k],p[i]]
    
    delta = a[i,i]*(b[pj,pj]-b[pi,pi]) + a[i,j]*(b[pj,pi]-b[pi,pj])
          + sum_{k!=i,j} a[i,k]*(b[pj,pk]-b[pi,pk])
          + a[j,i]*(b[pi,pj]-b[pj,pi]) + a[j,j]*(b[pi,pi]-b[pj,pj])
          + sum_{k!=i,j} a[j,k]*(b[pi,pk]-b[pj,pk])
          + sum_{k!=i,j} a[k,i]*(b[pk,pj]-b[pk,pi])
          + sum_{k!=i,j} a[k,j]*(b[pk,pi]-b[pk,pj])
    
    Note: a[i,j]*(b[pj,pi]-b[pi,pj]) + a[j,i]*(b[pi,pj]-b[pj,pi]) 
         = (a[i,j]-a[j,i])*(b[pj,pi]-b[pi,pj])
    
    And a[i,i]*(b[pj,pj]-b[pi,pi]) + a[j,j]*(b[pi,pi]-b[pj,pj])
         = (a[i,i]-a[j,j])*(b[pj,pj]-b[pi,pi])
    
    For the cross terms with k:
    sum_{k!=i,j} [a[i,k]*(b[pj,pk]-b[pi,pk]) + a[j,k]*(b[pi,pk]-b[pj,pk]) 
                  + a[k,i]*(b[pk,pj]-b[pk,pi]) + a[k,j]*(b[pk,pi]-b[pk,pj])]
    
    Let me define:
    S1 = sum_{k!=i,j} a[i,k]*(b[pj,pk]-b[pi,pk])
    S2 = sum_{k!=i,j} a[j,k]*(b[pi,pk]-b[pj,pk]) = -sum_{k!=i,j} a[j,k]*(b[pj,pk]-b[pi,pk])
    S3 = sum_{k!=i,j} a[k,i]*(b[pk,pj]-b[pk,pi])
    S4 = sum_{k!=i,j} a[k,j]*(b[pk,pi]-b[pk,pj]) = -sum_{k!=i,j} a[k,j]*(b[pk,pj]-b[pk,pi])
    
    So delta = (a[i,i]-a[j,j])*(b[pj,pj]-b[pi,pi]) + (a[i,j]-a[j,i])*(b[pj,pi]-b[pi,pj])
             + S1 + S2 + S3 + S4
    
    For efficiency, I'll precompute some things or just compute directly.
    """
    pi = p[i]
    pj = p[j]
    
    delta = 0.0
    
    # Diagonal terms
    delta += (a[i, i] - a[j, j]) * (b[pj, pj] - b[pi, pi])
    
    # Off-diagonal i,j terms
    delta += (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    
    # Cross terms with k != i, j
    # S1: sum_{k!=i,j} a[i,k] * (b[pj,pk] - b[pi,pk])
    # S2: -sum_{k!=i,j} a[j,k] * (b[pj,pk] - b[pi,pk])
    # S3: sum_{k!=i,j} a[k,i] * (b[pk,pj] - b[pk,pi])
    # S4: -sum_{k!=i,j} a[k,j] * (b[pk,pj] - b[pk,pi])
    
    # Combine S1 and S2:
    # sum_{k!=i,j} (a[i,k] - a[j,k]) * (b[pj,pk] - b[pi,pk])
    
    # Combine S3 and S4:
    # sum_{k!=i,j} (a[k,i] - a[k,j]) * (b[pk,pj] - b[pk,pi])
    
    # For k != i, j:
    mask = np.ones(len(p), dtype=bool)
    mask[i] = False
    mask[j] = False
    ks = np.where(mask)[0]
    
    if len(ks) > 0:
        # b[pj, pk] - b[pi, pk] for all k in ks
        diff1 = b[pj, p[ks]] - b[pi, p[ks]]
        # a[i, k] - a[j, k] for all k in ks
        diff2 = a[i, ks] - a[j, ks]
        delta += np.dot(diff2, diff1)
        
        # b[pk, pj] - b[pk, pi] for all k in ks
        diff3 = b[p[ks], pj] - b[p[ks], pi]
        # a[k, i] - a[k, j] for all k in ks
        diff4 = a[ks, i] - a[ks, j]
        delta += np.dot(diff4, diff3)
    
    return delta


def _local_search(a, b, p, max_time=100.0):
    """
    Perform 2-opt local search on the QAP assignment.
    p is the permutation array.
    Returns improved p and its cost.
    """
    n = len(p)
    p = p.copy()
    start_time = time.time()
    
    # Compute initial cost
    cost = _qap_cost(a, b, p)
    
    # Precompute C for potential use, but we'll compute deltas directly
    # Actually, let's just iterate over all pairs and find the best improving move
    
    improved = True
    while improved:
        improved = False
        best_delta = 0
        best_i = -1
        best_j = -1
        
        # Check time
        if time.time() - start_time > max_time:
            break
        
        # Find best improving swap
        for i in range(n):
            if time.time() - start_time > max_time:
                break
            for j in range(i + 1, n):
                delta = _swap_delta(None, a, b, p, i, j)
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
        
        if best_i >= 0:
            # Perform the swap
            p[best_i], p[best_j] = p[best_j], p[best_i]
            cost += best_delta
            improved = True
    
    return p, cost


def _anneal(a, b, p, start_temp=1.0, end_temp=0.01, max_time=100.0):
    """
    Simple simulated annealing with 2-opt moves.
    """
    n = len(p)
    p = p.copy()
    start_time = time.time()
    cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = cost
    
    temp = start_temp
    cooling_rate = 0.999
    
    while time.time() - start_time < max_time:
        # Pick random pair
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        
        delta = _swap_delta(None, a, b, p, i, j)
        
        if delta < 0 or np.random.random() < np.exp(-delta / temp):
            p[i], p[j] = p[j], p[i]
            cost += delta
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()
        
        temp *= cooling_rate
        if temp < end_temp:
            break
    
    return best_p, best_cost


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Rank-matching start
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Local search
    if time.time() < deadline - 10:
        p, cost = _local_search(a, b, p, max_time=deadline - time.time() - 5)
    
    # If we have more time, try a few restarts with perturbations
    best_p = p.copy()
    best_cost = cost
    
    if time.time() < deadline - 15:
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            # Perturb: random swap
            p2 = best_p.copy()
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i != j:
                p2[i], p2[j] = p2[j], p2[i]
            
            p2, cost2 = _local_search(a, b, p2, max_time=deadline - time.time() - 3)
            if cost2 < best_cost:
                best_cost = cost2
                best_p = p2
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
