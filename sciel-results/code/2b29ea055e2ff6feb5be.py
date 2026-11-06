import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_n = 5
    best_edges = []
    
    # Precompute combinations for k=6 up to n=22 to avoid overhead in loop
    # But n varies, so we compute inside the loop for the specific n
    
    # Target n=21 first (ambitious), then 20, 19...
    # The task says 20 is known, 21 would show G(6) >= 22.
    # Let's try 21, 20, 19, 18
    
    for n in [21, 20, 19, 18, 17, 16]:
        if time.time() > deadline - 10:
            break
            
        # Multiple restarts
        # With fast verification, we can afford more restarts
        max_restarts = 50 if n >= 20 else 100
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # 1. Generate Random Graph
            # Use p=0.5 as baseline, or vary it
            p = 0.5
            A = np.zeros((n, n), dtype=np.int8)
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        A[i, j] = 1
                        A[j, i] = 1
                        
            # Optional: Spectral Gap Filter?
            # Computing eigenvalues is O(n^3). For n=20, it's fast (~ms).
            # But if we do it for every restart, it might add up.
            # Let's try 10 random graphs, pick the one with largest spectral gap.
            # Actually, let's just do local search on the first one. If it fails, try next.
            # The local search is the main workhorse.
            
            adj = A.copy()
            
            # Local Search / Greedy Repair
            # We will iterate: find violation, flip edge in it, repeat.
            # To avoid getting stuck, we can add some randomness or simulated annealing logic,
            # but simple greedy often works if the verifier is fast.
            
            max_iter_ls = 2000
            it = 0
            found_valid = False
            
            while it < max_iter_ls:
                if time.time() > deadline - 2:
                    break
                
                # Check for violations of size 6
                # Vectorized check
                comb_6 = np.array(list(itertools.combinations(range(n), 6)))
                if len(comb_6) == 0:
                    break # n < 6
                    
                # Create mask matrix: (M, n)
                # M = len(comb_6)
                mask = np.zeros((len(comb_6), n), dtype=np.int8)
                for idx, combo in enumerate(comb_6):
                    mask[idx, combo] = 1
                
                # Degrees of each vertex in each subset: (M, n)
                # D[i, v] = sum_{u in combo_i} A[v, u]
                # This is mask @ A
                D = mask @ adj
                
                # Extract degrees for the vertices in the subset: (M, 6)
                # degrees[i, j] = D[i, comb_6[i, j]]
                degrees = D[np.arange(len(comb_6))[:, None], comb_6]
                
                # Check if all 6 degrees are equal
                # Compare each column to the first column
                is_regular = np.all(degrees == degrees[:, 0:1], axis=1)
                
                violating_indices = np.where(is_regular)[0]
                
                if len(violating_indices) == 0:
                    # No size-6 violations. Check size >= 7
                    # This is slower, so we only do it if 6 is clear.
                    valid = True
                    for k in range(7, n + 1):
                        comb_k = list(itertools.combinations(range(n), k))
                        if not comb_k:
                            continue
                        # For k > 6, we can do a simpler check or just loop
                        # Since k=7,8... are fewer, a python loop is acceptable here
                        # or we can vectorize similarly.
                        # Let's use a helper for k>6
                        if _has_violation_larger(adj, n, k):
                            valid = False
                            # Find one violation to fix
                            viol_sub = _find_violation_larger(adj, n, k)
                            if viol_sub:
                                # Flip an edge in viol_sub
                                u, v = random.choice([(a, b) for a in range(len(viol_sub)) for b in range(a+1, len(viol_sub))])
                                u, v = viol_sub[u], viol_sub[v]
                                adj[u, v] ^= 1
                                adj[v, u] ^= 1
                            break
                        else:
                            continue
                    
                    if valid:
                        found_valid = True
                        break
                    else:
                        # If we found a larger violation, we already flipped an edge.
                        # Continue loop to re-check 6s (flipping might create 6s)
                        it += 1
                        continue
                else:
                    # Found size-6 violations
                    # Pick one random violation
                    idx = random.choice(violating_indices)
                    subset = comb_6[idx]
                    
                    # Flip an edge inside this subset
                    # Random pair from subset
                    u, v = random.sample(list(subset), 2)
                    
                    # Flip
                    adj[u, v] ^= 1
                    adj[v, u] ^= 1
                    
                    it += 1
            
            if found_valid:
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i, j]:
                            edges.append([i, j])
                if n > best_n:
                    best_n = n
                    best_edges = edges
                return {'n': best_n, 'edges': best_edges}

    # Fallback: return best found or default
    if best_n == 5:
        # Try to find any small valid graph
        for n in [10, 9, 8, 7, 6]:
            if time.time() > deadline - 2:
                break
            for _ in range(20):
                A = np.zeros((n, n), dtype=np.int8)
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < 0.5:
                            A[i, j] = 1
                            A[j, i] = 1
                if not _has_violation_larger(A, n, 6) and not _check_6(A, n):
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if A[i, j]:
                                edges.append([i, j])
                    best_n = n
                    best_edges = edges
                    break
            if best_n > 5:
                break

    return {'n': best_n, 'edges': best_edges}

def _check_6(adj, n):
    """Vectorized check for any regular induced subgraph of size 6."""
    if n < 6:
        return False
    comb_6 = np.array(list(itertools.combinations(range(n), 6)))
    if len(comb_6) == 0:
        return False
    mask = np.zeros((len(comb_6), n), dtype=np.int8)
    for idx, combo in enumerate(comb_6):
        mask[idx, combo] = 1
    D = mask @ adj
    degrees = D[np.arange(len(comb_6))[:, None], comb_6]
    is_regular = np.all(degrees == degrees[:, 0:1], axis=1)
    return np.any(is_regular)

def _has_violation_larger(adj, n, k):
    """Check for regular induced subgraphs of size k (k > 6)."""
    if n < k:
        return False
    # For k > 6, the number of combinations is smaller.
    # We can use a similar vectorized approach or a simple loop.
    # Simple loop is safer for memory if k is close to n.
    for combo in itertools.combinations(range(n), k):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        d0 = (adj[combo[0]] & mask).bit_count()
        ok = True
        for v in combo[1:]:
            if (adj[v] & mask).bit_count() != d0:
                ok = False
                break
        if ok:
            return True
    return False

def _find_violation_larger(adj, n, k):
    """Find one regular induced subgraph of size k."""
    if n < k:
        return None
    for combo in itertools.combinations(range(n), k):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        d0 = (adj[combo[0]] & mask).bit_count()
        ok = True
        for v in combo[1:]:
            if (adj[v] & mask).bit_count() != d0:
                ok = False
                break
        if ok:
            return combo
    return None
