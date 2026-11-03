"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    # Precompute useful arrays
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)

    def compute_cost(p):
        """Compute exact cost: sum_{i,j} a[i,j] * b[p[i], p[j]]"""
        # Vectorized: for each i, sum_j a[i,j] * b[p[i], p[j]]
        # = sum_i dot(a[i,:], b[p[i],:])
        # Use einsum for efficiency
        # b[np.ix_(p, p)] gives the permuted matrix
        # cost = sum(a * b[np.ix_(p,p)])
        # But that creates a full n x n array. For n=150, that's 22500 elements, fine.
        bp = b[np.ix_(p, p)]
        return int(np.sum(a * bp))

    def compute_swap_delta(i, j, p):
        """
        Compute the change in cost when swapping locations of facilities i and j.
        p[i] and p[j] are swapped.
        """
        li = p[i]
        lj = p[j]

        if li == lj:
            return 0

        # The cost is sum_{u,v} a[u,v] * b[p[u], p[v]]
        # When we swap p[i] and p[j], only terms where u or v is i or j change.
        
        # Let's derive this carefully.
        # Old cost contribution from terms involving i or j:
        # Terms where u=i: sum_v a[i,v] * b[li, p[v]]
        # Terms where v=i: sum_u a[u,i] * b[p[u], li]  (note: u=i term counted twice if we're not careful, but a[i,i]b[li,li] appears in both)
        # Actually, let's just compute the difference directly.
        
        # New p: p'[i]=lj, p'[j]=li, p'[k]=p[k] for k!=i,j
        
        # Delta = new_cost - old_cost
        # Only terms where at least one index is i or j change.
        
        # Let's group:
        # 1. a[i,i]*b[li,li] -> a[i,i]*b[lj,lj]
        # 2. a[j,j]*b[lj,lj] -> a[j,j]*b[li,li]
        # 3. a[i,j]*b[li,lj] -> a[i,j]*b[lj,li]
        # 4. a[j,i]*b[lj,li] -> a[j,i]*b[li,lj]
        # 5. For k not in {i,j}:
        #    a[i,k]*b[li,p[k]] -> a[i,k]*b[lj,p[k]]
        #    a[k,i]*b[p[k],li] -> a[k,i]*b[p[k],lj]
        #    a[j,k]*b[lj,p[k]] -> a[j,k]*b[li,p[k]]
        #    a[k,j]*b[p[k],lj] -> a[k,j]*b[p[k],li]
        
        delta = 0
        
        # Diagonal terms
        delta += a[i,i] * (b[lj,lj] - b[li,li])
        delta += a[j,j] * (b[li,li] - b[lj,lj])
        
        # Cross terms between i and j
        delta += a[i,j] * (b[lj,li] - b[li,lj])
        delta += a[j,i] * (b[li,lj] - b[lj,li])
        
        # Terms with other facilities
        # Vectorized over all k != i, j
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        ks = np.where(mask)[0]
        
        if len(ks) > 0:
            pk = p[ks]
            
            # a[i,k] * (b[lj,pk] - b[li,pk])
            delta += np.sum(a[i, ks] * (b[lj, pk] - b[li, pk]))
            # a[k,i] * (b[pk,lj] - b[pk,li])
            delta += np.sum(a[ks, i] * (b[pk, lj] - b[pk, li]))
            # a[j,k] * (b[li,pk] - b[lj,pk])
            delta += np.sum(a[j, ks] * (b[li, pk] - b[lj, pk]))
            # a[k,j] * (b[pk,li] - b[pk,lj])
            delta += np.sum(a[ks, j] * (b[pk, li] - b[pk, lj]))
        
        return int(delta)

    # Initial heuristic: rank matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    best_p = p.copy()
    best_cost = compute_cost(p)

    # Tabu search with 2-opt swaps
    tabu_list = {}  # (i, j) -> iteration when it becomes non-tabu
    tabu_tenure = 20
    current_iter = 0
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_iter += 1
        
        # Find best improving (or best non-tabu) swap
        best_delta = 0
        best_i = -1
        best_j = -1
        
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i+1, n):
                if time.time() > deadline:
                    break
                    
                # Check tabu
                key = (i, j)
                if key in tabu_list and tabu_list[key] > current_iter:
                    continue
                
                delta = compute_swap_delta(i, j, p)
                
                # Accept if improving, or if best non-tabu move
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
        
        if best_i >= 0 and best_delta < 0:
            # Perform the swap
            p[best_i], p[best_j] = p[best_j], p[best_i]
            best_cost += best_delta
            improved = True
            
            # Update tabu list
            tabu_list[(best_i, best_j)] = current_iter + tabu_tenure
            # Also mark the reverse as tabu
            tabu_list[(best_j, best_i)] = current_iter + tabu_tenure
            
            # Update best
            if best_cost < compute_cost(best_p):
                best_p = p.copy()
            
            # Occasionally clear old tabu entries
            if current_iter % 100 == 0:
                tabu_list = {k: v for k, v in tabu_list.items() if v > current_iter}
    
    # If time remains, try a few more restarts with different heuristics
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        rng = np.random.RandomState(42)
        
        for restart in range(3):
            if time.time() > deadline - 5:
                break
            
            if restart == 0:
                # Random permutation
                p = np.random.permutation(n).astype(np.int64)
            elif restart == 1:
                # Reverse rank matching
                heavy = np.argsort(-a.sum(axis=1), kind="stable")
                central = np.argsort(b.sum(axis=1), kind="stable")
                p = np.empty(n, dtype=np.int64)
                p[heavy] = central[::-1]
            else:
                # Random with some structure
                p = np.random.permutation(n).astype(np.int64)
            
            current_cost = compute_cost(p)
            
            # Quick local search
            tabu_list = {}
            current_iter = 0
            for _ in range(50):  # Limited iterations for restarts
                if time.time() > deadline - 5:
                    break
                
                best_delta = 0
                best_i = -1
                best_j = -1
                
                for i in range(n):
                    if time.time() > deadline - 5:
                        break
                    for j in range(i+1, n):
                        if time.time() > deadline - 5:
                            break
                        key = (i, j)
                        if key in tabu_list and tabu_list[key] > current_iter:
                            continue
                        delta = compute_swap_delta(i, j, p)
                        if delta < best_delta:
                            best_delta = delta
                            best_i = i
                            best_j = j
                
                if best_i >= 0 and best_delta < 0:
                    p[best_i], p[best_j] = p[best_j], p[best_i]
                    current_cost += best_delta
                    current_iter += 1
                    tabu_list[(best_i, best_j)] = current_iter + tabu_tenure
                    tabu_list[(best_j, best_i)] = current_iter + tabu_tenure
                else:
                    break
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()

    return best_p.tolist()
# EVOLVE-BLOCK-END
