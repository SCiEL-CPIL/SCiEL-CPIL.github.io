import math
import random

def construct():
    N = 512
    # Precompute LCMs? No, compute on fly or precompute compatibility.
    # Compatibility matrix is 512x512. 262k entries. Fine.
    # compat[i][j] = True if lcm(i,j) <= N
    
    # To save time, we can just check on the fly in the greedy loop.
    # But for local search, we need quick checks.
    
    # 1. Generate candidates
    candidates = list(range(1, N + 1))
    
    # 2. Precompute conflict degrees for sorting heuristic
    # deg[i] = number of j such that lcm(i,j) > N
    deg = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            # Calculate LCM
            g = math.gcd(i, j)
            l = (i // g) * j
            if l > N:
                deg[i] += 1
                deg[j] += 1
                
    # 3. Greedy construction
    def greedy_solve(order):
        S = set()
        for n in order:
            # Check if n is compatible with all in S
            compatible = True
            for m in S:
                g = math.gcd(n, m)
                l = (n // g) * m
                if l > N:
                    compatible = False
                    break
            if compatible:
                S.add(n)
        return S
        
    best_S = set()
    
    # Run multiple times with different orderings
    # Ordering 1: Ascending degree
    order1 = sorted(candidates, key=lambda x: deg[x])
    S1 = greedy_solve(order1)
    if len(S1) > len(best_S):
        best_S = S1
        
    # Ordering 2: Descending degree (risky, but sometimes finds dense cores)
    order2 = sorted(candidates, key=lambda x: -deg[x])
    S2 = greedy_solve(order2)
    if len(S2) > len(best_S):
        best_S = S2
        
    # Ordering 3: Randomized perturbations of order1
    for _ in range(50):
        # Shuffle the low-degree part? Or just random sample?
        # Let's try randomizing the order of elements with similar degrees
        # Or just a random permutation
        order_rand = candidates[:]
        random.shuffle(order_rand)
        # But random permutation is bad. 
        # Better: Start with order1, but randomly swap adjacent elements with prob p
        order_perturbed = order1[:]
        for i in range(len(order_perturbed) - 1):
            if random.random() < 0.1:
                order_perturbed[i], order_perturbed[i+1] = order_perturbed[i+1], order_perturbed[i]
        S_rand = greedy_solve(order_perturbed)
        if len(S_rand) > len(best_S):
            best_S = S_rand
            
    # Local Search: Try to add any remaining node
    # If no single node can be added, try to swap one out and one in?
    # Or try to remove one and add two?
    
    # Simple improvement: Try to add nodes not in best_S
    # If we can add, add it.
    changed = True
    while changed:
        changed = False
        for n in candidates:
            if n not in best_S:
                compatible = True
                for m in best_S:
                    g = math.gcd(n, m)
                    l = (n // g) * m
                    if l > N:
                        compatible = False
                        break
                if compatible:
                    best_S.add(n)
                    changed = True
                    
    # Try to swap: for u in best_S, for v not in best_S:
    # If best_S - {u} + {v} is valid, do it? 
    # This keeps size same. We want to increase size.
    # This local search might be slow if done naively O(N^2 * |S|).
    # |S| ~ 30. N=512. 512*30*512 ~ 7M ops. OK.
    
    # Let's do a few iterations of swap to potentially enable adding more later?
    # Actually, if we swap u->v, size is same. 
    # But v might have fewer conflicts, allowing us to add w later.
    
    for _ in range(10): # Iterations
        improved = False
        for u in list(best_S):
            for v in candidates:
                if v in best_S:
                    continue
                # Check if S - {u} + {v} is valid
                # Remove u, check v against rest
                valid = True
                for m in best_S:
                    if m == u:
                        continue
                    g = math.gcd(v, m)
                    l = (v // g) * m
                    if l > N:
                        valid = False
                        break
                if valid:
                    # Perform swap
                    best_S.remove(u)
                    best_S.add(v)
                    improved = True
                    # Try to add more after swap
                    # (The 'add' loop above handles this if we re-run it, 
                    # but let's just break and let the outer loop handle it? 
                    # No, just break inner loops to restart check)
                    break
            if improved:
                break
        if not improved:
            break
            
    # Final pass to add any compatible nodes
    changed = True
    while changed:
        changed = False
        for n in candidates:
            if n not in best_S:
                compatible = True
                for m in best_S:
                    g = math.gcd(n, m)
                    l = (n // g) * m
                    if l > N:
                        compatible = False
                        break
                if compatible:
                    best_S.add(n)
                    changed = True

    return list(best_S)
