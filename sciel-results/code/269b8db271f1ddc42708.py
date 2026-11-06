import math
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # 1. Generate all candidate classes (m, r)
    # m in [20, 200], r in [0, m)
    candidates = []
    weights = []
    for m in range(20, 201):
        for r in range(m):
            candidates.append((m, r))
            weights.append(1.0 / m)
            
    n = len(candidates)
    
    # 2. Build Conflict Graph
    # Two classes (m1, r1) and (m2, r2) conflict if they are NOT disjoint.
    # They are disjoint iff r1 % gcd(m1, m2) != r2 % gcd(m1, m2).
    # So they conflict if r1 % gcd(m1, m2) == r2 % gcd(m1, m2).
    
    # To speed up graph construction, group candidates by modulus.
    # If m1 == m2, gcd is m1. Conflict if r1 == r2. 
    # But we have distinct moduli in the final set? 
    # The task says "distinct moduli". 
    # So we cannot pick two classes with the same m.
    # We can enforce this by adding edges between all pairs with same m, or just picking one.
    # Actually, the problem statement says "distinct moduli 20 <= m <= 200".
    # So we must pick at most one residue per modulus.
    # This simplifies things: we can treat this as a Maximum Weight Independent Set 
    # on a graph where vertices are (m, r), but we have the constraint that 
    # at most one vertex per m can be chosen.
    # Alternatively, we can just build the full graph including same-m conflicts 
    # and let the solver handle it. Since r1 != r2 for same m is required for 
    # disjointness? No, if m1=m2, gcd=m. r1 % m == r2 % m implies r1==r2.
    # So yes, (m, r1) and (m, r2) conflict if r1 != r2? 
    # Wait. Disjointness condition: r1 != r2 mod gcd(m1, m2).
    # If m1=m2=m, gcd=m. Condition: r1 != r2 mod m. 
    # Since 0 <= r < m, this means r1 != r2.
    # So any two different residues for the same modulus are disjoint?
    # No. The sets {x | x ≡ r1 mod m} and {x | x ≡ r2 mod m} are disjoint 
    # if and only if r1 != r2 mod m.
    # So yes, (m, 0) and (m, 1) are disjoint.
    # BUT the task requires "distinct moduli".
    # "construct() -> classes [[m, r], ...] with distinct moduli 20 <= m <= 200"
    # This implies we can only pick ONE class per modulus m.
    # So we can reduce the search space by picking the "best" residue for each m 
    # or just treating it as a set packing where we pick subsets of moduli and 
    # assign residues.
    
    # However, the residue choice affects compatibility with OTHER moduli.
    # So we must keep all (m, r) as nodes.
    # But we add edges between all (m, r1) and (m, r2) for r1 != r2?
    # No, if we pick (m, r1) and (m, r2), the moduli are not distinct. 
    # The output must have distinct moduli.
    # So we should enforce that we pick at most one node per m.
    # We can do this by adding edges between all nodes with the same m.
    # This makes them mutually exclusive in the independent set.
    
    # Let's build the adjacency list.
    # Optimization: Precompute gcds and group by m.
    adj = [[] for _ in range(n)]
    
    # Map index to (m, r)
    # To find nodes with same m quickly, group indices by m.
    indices_by_m = {}
    for i, (m, r) in enumerate(candidates):
        if m not in indices_by_m:
            indices_by_m[m] = []
        indices_by_m[m].append(i)
        
    # Add edges for same modulus (mutual exclusion due to distinct moduli constraint)
    for m, inds in indices_by_m.items():
        for i in range(len(inds)):
            for j in range(i+1, len(inds)):
                u = inds[i]
                v = inds[j]
                adj[u].append(v)
                adj[v].append(u)
                
    # Add edges for different moduli
    # For each pair of moduli m1, m2, check all pairs of residues.
    # To avoid O(N^2) which is 18k^2 = 324M, we can be smarter.
    # But 324M simple integer ops in Python might be too slow.
    # Let's estimate. 18k nodes. 
    # We can iterate over moduli pairs.
    # Number of moduli is 181. Pairs ~ 16k.
    # For each pair (m1, m2), we have m1*m2 residue pairs.
    # Sum over all pairs of m1*m2.
    # This is roughly (Sum m)^2 / 2. Sum m ~ 181*110 ~ 20k. 20k^2 / 2 = 200M.
    # Still large.
    
    # Alternative: Iterate over all node pairs (i, j) where m_i < m_j.
    # 18k nodes. 18k^2 / 2 = 162M pairs.
    # For each pair, compute gcd and check remainder.
    # math.gcd is fast. 
    # 162M iterations in Python might take 10-30 seconds. Acceptable?
    # Let's try to optimize.
    # We can precompute gcd for all modulus pairs.
    # gcds = {}
    # for m1 in range(20, 201):
    #   for m2 in range(m1+1, 201):
    #     gcds[(m1, m2)] = math.gcd(m1, m2)
    
    # Then for nodes i, j:
    # g = gcds[(m_i, m_j)]
    # if r_i % g == r_j % g: conflict
    
    # Let's do this.
    gcd_cache = {}
    for m1 in range(20, 201):
        for m2 in range(m1 + 1, 201):
            gcd_cache[(m1, m2)] = math.gcd(m1, m2)
            
    # Build adjacency
    for i in range(n):
        m1, r1 = candidates[i]
        for j in range(i + 1, n):
            m2, r2 = candidates[j]
            if m1 == m2:
                # Already handled by same-m edges? 
                # Yes, we added edges for same m. 
                # So we can skip here to avoid duplicates, 
                # or just check if m1 != m2.
                continue
            # Ensure m1 < m2 for cache lookup
            if m1 > m2:
                m1, m2 = m2, m1
                r1, r2 = r2, r1 # Order doesn't matter for equality check
            
            g = gcd_cache[(m1, m2)]
            if r1 % g == r2 % g:
                adj[i].append(j)
                adj[j].append(i)
                
    # 3. Solve Maximum Weight Independent Set
    # We use a branch and bound algorithm.
    # Since the graph is large, we need a good heuristic for ordering and pruning.
    
    # Convert adj to set for O(1) lookup? Or just list and check membership?
    # For 18k nodes, degree might be high.
    # Let's use a simple recursive backtracking with memoization? 
    # No, state space is too large.
    # We use a greedy approximation for the bound.
    
    # Heuristic: 
    # 1. Sort nodes by weight descending.
    # 2. In the search tree, at each node, we have a set of available nodes.
    # 3. We pick a node v. 
    # 4. Branch 1: Include v. Add weight[v]. Remove v and its neighbors from available.
    # 5. Branch 2: Exclude v. Remove v from available.
    
    # To make this fast, we need a good upper bound.
    # Upper bound = sum of weights of all available nodes.
    # This is weak.
    # Better bound: Greedy packing of available nodes.
    # Or: Weighted degree heuristic.
    
    # Given the time limit (115s) and N=18k, a pure Python B&B might be too slow 
    # unless the graph structure allows very strong pruning.
    # The graph is dense for small moduli? 
    # Small moduli have many small factors, so many conflicts.
    # Large moduli are often prime or have large factors, so fewer conflicts?
    # If m is prime, gcd(m, m2) is 1 or m. 
    # If gcd=1, they are always disjoint? 
    # Wait. If gcd(m1, m2)=1, then r1 % 1 == 0 and r2 % 1 == 0. 
    # So they ALWAYS conflict? 
    # Let's re-verify.
    # Disjoint iff r1 != r2 mod g.
    # If g=1, r1%1=0, r2%1=0. 0!=0 is False.
    # So they are NOT disjoint.
    # So ANY two coprime moduli conflict?
    # Yes. 
    # This means the conflict graph is VERY dense.
    # Most pairs of moduli are coprime.
    # So the independent set will be small?
    # If we pick m1, we cannot pick any m2 coprime to m1.
    # This restricts us to moduli sharing factors with m1.
    
    # This structure suggests that the optimal set consists of moduli that are 
    # all multiples of some small set of primes, or specifically, 
    # we can pick a set of moduli that are "pairwise non-coprime".
    
    # Let's try a different approach: 
    # Since the graph is dense, the independent set size is likely small (maybe < 20-30).
    # We can use a maximum clique solver on the complement graph? 
    # Or just stick to B&B with a strong bound.
    
    # Let's try a greedy local search first to get a good initial solution, 
    # then use B&B to prove optimality or find better.
    
    # Greedy:
    # Start with empty set.
    # Repeatedly add the node (m, r) with highest weight 1/m 
    # that is compatible with current set.
    # This is O(N * K * K) where K is set size.
    
    best_solution = []
    best_weight = 0.0
    
    # Pre-sort candidates by weight descending (m ascending)
    sorted_indices = sorted(range(n), key=lambda i: -weights[i])
    
    # Run greedy multiple times with random perturbations?
    # Or just one good greedy.
    
    # Greedy 1: Strict weight order
    current_set = set()
    for idx in sorted_indices:
        m, r = candidates[idx]
        compatible = True
        for other_idx in current_set:
            om, or_ = candidates[other_idx]
            g = math.gcd(m, om)
            if r % g == or_ % g:
                compatible = False
                break
        if compatible:
            current_set.add(idx)
            
    w = sum(weights[i] for i in current_set)
    if w > best_weight:
        best_weight = w
        best_solution = list(current_set)
        
    # Try to improve with local search?
    # Swap one node out for two nodes in?
    # Or just accept the greedy result if B&B is too slow.
    
    # Given the density, the greedy result is likely close to optimal.
    # Let's try a simple B&B with a time limit.
    
    # To make B&B work, we need a fast bound.
    # Bound: Sum of weights of all nodes in the available set that are not adjacent to each other? 
    # Too slow to compute.
    # Bound: Sum of all weights. (Weak)
    
    # Alternative: Since we suspect the set is small, we can use 
    # "Maximum Weight Clique" on the compatibility graph (complement of conflict graph).
    # But the compatibility graph is sparse?
    # If conflict graph is dense, compatibility graph is sparse.
    # Yes! 
    # Two nodes are compatible if they are disjoint.
    # We want a Maximum Weight Clique in the compatibility graph.
    
    # Let's build the compatibility graph adjacency.
    # compat_adj[i] = list of j such that i and j are compatible.
    
    # Building compat_adj is O(N^2).
    # We already computed conflicts. 
    # compat_adj[i] = all j != i not in adj[i].
    
    # This is expensive to store explicitly if N=18k.
    # But we can compute it on the fly or store as bitsets?
    # Python doesn't have native bitsets for this.
    
    # Let's stick to the greedy result. 
    # The greedy result with weight ordering is usually very good for this type of problem.
    # Let's add a second greedy pass with a different ordering (e.g., random) 
    # to see if we can beat it.
    
    import random
    for _ in range(10):
        if time.time() > deadline - 5:
            break
        random.shuffle(sorted_indices)
        current_set = set()
        for idx in sorted_indices:
            m, r = candidates[idx]
            compatible = True
            for other_idx in current_set:
                om, or_ = candidates[other_idx]
                g = math.gcd(m, om)
                if r % g == or_ % g:
                    compatible = False
                    break
            if compatible:
                current_set.add(idx)
        w = sum(weights[i] for i in current_set)
        if w > best_weight:
            best_weight = w
            best_solution = list(current_set)
            
    # Convert indices back to [[m, r], ...]
    result = []
    for idx in best_solution:
        m, r = candidates[idx]
        result.append([m, r])
        
    # Sort by m for consistent output? Not required, but nice.
    result.sort()
    
    return result

