import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    # All candidate moduli in [20, 200]
    all_moduli = list(range(20, 201))
    n = len(all_moduli)
    
    # Precompute gcd matrix and weights
    gcd_mat = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            gcd_mat[i][j] = gcd_mat[j][i] = gcd(all_moduli[i], all_moduli[j])
    
    weights = [1.0/all_moduli[i] for i in range(n)]
    
    # Determine compatibility: two moduli are compatible if they can be made disjoint
    # Incompatible if gcd=1 (must be different mod 1, always same) 
    # Actually gcd=1 means r1 mod 1 = r2 mod 1 = 0, so they're always "same" mod 1
    # Wait: r1 != r2 mod gcd. If gcd=1, r1 mod 1 = 0, r2 mod 1 = 0, so 0 != 0 is false.
    # So gcd=1 pairs are INCOMPATIBLE.
    # If gcd=2 and both even: r1 mod 2 and r2 mod 2 can be 0 or 1. We can choose different.
    # Actually for any gcd >= 2, we can always find residues that differ mod gcd.
    # So only gcd=1 pairs are incompatible.
    
    # Wait, let me re-examine. The constraint is r1 != r2 mod gcd(m1,m2).
    # If gcd=1: r1 mod 1 = 0, r2 mod 1 = 0, so 0 != 0 is FALSE. Incompatible.
    # If gcd>=2: We can pick r1=0, r2=1, and 0 != 1 mod gcd (for gcd>=2). Compatible.
    
    # So the conflict graph has edges between pairs with gcd=1.
    # We need maximum weight independent set in this graph.
    
    # Build adjacency: adj[i] is set of j incompatible with i
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if gcd_mat[i][j] == 1:
                adj[i].add(j)
                adj[j].add(i)
    
    # Sort nodes by weight descending for better branching
    order = sorted(range(n), key=lambda i: -weights[i])
    # Remap: new index position in sorted order
    pos = [0]*n
    for new_idx, old_idx in enumerate(order):
        pos[old_idx] = new_idx
    
    # Rebuild in sorted order
    sorted_weights = [weights[order[i]] for i in range(n)]
    sorted_moduli = [all_moduli[order[i]] for i in range(n)]
    
    # Rebuild adjacency in sorted order
    sorted_adj = [set() for _ in range(n)]
    for i in range(n):
        for j in adj[order[i]]:
            sorted_adj[i].add(pos[j])
    
    # Precompute for each node, the set of neighbors with higher index (for pruning)
    # and for each node, neighbors that are "later" in the order
    
    # Branch and bound for maximum weight independent set
    best_weight = 0.0
    best_set = []
    
    # Use a recursive branch and bound with memoization-like pruning
    # State: (idx, selected_frozen_set) where we decide for nodes idx..n-1
    # But this is exponential. Let's use a smarter approach.
    
    # Alternative: Since the conflict graph is defined by gcd=1, and many numbers
    # in [20,200] share factors, the independent set is actually quite large.
    
    # Let me think about this differently. Two numbers are incompatible iff gcd=1.
    # So compatible iff gcd>=2. We want max weight subset where all pairs have gcd>=2.
    
    # This is a maximum weight clique in the compatibility graph (complement of conflict).
    # Or equivalently, maximum weight independent set in the conflict graph.
    
    # Let's use a greedy + local search approach that's more effective:
    # 1. Start with greedy: add moduli in order of 1/m if compatible with all selected
    # 2. Local search: try to add unselected moduli by removing conflicting selected ones
    # 3. Randomized restarts
    
    import random
    
    def greedy_construction(order_list, adj_sets):
        """Greedy: add nodes in given order if compatible with all selected."""
        selected = set()
        for i in order_list:
            compatible = True
            for j in selected:
                if j in adj_sets[i]:
                    compatible = False
                    break
            if compatible:
                selected.add(i)
        return selected
    
    def local_search(selected, adj_sets, max_iter=5000):
        """Try to improve by swapping or adding."""
        selected = set(selected)
        current_weight = sum(sorted_weights[i] for i in selected)
        unselected = set(range(n)) - selected
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try to add an unselected node
            for u in unselected:
                # Check if u is compatible with all selected
                can_add = all(u not in adj_sets[s] for s in selected)
                if can_add:
                    selected.add(u)
                    unselected.remove(u)
                    current_weight += sorted_weights[u]
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swap: remove one selected, add one unselected
            best_gain = 0
            best_swap = None
            for s in list(selected):
                for u in unselected:
                    # Check if u is compatible with all selected except s
                    can_add = all(u not in adj_sets[s2] for s2 in selected if s2 != s)
                    if can_add:
                        gain = sorted_weights[u] - sorted_weights[s]
                        if gain > best_gain:
                            best_gain = gain
                            best_swap = (s, u)
            
            if best_swap is not None:
                s, u = best_swap
                selected.remove(s)
                unselected.remove(u)
                selected.add(u)
                current_weight += best_gain
                improved = True
        
        return selected, current_weight
    
    # Run multiple randomized restarts
    best_result_weight = 0.0
    best_result_set = set()
    
    # First, try a deterministic greedy in weight order
    initial = greedy_construction(list(range(n)), sorted_adj)
    initial, w = local_search(initial, sorted_adj)
    best_result_set = initial
    best_result_weight = w
    
    # Randomized restarts
    rng = random.Random(42)
    num_restarts = 50
    restarts_done = 0
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Create a random perturbation of the order
        perm = list(range(n))
        # Partially shuffle: keep top 20% in order, shuffle rest
        cutoff = n // 5
        perm[cutoff:] = perm[cutoff:]
        rng.shuffle(perm[cutoff:])
        
        current = greedy_construction(perm, sorted_adj)
        current, w = local_search(current, sorted_adj, max_iter=3000)
        
        if w > best_result_weight:
            best_result_weight = w
            best_result_set = current
        
        restarts_done += 1
    
    # Final local search with more iterations
    best_result_set, best_result_weight = local_search(best_result_set, sorted_adj, max_iter=10000)
    
    # Assign residues: for each selected modulus, pick a residue that's compatible with all others
    # Since we've verified pairwise compatibility, we can use a simple greedy residue assignment
    selected_indices = sorted(best_result_set)
    
    # Assign residues greedily
    residues = {}
    for idx in selected_indices:
        m = sorted_moduli[idx]
        # Try residues 0, 1, 2, ... until we find one compatible with all assigned
        found = False
        for r in range(m):
            compatible = True
            for other_idx in selected_indices:
                if other_idx == idx:
                    continue
                m_other = sorted_moduli[other_idx]
                r_other = residues[other_idx]
                g = gcd(m, m_other)
                if r % g == r_other % g:
                    compatible = False
                    break
            if compatible:
                residues[idx] = r
                found = True
                break
        if not found:
            # This shouldn't happen for pairwise compatible sets, but handle gracefully
            residues[idx] = 0
    
    # Build result
    result = []
    for idx in selected_indices:
        m = sorted_moduli[idx]
        r = residues[idx]
        result.append([m, r])
    
    return result
