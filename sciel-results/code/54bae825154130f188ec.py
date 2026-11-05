import time
import random
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    """
    Find a large subset of [1, 128] with no three distinct elements whose
    three pairwise gcds are equal.
    
    Strategy: Model as 3-uniform hypergraph, use weighted greedy (lowest degree first)
    to build a large independent set, then refine with local search swaps.
    """
    
    DEADLINE = time.time() + 82.0  # safety margin under 90s
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute gcd matrix
    gcd_mat = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g
    
    # Precompute all bad triples and their structure
    bad_triples = []
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            for k in range(j + 1, N + 1):
                if gcd_mat[i][j] == gcd_mat[i][k] == gcd_mat[j][k]:
                    bad_triples.append((i, j, k))
    
    # Compute hypergraph degree: number of bad triples each vertex participates in
    degree = [0] * (N + 1)
    for (a, b, c) in bad_triples:
        degree[a] += 1
        degree[b] += 1
        degree[c] += 1
    
    # For fast conflict checking: for each pair (i,j), store the gcd value
    # and for each element x, we can quickly check if adding x creates a bad triple
    
    def can_add(x, selected):
        """Check if x can be added to selected without creating a bad triple."""
        sel = list(selected)
        n = len(sel)
        for i in range(n):
            for j in range(i + 1, n):
                u, v = sel[i], sel[j]
                if gcd_mat[x][u] == gcd_mat[x][v] == gcd_mat[u][v]:
                    return False
        return True
    
    def greedy_build_lowest_degree(order):
        """Build a valid set by adding elements in the given order."""
        selected = set()
        for x in order:
            if can_add(x, selected):
                selected.add(x)
        return selected
    
    def local_search_add_only(selected, deadline=DEADLINE):
        """Try to add excluded elements one at a time."""
        selected = set(selected)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            excluded = [x for x in range(1, N + 1) if x not in selected]
            for x in excluded:
                if time.time() > deadline:
                    break
                if can_add(x, selected):
                    selected.add(x)
                    improved = True
        return selected
    
    def local_search_swap(selected, max_rounds=50, deadline=DEADLINE):
        """Improve by removing one element and trying to add multiple excluded ones."""
        selected = set(selected)
        current_size = len(selected)
        
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            
            best_new = None
            best_gain = 0
            
            excluded = [x for x in range(1, N + 1) if x not in selected]
            
            # Try removing each element in selected
            for x in list(selected):
                if time.time() > deadline:
                    break
                
                temp = selected - {x}
                new_sel = set(temp)
                
                # Try adding excluded elements to temp
                for y in excluded:
                    if time.time() > deadline:
                        break
                    if can_add(y, new_sel):
                        new_sel.add(y)
                
                gain = len(new_sel) - current_size
                if gain > best_gain:
                    best_gain = gain
                    best_new = new_sel
            
            if best_new is not None:
                selected = best_new
                current_size = len(selected)
            else:
                break
        
        return selected
    
    def weighted_greedy_with_tiebreak(seed_val):
        """
        Weighted greedy: prioritize vertices with lowest degree.
        Use seed for deterministic tie-breaking variations.
        """
        rng = random.Random(seed_val)
        # Sort by (degree, random tiebreak)
        order = sorted(elements, key=lambda x: (degree[x], rng.random()))
        return greedy_build_lowest_degree(order)
    
    def multi_start_local_search(initial_set, n_starts=10, deadline=DEADLINE):
        """Run local search from multiple perturbations of initial_set."""
        best = set(initial_set)
        best_size = len(initial_set)
        
        for s in range(n_starts):
            if time.time() > deadline:
                break
            
            # Perturb: remove a few random elements
            perturbed = set(initial_set)
            n_remove = min(3, len(perturbed))
            to_remove = random.sample(list(perturbed), n_remove)
            for x in to_remove:
                perturbed.discard(x)
            
            # Add back what we can
            perturbed = local_search_add_only(perturbed, deadline)
            perturbed = local_search_swap(perturbed, max_rounds=20, deadline=deadline)
            
            if len(perturbed) > best_size:
                best_size = len(perturbed)
                best = perturbed
        
        return best
    
    # --- Main search ---
    
    best_set = set()
    best_size = 0
    
    # Phase 1: Weighted greedy with lowest degree (multiple tie-breaks)
    for seed in range(10):
        if time.time() > DEADLINE:
            break
        s = weighted_greedy_with_tiebreak(seed)
        s = local_search_add_only(s)
        s = local_search_swap(s, max_rounds=30)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
    
    if time.time() > DEADLINE:
        return sorted(best_set) if best_set else [1]
    
    # Phase 2: Simple degree-ordered greedy
    order_by_degree = sorted(elements, key=lambda x: degree[x])
    s = greedy_build_lowest_degree(order_by_degree)
    s = local_search_add_only(s)
    s = local_search_swap(s, max_rounds=30)
    if len(s) > best_size:
        best_size = len(s)
        best_set = s
    
    if time.time() > DEADLINE:
        return sorted(best_set) if best_set else [1]
    
    # Phase 3: Randomized greedy builds
    for trial in range(30):
        if time.time() > DEADLINE:
            break
        order = list(elements)
        random.shuffle(order)
        s = greedy_build_lowest_degree(order)
        s = local_search_add_only(s)
        s = local_search_swap(s, max_rounds=20)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
    
    if time.time() > DEADLINE:
        return sorted(best_set) if best_set else [1]
    
    # Phase 4: Multi-start local search from best
    s = multi_start_local_search(best_set, n_starts=15, deadline=DEADLINE)
    if len(s) > best_size:
        best_size = len(s)
        best_set = s
    
    if time.time() > DEADLINE:
        return sorted(best_set) if best_set else [1]
    
    # Phase 5: Final polish - more aggressive local search
    s = local_search_add_only(best_set)
    s = local_search_swap(s, max_rounds=100, deadline=DEADLINE)
    if len(s) > best_size:
        best_size = len(s)
        best_set = s
    
    # Phase 6: One more round of multi-start
    if time.time() < DEADLINE - 5:
        s = multi_start_local_search(best_set, n_starts=10, deadline=DEADLINE)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
    
    if not best_set:
        best_set = {1}
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
