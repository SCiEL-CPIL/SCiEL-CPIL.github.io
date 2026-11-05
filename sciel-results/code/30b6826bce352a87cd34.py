import time
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    """
    Find a large subset of [1, 128] with no three distinct elements whose
    three pairwise gcds are equal.
    
    Strategy: Model as hypergraph independent set problem where each "bad triple"
    is a hyperedge. Use degree-aware greedy initialization + 1-1 and 2-1 exchange
    local search with multiple randomized restarts.
    """
    
    N = 128
    DEADLINE = time.time() + 85.0  # Safety margin below 90s limit
    
    # Precompute GCD table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Precompute all bad triples
    # A triple (a,b,c) is bad if gcd(a,b) == gcd(a,c) == gcd(b,c)
    bad_triples = []
    # Also build conflict structure: for each element, list of bad triples containing it
    triples_by_elem = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = gcd_table[a][b]
            for c in range(b + 1, N + 1):
                if g_ab == gcd_table[a][c] == gcd_table[b][c]:
                    triple = (a, b, c)
                    bad_triples.append(triple)
                    triples_by_elem[a].append(len(bad_triples) - 1)
                    triples_by_elem[b].append(len(bad_triples) - 1)
                    triples_by_elem[c].append(len(bad_triples) - 1)
    
    def is_valid(x, selected):
        """Check if adding x to selected creates no bad triple."""
        # Need to check all pairs in selected: does (x, a, b) form a bad triple?
        # Precomputed: for each pair (a,b) in selected, check if gcd(x,a)==gcd(x,b)==gcd(a,b)
        sel_list = list(selected)
        n = len(sel_list)
        for i in range(n):
            a = sel_list[i]
            g_xa = gcd_table[x][a]
            for j in range(i + 1, n):
                b = sel_list[j]
                if g_xa == gcd_table[x][b] == gcd_table[a][b]:
                    return False
        return True
    
    def count_conflicts(x, selected):
        """Count how many bad triples would be created by adding x."""
        sel_list = list(selected)
        n = len(sel_list)
        conflicts = 0
        for i in range(n):
            a = sel_list[i]
            g_xa = gcd_table[x][a]
            for j in range(i + 1, n):
                b = sel_list[j]
                if g_xa == gcd_table[x][b] == gcd_table[a][b]:
                    conflicts += 1
        return conflicts
    
    def greedy_build(order, selected=None):
        """Build a valid set by adding elements in the given order if no conflict."""
        if selected is None:
            selected = set()
        for x in order:
            if x not in selected:
                if count_conflicts(x, selected) == 0:
                    selected.add(x)
        return selected
    
    def local_search(selected, max_rounds=50):
        """Improve set via add, 1-1 swap, and 2-1 exchange moves."""
        selected = set(selected)
        excluded = set(range(1, N + 1)) - selected
        
        for _ in range(max_rounds):
            if time.time() > DEADLINE:
                break
            improved = False
            
            # Try adding excluded elements (sorted by conflict count ascending)
            add_candidates = sorted(excluded, key=lambda x: count_conflicts(x, selected))
            for x in add_candidates:
                if count_conflicts(x, selected) == 0:
                    selected.add(x)
                    excluded.remove(x)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try 1-1 swaps: remove r from selected, add a from excluded
            # Only accept if it doesn't reduce size (it won't) and potentially enables future adds
            # Actually for pure size maximization, 1-1 swaps don't help directly.
            # Skip pure 1-1 swaps and go to 2-1 exchanges.
            
            # Try 2-1 exchanges: remove 2 elements, add 1 excluded element
            # This can only help if the 2 removed were blocking 1 that we can add
            # Net change: -2 + 1 = -1, so this reduces size. Not useful for max size.
            
            # Better: try removing 1, then greedily add as many as possible
            # If we can add >= 1, we maintain or grow size
            for r in list(selected):
                if time.time() > DEADLINE:
                    break
                temp = selected - {r}
                # Try to add excluded elements to temp
                added = []
                for a in sorted(excluded - {r}, key=lambda x: count_conflicts(x, temp)):
                    if count_conflicts(a, temp | set(added)) == 0:
                        added.append(a)
                if len(added) >= 1:
                    new_set = temp | set(added)
                    if len(new_set) > len(selected):
                        selected = new_set
                        excluded = set(range(1, N + 1)) - selected
                        improved = True
                        break
                    elif len(added) == 1:
                        # Same size; accept if it changes composition (for diversity)
                        # Only accept occasionally to avoid cycling
                        if added[0] != r:  # Always true since r was removed
                            pass  # Skip same-size swaps for now to ensure progress
            
            if not improved:
                break
        
        return selected
    
    # --- Main search ---
    best_set = set()
    best_size = 0
    
    import random
    random.seed(12345)
    
    # Phase 1: Degree-aware greedy (add elements with fewest bad triples first)
    # Elements with fewer bad triples are "safer" to include
    elem_degree = [len(triples_by_elem[i]) for i in range(N + 1)]
    order1 = sorted(range(1, N + 1), key=lambda x: (elem_degree[x], -x))
    s1 = greedy_build(order1)
    s1 = local_search(s1)
    if len(s1) > best_size:
        best_size = len(s1)
        best_set = s1
    
    if time.time() > DEADLINE:
        return sorted(best_set) if best_set else [1]
    
    # Phase 2: Multiple randomized restarts
    for trial in range(300):
        if time.time() > DEADLINE:
            break
        
        # Randomized ordering with some bias toward low-degree elements
        order = list(range(1, N + 1))
        # Sort by (degree + random noise) to get varied but structured orderings
        noisy_order = sorted(order, key=lambda x: elem_degree[x] + random.random() * 3)
        
        s = greedy_build(noisy_order)
        s = local_search(s)
        
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
    
    # Phase 3: Aggressive local search on best solution
    if time.time() < DEADLINE:
        s_final = local_search(best_set, max_rounds=100)
        if len(s_final) > best_size:
            best_size = len(s_final)
            best_set = s_final
    
    if not best_set:
        best_set = {1}
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
