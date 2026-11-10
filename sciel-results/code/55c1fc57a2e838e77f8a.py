import time
from math import gcd
from collections import defaultdict

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85
    n = 128
    
    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        lcm_table[i][i] = i
        for j in range(i + 1, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Find all bad triples: (a,b,c) with a<b<c where lcm(a,b)==lcm(a,c)==lcm(b,c)
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))
    
    num_bad = len(bad_triples)
    
    # Build conflict graph: edge (x,y) exists if x,y appear together in some bad triple
    # Use bitmasks for fast operations
    # Map elements to 0-indexed
    elem_to_idx = {x: x - 1 for x in range(1, n + 1)}
    
    # conflict_mask[i] is a bitmask of all elements j that conflict with element i
    conflict_mask = [0] * n
    
    for (a, b, c) in bad_triples:
        ia, ib, ic = a - 1, b - 1, c - 1
        # a conflicts with b and c
        conflict_mask[ia] |= (1 << ib) | (1 << ic)
        conflict_mask[ib] |= (1 << ia) | (1 << ic)
        conflict_mask[ic] |= (1 << ia) | (1 << ib)
    
    # Also need to track which triples each pair is in (for verification)
    # But for the independent set on 2-section, we just need: no edge in selected set
    # Wait - that's not quite right. The constraint is no THREE elements all mutually connected.
    # Actually, if we have a bad triple (a,b,c), then a,b,c form a triangle in the conflict graph.
    # We need no triangle in the selected set.
    # This is NOT the same as independent set!
    # Independent set means no edges at all. We need no triangles (3-cliques).
    
    # So this is: maximum subset with no 3-clique (triangle-free)
    # This is different from independent set.
    
    # Let me reconsider. The problem is: no three distinct elements whose pairwise LCMs are equal.
    # So we need: for any three selected elements a,b,c, NOT(lcm(a,b)==lcm(a,c)==lcm(b,c))
    
    # This is a 3-uniform hypergraph independent set problem.
    # The "edges" are the bad triples.
    
    # For a 3-uniform hypergraph, we can use:
    # 1. Branch and bound on hypergraph independent set
    # 2. Or reformulate: for each bad triple, at most 2 of its 3 elements can be selected
    
    # Let me use a different approach: branch and bound with the following structure
    # - Order elements
    # - For each element, decide include/exclude
    # - Prune when a bad triple is fully included
    
    # Precompute for each element, the list of bad triples it participates in
    elem_triples = [[] for _ in range(n)]
    for i, (a, b, c) in enumerate(bad_triples):
        elem_triples[a-1].append(i)
        elem_triples[b-1].append(i)
        elem_triples[c-1].append(i)
    
    # For each bad triple, store the three elements (0-indexed)
    triple_elems = [(a-1, b-1, c-1) for (a, b, c) in bad_triples]
    
    # Strategy: Use a greedy + local search approach with better heuristics
    # First, identify elements that are in NO bad triples - they can always be included
    safe_elements = [i for i in range(n) if len(elem_triples[i]) == 0]
    
    # Elements that are in bad triples
    constrained_elements = [i for i in range(n) if len(elem_triples[i]) > 0]
    
    # For constrained elements, use branch and bound
    # But first, let's try a smarter greedy approach
    
    # Key insight: If we sort elements by the number of bad triples they're in,
    # and greedily include elements that don't create violations, we get a good solution.
    
    # Even better: use the fact that the hypergraph is sparse
    
    # Let me implement a proper branch-and-bound for the hypergraph independent set
    # with memoization and good pruning
    
    # Actually, for 128 elements, let me use a different strategy:
    # 1. Start with all elements
    # 2. While there's a violation (bad triple fully selected), remove one element
    # 3. Use a smart removal strategy: remove the element that appears in the most bad triples
    
    # But this might not be optimal. Let me use a more sophisticated approach.
    
    # Approach: Local search with simulated annealing-like moves
    # State: which elements are selected
    # Objective: maximize |selected| subject to no bad triple fully selected
    
    # Let me implement a robust local search with multiple restarts
    
    def count_violations(selected):
        """Count the number of bad triples that are fully selected."""
        cnt = 0
        for t in range(num_bad):
            a, b, c = triple_elems[t]
            if selected[a] and selected[b] and selected[c]:
                cnt += 1
        return cnt
    
    def add_violations(selected, x):
        """If we add element x, how many new violations are created?"""
        cnt = 0
        for t in elem_triples[x]:
            a, b, c = triple_elems[t]
            others = [v for v in (a, b, c) if v != x]
            if selected[others[0]] and selected[others[1]]:
                cnt += 1
        return cnt
    
    # Phase 1: Greedy construction
    # Sort elements by degree (number of bad triples they're in)
    # Include elements in order of increasing degree (less constrained first)
    
    degree = [len(elem_triples[i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: (degree[i], i))
    
    selected = [False] * n
    for x in order:
        if add_violations(selected, x) == 0:
            selected[x] = True
    
    cur_size = sum(selected)
    
    # Phase 2: Local search - try to add elements, and do 1-remove/k-add moves
    # Precompute for speed
    
    def try_add(selected, x):
        """Try to add element x. Return True if added."""
        if selected[x]:
            return False
        if add_violations(selected, x) == 0:
            selected[x] = True
            return True
        return False
    
    # Greedy addition pass
    changed = True
    while changed and time.time() < deadline - 30:
        changed = False
        for x in range(n):
            if not selected[x] and add_violations(selected, x) == 0:
                selected[x] = True
                changed = True
    
    cur_size = sum(selected)
    best_size = cur_size
    best_sel = selected[:]
    
    # Phase 3: Local search with 1-remove, multi-add moves
    # For each element r in selected, try removing it and then greedily adding back
    # all elements that can be added without violation
    
    for ls_iter in range(200):
        if time.time() > deadline - 20:
            break
        improved = False
        
        for r in range(n):
            if not selected[r]:
                continue
            if time.time() > deadline - 20:
                break
            
            # Remove r
            selected[r] = False
            
            # Greedily add all elements that can be added
            added = []
            for x in range(n):
                if not selected[x] and add_violations(selected, x) == 0:
                    selected[x] = True
                    added.append(x)
            
            net_gain = len(added) - 1
            if net_gain > 0:
                cur_size += net_gain
                if cur_size > best_size:
                    best_size = cur_size
                    best_sel = selected[:]
                improved = True
            else:
                # Revert
                for x in added:
                    selected[x] = False
                selected[r] = True
        
        if not improved:
            break
    
    # Phase 4: More aggressive local search - try 2-remove, k-add moves
    # For each pair (r1, r2) in selected, try removing both and adding back greedily
    
    for ls_iter in range(100):
        if time.time() > deadline - 15:
            break
        improved = False
        
        for r1 in range(n):
            if not selected[r1]:
                continue
            for r2 in range(r1 + 1, n):
                if not selected[r2]:
                    continue
                if time.time() > deadline - 15:
                    break
                
                # Remove r1 and r2
                selected[r1] = False
                selected[r2] = False
                
                # Greedily add all elements that can be added
                added = []
                for x in range(n):
                    if not selected[x] and add_violations(selected, x) == 0:
                        selected[x] = True
                        added.append(x)
                
                net_gain = len(added) - 2
                if net_gain > 0:
                    cur_size += net_gain
                    if cur_size > best_size:
                        best_size = cur_size
                        best_sel = selected[:]
                    improved = True
                else:
                    # Revert
                    for x in added:
                        selected[x] = False
                    selected[r1] = True
                    selected[r2] = True
            
            if time.time() > deadline - 15:
                break
        
        if not improved:
            break
    
    # Phase 5: Random restarts with different orderings
    import random
    random.seed(42)
    
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        
        # Random ordering
        order = list(range(n))
        random.shuffle(order)
        
        sel = [False] * n
        for x in order:
            if add_violations(sel, x) == 0:
                sel[x] = True
        
        # Greedy addition
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for x in range(n):
                if not sel[x] and add_violations(sel, x) == 0:
                    sel[x] = True
                    changed = True
        
        sz = sum(sel)
        if sz > best_size:
            best_size = sz
            best_sel = sel[:]
        
        # Quick local search
        for ls_iter in range(50):
            if time.time() > deadline - 5:
                break
            improved = False
            for r in range(n):
                if not sel[r]:
                    continue
                sel[r] = False
                added = []
                for x in range(n):
                    if not sel[x] and add_violations(sel, x) == 0:
                        sel[x] = True
                        added.append(x)
                net_gain = len(added) - 1
                if net_gain > 0:
                    sz += net_gain
                    if sz > best_size:
                        best_size = sz
                        best_sel = sel[:]
                    improved = True
                else:
                    for x in added:
                        sel[x] = False
                    sel[r] = True
            if not improved:
                break
    
    # Final polish: try to add any remaining elements
    for x in range(n):
        if not best_sel[x] and add_violations(best_sel, x) == 0:
            best_sel[x] = True
    
    # Convert back to 1-indexed
    result = [x + 1 for x in range(n) if best_sel[x]]
    if not result:
        result = [1]
    
    return result
