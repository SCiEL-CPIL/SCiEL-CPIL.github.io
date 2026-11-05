"""
Strategy: Branch-and-Bound over the "small" half of the domain [1, 90].
For each candidate set A of small elements, determine the constraints on the "large" half [91, 180].
The constraints on the large half form a graph (edges between b,c if a in A creates a triple).
Solve the Maximum Independent Set on this graph (using a simple B&B or greedy approximation if too slow, 
but N_large=90 is small enough for a fast B&B or even exhaustive search with pruning if the graph is sparse).
Actually, solving MIS for N=90 can be hard in worst case, but these graphs are likely sparse/structured.
We will use a greedy coloring bound for the inner MIS to keep it fast, or a simple B&B.
Given the time limit, we'll use a heuristic for the inner MIS (greedy + local search) to evaluate A, 
and use B&B to search over A.
"""

import time
import sys
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    N = 180
    S_small = list(range(1, 91))
    S_large = list(range(91, 181))
    
    # Precompute all triples (a, b, c) with a < b, c and 1/a = 1/b + 1/c
    # a = bc / (b+c) => (b-a)(c-a) = a^2
    triples = []
    for a in range(1, N + 1):
        a_sq = a * a
        for d in range(1, int(a_sq**0.5) + 1):
            if a_sq % d == 0:
                e = a_sq // d
                b1 = a + d
                c1 = a + e
                if b1 <= N and c1 <= N and b1 != c1:
                    triples.append((a, min(b1, c1), max(b1, c1)))
                if d != e:
                    b2 = a + e
                    c2 = a + d
                    if b2 <= N and c2 <= N and b2 != c2:
                        triples.append((a, min(b2, c2), max(b2, c2)))
    
    triples = list(set(triples))
    
    # Index triples by 'a'
    # For a given a, list of (b, c)
    triples_by_a = {a: [] for a in range(1, N + 1)}
    for a, b, c in triples:
        triples_by_a[a].append((b, c))
        
    # For checking validity of A (small set), we need to know if any triple is fully in A.
    # Also, for the large set constraints:
    # If a in A, and (b, c) in triples_by_a[a]:
    #   If b in A and c in A: Invalid A (but b,c > a, so if a is small, b,c might be large or small)
    #   Wait, a < b, c. If a <= 90, b and c can be up to 180.
    #   So b and c can be in S_small or S_large.
    
    # Let's re-evaluate the split.
    # If a is in S_small, b and c are > a.
    # If b is in S_small, then a, b are in S_small.
    # If c is in S_small, then a, c are in S_small.
    
    # Let's define the state of the B&B on S_small.
    # We process elements in S_small in increasing order.
    # State: (idx, current_A, current_size_A)
    # We need to quickly check if adding x to A is valid.
    # Adding x to A is invalid if:
    # 1. There exists a triple (a, b, c) with a, b, c in A U {x}.
    #    Since we process in order, if x is the largest, then a, b < x.
    #    If x is not the largest, say b=x, then a < x and c > x (or c < x).
    #    Actually, it's easier to just check all triples involving x.
    
    # Precompute for each x in S_small:
    # - triples where x is a: (x, b, c)
    # - triples where x is b: (a, x, c)
    # - triples where x is c: (a, b, x)
    
    # For the "Large" part optimization:
    # Given A, we need to find max B subset of S_large such that:
    # 1. No triple (a, b, c) with a in A, b in B, c in B.
    # 2. No triple (a, b, c) with a in A, b in A, c in B. (c is banned)
    # 3. No triple (a, b, c) with a in A, b in B, c in A. (b is banned)
    # Note: If b in A, it's handled by A's validity.
    
    # Let's precompute for each a in S_small:
    # - pairs (b, c) where b, c in S_large
    # - c in S_large where b in S_small (banned if b in A)
    # - b in S_large where c in S_small (banned if c in A)
    
    # This depends on A.
    
    # To make the B&B fast, we need a good upper bound.
    # Upper bound = current_size_A + (number of remaining small elements) + (max possible large elements)
    # Max possible large elements <= |S_large| - (banned count)
    # A tighter bound for large elements is the independence number of the graph formed by pairs (b,c) from a in A.
    # Calculating alpha(G) is NP-hard, but for N=90, a greedy coloring bound is fast.
    
    # We will implement a function `estimate_max_large(A)` that returns an upper bound on the size of B.
    # And a function `solve_large(A)` that finds a good B (greedy + local search).
    
    # Precomputation for speed:
    # For each a in S_small:
    #   large_pairs[a] = list of (b, c) in S_large x S_large
    #   large_bans_from_small_b[a] = set of c in S_large such that there is b in S_small with (a, b, c) triple
    #   large_bans_from_small_c[a] = set of b in S_large such that there is c in S_small with (a, b, c) triple
    
    large_pairs = {a: [] for a in S_small}
    # For banning, we need to know, for a given a, which large elements are banned if specific small elements are in A.
    # Actually, if a is in A, and b is in A, then c is banned.
    # So for a fixed A, banned_large = union over a in A, b in A (b != a) of {c | (a,b,c) is triple}
    # Plus union over a in A, c in A (c != a) of {b | (a,b,c) is triple}
    
    # Let's precompute `bans[a][b]` = set of c in S_large such that (a,b,c) is a triple.
    # Here a, b are in S_small.
    bans = {a: {b: set() for b in S_small if b > a} for a in S_small}
    for a in S_small:
        for b, c in triples_by_a[a]:
            if b in S_small and c in S_large:
                bans[a][b].add(c)
            elif c in S_small and b in S_large:
                # (a, b, c) with b large, c small.
                # If c is in A, b is banned.
                # So bans[a][c].add(b)
                if c in S_small:
                    bans[a][c].add(b)
                    
    # Also, pairs in large:
    for a in S_small:
        for b, c in triples_by_a[a]:
            if b in S_large and c in S_large:
                large_pairs[a].append((b, c))

    def is_valid_A(A):
        # Check if A has no triples.
        # A is a set of small numbers.
        # Triple (a, b, c) in A.
        # Since a < b, c, we can just check for each a in A, if any (b, c) in triples_by_a[a] are both in A.
        for a in A:
            for b, c in triples_by_a[a]:
                if b in A and c in A:
                    return False
        return True

    def get_banned_large(A):
        banned = set()
        for a in A:
            for b in A:
                if b != a and b in bans[a]:
                    banned.update(bans[a][b])
        return banned

    def solve_large(A, time_limit=0.5):
        # Find a large independent set B.
        # Graph G on S_large.
        # Edge (b, c) if exists a in A such that (b, c) in large_pairs[a].
        # Vertex banned if in get_banned_large(A).
        
        start = time.time()
        banned = get_banned_large(A)
        available = [x for x in S_large if x not in banned]
        
        if not available:
            return set()
            
        # Build adjacency list for available nodes
        # Edge if there is a common a in A connecting them.
        # To be fast, we can just check on the fly or precompute.
        # Given N_large=90, we can build the graph.
        
        # Map available to indices
        idx_map = {x: i for i, x in enumerate(available)}
        n = len(available)
        adj = [set() for _ in range(n)]
        
        for a in A:
            for b, c in large_pairs[a]:
                if b in idx_map and c in idx_map:
                    i, j = idx_map[b], idx_map[c]
                    adj[i].add(j)
                    adj[j].add(i)
                    
        # Greedy coloring bound for MIS
        # We want to find a large independent set.
        # Simple heuristic: Greedy pick.
        # Better: Local search on the graph.
        
        # Let's use a simple B&B for MIS if n is small, else greedy.
        # n <= 90.
        
        # Greedy: pick node with min degree, remove it and its neighbors.
        # This gives a maximal independent set.
        
        # To improve, we can do a local search: try to swap nodes.
        
        # For now, let's use a robust greedy + random restarts for the inner problem.
        
        best_B = set()
        
        # Heuristic: 
        # 1. Greedy by degree
        # 2. Random order greedy
        
        def greedy_is(adj, n, order):
            selected = set()
            removed = set()
            for u in order:
                if u in removed:
                    continue
                # Check if u is still available (not adjacent to any selected)
                # Actually, if we remove neighbors, we just check if u is not in removed.
                # But we need to ensure u is not adjacent to selected.
                # Standard greedy:
                # If u is not adjacent to any selected, select it.
                # Remove u and its neighbors from consideration.
                
                if any(v in selected for v in adj[u]):
                    continue
                selected.add(u)
                for v in adj[u]:
                    removed.add(v)
                removed.add(u)
            return selected

        # Try a few random orders
        best_size = 0
        best_sel = set()
        
        for _ in range(20):
            if time.time() - start > time_limit:
                break
            order = available[:]
            random.shuffle(order)
            sel = greedy_is(adj, n, order)
            # Convert back to values
            vals = [available[i] for i in sel]
            if len(vals) > best_size:
                best_size = len(vals)
                best_sel = set(vals)
                
        # Local search on best_sel
        # Try to add a node, or swap.
        # This is getting complex. Let's stick to a good greedy.
        
        return best_sel

    # B&B on S_small
    best_solution = set()
    best_size = 0
    
    # Initial solution: Greedy on all N
    # We can just use the result of solve_large on A = greedy_small
    # Let's do a quick greedy on S_small to get a starting A.
    A_init = set()
    for a in S_small:
        # Check if adding a is valid
        valid = True
        for b, c in triples_by_a[a]:
            if b in A_init and c in A_init:
                valid = False
                break
        if valid:
            # Also check triples where a is b or c?
            # If a is b, then (a', a, c). a' < a.
            # If a is c, then (a', b, a). a' < a, b < a.
            # So we need to check all triples involving a.
            for (x, y, z) in triples:
                if a in (x, y, z):
                    others = [v for v in (x, y, z) if v != a]
                    if all(o in A_init for o in others):
                        valid = False
                        break
        if valid:
            A_init.add(a)
            
    B_init = solve_large(A_init, 1.0)
    best_solution = A_init | B_init
    best_size = len(best_solution)
    
    # B&B
    # Order: increasing
    # State: (idx, current_A)
    # We use a stack to avoid recursion depth issues, or recursion with limit.
    # N_small = 90. Recursion depth 90 is fine.
    
    def bound(idx, current_A):
        # Upper bound on total size
        # current_size = len(current_A)
        # Remaining small: N_small - idx
        # Max large: N_large
        # This is too loose.
        # Better: current_size + (N_small - idx) + N_large
        # Even better: estimate max large given current_A.
        # But that's expensive.
        # Let's use: current_size + (N_small - idx) + N_large
        return len(current_A) + (len(S_small) - idx) + len(S_large)

    def branch_and_bound(idx, current_A):
        nonlocal best_solution, best_size
        
        if time.time() > deadline - 5.0:
            return
            
        if idx == len(S_small):
            # Evaluate current_A
            B = solve_large(current_A, 0.5)
            total = current_A | B
            if len(total) > best_size:
                best_size = len(total)
                best_solution = total
            return
            
        # Prune
        if bound(idx, current_A) <= best_size:
            return
            
        # Option 1: Exclude S_small[idx]
        branch_and_bound(idx + 1, current_A)
        
        # Option 2: Include S_small[idx]
        x = S_small[idx]
        # Check if x can be added
        valid = True
        # Check triples involving x and elements in current_A
        # x is the largest in current_A U {x} because we process in order.
        # So we only need to check triples where x is b or c, and a is in current_A.
        # Or x is a, and b, c in current_A? No, b,c > x.
        # So:
        # 1. x is b: (a, x, c). a in current_A, c > x (not in current_A). 
        #    Wait, if c is in current_A, then c < x. But c > a.
        #    Triple (a, b, c) with a < b, c.
        #    If x is b, then a < x and c > x (or c < x).
        #    If c < x, then c is in current_A.
        #    So we need to check:
        #    - Triples (a, x, c) with a in current_A and c in current_A.
        #    - Triples (a, c, x) with a in current_A and c in current_A. (Here x is c)
        #    - Triples (x, b, c) with b, c in current_A. (Here x is a). But b,c > x, so not in current_A.
        
        for a in current_A:
            # Check if (a, x, c) or (a, c, x) is a triple with c in current_A
            # We can just iterate through triples involving x and a.
            pass
            
        # Simpler: Just check all triples in `triples` that contain x.
        # For each triple (u, v, w) containing x:
        #   If the other two are in current_A, then invalid.
        
        for (u, v, w) in triples:
            if x in (u, v, w):
                others = [val for val in (u, v, w) if val != x]
                if all(o in current_A for o in others):
                    valid = False
                    break
                    
        if valid:
            new_A = set(current_A)
            new_A.add(x)
            branch_and_bound(idx + 1, new_A)

    # Start B&B
    # To make it more effective, we can order the B&B to try including promising elements first.
    # But increasing order is natural for the "a" structure.
    # Let's run it.
    
    branch_and_bound(0, set())
    
    if not best_solution:
        return list(range(91, 181))
        
    return sorted(best_solution)

# EVOLVE-BLOCK-END
