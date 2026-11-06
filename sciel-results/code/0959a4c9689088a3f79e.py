import time
import random
import itertools
import numpy as np

def construct():
    deadline = time.time() + 80  # Safety margin for 90s limit
    
    N = 180
    universe = list(range(1, N + 1))
    
    # Precompute all conflicting triples
    # A triple (a, b, c) with distinct elements conflicts if 1/a = 1/b + 1/c
    # which is equivalent to bc = a(b+c)
    # We need to avoid having all three in the set.
    
    # For efficiency, precompute for each pair (u, v) whether adding a third element x
    # would create a conflict. But a better approach: precompute all conflicting triples.
    
    # Actually, let's precompute a conflict structure.
    # For each element x, precompute which pairs (u, v) would form a conflict with x.
    # Or better: precompute all triples that conflict.
    
    # Let's think about it differently. For a set S, a conflict exists if there are
    # distinct a, b, c in S such that bc = a(b+c).
    # 
    # Key insight: if we fix b and c, then a = bc/(b+c). So for each pair (b,c),
    # there's at most one a that conflicts. And a must be an integer in [1, N].
    
    # Let's precompute: for each pair (b, c) with b != c, compute a = bc/(b+c).
    # If a is an integer and a != b and a != c and 1 <= a <= N, then (a, b, c) is a
    # conflicting triple.
    
    # Store conflicts as: for each element a, the set of pairs (b, c) that conflict with a.
    # Or: for each element x, the set of other elements that cannot be in the set together
    # with x in a way that creates a triple.
    
    # Actually, the cleanest precomputation:
    # conflict_triples: list of frozensets {a, b, c} where 1/a = 1/b + 1/c
    # Then the problem is: find the largest subset with no conflicting triple.
    
    # Let's precompute all conflicting triples.
    all_triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                # Check if any permutation satisfies the equation
                # 1/a = 1/b + 1/c  =>  bc = a(b+c)
                # 1/b = 1/a + 1/c  =>  ac = b(a+c)
                # 1/c = 1/a + 1/b  =>  ab = c(a+b)
                if b * c == a * (b + c):
                    all_triples.append((a, b, c))
                elif a * c == b * (a + c):
                    all_triples.append((a, b, c))
                elif a * b == c * (a + b):
                    all_triples.append((a, b, c))
    
    # For each element, precompute which other elements it conflicts with in pairs
    # conflict_pairs[x] = set of (u, v) such that {x, u, v} is a conflicting triple
    # But this is expensive to store. Instead, let's use a different approach.
    
    # For local search, we need to quickly check if adding x to S creates a conflict.
    # This happens iff there exist u, v in S such that {x, u, v} is a conflicting triple.
    
    # Precompute: for each x, the list of pairs (u, v) with u < v, u != x, v != x,
    # such that {x, u, v} is a conflicting triple.
    x_conflicts = {i: [] for i in range(1, N + 1)}
    for (a, b, c) in all_triples:
        for x, (u, v) in [(a, (b, c)), (b, (a, c)), (c, (a, b))]:
            x_conflicts[x].append((u, v))
    
    # Now, for a given set S and element x not in S, adding x creates a conflict
    # iff there exists (u, v) in x_conflicts[x] such that u in S and v in S.
    
    # For greedy construction, we can maintain for each x not in S, a count of how many
    # of its conflicting pairs are "active" (both elements in S).
    
    # Let's do a greedy approach with a good ordering, then local search.
    
    def can_add(S, x, active_counts):
        """Check if x can be added to S without creating a conflict."""
        for (u, v) in x_conflicts[x]:
            if u in S and v in S:
                return False
        return True
    
    def add_element(S, x, active_counts):
        """Add x to S and update active_counts for all other elements."""
        S.add(x)
        for y in range(1, N + 1):
            if y == x or y in S:
                continue
            for (u, v) in x_conflicts[y]:
                if (u == x and v in S) or (v == x and u in S):
                    active_counts[y] += 1
    
    def remove_element(S, x, active_counts):
        """Remove x from S and update active_counts."""
        S.discard(x)
        for y in range(1, N + 1):
            if y == x or y in S:
                continue
            for (u, v) in x_conflicts[y]:
                if (u == x and v in S) or (v == x and u in S):
                    active_counts[y] -= 1
    
    best_set = set()
    best_size = 0
    
    # Try several greedy orderings
    orderings = [
        sorted(universe, reverse=True),
        sorted(universe),
        random.sample(universe, N),
        random.sample(universe, N),
        random.sample(universe, N),
    ]
    
    for order in orderings:
        if time.time() > deadline - 10:
            break
        S = set()
        active_counts = {i: 0 for i in range(1, N + 1)}
        
        for x in order:
            if not can_add(S, x, active_counts):
                continue
            add_element(S, x, active_counts)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
    
    # Local search: try to improve by removing one element and adding another
    if time.time() > deadline - 5:
        return list(best_set)
    
    S = best_set.copy()
    active_counts = {i: 0 for i in range(1, N + 1)}
    # Rebuild active_counts
    for x in S:
        for y in range(1, N + 1):
            if y == x or y not in S:
                continue
            for (u, v) in x_conflicts[y]:
                if (u == x and v in S) or (v == x and u in S):
                    active_counts[y] += 1
    
    # Actually, let's rebuild active_counts properly
    active_counts = {i: 0 for i in range(1, N + 1)}
    for y in range(1, N + 1):
        if y in S:
            continue
        for (u, v) in x_conflicts[y]:
            if u in S and v in S:
                active_counts[y] += 1
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Try removing each element in S and see if we can add more
        for x in list(S):
            if time.time() > deadline - 5:
                break
            # Remove x
            remove_element(S, x, active_counts)
            
            # Try to add elements not in S
            added_any = False
            for y in range(1, N + 1):
                if y in S or y == x:
                    continue
                if active_counts[y] == 0:
                    add_element(S, y, active_counts)
                    added_any = True
            
            if added_any:
                improved = True
            else:
                # Put x back
                add_element(S, x, active_counts)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
    
    return list(best_set)
