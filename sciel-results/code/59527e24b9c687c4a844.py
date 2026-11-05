"""Improved program: Branch-and-Bound for max independent set in 3-uniform hypergraph."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import lru_cache

def construct():
    t0 = time.time()
    DEADLINE = 80.0  # safety margin under 90s wall-clock

    N = 128
    elements = list(range(1, N + 1))

    # Precompute lcm table
    lcm_table = {}
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = gcd(i, j)
            lcm_table[(i, j)] = i // g * j

    # A triple (a,b,c) is "bad" if lcm(a,b)==lcm(a,c)==lcm(b,c)
    # For each pair (a,b), find all c > b such that lcm(a,c)==lcm(b,c)==lcm(a,b)
    # We can group by lcm value.
    
    # Better approach: for each possible LCM value L, find all pairs (a,b) with lcm(a,b)=L.
    # If there are 3+ elements in a set S_L where every pair has lcm L, then any triple from S_L is bad.
    # Actually the condition is: three distinct a,b,c such that lcm(a,b)=lcm(a,c)=lcm(b,c).
    # This means all three pairwise LCMs are equal to some value L.
    # So a,b,c must all be in the set of numbers whose pairwise lcms with each other equal L.
    
    # Let's think differently. For a fixed L, define S(L) = {x in [1,128] : x divides L}.
    # For a,b,c in S(L), lcm(a,b)=L iff a and b "cover" L in the sense that for every prime power p^k || L, at least one of a,b has p^k.
    # This is complex. Let's just precompute all bad triples.
    
    # Precompute: for each pair (a,b) with a<b, compute L = lcm(a,b).
    # Then for c > b, check if lcm(a,c)==L and lcm(b,c)==L.
    # Group by (a,b) -> list of c's. But that's O(N^3) = 128^3 = 2M, which is fine.
    
    # Build conflict structure: a triple (a,b,c) with a<b<c is bad if lcm(a,b)==lcm(a,c)==lcm(b,c).
    # For branch-and-bound, we need: given a partial set S, can we add x?
    # x conflicts with S if there exist a,b in S such that (a,b,x) is a bad triple.
    
    # Precompute for each x, the set of pairs (a,b) with a<b<x such that (a,b,x) is bad.
    # Actually, for efficiency: for each x, precompute which pairs (a,b) in the remaining candidates would form a bad triple with x.
    
    # Let's precompute: bad_pairs_with[x] = set of (a,b) such that a<b<x and lcm(a,b)==lcm(a,x)==lcm(b,x)
    # This is O(N^3) total.
    
    bad_pairs_with = [set() for _ in range(N + 1)]
    for c in range(1, N + 1):
        for b in range(1, c):
            lb = lcm_table[(b, c)] if b < c else lcm_table[(c, b)]
            # Wait, lcm_table is for pairs (i,j) with i<j. Let me fix.
            pass
    
    # Redo: lcm_table[(i,j)] for i<j
    # For triple (a,b,c) with a<b<c:
    #   lcm(a,b), lcm(a,c), lcm(b,c)
    #   All three must be equal.
    
    bad_pairs_with = [set() for _ in range(N + 1)]
    for c in range(3, N + 1):
        for b in range(2, c):
            lbc = lcm_table[(b, c)]
            for a in range(1, b):
                lab = lcm_table[(a, b)]
                if lab != lbc:
                    continue
                lac = lcm_table[(a, c)]
                if lac == lbc:
                    bad_pairs_with[c].add((a, b))
    
    # Now, for branch and bound:
    # We process elements in order. State: current set S, next candidate index.
    # Upper bound: |S| + number of remaining candidates that don't conflict.
    # Conflict: x can be added to S iff for all (a,b) in S with a<b, (a,b) not in bad_pairs_with[x].
    # i.e., no pair in S is in bad_pairs_with[x].
    
    # Greedy coloring upper bound on the "conflict graph" of remaining candidates:
    # Two remaining candidates x,y conflict if they can't both be added (i.e., adding both would create a bad triple).
    # Actually, x and y conflict if: there exist a,b in S such that (a,b,x) bad OR (a,b,y) bad... no.
    # x and y are compatible with S if individually they don't conflict with S.
    # x and y conflict with each other (given S) if adding both x,y to S creates a bad triple.
    # This happens if there exist a in S such that (a,x,y) is bad, OR (a,b,x) bad with b in S... 
    # Wait, the triple must be three distinct elements. If we add x and y to S, a bad triple could be:
    #   (a,b,c) where a,b,c are from S∪{x,y}.
    # Since S is already independent, the new bad triple must include at least one of {x,y}.
    # Cases:
    #   1. (x, a, b) where a,b in S: bad if (a,b) in bad_pairs_with[x]
    #   2. (y, a, b) where a,b in S: bad if (a,b) in bad_pairs_with[y]
    #   3. (x, y, a) where a in S: bad if (a,x) in bad_pairs_with[y] or (a,y) in bad_pairs_with[x]... 
    #      Actually (x,y,a) with a<x<y: bad if lcm(x,y)==lcm(x,a)==lcm(y,a).
    #      This is equivalent to: (a,x) in bad_pairs_with[y] means lcm(a,x)==lcm(a,y)==lcm(x,y). Yes.
    #      Or (a,y) in bad_pairs_with[x]... but a<y and x<y, so if x<a<y, then (a,y) with x: lcm(a,y)==lcm(a,x)==lcm(x,y)?
    #      Let's just check: (x,y,a) is bad iff the three pairwise lcms are equal.
    
    # For simplicity and speed, let's use a simpler upper bound:
    # UB = |S| + (number of remaining candidates that are individually compatible with S)
    # This is a valid upper bound (optimistic).
    
    # Better: greedy coloring. Build a conflict graph among remaining compatible candidates.
    # Two candidates x,y (both compatible with S individually) conflict if adding both creates a bad triple.
    
    def is_compatible(x, S):
        """Check if x can be added to S without creating a bad triple."""
        for (a, b) in bad_pairs_with[x]:
            if a in S and b in S:
                return False
        return True
    
    def conflict_with_both(x, y, S):
        """Check if adding both x and y to S creates a bad triple."""
        # Assume x < y. Check triples involving x, y and one element from S.
        for a in S:
            if a < x:
                # Triple (a, x, y): bad if lcm(a,x)==lcm(a,y)==lcm(x,y)
                # This is equivalent to (a,x) in bad_pairs_with[y]
                if (a, x) in bad_pairs_with[y]:
                    return True
            elif x < a < y:
                # Triple (x, a, y): bad if lcm(x,a)==lcm(x,y)==lcm(a,y)
                # Equivalent to (x,a) in bad_pairs_with[y]
                if (x, a) in bad_pairs_with[y]:
                    return True
            elif a > y:
                # Triple (x, y, a): bad if lcm(x,y)==lcm(x,a)==lcm(y,a)
                # Equivalent to (x,y) in bad_pairs_with[a]... but we're checking from x,y perspective.
                # Actually (x,y) in bad_pairs_with[a] means lcm(x,y)==lcm(x,a)==lcm(y,a). Yes.
                if (x, y) in bad_pairs_with[a]:
                    return True
        return False
    
    # For efficiency, let's precompute more.
    # Actually, let's just do a clean B&B with a good upper bound.
    
    # Sort elements. We'll use index-based.
    # Precompute for each x, the set of pairs in bad_pairs_with[x] as a frozenset for fast lookup.
    # Actually, let's convert to a set of frozensets or just use the tuple set.
    
    # Let's precompute: for each x, bad_pairs_with[x] as a set of (a,b) tuples. Already done.
    
    # For the conflict check between x and y given S, we need fast access.
    # Let's precompute: for each pair (x,y) with x<y, the set of a's such that (x,y,a) is a bad triple.
    # triple_conflict[x][y] = set of a's where (min(x,y,a), mid, max) is bad.
    
    triple_conflict = {}
    for c in range(3, N + 1):
        for b in range(2, c):
            for a in range(1, b):
                if (a, b) in bad_pairs_with[c]:
                    # (a,b,c) is bad
                    triple_conflict.setdefault((a, b), set()).add(c)
                    triple_conflict.setdefault((a, c), set()).add(b)
                    triple_conflict.setdefault((b, c), set()).add(a)
    
    # Now conflict_with_both(x, y, S):
    # Adding x,y to S creates bad triple iff:
    #   - There's a in S with (x,y,a) bad, i.e., a in triple_conflict.get((min(x,y),max(x,y)), set())
    #   - Or there's a,b in S with (a,b,x) bad, but that's already checked by individual compatibility.
    #   - Or (a,x,y) bad which is same as (x,y,a) bad.
    # So: conflict iff S intersects triple_conflict.get((min(x,y),max(x,y)), set())
    
    def conflict_xy(x, y, S):
        key = (min(x, y), max(x, y))
        conflict_set = triple_conflict.get(key, set())
        return bool(conflict_set & S)
    
    # Now B&B:
    # We process in order 1..128. At each step, decide include/exclude.
    # Use upper bound: |S| + greedy_coloring_bound(remaining_compatible_candidates, S)
    
    best = []
    
    def branch_and_bound(idx, S, S_set):
        nonlocal best
        if time.time() > DEADLINE:
            return
        if len(S) > len(best):
            best = list(S)
        
        if idx > N:
            return
        
        # Upper bound
        # Count remaining candidates that are individually compatible with S
        remaining_compatible = []
        for j in range(idx, N + 1):
            if is_compatible(j, S_set):
                remaining_compatible.append(j)
        
        if len(S) + len(remaining_compatible) <= len(best):
            return
        
        # Greedy coloring upper bound on the conflict graph of remaining_compatible
        # Build conflict graph: x,y in remaining_compatible conflict if conflict_xy(x,y,S_set)
        # Greedy coloring: order by degree (descending), assign colors.
        if len(remaining_compatible) > 1:
            # Build adjacency
            adj = {x: set() for x in remaining_compatible}
            for i in range(len(remaining_compatible)):
                for j in range(i + 1, len(remaining_compatible)):
                    x, y = remaining_compatible[i], remaining_compatible[j]
                    if conflict_xy(x, y, S_set):
                        adj[x].add(y)
                        adj[y].add(x)
            
            # Order by degree descending
            ordered = sorted(remaining_compatible, key=lambda x: -len(adj[x]))
            
            # Greedy coloring
            color = {}
            next_color = 0
            for x in ordered:
                used = set()
                for y in adj[x]:
                    if y in color:
                        used.add(color[y])
                c = 0
                while c in used:
                    c += 1
                color[x] = c
                next_color = max(next_color, c + 1)
            
            ub = len(S) + next_color
        else:
            ub = len(S) + len(remaining_compatible)
        
        if ub <= len(best):
            return
        
        # Branch: include idx if possible
        x = idx
        if is_compatible(x, S_set):
            S.append(x)
            S_set.add(x)
            branch_and_bound(idx + 1, S, S_set)
            S.pop()
            S_set.remove(x)
        
        # Exclude idx
        branch_and_bound(idx + 1, S, S_set)
    
    branch_and_bound(1, [], set())
    
    if not best:
        return [1]
    return best
# EVOLVE-BLOCK-END
