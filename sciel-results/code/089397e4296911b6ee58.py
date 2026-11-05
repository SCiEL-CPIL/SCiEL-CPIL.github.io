"""Rewritten program: local-search with 1-2 exchange moves to maximize a reciprocal-Schur-free subset of [1,180]."""
import time
import itertools

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin under 90s

    N = 180
    # Precompute all "forbidden triples" (a, b, c) of distinct elements in [1..N]
    # such that 1/a = 1/b + 1/c  <=>  a = b*c / (b+c)
    # So b+c must divide b*c, and a = b*c/(b+c) must be an integer in [1..N], distinct from b and c.
    
    # For a pair (b, c) with b < c, check if a = b*c/(b+c) is integer and distinct
    # We store for each element x, the set of pairs (i, j) that would form a forbidden triple with x
    # i.e., if x is in the set, then we cannot have both i and j in the set (when i != j != x)
    
    # Actually, let's think differently:
    # A set S is valid iff there do NOT exist distinct a, b, c in S with 1/a = 1/b + 1/c.
    # Equivalently, for each pair (b, c) in S with b != c, if a = b*c/(b+c) is an integer and a in S and a != b and a != c, then invalid.
    
    # Precompute: for each unordered pair (b, c), b < c, compute a = b*c/(b+c) if it's a valid integer in [1..N], a != b, a != c.
    # Store as a mapping: for each a, the list of pairs (b, c) that conflict with a.
    # And for each pair (b, c), the a that conflicts.
    
    # Let's precompute all forbidden triples
    forbidden = {}  # a -> list of (b, c) such that 1/a = 1/b + 1/c
    pair_to_a = {}  # (b, c) with b < c -> a (if valid)
    
    for b in range(1, N + 1):
        for c in range(b + 1, N + 1):
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if 1 <= a <= N and a != b and a != c:
                    if a not in forbidden:
                        forbidden[a] = []
                    forbidden[a].append((b, c))
                    pair_to_a[(b, c)] = a
    
    # Also, for each a, we need to know which pairs (b, c) conflict.
    # A set S is invalid if there exists a in S and (b, c) in forbidden[a] such that b in S and c in S.
    
    # Strategy: local search.
    # Start with a large set, e.g., all elements in [91, 180] (size 90).
    # Then try to add elements from [1, 90] if they don't create a forbidden triple.
    # Then try 1-for-1 swaps: remove one element, add another.
    # Then try 2-for-1: remove two, add one.
    # Then try 1-for-2: remove one, add two.
    # Iterate until no improvement or time limit.
    
    # Use a boolean array for membership for fast checks.
    def is_valid_set(S):
        """Check if set S (as a set of ints) is valid."""
        S_set = set(S)
        for a in S:
            if a in forbidden:
                for (b, c) in forbidden[a]:
                    if b in S_set and c in S_set:
                        return False
        return True
    
    def can_add(S_bool, x):
        """Check if adding x to S (represented as boolean array) keeps it valid."""
        if S_bool[x]:
            return True  # already in set
        # Check all forbidden triples involving x
        if x in forbidden:
            for (b, c) in forbidden[x]:
                if S_bool[b] and S_bool[c]:
                    return False
        # Also check if x is the 'a' in some triple where b, c are in S
        # Wait, the above already covers: if x is in forbidden, it means x is the 'a' value.
        # But we also need to check: is there some a in S such that (x, c) or (b, x) is in forbidden[a]?
        # i.e., if we add x, does it form a forbidden triple with two existing elements?
        # The forbidden dict is keyed by 'a'. So for a existing element a, we need to check if (x, c) is in forbidden[a] for some c in S, or (b, x) for some b in S.
        
        # Let's restructure: precompute for each element x, the list of (a, other) such that {a, x, other} is a forbidden triple.
        # Actually, let's precompute a different structure.
        
        # For each element x, precompute the set of pairs (y, z) such that {x, y, z} is a forbidden triple.
        # Then can_add(x) checks: for all (y, z) in conflict_pairs[x], not (S_bool[y] and S_bool[z]).
        
        # Let's rebuild the precomputation.
        return True  # placeholder
    
    # Let's redo the precomputation more carefully.
    # For each element x, conflict_pairs[x] = list of (y, z) with y < z, y != x, z != x, such that {x, y, z} is a forbidden triple.
    
    conflict_pairs = {i: [] for i in range(1, N + 1)}
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                # Check if 1/a = 1/b + 1/c or 1/b = 1/a + 1/c or 1/c = 1/a + 1/b
                # 1/a = 1/b + 1/c => a = b*c/(b+c)
                # 1/b = 1/a + 1/c => b = a*c/(a+c)
                # 1/c = 1/a + 1/b => c = a*b/(a+b)
                
                # Check all three
                num1 = b * c
                den1 = b + c
                if num1 % den1 == 0:
                    a_val = num1 // den1
                    if a_val == a:
                        # a is the "small" one, b and c are the "large" ones
                        conflict_pairs[a].append((b, c))
                        conflict_pairs[b].append((a, c))
                        conflict_pairs[c].append((a, b))
                
                num2 = a * c
                den2 = a + c
                if num2 % den2 == 0:
                    b_val = num2 // den2
                    if b_val == b:
                        # b is the "small" one
                        conflict_pairs[b].append((a, c))
                        conflict_pairs[a].append((b, c))
                        conflict_pairs[c].append((a, b))
                
                num3 = a * b
                den3 = a + b
                if num3 % den3 == 0:
                    c_val = num3 // den3
                    if c_val == c:
                        # c is the "small" one
                        conflict_pairs[c].append((a, b))
                        conflict_pairs[a].append((b, c))
                        conflict_pairs[b].append((a, c))
    
    # Remove duplicates in conflict_pairs
    for x in conflict_pairs:
        conflict_pairs[x] = list(set(conflict_pairs[x]))
    
    def can_add(S_bool, x):
        """Check if adding x to S keeps it valid."""
        if S_bool[x]:
            return True
        for (y, z) in conflict_pairs[x]:
            if S_bool[y] and S_bool[z]:
                return False
        return True
    
    def can_remove_keep_valid(S_bool, x):
        """After removing x, the set is always valid (removing elements can't create new forbidden triples)."""
        return True
    
    def is_valid_bool(S_bool):
        """Check validity of a boolean array."""
        for x in range(1, N + 1):
            if S_bool[x]:
                for (y, z) in conflict_pairs[x]:
                    if S_bool[y] and S_bool[z]:
                        return False
        return True
    
    # Start with [91, 180]
    S_bool = [False] * (N + 1)
    for i in range(91, N + 1):
        S_bool[i] = True
    
    best_size = sum(S_bool)
    best_S = S_bool[:]
    
    # Phase 1: Greedy add
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for x in range(1, N + 1):
            if not S_bool[x] and can_add(S_bool, x):
                S_bool[x] = True
                changed = True
                if sum(S_bool) > best_size:
                    best_size = sum(S_bool)
                    best_S = S_bool[:]
    
    # Phase 2: Local search with 1-for-1 swaps
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in range(1, N + 1):
            if S_bool[x]:
                for y in range(1, N + 1):
                    if not S_bool[y]:
                        # Try swapping x for y
                        S_bool[x] = False
                        if can_add(S_bool, y):
                            S_bool[y] = True
                            improved = True
                            if sum(S_bool) > best_size:
                                best_size = sum(S_bool)
                                best_S = S_bool[:]
                        else:
                            S_bool[x] = True
                        if time.time() > deadline:
                            break
                if time.time() > deadline:
                    break
        if time.time() > deadline:
            break
    
    # Phase 3: 2-for-1 moves (remove two, add one)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S_list = [i for i in range(1, N + 1) if S_bool[i]]
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                x = S_list[i]
                y = S_list[j]
                S_bool[x] = False
                S_bool[y] = False
                for z in range(1, N + 1):
                    if not S_bool[z] and can_add(S_bool, z):
                        S_bool[z] = True
                        # Size stays the same (removed 2, added 1 => net -1)
                        # This doesn't increase size, but might enable future improvements
                        # Actually, this decreases size by 1. Only do this if it enables a larger set later.
                        # For now, skip 2-for-1 as it reduces size.
                        S_bool[z] = False
                        break
                S_bool[x] = True
                S_bool[y] = True
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Phase 4: 1-for-2 moves (remove one, add two)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S_list = [i for i in range(1, N + 1) if S_bool[i]]
        for x in S_list:
            S_bool[x] = False
            # Try to add two elements
            added = []
            for y in range(1, N + 1):
                if not S_bool[y] and can_add(S_bool, y):
                    S_bool[y] = True
                    added.append(y)
                    if len(added) == 2:
                        break
            if len(added) == 2:
                improved = True
                if sum(S_bool) > best_size:
                    best_size = sum(S_bool)
                    best_S = S_bool[:]
            else:
                # Revert
                for y in added:
                    S_bool[y] = False
            S_bool[x] = True
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Use the best set found
    final_S = best_S if best_size >= sum(S_bool) else S_bool
    
    return [i for i in range(1, N + 1) if final_S[i]]

# EVOLVE-BLOCK-END
