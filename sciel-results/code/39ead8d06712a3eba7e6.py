"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from math import lcm
    import random

    N = 128
    # Precompute LCM matrix
    lcms = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            lcms[i][j] = lcm(i, j)
            lcms[j][i] = lcms[i][j]

    # Precompute bad triples as a list of sets for each pair
    # bad[a][b] contains c if (a,b,c) is a bad triple
    bad = [[set() for _ in range(N + 1)] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            l_ab = lcms[a][b]
            for c in range(b + 1, N + 1):
                if lcms[a][c] == l_ab and lcms[b][c] == l_ab:
                    bad[a][b].add(c)
                    bad[a][c].add(b)
                    bad[b][c].add(a)
                    bad[b][a].add(c)
                    bad[c][a].add(b)
                    bad[c][b].add(a)

    def can_add(x, current_set):
        """Check if x can be added to current_set without forming a bad triple."""
        # For any pair (a, b) in current_set, x must not be in bad[a][b]
        # Since current_set is a set of ints, we iterate
        # Optimization: If current_set is small, this is fast.
        # We can also precompute for each x the list of pairs that conflict with it?
        # No, just check.
        for a in current_set:
            for b in current_set:
                if a == b:
                    continue
                if x in bad[a][b]:
                    return False
        return True

    # Helper to count distinct prime factors
    def num_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            if n % d == 0:
                count += 1
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            count += 1
        return count

    best_set = set()
    
    # Strategy: Try multiple orderings
    # 1. Sorted by number of prime factors (ascending) - favors prime powers
    # 2. Natural order
    # 3. Randomized shuffles
    
    orders = []
    
    # Order 1: Prime factors ascending
    nums = list(range(1, N + 1))
    nums.sort(key=num_prime_factors)
    orders.append(nums)
    
    # Order 2: Natural
    orders.append(list(range(1, N + 1)))
    
    # Order 3-10: Random
    random.seed(42)
    for _ in range(8):
        shuffled = list(range(1, N + 1))
        random.shuffle(shuffled)
        orders.append(shuffled)

    for order in orders:
        current_set = set()
        for x in order:
            if can_add(x, current_set):
                current_set.add(x)
        
        if len(current_set) > len(best_set):
            best_set = current_set.copy()

    # Local Search: Try to improve best_set
    # Try to add any x not in best_set
    # If conflict, try to remove one conflicting element
    
    improved = True
    while improved:
        improved = False
        current_best = set(best_set)
        
        for x in range(1, N + 1):
            if x in current_best:
                continue
            
            # Find conflicts
            conflicts = []
            for a in current_best:
                for b in current_best:
                    if a == b: continue
                    if x in bad[a][b]:
                        conflicts.append((a, b))
            
            if not conflicts:
                current_best.add(x)
                best_set = current_best.copy()
                improved = True
                break
            
            # Try to resolve by removing one element from the first conflict
            # This is a heuristic. We try removing 'a' or 'b' from the first conflicting pair.
            # We must ensure the new set is valid.
            a, b = conflicts[0]
            for y in [a, b]:
                new_set = set(current_best)
                new_set.remove(y)
                new_set.add(x)
                
                # Check if new_set is valid
                # We can reuse can_add logic by checking if any triple in new_set is bad
                # But can_add(x, new_set - {x}) is not enough because we removed y.
                # We need to check if new_set is valid.
                # Since we only changed by removing y and adding x, we can check:
                # 1. Is x compatible with (new_set - {x})? Yes, by construction we tried to resolve conflicts.
                #    But removing y might have resolved conflicts with OTHER pairs?
                #    Actually, if x conflicted with (a,b), removing a resolves that specific conflict.
                #    But x might conflict with other pairs in current_best that we didn't remove.
                #    So we must check if x is compatible with the rest of the set.
                
                # Let's just do a full check on new_set for efficiency if it's small.
                # Or use can_add(x, new_set - {x})
                if can_add(x, new_set - {x}):
                    # Also need to ensure that removing y didn't break anything?
                    # Removing elements never creates bad triples.
                    # So if x is compatible with the rest, the set is valid.
                    best_set = new_set
                    improved = True
                    break
            
            if improved:
                break

    return list(best_set)
# EVOLVE-BLOCK-END
