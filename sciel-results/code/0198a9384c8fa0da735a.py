"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    t0 = time.time()
    DEADLINE = t0 + 85.0  # safety margin before 90s kill
    
    N = 256
    all_nums = list(range(1, N + 1))
    
    # Precompute conflict matrix: conflict[a][b] = True if (a+b) | (a*b)
    # For distinct a, b
    # (a*b) % (a+b) == 0 means conflict
    # 
    # Key insight: If both a and b are odd, then a+b is even.
    # a*b is odd (product of two odds).
    # An even number cannot divide an odd number (unless the odd number is 0, which it isn't).
    # So (a*b) % (a+b) != 0 is ALWAYS true for distinct odd a, b.
    # Thus, all odd numbers can be in the set together.
    
    # What about evens?
    # If a is even and b is odd: a+b is odd. a*b is even.
    # (a+b) can divide a*b.
    # If a is even and b is even: a+b is even. a*b is even.
    # (a+b) can divide a*b.
    
    # So the set of all odds is a valid base. Size = 128.
    # Can we add any evens?
    # If we add an even number e, it must not conflict with any odd in the set.
    # And it must not conflict with any other even we add.
    
    # Let's check if we can add any evens to the set of all odds.
    # For an even e and an odd o, conflict if (e+o) | (e*o).
    # e*o / (e+o) = k => e*o = k*e + k*o => e*o - k*e - k*o = 0 => e*o - k*e - k*o + k^2 = k^2
    # (e-k)(o-k) = k^2
    # This is a standard number theory transformation.
    # For a fixed even e, we need to check if there exists an odd o in [1, 256] such that (e+o) divides e*o.
    # Actually, we want to AVOID conflicts. So we want (e+o) NOT dividing e*o.
    # But we want to MAXIMIZE the set. So we want to include as many evens as possible,
    # subject to them not conflicting with the odds (and each other).
    
    # Wait, if we include ALL odds, we can only include an even e if it does NOT conflict with ANY odd.
    # Let's check if there are any evens that don't conflict with ANY odd.
    
    # Let's precompute conflicts between evens and odds.
    odds = [x for x in all_nums if x % 2 == 1]
    evens = [x for x in all_nums if x % 2 == 0]
    
    # conflict_even_odd[e][o] = True if (e+o) | (e*o)
    # We want to find evens e such that for ALL o in odds, (e+o) does NOT divide e*o.
    
    # Let's compute for each even e, the number of odds it conflicts with.
    # If it conflicts with 0 odds, we can add it.
    
    # But also, if we add multiple evens, they must not conflict with each other.
    
    # Let's first find the "safe" evens: those that don't conflict with any odd.
    safe_evens = []
    for e in evens:
        is_safe = True
        for o in odds:
            if (e * o) % (e + o) == 0:
                is_safe = False
                break
        if is_safe:
            safe_evens.append(e)
    
    # Now, among safe_evens, we need to find the largest subset such that no two conflict.
    # Conflict between two evens e1, e2: (e1*e2) % (e1+e2) == 0.
    
    # Let's check how many safe evens there are.
    # If there are few, we can use brute force or greedy.
    # If there are many, we need a better approach.
    
    # Let's print or track the count.
    # For now, let's assume we can add all safe evens if they don't conflict with each other.
    # But they might. So we need a maximum independent set on the graph of safe evens,
    # where edges represent conflicts.
    
    # Given the small size (at most 128 evens, likely much fewer safe ones),
    # we can try a greedy approach or even brute force if the number is small.
    
    # Let's build the conflict graph among safe evens.
    n_safe = len(safe_evens)
    if n_safe == 0:
        return odds
    
    # conflict_safe[i][j] = True if safe_evens[i] and safe_evens[j] conflict
    conflict_safe = [[False] * n_safe for _ in range(n_safe)]
    for i in range(n_safe):
        for j in range(i + 1, n_safe):
            e1 = safe_evens[i]
            e2 = safe_evens[j]
            if (e1 * e2) % (e1 + e2) == 0:
                conflict_safe[i][j] = True
                conflict_safe[j][i] = True
    
    # Now find the maximum independent set in this graph.
    # Since n_safe is likely small, we can use a simple branch and bound or even brute force if n_safe <= 20.
    # But let's use a greedy heuristic first, then maybe local search.
    
    # Greedy: sort by degree (number of conflicts), pick the one with lowest degree, remove it and its neighbors.
    # This gives a maximal independent set, not necessarily maximum.
    
    # Let's try a simple local search on the independent set.
    # Start with greedy, then try to swap elements to increase size.
    
    # First, greedy initialization.
    degrees = [sum(conflict_safe[i]) for i in range(n_safe)]
    order = sorted(range(n_safe), key=lambda x: degrees[x])
    
    selected = set()
    for idx in order:
        if idx in selected:
            continue
        # Check if idx conflicts with any already selected
        can_add = True
        for s in selected:
            if conflict_safe[idx][s]:
                can_add = False
                break
        if can_add:
            selected.add(idx)
    
    # Local search: try to add any unselected vertex that doesn't conflict with selected.
    # (This is already done in greedy, but let's be sure.)
    # Then try to swap: remove one selected, add two unselected? Or just add one if possible.
    
    # Actually, the greedy above already finds a maximal independent set.
    # To improve, we can try:
    # 1. For each selected vertex, remove it, then try to add as many unselected as possible.
    # 2. If the new set is larger, keep it.
    
    best_set = set(selected)
    best_size = len(best_set)
    
    # Simple local search: try removing each element and re-greedy
    for i in range(n_safe):
        if i in best_set:
            new_set = set(best_set)
            new_set.remove(i)
            # Re-greedy on the remaining vertices, starting from new_set
            # Actually, let's just try to add any vertex that can be added
            improved = True
            while improved:
                improved = False
                for j in range(n_safe):
                    if j in new_set:
                        continue
                    can_add = True
                    for s in new_set:
                        if conflict_safe[j][s]:
                            can_add = False
                            break
                    if can_add:
                        new_set.add(j)
                        improved = True
            if len(new_set) > best_size:
                best_set = new_set
                best_size = len(new_set)
    
    # Convert best_set indices back to values
    added_evens = [safe_evens[i] for i in best_set]
    
    result = odds + added_evens
    return result
# EVOLVE-BLOCK-END
