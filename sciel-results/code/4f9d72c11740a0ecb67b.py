"""Rewritten program: simulated annealing over a 192-bit state with precomputed conflict bitsets."""
import time
import random
from itertools import combinations

# Precompute conflict bitsets:
# For each candidate x (1..192), conflict_bits[x] is a 192-bit int where bit j is set
# if there exist 3 DISTINCT OTHER selected integers (from the full set 1..192) whose sum is divisible by x.
# But that's not quite right. The condition is: x must NOT divide the sum of any 3 distinct OTHER selected integers.
# So for a given set S containing x, x is invalid if there exist a,b,c in S\{x} with (a+b+c) % x == 0.
#
# We can precompute for each pair (x, triple) whether the triple's sum is divisible by x.
# But triples are C(192,3) ~ 1.16M, which is manageable.
#
# Better approach: For each x, precompute a set of "forbidden triples" - but that's too large.
#
# Alternative: For a given state (set of selected numbers), to check if x is valid, we need to check
# all triples from the OTHER selected numbers. This is O(k^3) per check where k is the set size.
#
# For simulated annealing, we need fast validity checks. Let's precompute:
# For each x in 1..192, and for each pair (a,b) with a<b, both != x, compute (a+b) mod x.
# Then for a given set S containing x, x is invalid iff there exists c in S\{x} such that
# (a+b+c) % x == 0 for some a,b in S\{x}, a!=b, a!=c, b!=c.
#
# Actually, let's think differently. For a fixed x, define for each pair (a,b) in [1,192]\{x}, a!=b:
#   need_c = (-a-b) mod x
# Then x is invalid in set S if there exist a,b,c in S\{x}, distinct, with c ≡ need_c (mod x).
#
# For efficiency in the SA loop, let's precompute for each x:
#   For each residue r in 0..x-1, the list of numbers in [1,192]\{x} that are ≡ r (mod x).
#
# Then for a given state S, to check if x is valid:
#   For each pair (a,b) in S\{x}, a<b, compute r = (-a-b) mod x.
#   Check if there exists c in S\{x}, c != a, c != b, with c ≡ r (mod x).
#
# This is still O(k^2) per x check, and O(k^3) total for all x in S.
#
# For k up to 48, that's ~48^3/6 ≈ 18k operations per full validation. That's fine.
#
# Let's just do direct checking with early termination.

def is_valid_set(selected):
    """Check if no element in selected divides the sum of any 3 distinct other elements in selected."""
    sel = set(selected)
    n = len(selected)
    if n < 4:
        return True
    for i, x in enumerate(selected):
        others = [v for j, v in enumerate(selected) if j != i]
        # Check all triples from others
        for a in range(len(others)):
            for b in range(a+1, len(others)):
                for c in range(b+1, len(others)):
                    if (others[a] + others[b] + others[c]) % x == 0:
                        return False
    return True

def count_violations(selected):
    """Count the number of violating (x, triple) pairs."""
    sel = set(selected)
    n = len(selected)
    if n < 4:
        return 0
    violations = 0
    for i, x in enumerate(selected):
        others = [v for j, v in enumerate(selected) if j != i]
        for a in range(len(others)):
            for b in range(a+1, len(others)):
                for c in range(b+1, len(others)):
                    if (others[a] + others[b] + others[c]) % x == 0:
                        violations += 1
    return violations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Start with the greedy solution from the seed
    selected = []
    for x in range(192, 0, -1):
        candidate = selected + [x]
        if all(sum(t) % a for a in candidate for t in combinations([v for v in candidate if v != a], 3)):
            selected.append(x)
    
    best_set = set(selected)
    best_size = len(selected)
    
    # Simulated annealing
    random.seed(42)
    current = list(selected)
    current_set = set(current)
    current_violations = count_violations(current)
    
    T = 10.0
    T_min = 0.01
    cooling = 0.9995
    steps = 0
    
    while time.time() < deadline:
        steps += 1
        # Choose a move: remove one, add one, or remove one add two, or swap
        move_type = random.random()
        if move_type < 0.3:
            # Remove one element
            if not current:
                continue
            idx = random.randint(0, len(current)-1)
            removed = current.pop(idx)
            current_set.discard(removed)
            new_violations = count_violations(current)
            delta_v = new_violations - current_violations
            delta_size = len(current) - best_size  # not quite, we compare to current
            # Acceptance: based on violation change
            if delta_v <= 0 or (T > T_min and random.random() < min(1.0, (-delta_v) / T if delta_v > 0 else 1.0)):
                current_violations = new_violations
                if current_violations == 0 and len(current) > best_size:
                    best_size = len(current)
                    best_set = set(current)
            else:
                current.append(removed)
                current_set.add(removed)
        elif move_type < 0.6:
            # Add one element
            candidates = [x for x in range(1, 193) if x not in current_set]
            if not candidates:
                continue
            added = random.choice(candidates)
            current.append(added)
            current_set.add(added)
            new_violations = count_violations(current)
            delta_v = new_violations - current_violations
            if delta_v <= 0 or (T > T_min and random.random() < min(1.0, (-delta_v) / T if delta_v > 0 else 1.0)):
                current_violations = new_violations
                if current_violations == 0 and len(current) > best_size:
                    best_size = len(current)
                    best_set = set(current)
            else:
                current.remove(added)
                current_set.discard(added)
        elif move_type < 0.85:
            # Remove one, add two
            if not current:
                continue
            idx = random.randint(0, len(current)-1)
            removed = current.pop(idx)
            current_set.discard(removed)
            candidates = [x for x in range(1, 193) if x not in current_set and x != removed]
            if len(candidates) < 2:
                current.append(removed)
                current_set.add(removed)
                continue
            add1, add2 = random.sample(candidates, 2)
            current.extend([add1, add2])
            current_set.add(add1)
            current_set.add(add2)
            new_violations = count_violations(current)
            delta_v = new_violations - current_violations
            if delta_v <= 0 or (T > T_min and random.random() < min(1.0, (-delta_v) / T if delta_v > 0 else 1.0)):
                current_violations = new_violations
                if current_violations == 0 and len(current) > best_size:
                    best_size = len(current)
                    best_set = set(current)
            else:
                current.remove(removed)
                current.remove(add1)
                current.remove(add2)
                current_set.add(removed)
                current_set.discard(add1)
                current_set.discard(add2)
        else:
            # Swap: remove one, add one
            if not current:
                continue
            idx = random.randint(0, len(current)-1)
            removed = current[idx]
            candidates = [x for x in range(1, 193) if x not in current_set]
            if not candidates:
                continue
            added = random.choice(candidates)
            current[idx] = added
            current_set.discard(removed)
            current_set.add(added)
            new_violations = count_violations(current)
            delta_v = new_violations - current_violations
            if delta_v <= 0 or (T > T_min and random.random() < min(1.0, (-delta_v) / T if delta_v > 0 else 1.0)):
                current_violations = new_violations
                if current_violations == 0 and len(current) > best_size:
                    best_size = len(current)
                    best_set = set(current)
            else:
                current[idx] = removed
                current_set.discard(added)
                current_set.add(removed)
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Return the best valid set found
    result = list(best_set)
    # Verify it's valid
    if not is_valid_set(result):
        # Fallback to greedy
        result = []
        for x in range(192, 0, -1):
            candidate = result + [x]
            if all(sum(t) % a for a in candidate for t in combinations([v for v in candidate if v != a], 3)):
                result.append(x)
    return result
