import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    n = 12
    m = 40
    
    def count_comparable(sets):
        """Count unordered pairs where one is strict subset of the other."""
        count = 0
        for i in range(len(sets)):
            si = frozenset(sets[i])
            for j in range(i + 1, len(sets)):
                sj = frozenset(sets[j])
                if si < sj or sj < si:
                    count += 1
        return count
    
    # Initial solution: empty set + all 12 singletons + 27 large sets
    # Strategy: use sets that are supersets of many singletons
    initial_sets = []
    initial_sets.append(frozenset())  # empty set
    for i in range(n):
        initial_sets.append(frozenset([i]))  # singletons
    
    # Add large sets - use a greedy approach to pick sets that maximize comparability
    # Start with the full set, then remove elements one at a time
    full_set = frozenset(range(n))
    initial_sets.append(full_set)
    
    # Add size-11 sets (each missing one element)
    for i in range(n):
        s = full_set - {i}
        initial_sets.append(s)
    
    # We have 1 + 12 + 1 + 12 = 26 sets, need 14 more
    # Add size-10 sets strategically
    used = set(initial_sets)
    all_size10 = [frozenset(range(n)) - {i, j} for i in range(n) for j in range(i+1, n)]
    
    # Greedily pick size-10 sets that maximize comparability with existing sets
    for _ in range(14):
        best_set = None
        best_gain = -1
        for candidate in all_size10:
            if candidate in used:
                continue
            # Count how many existing sets this would be comparable with
            gain = 0
            for existing in initial_sets:
                if candidate < existing or existing < candidate:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_set = candidate
        if best_set is not None:
            initial_sets.append(best_set)
            used.add(best_set)
        else:
            break
    
    # Ensure we have exactly 40 sets
    while len(initial_sets) < m:
        # Add random sets of various sizes
        size = random.randint(5, 9)
        s = frozenset(random.sample(range(n), size))
        if s not in used:
            initial_sets.append(s)
            used.add(s)
    
    # Convert to list of lists
    current = [list(s) for s in initial_sets[:m]]
    best_score = count_comparable(current)
    best_solution = [list(s) for s in current]
    
    # Local search: try swapping elements in sets to improve comparability
    random.seed(42)
    iteration = 0
    while time.time() < deadline and iteration < 50000:
        iteration += 1
        if time.time() > deadline:
            break
        
        # Pick a random set to modify
        idx = random.randint(0, m - 1)
        old_set = set(current[idx])
        
        # Try removing an element
        if len(old_set) > 0:
            elem_to_remove = random.choice(list(old_set))
            new_set = old_set - {elem_to_remove}
            if new_set:  # Don't create empty set if already have one
                # Check if this set already exists
                new_frozen = frozenset(new_set)
                exists = False
                for i, s in enumerate(current):
                    if i != idx and frozenset(s) == new_frozen:
                        exists = True
                        break
                if not exists:
                    old = current[idx]
                    current[idx] = list(new_set)
                    new_score = count_comparable(current)
                    if new_score > best_score:
                        best_score = new_score
                        best_solution = [list(s) for s in current]
                    else:
                        current[idx] = old
        
        # Try adding an element
        missing = [i for i in range(n) if i not in old_set]
        if missing and len(old_set) < n:
            elem_to_add = random.choice(missing)
            new_set = old_set | {elem_to_add}
            new_frozen = frozenset(new_set)
            exists = False
            for i, s in enumerate(current):
                if i != idx and frozenset(s) == new_frozen:
                    exists = True
                    break
            if not exists:
                old = current[idx]
                current[idx] = list(new_set)
                new_score = count_comparable(current)
                if new_score > best_score:
                    best_score = new_score
                    best_solution = [list(s) for s in current]
                else:
                    current[idx] = old
    
    # Ensure all elements are Python int (not bool or numpy)
    result = []
    for s in best_solution:
        result.append([int(x) for x in s])
    
    return result
