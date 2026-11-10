# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero
    quadratic residue modulo 205. Uses bitmask-based local search with
    simulated annealing to maximize the set size.
    """
    m = 205
    start_time = time.time()
    deadline = start_time + 580  # 580s safety margin for 630s limit

    # Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)

    # Precompute forbidden as a boolean array for speed
    forbidden_arr = np.zeros(m, dtype=bool)
    for f in forbidden:
        forbidden_arr[f] = True

    # Precompute adjacency bitmasks for fast conflict checking
    # adj_mask[i] is a 205-bit integer where bit j is set if (j-i) % m is forbidden
    adj_mask = [0] * m
    for i in range(m):
        mask = 0
        for f in forbidden:
            j = (i + f) % m
            mask |= (1 << j)
        adj_mask[i] = mask

    def is_valid_pair(a, b):
        d = (a - b) % m
        return not forbidden_arr[d]

    def greedy_init(order):
        """Greedy: process candidates in given order, add if compatible."""
        selected = []
        selected_mask = 0
        for candidate in order:
            # Check if candidate conflicts with any selected element
            # A conflict exists if candidate is in the adj_mask of any selected element
            # OR equivalently, if any selected element is in adj_mask[candidate]
            conflict = False
            for s in selected:
                if (adj_mask[s] >> candidate) & 1:
                    conflict = True
                    break
            if not conflict:
                selected.append(candidate)
                selected_mask |= (1 << candidate)
        return selected

    def local_search(subset, time_limit):
        """
        Local search with add, remove, swap, and 2-for-1 moves.
        Uses bitmask representation for fast conflict checking.
        """
        current = list(subset)
        current_mask = 0
        for x in current:
            current_mask |= (1 << x)

        best = list(current)
        best_size = len(current)
        best_mask = current_mask

        def count_conflicts(lst):
            """Count total number of conflicting pairs in lst."""
            count = 0
            n = len(lst)
            for i in range(n):
                for j in range(i + 1, n):
                    if (adj_mask[lst[i]] >> lst[j]) & 1:
                        count += 1
            return count

        # Phase 1: Greedy add - keep adding elements that have no conflicts
        improved = True
        while improved:
            if time.time() > time_limit:
                break
            improved = False
            for candidate in range(m):
                if (current_mask >> candidate) & 1:
                    continue
                # Check if candidate has no conflicts with current set
                conflict = False
                for s in current:
                    if (adj_mask[s] >> candidate) & 1:
                        conflict = True
                        break
                if not conflict:
                    current.append(candidate)
                    current_mask |= (1 << candidate)
                    improved = True
                    if len(current) > best_size:
                        best_size = len(current)
                        best = list(current)
                        best_mask = current_mask

        # Phase 2: Simulated annealing with various moves
        temperature = 10.0
        cooling_rate = 0.995
        min_temperature = 0.1
        rng = random.Random(42)

        while temperature > min_temperature and time.time() < time_limit:
            # Pick a random move type
            move_type = rng.random()
            
            if move_type < 0.3 and len(current) > 0:
                # Move: remove one element
                idx = rng.randint(0, len(current) - 1)
                removed = current[idx]
                current.pop(idx)
                current_mask &= ~(1 << removed)
                
                # Try to add a different element
                added = False
                for _ in range(30):
                    candidate = rng.randint(0, m - 1)
                    if (current_mask >> candidate) & 1:
                        continue
                    conflict = False
                    for s in current:
                        if (adj_mask[s] >> candidate) & 1:
                            conflict = True
                            break
                    if not conflict:
                        current.append(candidate)
                        current_mask |= (1 << candidate)
                        added = True
                        break
                
                if len(current) > best_size:
                    best_size = len(current)
                    best = list(current)
                    best_mask = current_mask
                    
                if not added:
                    # Revert
                    current.append(removed)
                    current_mask |= (1 << removed)

            elif move_type < 0.5:
                # Move: swap - remove one, add one
                if len(current) == 0:
                    temperature *= cooling_rate
                    continue
                idx = rng.randint(0, len(current) - 1)
                removed = current[idx]
                current.pop(idx)
                current_mask &= ~(1 << removed)
                
                added = False
                for _ in range(30):
                    candidate = rng.randint(0, m - 1)
                    if candidate == removed or (current_mask >> candidate) & 1:
                        continue
                    conflict = False
                    for s in current:
                        if (adj_mask[s] >> candidate) & 1:
                            conflict = True
                            break
                    if not conflict:
                        current.append(candidate)
                        current_mask |= (1 << candidate)
                        added = True
                        break
                
                if len(current) > best_size:
                    best_size = len(current)
                    best = list(current)
                    best_mask = current_mask
                    
                if not added:
                    # Revert
                    current.append(removed)
                    current_mask |= (1 << removed)

            elif move_type < 0.7:
                # Move: 2-for-1 - remove two, add two
                if len(current) < 2:
                    temperature *= cooling_rate
                    continue
                idx1 = rng.randint(0, len(current) - 1)
                idx2 = rng.randint(0, len(current) - 1)
                while idx2 == idx1:
                    idx2 = rng.randint(0, len(current) - 1)
                
                # Ensure idx1 < idx2 for consistent removal
                if idx1 > idx2:
                    idx1, idx2 = idx2, idx1
                
                removed1 = current[idx1]
                removed2 = current[idx2]
                # Remove in reverse order to maintain indices
                current.pop(idx2)
                current.pop(idx1)
                current_mask &= ~(1 << removed1)
                current_mask &= ~(1 << removed2)
                
                added = []
                for _ in range(50):
                    candidate = rng.randint(0, m - 1)
                    if candidate in (removed1, removed2) or (current_mask >> candidate) & 1:
                        continue
                    if candidate in added:
                        continue
                    conflict = False
                    for s in current:
                        if (adj_mask[s] >> candidate) & 1:
                            conflict = True
                            break
                    if not conflict:
                        added.append(candidate)
                        if len(added) == 2:
                            break
                
                if len(added) == 2:
                    current.extend(added)
                    for a in added:
                        current_mask |= (1 << a)
                    
                    if len(current) > best_size:
                        best_size = len(current)
                        best = list(current)
                        best_mask = current_mask
                else:
                    # Revert
                    current.extend([removed1, removed2])
                    current_mask |= (1 << removed1)
                    current_mask |= (1 << removed2)

            else:
                # Move: add one element (even if it causes a conflict, for exploration)
                candidate = rng.randint(0, m - 1)
                if (current_mask >> candidate) & 1:
                    temperature *= cooling_rate
                    continue
                
                # Check if it actually conflicts
                conflict_count = 0
                for s in current:
                    if (adj_mask[s] >> candidate) & 1:
                        conflict_count += 1
                
                if conflict_count == 0:
                    current.append(candidate)
                    current_mask |= (1 << candidate)
                    if len(current) > best_size:
                        best_size = len(current)
                        best = list(current)
                        best_mask = current_mask
                else:
                    # Accept with probability based on temperature
                    # This allows escaping local optima
                    delta = conflict_count
                    if rng.random() < np.exp(-delta / max(temperature, 0.01)):
                        current.append(candidate)
                        current_mask |= (1 << candidate)
                        if len(current) > best_size:
                            best_size = len(current)
                            best = list(current)
                            best_mask = current_mask

            temperature *= cooling_rate

        # Final cleanup: remove any conflicting elements
        clean = []
        clean_mask = 0
        for x in current:
            conflict = False
            for y in clean:
                if (adj_mask[y] >> x) & 1:
                    conflict = True
                    break
            if not conflict:
                clean.append(x)
                clean_mask |= (1 << x)
        
        if len(clean) > best_size:
            best = list(clean)
            best_size = len(clean)

        return best

    # Generate multiple initial solutions and run local search on each
    best_solution = []
    best_size = 0
    rng_main = random.Random(123)

    # Greedy init with natural order
    sol = greedy_init(list(range(m)))
    sol = local_search(sol, deadline - 60)
    if len(sol) > best_size:
        best_size = len(sol)
        best_solution = list(sol)

    # Try different random orderings for greedy
    for trial in range(50):
        if time.time() > deadline - 60:
            break
        order = list(range(m))
        rng_main.shuffle(order)
        sol = greedy_init(order)
        sol = local_search(sol, deadline - 60)
        if len(sol) > best_size:
            best_size = len(sol)
            best_solution = list(sol)

    # Random restarts with local search
    for trial in range(30):
        if time.time() > deadline - 30:
            break
        # Start with a random subset
        size = rng_main.randint(5, 15)
        sol = rng_main.sample(range(m), size)
        # Remove conflicts
        clean = []
        for x in sol:
            ok = True
            for y in clean:
                if not is_valid_pair(x, y):
                    ok = False
                    break
            if ok:
                clean.append(x)
        sol = local_search(clean, deadline - 30)
        if len(sol) > best_size:
            best_size = len(sol)
            best_solution = list(sol)

    return np.array(sorted(best_solution), dtype=np.int64)


# EVOLVE-BLOCK-END
