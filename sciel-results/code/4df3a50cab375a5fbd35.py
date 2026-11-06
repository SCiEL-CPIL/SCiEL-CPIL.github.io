"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import time
    import bisect
    
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    denominators = list(range(2, 49))  # 2..48, 47 items
    n = len(denominators)
    reciprocals = [1.0 / d for d in denominators]
    
    # Greedy baseline
    greedy_selected, greedy_total = [], Fraction(0)
    for i, d in enumerate(denominators):
        if greedy_total + Fraction(1, d) < 1:
            greedy_selected.append(i)
            greedy_total += Fraction(1, d)
    
    best_indices = list(greedy_selected)
    best_deficit = 1 - greedy_total
    
    # Meet-in-the-middle with floating point
    mid = n // 2  # 23 and 24
    left_items = list(range(mid))
    right_items = list(range(mid, n))
    
    # Enumerate left subsets
    left_sums = []
    for mask in range(1 << len(left_items)):
        s = 0.0
        for j in range(len(left_items)):
            if mask & (1 << j):
                s += reciprocals[left_items[j]]
        left_sums.append((s, mask))
    
    if time.time() > deadline:
        return best_indices
    
    # Enumerate right subsets
    right_sums = []
    for mask in range(1 << len(right_items)):
        s = 0.0
        for j in range(len(right_items)):
            if mask & (1 << j):
                s += reciprocals[right_items[j]]
        right_sums.append((s, mask))
    
    if time.time() > deadline:
        return best_indices
    
    right_sums.sort(key=lambda x: x[0])
    right_sum_values = [rs[0] for rs in right_sums]
    
    # For each left subset, find best right subset
    best_total_float = 0.0
    best_left_mask = 0
    best_right_mask = 0
    
    for ls, lmask in left_sums:
        if ls >= 1.0:
            continue
        target = 1.0 - ls
        idx = bisect.bisect_left(right_sum_values, target)
        if idx > 0:
            rs, rmask = right_sums[idx - 1]
            total = ls + rs
            if total < 1.0 and total > best_total_float:
                best_total_float = total
                best_left_mask = lmask
                best_right_mask = rmask
    
    if time.time() > deadline:
        return best_indices
    
    # Decode and verify with exact arithmetic
    candidate_indices = []
    for j in range(len(left_items)):
        if best_left_mask & (1 << j):
            candidate_indices.append(left_items[j])
    for j in range(len(right_items)):
        if best_right_mask & (1 << j):
            candidate_indices.append(right_items[j])
    
    if len(candidate_indices) > 0:
        exact_total = sum(Fraction(1, denominators[i]) for i in candidate_indices)
        if exact_total < 1:
            best_indices = candidate_indices
            best_deficit = 1 - exact_total
    
    # Local search to improve
    def get_exact_total(indices):
        return sum(Fraction(1, denominators[i]) for i in indices)
    
    current_indices = set(best_indices)
    current_total = get_exact_total(list(current_indices))
    current_deficit = 1 - current_total
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing one element
        for i in list(current_indices):
            new_indices = current_indices - {i}
            if not new_indices:
                continue
            new_total = get_exact_total(list(new_indices))
            if new_total < 1:
                new_deficit = 1 - new_total
                if new_deficit < current_deficit:
                    current_indices = new_indices
                    current_total = new_total
                    current_deficit = new_deficit
                    improved = True
        # Try adding one element
        for i in range(n):
            if i not in current_indices:
                new_indices = current_indices | {i}
                new_total = get_exact_total(list(new_indices))
                if new_total < 1:
                    new_deficit = 1 - new_total
                    if new_deficit < current_deficit:
                        current_indices = new_indices
                        current_total = new_total
                        current_deficit = new_deficit
                        improved = True
        # Try swapping one element
        for i in list(current_indices):
            for j in range(n):
                if j not in current_indices:
                    new_indices = current_indices - {i} | {j}
                    if not new_indices:
                        continue
                    new_total = get_exact_total(list(new_indices))
                    if new_total < 1:
                        new_deficit = 1 - new_total
                        if new_deficit < current_deficit:
                            current_indices = new_indices
                            current_total = new_total
                            current_deficit = new_deficit
                            improved = True
                            break
            if improved:
                break
    
    best_indices = list(current_indices)
    return best_indices

# EVOLVE-BLOCK-END
