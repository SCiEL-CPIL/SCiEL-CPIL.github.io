"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    N = 100
    target = 200

    def compute_reachable_bitset(elements):
        """Compute bitset of reachable subset sums. Bit s is set if sum s is reachable."""
        bits = 1  # bit 0 set
        for x in elements:
            if x > target:
                continue
            bits |= (bits << x) & ((1 << (target + 1)) - 1)
        return bits

    def has_subset_sum_200(elements):
        """Return True if any subset of elements sums to exactly target."""
        bits = 1
        for x in elements:
            if x > target:
                continue
            bits |= (bits << x) & ((1 << (target + 1)) - 1)
        return bool(bits & (1 << target))

    def find_subset_sum(elements):
        """Find a subset of elements summing to target, return it or None."""
        # DP with parent tracking
        dp = [-1] * (target + 1)
        dp[0] = -1  # sum 0 reachable with no elements
        parent = {}
        for i, x in enumerate(elements):
            for s in range(target - x, -1, -1):
                if dp[s] != -1 and dp[s + x] == -1:
                    dp[s + x] = i
                    parent[(s + x, i)] = s
            if dp[target] != -1:
                subset = []
                cur = target
                while cur > 0:
                    i = dp[cur]
                    subset.append(elements[i])
                    cur = parent[(cur, i)]
                return subset
        return None

    def greedy_largest_first():
        """Greedy: add elements from largest to smallest if it doesn't create sum 200."""
        selected = []
        bits = 1
        for x in range(N, 0, -1):
            new_bits = bits | ((bits << x) & ((1 << (target + 1)) - 1))
            if not (new_bits & (1 << target)):
                selected.append(x)
                bits = new_bits
        return selected

    def greedy_smallest_first():
        """Greedy: add elements from smallest to largest if it doesn't create sum 200."""
        selected = []
        bits = 1
        for x in range(1, N + 1):
            new_bits = bits | ((bits << x) & ((1 << (target + 1)) - 1))
            if not (new_bits & (1 << target)):
                selected.append(x)
                bits = new_bits
        return selected

    def iterative_remove(remaining, strategy='largest', max_iter=500):
        """Iteratively remove elements to break all subset-sum-200 paths."""
        remaining = list(remaining)
        for _ in range(max_iter):
            if time.time() > deadline - 3:
                break
            subset = find_subset_sum(remaining)
            if subset is None:
                break
            if strategy == 'largest':
                to_remove = max(subset)
            elif strategy == 'smallest':
                to_remove = min(subset)
            elif strategy == 'random':
                to_remove = random.choice(subset)
            elif strategy == 'critical':
                sorted_subset = sorted(subset)
                mid = len(sorted_subset) // 2
                to_remove = sorted_subset[mid]
            else:
                to_remove = max(subset)
            remaining = [x for x in remaining if x != to_remove]
        return remaining

    def local_search_add(current_set, bits, max_rounds=100, order='random'):
        """Try to add back elements that don't create sum 200."""
        current_set = set(current_set)
        all_elements = set(range(1, N + 1))
        
        for _ in range(max_rounds):
            if time.time() > deadline - 2:
                break
            improved = False
            candidates = list(all_elements - current_set)
            if order == 'random':
                random.shuffle(candidates)
            elif order == 'descending':
                candidates.sort(reverse=True)
            elif order == 'ascending':
                candidates.sort()
            
            for x in candidates:
                if time.time() > deadline - 2:
                    break
                new_bits = bits | ((bits << x) & ((1 << (target + 1)) - 1))
                if not (new_bits & (1 << target)):
                    current_set.add(x)
                    bits = new_bits
                    improved = True
                    break
            if not improved:
                break
        
        return list(current_set), bits

    def degree_guided_remove(remaining, max_iter=200):
        """Remove the element that appears in the most target-summing subsets."""
        remaining = list(remaining)
        for _ in range(max_iter):
            if time.time() > deadline - 3:
                break
            subset = find_subset_sum(remaining)
            if subset is None:
                break
            # For each element in the subset, estimate its "degree"
            best_elem = None
            best_score = -1
            for x in set(subset):
                # Count how many ways we can form target-x from remaining without x
                # Use a quick DP estimate
                ways = np.zeros(target + 1, dtype=np.int64)
                ways[0] = 1
                for y in remaining:
                    if y == x:
                        continue
                    if y > target:
                        continue
                    ways[y:] += ways[:-y]
                deg = int(ways[target - x]) if target - x >= 0 else 0
                if deg > best_score:
                    best_score = deg
                    best_elem = x
            if best_elem is None:
                best_elem = max(subset)
            remaining = [x for x in remaining if x != best_elem]
        return remaining

    # Phase 1: Greedy initial solutions
    best_selected = greedy_largest_first()
    best_size = len(best_selected)

    if time.time() > deadline:
        return best_selected

    sel2 = greedy_smallest_first()
    if len(sel2) > best_size:
        best_selected = sel2
        best_size = len(best_selected)

    if time.time() > deadline:
        return best_selected

    # Phase 2: Iterative removal with different strategies
    for strategy in ['largest', 'smallest', 'critical']:
        if time.time() > deadline - 3:
            break
        cand = iterative_remove(list(range(1, N + 1)), strategy=strategy, max_iter=300)
        if len(cand) > best_size:
            best_selected = cand
            best_size = len(best_selected)

    if time.time() > deadline:
        return best_selected

    # Phase 3: Degree-guided removal
    if time.time() < deadline - 5:
        cand = degree_guided_remove(list(range(1, N + 1)), max_iter=150)
        if len(cand) > best_size:
            best_selected = cand
            best_size = len(best_selected)

    if time.time() > deadline:
        return best_selected

    # Phase 4: Random restarts with random removal
    random.seed(42)
    for trial in range(100):
        if time.time() > deadline - 3:
            break
        cand = iterative_remove(list(range(1, N + 1)), strategy='random', max_iter=200)
        if len(cand) > best_size:
            best_selected = cand
            best_size = len(best_selected)

    if time.time() > deadline:
        return best_selected

    # Phase 5: Local search - try to add back elements
    current_bits = compute_reachable_bitset(best_selected)
    for order in ['random', 'descending', 'ascending']:
        if time.time() > deadline - 2:
            break
        cand, new_bits = local_search_add(best_selected, current_bits, max_rounds=50, order=order)
        if len(cand) > best_size:
            best_selected = cand
            best_size = len(best_selected)
            current_bits = new_bits

    if time.time() > deadline:
        return best_selected

    # Phase 6: Multiple local search passes with random restarts
    random.seed(123)
    for trial in range(50):
        if time.time() > deadline - 2:
            break
        # Start from a random subset of the current best
        subset_size = max(1, best_size - random.randint(0, 5))
        random_subset = random.sample(best_selected, min(subset_size, len(best_selected)))
        random_bits = compute_reachable_bitset(random_subset)
        cand, new_bits = local_search_add(random_subset, random_bits, max_rounds=30, order='random')
        if len(cand) > best_size:
            best_selected = cand
            best_size = len(best_selected)
            current_bits = new_bits

    if time.time() > deadline:
        return best_selected

    # Phase 7: Final local search with multiple orderings
    current_bits = compute_reachable_bitset(best_selected)
    for order in ['descending', 'ascending', 'random']:
        if time.time() > deadline - 1:
            break
        cand, new_bits = local_search_add(best_selected, current_bits, max_rounds=30, order=order)
        if len(cand) > best_size:
            best_selected = cand
            best_size = len(best_selected)
            current_bits = new_bits

    # Final verification
    if has_subset_sum_200(best_selected):
        safe_selected = greedy_largest_first()
        if len(safe_selected) > 0:
            best_selected = safe_selected
            best_size = len(best_selected)

    if not best_selected:
        best_selected = [1]
        best_size = 1

    return best_selected
# EVOLVE-BLOCK-END
