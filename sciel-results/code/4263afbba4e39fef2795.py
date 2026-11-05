"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 140  # Safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def can_add(S, S_set, x):
        """Check if adding x to S keeps it sum-free."""
        # x + x must not be in S
        if 2 * x in S_set:
            return False
        # For each a in S, a + x must not be in S
        for a in S:
            if a + x in S_set:
                return False
        return True

    def greedy_augment(S, all_elements):
        """Try to add elements from all_elements not in S, in a given order."""
        S_set = set(S)
        S_list = list(S)
        # Try adding each excluded element
        for x in all_elements:
            if x not in S_set:
                if can_add(S_list, S_set, x):
                    S_set.add(x)
                    S_list.append(x)
        return S_list

    def try_swaps(S, all_elements, max_iters=200):
        """Try small swaps: remove one, add another, if it improves size."""
        S_set = set(S)
        S_list = list(S)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            for i, x in enumerate(S_list):
                for y in all_elements:
                    if y in S_set or y == x:
                        continue
                    # Try removing x and adding y
                    temp_set = S_set - {x}
                    if can_add(list(temp_set), temp_set, y):
                        # Check if this leads to a larger set after augmentation
                        new_S = temp_set | {y}
                        new_list = list(new_S)
                        # Quick check: if we can add more after this swap
                        aug_list = greedy_augment(new_list, all_elements)
                        if len(aug_list) > len(S_list):
                            S_list = aug_list
                            S_set = set(aug_list)
                            improved = True
                            break
                if improved:
                    break
        return S_list

    def solve_one(vals):
        """Find a large sum-free subset of vals."""
        n = len(vals)
        
        # Start with odd numbers, which are always sum-free
        odd = [x for x in vals if x % 2 != 0]
        
        # Try to augment the odd set with even numbers
        # This is safe because can_add checks for conflicts
        best = greedy_augment(odd, vals)
        
        # Try a few other simple heuristics that are likely sum-free or fixable
        # 1. Largest absolute values (often sum-free if gaps are large)
        # 2. Positive numbers only (if we take the largest half, often sum-free)
        
        # Let's try starting from empty and adding greedily in order of abs value
        # This might find a larger set than odd+augment
        greedy_from_empty = greedy_augment([], vals)
        
        if len(greedy_from_empty) > len(best):
            best = greedy_from_empty
            
        # Final safety check: if best is not sum-free, fall back to odd
        if not is_sum_free(best):
            best = odd
            
        return best

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() - start_time > DEADLINE - 5:
            # Not enough time, return odd numbers as fallback
            vals = instance(size, seed)
            results.append([x for x in vals if x % 2 != 0])
            continue
        vals = instance(size, seed)
        result = solve_one(vals)
        results.append(result)

    return results
# EVOLVE-BLOCK-END
