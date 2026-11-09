"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # seconds, safety margin before 60s kill
    
    def elapsed():
        return time.time() - start_time
    
    N = 256
    numbers = list(range(2, N + 1))
    n = len(numbers)
    
    # Precompute conflict graph using bitmasks for speed
    conflict_mask = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            a, b = numbers[i], numbers[j]
            g = gcd(a, b)
            l = a * b // g
            if l <= N:  # lcm <= 256, conflict
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    
    # Reciprocal weights
    weights = [1.0 / numbers[i] for i in range(n)]
    
    def can_add(i, selected_mask):
        return (conflict_mask[i] & selected_mask) == 0
    
    def reciprocal_sum_from_mask(mask):
        total = 0.0
        for i in range(n):
            if mask & (1 << i):
                total += weights[i]
        return total
    
    # Greedy construction
    def greedy_construct(order):
        selected_mask = 0
        for i in order:
            if can_add(i, selected_mask):
                selected_mask |= (1 << i)
        return selected_mask
    
    # Fast local search: try adding all possible, then try 1-for-1 and 1-for-2 swaps
    def local_search(mask, max_iters=50):
        current_sum = reciprocal_sum_from_mask(mask)
        for _ in range(max_iters):
            if elapsed() > DEADLINE:
                break
            improved = False
            
            # Phase 1: Add all possible nodes
            changed = True
            while changed:
                changed = False
                for i in range(n):
                    if not (mask & (1 << i)) and can_add(i, mask):
                        mask |= (1 << i)
                        changed = True
                        improved = True
            
            # Phase 2: Try removing one node and adding others
            for i in range(n):
                if not (mask & (1 << i)):
                    continue
                if elapsed() > DEADLINE:
                    break
                
                mask_without_i = mask & ~(1 << i)
                new_mask = mask_without_i
                
                # Greedily add back as many as possible, prioritizing high weight
                candidates = [j for j in range(n) if not (new_mask & (1 << j)) and j != i and can_add(j, new_mask)]
                candidates.sort(key=lambda j: -weights[j])
                for j in candidates:
                    if can_add(j, new_mask):
                        new_mask |= (1 << j)
                
                new_sum = reciprocal_sum_from_mask(new_mask)
                if new_sum > current_sum + 1e-12:
                    current_sum = new_sum
                    mask = new_mask
                    improved = True
                    break  # Restart from new state
            
            if not improved:
                break
        
        return mask, current_sum
    
    best_mask = 0
    best_sum = 0.0
    
    # Strategy 1: Weight-descending greedy
    order1 = sorted(range(n), key=lambda i: (-weights[i], conflict_mask[i].bit_count()))
    mask1 = greedy_construct(order1)
    mask1, sum1 = local_search(mask1)
    if sum1 > best_sum:
        best_sum = sum1
        best_mask = mask1
    
    # Strategy 2: Weight/conflict-ratio greedy
    order2 = sorted(range(n), key=lambda i: (-(weights[i] / (1 + conflict_mask[i].bit_count())), -weights[i]))
    mask2 = greedy_construct(order2)
    mask2, sum2 = local_search(mask2)
    if sum2 > best_sum:
        best_sum = sum2
        best_mask = mask2
    
    # Strategy 3-8: Randomized restarts with different biases
    random.seed(42)
    for trial in range(8):
        if elapsed() > DEADLINE - 2:
            break
        
        if trial == 0:
            # Pure random shuffle
            order = list(range(n))
            random.shuffle(order)
        elif trial == 1:
            # Bias toward small numbers
            order = list(range(n))
            for i in range(n - 1, 0, -1):
                j = random.randint(0, i)
                if random.random() < 0.7:  # Favor keeping smaller numbers earlier
                    order[i], order[j] = order[j], order[i]
        elif trial == 2:
            # Bias toward low-conflict nodes
            order = list(range(n))
            random.shuffle(order)
            order.sort(key=lambda i: (random.random() * 0.5 + conflict_mask[i].bit_count() / n))
        elif trial == 3:
            # Weight-biased random
            order = list(range(n))
            for i in range(n - 1, 0, -1):
                j = random.randint(0, i)
                w_i = weights[order[i]]
                w_j = weights[order[j]]
                if random.random() < w_i / (w_i + w_j + 1e-10):
                    order[i], order[j] = order[j], order[i]
        elif trial == 4:
            # Reverse weight order with noise
            order = list(range(n))
            order.sort(key=lambda i: weights[i])  # Small weights first
            for i in range(0, n - 1, 2):
                if random.random() < 0.5:
                    order[i], order[i+1] = order[i+1], order[i]
        elif trial == 5:
            # Conflict-free priority
            low_conf = [i for i in range(n) if conflict_mask[i].bit_count() < n * 0.1]
            high_conf = [i for i in range(n) if conflict_mask[i].bit_count() >= n * 0.1]
            random.shuffle(low_conf)
            random.shuffle(high_conf)
            order = low_conf + high_conf
        elif trial == 6:
            # Geometric spacing bias
            order = list(range(n))
            random.shuffle(order)
            order.sort(key=lambda i: (numbers[i] % 16, random.random()))
        else:
            # Mixed: half weight-sorted, half random
            half = n // 2
            order_a = sorted(range(n), key=lambda i: -weights[i])[:half]
            order_b = sorted(range(n), key=lambda i: -weights[i])[half:]
            random.shuffle(order_b)
            order = order_a + order_b
        
        mask = greedy_construct(order)
        mask, s = local_search(mask)
        if s > best_sum:
            best_sum = s
            best_mask = mask
    
    # Final refinement: try more aggressive local search on best
    if elapsed() < DEADLINE - 5:
        for _ in range(3):
            if elapsed() > DEADLINE - 2:
                break
            # Perturb: remove a random subset and re-optimize
            perturbed = best_mask
            num_remove = max(1, int(best_mask.bit_count() * 0.15))
            selected_indices = [i for i in range(n) if best_mask & (1 << i)]
            to_remove = random.sample(selected_indices, min(num_remove, len(selected_indices)))
            for i in to_remove:
                perturbed &= ~(1 << i)
            perturbed, s = local_search(perturbed, max_iters=100)
            if s > best_sum + 1e-12:
                best_sum = s
                best_mask = perturbed
    
    # Convert back to numbers
    result = []
    for i in range(n):
        if best_mask & (1 << i):
            result.append(numbers[i])
    result.sort()
    return result
# EVOLVE-BLOCK-END
