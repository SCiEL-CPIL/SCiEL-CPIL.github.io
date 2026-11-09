"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    n = 256
    
    def can_add_fast(selected_set, x):
        """O(K) check: can x be added without creating a 3-AP?"""
        for y in selected_set:
            if y == x:
                continue
            z = 2 * x - y
            if z != x and z != y and z in selected_set:
                return False
            z2 = 2 * y - x
            if z2 != x and z2 != y and z2 in selected_set:
                return False
        return True
    
    def compute_weight(selected_set):
        """Compute sum of 1/x for x in selected_set."""
        total = Fraction(0, 1)
        for x in selected_set:
            total += Fraction(1, x)
        return total
    
    def is_3ap_free(selected_set):
        """Check if selected_set has no 3-AP using O(K^2) but with early exit."""
        s_list = sorted(selected_set)
        for i, a in enumerate(s_list):
            for b in s_list[i+1:]:
                c = 2 * b - a
                if c in selected_set and c != a and c != b:
                    return False
        return True
    
    def greedy_construct(order=None):
        """Greedy construction with optional custom order."""
        selected = set()
        if order is None:
            order = list(range(1, 257))
        for x in order:
            if can_add_fast(selected, x):
                selected.add(x)
        return selected
    
    # Multiple greedy initializations with different orderings
    best_set = set()
    best_weight = Fraction(0, 1)
    
    # Standard greedy (increasing order) - prioritizes small numbers for high reciprocal weight
    s1 = greedy_construct()
    w1 = compute_weight(s1)
    if w1 > best_weight:
        best_set = s1
        best_weight = w1
    
    # Greedy with randomization - multiple diverse starts
    for trial in range(12):
        if time.time() > deadline - 20:
            break
        random.seed(trial * 1000 + 42)
        random_order = list(range(1, 257))
        for i in range(len(random_order) - 1):
            if random.random() < 0.5:
                j = i + random.randint(1, len(random_order) - i - 1)
                random_order[i], random_order[j] = random_order[j], random_order[i]
        s = greedy_construct(random_order)
        w = compute_weight(s)
        if w > best_weight:
            best_set = s
            best_weight = w
    
    # Local search: iterative improvement with O(K) checks
    improved = True
    iteration = 0
    while improved and time.time() < deadline - 5:
        improved = False
        iteration += 1
        current_set = set(best_set)
        current_weight = best_weight
        
        # Phase 1: Try adding each unselected number (O(K) per check)
        for x in range(1, 257):
            if x not in current_set and can_add_fast(current_set, x):
                new_set = current_set | {x}
                new_weight = current_weight + Fraction(1, x)
                if new_weight > current_weight:
                    best_set = new_set
                    best_weight = new_weight
                    improved = True
        
        # Phase 2: Try single swaps (remove y, add x) - O(K) per check
        for y in sorted(list(best_set), key=lambda v: -1/v):
            for x in range(1, 257):
                if x not in best_set:
                    trial_set = (best_set - {y}) | {x}
                    if is_3ap_free(trial_set):
                        new_weight = best_weight - Fraction(1, y) + Fraction(1, x)
                        if new_weight > best_weight:
                            best_set = trial_set
                            best_weight = new_weight
                            improved = True
                            break
        
        # Phase 3: Try 2-for-1 swaps if time permits
        if time.time() < deadline - 10:
            for y1 in list(best_set):
                if time.time() > deadline - 5:
                    break
                for y2 in list(best_set):
                    if y1 >= y2:
                        continue
                    if time.time() > deadline - 5:
                        break
                    for x in range(1, 257):
                        if x not in best_set and x != y1 and x != y2:
                            trial_set = (best_set - {y1, y2}) | {x}
                            if is_3ap_free(trial_set):
                                new_weight = best_weight - Fraction(1, y1) - Fraction(1, y2) + Fraction(1, x)
                                if new_weight > best_weight:
                                    best_set = trial_set
                                    best_weight = new_weight
                                    improved = True
                                    break
                    if time.time() > deadline - 5:
                        break
        
        # Phase 4: Try 1-for-2 swaps (remove one, add two)
        if time.time() < deadline - 15:
            for y in list(best_set):
                if time.time() > deadline - 5:
                    break
                for x1 in range(1, 257):
                    if x1 in best_set or x1 == y:
                        continue
                    if time.time() > deadline - 5:
                        break
                    for x2 in range(x1+1, 257):
                        if x2 in best_set or x2 == y:
                            continue
                        if time.time() > deadline - 5:
                            break
                        trial_set = (best_set - {y}) | {x1, x2}
                        if is_3ap_free(trial_set):
                            new_weight = best_weight - Fraction(1, y) + Fraction(1, x1) + Fraction(1, x2)
                            if new_weight > best_weight:
                                best_set = trial_set
                                best_weight = new_weight
                                improved = True
                                break
                    if time.time() > deadline - 5:
                        break
    
    # Final polish: try all single additions one more time
    for x in range(1, 257):
        if x not in best_set and can_add_fast(best_set, x):
            best_set = best_set | {x}
            best_weight += Fraction(1, x)
    
    result = sorted(best_set)
    if not result:
        return [1]
    return result
# EVOLVE-BLOCK-END
