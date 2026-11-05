import time
import random
from fractions import Fraction

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    weights = [1.0 / i for i in range(n + 1)]
    
    def can_add(x, s_set):
        for a in s_set:
            if a == x:
                continue
            b1 = 2 * x - a
            if b1 != x and b1 in s_set:
                return False
            if (a + x) % 2 == 0:
                b2 = (a + x) // 2
                if b2 != x and b2 != a and b2 in s_set:
                    return False
            b3 = 2 * a - x
            if b3 != x and b3 != a and b3 in s_set:
                return False
        return True

    def is_3ap_free(lst):
        s = set(lst)
        for i, a in enumerate(lst):
            for b in lst[i+1:]:
                if (a + b) % 2 == 0:
                    mid = (a + b) // 2
                    if mid != a and mid != b and mid in s:
                        return False
        return True

    def greedy_fill(s_set):
        new_set = set(s_set)
        for x in range(1, n + 1):
            if x not in new_set:
                if can_add(x, new_set):
                    new_set.add(x)
        return new_set

    def local_polish(s_set):
        """Remove one element, greedy fill, keep if better. Repeat."""
        improved = True
        current_set = set(s_set)
        current_weight = sum(weights[x] for x in current_set)
        
        while improved and time.time() < deadline:
            improved = False
            # Try removing each element
            for x in list(current_set):
                if time.time() > deadline:
                    break
                trial_set = current_set - {x}
                new_set = greedy_fill(trial_set)
                new_weight = sum(weights[elem] for elem in new_set)
                if new_weight > current_weight + 1e-9:
                    current_set = new_set
                    current_weight = new_weight
                    improved = True
                    break
        return current_set, current_weight

    # 1. Initial Greedy Solution
    best_set = set()
    for x in range(1, n + 1):
        if can_add(x, best_set):
            best_set.add(x)
    
    best_set, best_weight = local_polish(best_set)
    
    # 2. Iterated Local Search
    if time.time() < deadline:
        while time.time() < deadline:
            current_set = set(best_set)
            
            # Perturb
            k_perturb = random.randint(2, 8)
            if len(current_set) > k_perturb:
                to_remove = random.sample(list(current_set), k_perturb)
                for x in to_remove:
                    current_set.remove(x)
            
            # Re-optimize
            new_set = greedy_fill(current_set)
            new_set, new_weight = local_polish(new_set)
            
            if new_weight > best_weight + 1e-9:
                best_set = new_set
                best_weight = new_weight
            
            # Occasionally try a larger perturbation
            if random.random() < 0.05:
                k_perturb = random.randint(10, 20)
                if len(best_set) > k_perturb:
                    to_remove = random.sample(list(best_set), k_perturb)
                    for x in to_remove:
                        current_set.remove(x)
                    new_set = greedy_fill(current_set)
                    new_set, new_weight = local_polish(new_set)
                    if new_weight > best_weight + 1e-9:
                        best_set = new_set
                        best_weight = new_weight

    result = sorted(best_set)
    assert len(result) > 0, "Set must be nonempty"
    assert is_3ap_free(result), "Set must be 3-AP-free"
    
    return result
