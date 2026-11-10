import time
import random

def construct():
    deadline = time.time() + 80
    n = 12
    
    # Theoretical lower bound for max element M:
    # Sum of largest n distinct integers <= M is n*M - n(n-1)/2
    # This sum must be >= 2^n - 1
    # 12*M - 66 >= 4095 => 12*M >= 4161 => M >= 347
    # Let's search in range [350, 1000] initially
    
    def get_subset_sums(elements):
        sums = {0}
        for e in elements:
            sums = sums | {s + e for s in sums}
        return sums

    def is_valid(elements):
        sums = get_subset_sums(elements)
        return len(sums) == (1 << n)

    best = None
    best_max = float('inf')
    
    # Initial random starts
    for _ in range(100):
        # Generate 12 distinct random integers
        # Range: try to be close to lower bound
        start_range = random.randint(350, 600)
        elements = random.sample(range(1, start_range), n)
        if is_valid(elements):
            curr_max = max(elements)
            if curr_max < best_max:
                best_max = curr_max
                best = elements[:]
                # If we found a very good one, tighten the range for next starts?
                # Or just continue.
                
    if best is None:
        # Fallback to powers of 2 if no valid set found in random search
        return [1 << i for i in range(n)]

    # Local search to improve best
    current = best[:]
    current_max = best_max
    
    while time.time() < deadline - 5:
        improved = False
        # Try to reduce the max element
        max_idx = current.index(max(current))
        
        # Try values slightly below current max
        # Or random values below current max
        for _ in range(1000):
            if time.time() > deadline - 5:
                break
            new_val = random.randint(1, current_max - 1)
            if new_val in current:
                continue
            candidate = current[:]
            candidate[max_idx] = new_val
            if is_valid(candidate):
                new_max = max(candidate)
                if new_max < current_max:
                    current = candidate
                    current_max = new_max
                    improved = True
                    break
        
        if not improved:
            # Try changing any element to a smaller value
            # Or swap
            for i in range(n):
                if time.time() > deadline - 5:
                    break
                for _ in range(100):
                    if time.time() > deadline - 5:
                        break
                    new_val = random.randint(1, current_max - 1)
                    if new_val in current:
                        continue
                    candidate = current[:]
                    candidate[i] = new_val
                    if is_valid(candidate):
                        new_max = max(candidate)
                        if new_max < current_max:
                            current = candidate
                            current_max = new_max
                            improved = True
                            break
                if improved:
                    break
                    
        if not improved:
            # Try random restarts within the current best range
            # Or just break if stuck?
            # Let's try a few more random perturbations
            for _ in range(100):
                if time.time() > deadline - 5:
                    break
                i = random.randint(0, n-1)
                new_val = random.randint(1, current_max)
                if new_val in current:
                    continue
                candidate = current[:]
                candidate[i] = new_val
                if is_valid(candidate):
                    new_max = max(candidate)
                    if new_max <= current_max:
                        current = candidate
                        current_max = new_max
                        improved = True
                        break
            if not improved:
                # Stuck, try a new random start closer to current_max
                elements = random.sample(range(1, current_max + 10), n)
                if is_valid(elements):
                    curr_max = max(elements)
                    if curr_max < current_max:
                        current = elements
                        current_max = curr_max
                    else:
                        # If not better, maybe we should keep searching?
                        # But if we can't improve, we might be at a local minimum.
                        pass

    return current
