import time
import sys
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    def try_greedy(start=1, end=4096, step=1, skip_initial=None):
        selected = []
        existing_sums = set()
        
        # Pre-allocate or just use list
        for x in range(start, end + 1, step):
            if time.time() > deadline:
                return selected
            
            if skip_initial and x in skip_initial:
                continue
                
            # Check if x can be added
            valid = True
            
            # 1. x, x, x -> 3x
            if 3 * x in existing_sums:
                valid = False
            
            # 2. x, x, y -> 2x + y
            if valid:
                for y in selected:
                    if 2 * x + y in existing_sums:
                        valid = False
                        break
            
            # 3. x, y, z -> x + y + z
            if valid:
                k = len(selected)
                for i in range(k):
                    y = selected[i]
                    # y <= z
                    for j in range(i, k):
                        z = selected[j]
                        if x + y + z in existing_sums:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                selected.append(x)
                # Update existing_sums
                # Add 3x
                existing_sums.add(3 * x)
                # Add 2x + y
                for y in selected[:-1]: # exclude x itself as y? No, y is from old set
                    existing_sums.add(2 * x + y)
                # Add x + y + z
                k_old = len(selected) - 1
                for i in range(k_old):
                    y = selected[i]
                    for j in range(i, k_old):
                        z = selected[j]
                        existing_sums.add(x + y + z)
        
        return selected

    # Standard greedy
    best_set = try_greedy(1, 4096, 1)
    
    # Try a few variations with small skips to see if we can get a larger set
    # Sometimes skipping a small number allows more numbers to fit in
    variations = [
        {2}, {3}, {4}, {2, 3}, {5}, {1, 2} # Note: skipping 1 is usually bad but let's check
    ]
    
    for skip in variations:
        if time.time() > deadline - 1.0:
            break
        candidate = try_greedy(1, 4096, 1, skip_initial=skip)
        if len(candidate) > len(best_set):
            best_set = candidate
            
    # If we still have time, try random restarts with random skips
    if time.time() < deadline - 2.0:
        # Try a few random sets of skipped numbers
        for _ in range(5):
            if time.time() > deadline - 1.0:
                break
            # Skip 1-3 random small numbers
            skip_count = random.randint(1, 3)
            skips = set(random.sample(range(1, 20), min(skip_count, 19)))
            candidate = try_greedy(1, 4096, 1, skip_initial=skips)
            if len(candidate) > len(best_set):
                best_set = candidate

    return best_set

if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res)
