import time

def construct():
    deadline = time.time() + 80  # 80s safety margin for 90s limit
    
    # Greedy: pick smallest positive integer not in previous subset sums
    # and not already used
    chosen = []
    subset_sums = {0}  # all subset sums of chosen elements
    
    for i in range(12):
        if time.time() > deadline:
            break
        candidate = 1
        while True:
            if candidate in subset_sums:
                candidate += 1
                continue
            # candidate is not a subset sum of previous elements
            # Check it's not already in chosen (shouldn't happen if we're increasing)
            if candidate in chosen:
                candidate += 1
                continue
            # Accept this candidate
            chosen.append(candidate)
            # Update subset sums: add candidate to all existing subset sums
            new_sums = set()
            for s in subset_sums:
                new_sums.add(s + candidate)
            subset_sums = subset_sums | new_sums
            break
    
    # If we didn't complete 12 elements due to timeout, fallback to powers of 2
    if len(chosen) < 12:
        return [1 << i for i in range(12)]
    
    return chosen
