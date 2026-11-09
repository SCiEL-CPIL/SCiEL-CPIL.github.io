import time
import random

def construct():
    L = 232792560
    elements = list(range(1, 23))
    weights = [L // i for i in elements]
    
    def is_valid(indices):
        if not indices:
            return True
        w = [weights[i] for i in indices]
        sums = {0}
        for x in w:
            new_sums = set()
            for s in sums:
                new_sums.add(s + x)
            if len(sums | new_sums) != len(sums) + len(new_sums):
                return False
            sums = sums | new_sums
        return len(sums) == (1 << len(w))

    best_subset = []
    best_size = 0
    start_time = time.time()
    deadline = start_time + 85.0

    # Heuristic 1: Descending weight (small i)
    order1 = sorted(range(22), key=lambda i: weights[i], reverse=True)
    curr = []
    for idx in order1:
        if is_valid(curr + [idx]):
            curr.append(idx)
    if len(curr) > best_size:
        best_size = len(curr)
        best_subset = curr[:]

    # Heuristic 2: Ascending weight (large i)
    order2 = sorted(range(22), key=lambda i: weights[i])
    curr = []
    for idx in order2:
        if is_valid(curr + [idx]):
            curr.append(idx)
    if len(curr) > best_size:
        best_size = len(curr)
        best_subset = curr[:]

    # Heuristic 3: Random restarts
    while time.time() < deadline:
        random.shuffle(elements)
        curr = []
        for elem in elements:
            idx = elem - 1
            if is_valid(curr + [idx]):
                curr.append(idx)
        if len(curr) > best_size:
            best_size = len(curr)
            best_subset = curr[:]
            
            # If we found a set of size 16+, we might be close to optimal.
            # We can continue to see if 17 is possible.
            
    result = [idx + 1 for idx in best_subset]
    if not result:
        return [1]
    return result
