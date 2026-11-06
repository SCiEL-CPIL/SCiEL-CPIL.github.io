import random
import time

def construct():
    N = 1000
    start_time = time.time()
    limit = 115  # Leave 5s buffer
    
    def is_valid_add(A, x, counts):
        if x in A:
            return False
        if counts.get(2*x, 0) >= 2:
            return False
        for y in A:
            if counts.get(x+y, 0) >= 2:
                return False
        return True

    def greedy_construct(order):
        A = []
        counts = {}
        A_set = set()
        for x in order:
            if is_valid_add(A, x, counts):
                A.append(x)
                A_set.add(x)
                counts[2*x] = counts.get(2*x, 0) + 1
                for y in A[:-1]:
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
        return A, counts, A_set

    best_A = []
    best_counts = {}
    best_A_set = set()
    best_size = 0
    
    # Phase 1: Multi-Restart Greedy
    # Try ascending
    order = list(range(1, N + 1))
    A, counts, A_set = greedy_construct(order)
    if len(A) > best_size:
        best_size = len(A)
        best_A = A
        best_counts = counts
        best_A_set = A_set
        
    # Try random permutations
    for _ in range(10):
        order = list(range(1, N + 1))
        random.shuffle(order)
        A, counts, A_set = greedy_construct(order)
        if len(A) > best_size:
            best_size = len(A)
            best_A = A
            best_counts = counts
            best_A_set = A_set

    A = best_A
    counts = best_counts
    A_set = best_A_set

    # Phase 2: Local Search
    while time.time() - start_time < limit:
        # Try to find any valid add
        valid_adds = []
        for x in range(1, N + 1):
            if x not in A_set and is_valid_add(A, x, counts):
                valid_adds.append(x)
                
        if valid_adds:
            x = random.choice(valid_adds)
            A.append(x)
            A_set.add(x)
            counts[2*x] = counts.get(2*x, 0) + 1
            for y in A[:-1]:
                s = x + y
                counts[s] = counts.get(s, 0) + 1
            best_size = len(A)
            best_A = list(A)
            best_counts = dict(counts)
            best_A_set = set(A_set)
        else:
            # Stuck. Try to swap or remove.
            # Attempt a swap
            swapped = False
            if A:
                x = random.choice(A)
                # Try a few random y's
                for _ in range(5):
                    y = random.randint(1, N)
                    if y in A_set:
                        continue
                    
                    # Check if swapping x with y is valid
                    # 1. Remove x
                    A.remove(x)
                    A_set.remove(x)
                    counts[2*x] -= 1
                    if counts[2*x] == 0: del counts[2*x]
                    for y_elem in A:
                        s = x + y_elem
                        counts[s] -= 1
                        if counts[s] == 0: del counts[s]
                        
                    # 2. Check y
                    if is_valid_add(A, y, counts):
                        A.append(y)
                        A_set.add(y)
                        counts[2*y] = counts.get(2*y, 0) + 1
                        for y_elem in A[:-1]:
                            s = y + y_elem
                            counts[s] = counts.get(s, 0) + 1
                        swapped = True
                        break
                    else:
                        # Revert
                        A.append(x)
                        A_set.add(x)
                        counts[2*x] = counts.get(2*x, 0) + 1
                        for y_elem in A[:-1]:
                            s = x + y_elem
                            counts[s] = counts.get(s, 0) + 1

            if not swapped:
                # Perform a random removal to escape
                if not A:
                    break
                x = random.choice(A)
                A.remove(x)
                A_set.remove(x)
                counts[2*x] -= 1
                if counts[2*x] == 0: del counts[2*x]
                for y_elem in A:
                    s = x + y_elem
                    counts[s] -= 1
                    if counts[s] == 0: del counts[s]

    return best_A
