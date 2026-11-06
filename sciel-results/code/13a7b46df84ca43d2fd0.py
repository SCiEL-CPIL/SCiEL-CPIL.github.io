import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin for safety
    
    def can_add(x, A, counts):
        if counts[2*x] >= 2:
            return False
        for y in A:
            if y < x:
                continue
            if counts[x+y] >= 2:
                return False
        return True

    def do_add(x, A, counts):
        for y in list(A):
            counts[x+y] += 1
        counts[2*x] += 1
        A.add(x)

    def do_remove(x, A, counts):
        for y in A:
            if y == x:
                continue
            counts[x+y] -= 1
        counts[2*x] -= 1
        A.remove(x)

    best_size = 0
    best_A = set()
    best_counts = [0] * 2001
    
    # Multiple restarts
    while time.time() < deadline:
        # Greedy initialization with random order
        A = set()
        counts = [0] * 2001
        candidates = list(range(1, 1001))
        random.shuffle(candidates)
        
        for x in candidates:
            if can_add(x, A, counts):
                do_add(x, A, counts)
        
        if len(A) > best_size:
            best_size = len(A)
            best_A = set(A)
            best_counts = counts.copy()
        
        # Local search
        unused = set(range(1, 1001)) - A
        no_improve_count = 0
        max_no_improve = 5000
        
        while time.time() < deadline - 5:
            added = False
            # Try to add
            # Pick random unused to avoid scanning all
            if unused:
                y = random.choice(list(unused))
                if can_add(y, A, counts):
                    do_add(y, A, counts)
                    unused.remove(y)
                    added = True
                    if len(A) > best_size:
                        best_size = len(A)
                        best_A = set(A)
                        best_counts = counts.copy()
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
            else:
                break
            
            if not added:
                # Try swap
                if A and unused:
                    x = random.choice(list(A))
                    y = random.choice(list(unused))
                    
                    do_remove(x, A, counts)
                    unused.add(x)
                    
                    if can_add(y, A, counts):
                        do_add(y, A, counts)
                        unused.remove(y)
                        no_improve_count += 1
                    else:
                        do_add(x, A, counts)
                        unused.remove(x)
                        no_improve_count += 1
                else:
                    break
                
                if no_improve_count > max_no_improve:
                    break

    return sorted(best_A)
