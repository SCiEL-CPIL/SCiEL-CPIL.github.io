import time
import random

N = 256

def count_3aps(A):
    """Count number of 3-term arithmetic progressions in A."""
    A_set = set(A)
    count = 0
    for a in A:
        for b in A:
            if a == b:
                continue
            c = 2*b - a
            if c in A_set and c > b:
                count += 1
    return count

def has_4ap_with_x(A, x):
    """Check if adding x to A creates a 4-AP containing x."""
    A_set = set(A)
    # x is 1st: x, x+d, x+2d, x+3d
    max_d = (N - x) // 3
    for d in range(1, max_d + 1):
        if x+d in A_set and x+2*d in A_set and x+3*d in A_set:
            return True
    # x is 2nd: x-d, x, x+d, x+2d
    max_d = min(x-1, (N-x)//2)
    for d in range(1, max_d + 1):
        if x-d in A_set and x+d in A_set and x+2*d in A_set:
            return True
    # x is 3rd: x-2d, x-d, x, x+d
    max_d = min((x-1)//2, N-x)
    for d in range(1, max_d + 1):
        if x-2*d in A_set and x-d in A_set and x+d in A_set:
            return True
    # x is 4th: x-3d, x-2d, x-d, x
    max_d = (x-1) // 3
    for d in range(1, max_d + 1):
        if x-3*d in A_set and x-2*d in A_set and x-d in A_set:
            return True
    return False

def delta_3aps_add(A, x):
    """Calculate increase in 3-AP count if x is added to A."""
    A_set = set(A)
    delta = 0
    # x is 1st: x, x+d, x+2d
    max_d = (N - x) // 2
    for d in range(1, max_d + 1):
        if x+d in A_set and x+2*d in A_set:
            delta += 1
    # x is 2nd: x-d, x, x+d
    max_d = min(x-1, N-x)
    for d in range(1, max_d + 1):
        if x-d in A_set and x+d in A_set:
            delta += 1
    # x is 3rd: x-2d, x-d, x
    max_d = (x-1) // 2
    for d in range(1, max_d + 1):
        if x-2*d in A_set and x-d in A_set:
            delta += 1
    return delta

def local_search(A):
    """Improve set A using local search."""
    A = set(A)
    current_score = count_3aps(A)
    improved = True
    while improved:
        improved = False
        # Add phase
        for x in range(1, N + 1):
            if x not in A:
                if not has_4ap_with_x(A, x):
                    new_score = current_score + delta_3aps_add(A, x)
                    if new_score > current_score:
                        A.add(x)
                        current_score = new_score
                        improved = True
        # Swap phase
        for x in list(A):
            for y in range(1, N + 1):
                if y not in A:
                    A_temp = A.copy()
                    A_temp.remove(x)
                    if not has_4ap_with_x(A_temp, y):
                        new_set = A_temp.copy()
                        new_set.add(y)
                        new_score = count_3aps(new_set)
                        if new_score > current_score:
                            A = new_set
                            current_score = new_score
                            improved = True
                            break
    return A, current_score

def greedy_init():
    """Greedy initialization similar to seed."""
    selected = set()
    for x in range(1, N + 1):
        if all(not all(x-j*d in selected for j in (1,2,3)) for d in range(1,(x-1)//3+1)):
            selected.add(x)
    return selected

def construct():
    start_time = time.time()
    max_time = 80
    best_set = greedy_init()
    best_score = count_3aps(best_set)
    
    # Try to improve with local search
    improved_set, improved_score = local_search(best_set)
    if improved_score > best_score:
        best_set = improved_set
        best_score = improved_score

    # Random restarts if time permits
    while time.time() - start_time < max_time:
        # Random valid set initialization
        A = set(random.sample(range(1, N+1), 40))
        # Repair
        valid = True
        while valid:
            valid = False
            for x in list(A):
                A_temp = A.copy()
                A_temp.remove(x)
                if has_4ap_with_x(A_temp, x):
                    A.remove(x)
                    valid = True
        
        if not A:
            continue
            
        # Local search
        A, score = local_search(A)
        if score > best_score:
            best_set = A
            best_score = score

    return sorted(best_set)
