import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 128
    TARGET_MIN = 128
    TARGET_MAX = 256
    targets = list(range(TARGET_MIN, TARGET_MAX + 1))
    
    def compute_reps(A_set):
        reps = {}
        for t in targets:
            count = 0
            for a in A_set:
                b = t - a
                if b in A_set and a <= b:
                    count += 1
            reps[t] = count
        return reps
    
    def is_valid(A_set):
        for t in targets:
            count = 0
            for a in A_set:
                b = t - a
                if b in A_set and a <= b:
                    count += 1
                    if count >= 1:
                        break
            if count == 0:
                return False
        return True
    
    # Greedy construction
    A = set()
    uncovered = set(targets)
    
    # Try adding elements greedily
    for _ in range(100):
        if time.time() > deadline:
            break
        if not uncovered:
            break
        # For each candidate element, compute how many uncovered targets it would help cover
        # An element a helps cover target t if there exists b in A such that a+b=t or b+a=t with a<=b
        # Actually, adding a new element x to A creates new pairs (x, y) for y in A with x<=y, and (x,x)
        # So new pairs are: for each y in A with x<=y: pair (x,y) covers target x+y
        # And pair (x,x) covers target 2x
        
        best_elem = None
        best_gain = 0
        for x in range(0, 129):
            if x in A:
                continue
            # Compute which uncovered targets would be covered
            gain = 0
            for y in A:
                if x <= y:
                    t = x + y
                    if t in uncovered:
                        gain += 1
                else:
                    t = x + y
                    if t in uncovered:
                        gain += 1
            t = 2*x
            if t in uncovered:
                gain += 1
            if gain > best_gain:
                best_gain = gain
                best_elem = x
        if best_elem is None or best_gain == 0:
            break
        A.add(best_elem)
        # Update uncovered
        for t in list(uncovered):
            count = 0
            for a in A:
                b = t - a
                if b in A and a <= b:
                    count += 1
                    break
            if count > 0:
                uncovered.discard(t)
    
    # Now try to improve via local search
    best_A = set(A)
    best_peak = max(compute_reps(best_A)[t] for t in targets)
    
    # Local search: try removing elements
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for elem in list(best_A):
            if time.time() > deadline:
                break
            A_try = best_A - {elem}
            if is_valid(A_try):
                p = max(compute_reps(A_try)[t] for t in targets)
                if p < best_peak:
                    best_peak = p
                    best_A = A_try
                    improved = True
                    break
    
    # Randomized swaps
    if time.time() < deadline:
        A = set(best_A)
        for _ in range(50000):
            if time.time() > deadline:
                break
            if len(A) < 2:
                break
            elem = random.choice(list(A))
            A_no = A - {elem}
            # Try adding a random element
            candidates = [x for x in range(0, 129) if x not in A_no]
            if not candidates:
                continue
            for _ in range(3):
                new_elem = random.choice(candidates)
                A_try = A_no | {new_elem}
                if is_valid(A_try):
                    p = max(compute_reps(A_try)[t] for t in targets)
                    if p < best_peak:
                        best_peak = p
                        best_A = A_try
                        A = A_try
                        break
    
    return list(best_A)
