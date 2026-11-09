import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_max_val = float('inf')
    best_set = None
    
    def is_valid_set(A):
        if 0 not in A:
            return False
        diffs = set()
        for i in range(len(A)):
            for j in range(i):
                d = A[i] - A[j]
                if d in diffs:
                    return False
                diffs.add(d)
        for d in range(1, 31):
            if d not in diffs:
                return False
        return True

    def get_initial_set():
        a = [0]
        def diffs_list(s):
            return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
        for d in range(1, 31):
            if d in diffs_list(sorted(a)):
                continue
            found = False
            for v in sorted(a):
                candidates = []
                if v + d not in a: candidates.append(v + d)
                if v - d >= 0 and v - d not in a: candidates.append(v - d)
                for x in candidates:
                    trial = sorted(a + [x])
                    dl = diffs_list(trial)
                    if len(set(dl)) == len(dl):
                        a = trial
                        found = True
                        break
                if found:
                    break
            if not found:
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y])
                    dl = diffs_list(trial)
                    if len(set(dl)) == len(dl):
                        a = trial
                        break
                    y += 1
        return a

    def local_search(A, max_iter=20000):
        current_A = A[:]
        current_max = max(current_A)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            idx = random.randint(1, len(current_A) - 1)
            old_val = current_A[idx]
            
            # Random step size
            step = random.randint(-20, 20)
            if step == 0: step = 1
            new_val = old_val + step
            
            if new_val < 1 or new_val > 10**6:
                continue
            if new_val in current_A:
                continue
                
            trial_A = current_A[:]
            trial_A[idx] = new_val
            trial_A.sort()
            
            if is_valid_set(trial_A):
                new_max = max(trial_A)
                if new_max < current_max:
                    current_A = trial_A
                    current_max = new_max
                elif new_max == current_max:
                    if random.random() < 0.3:
                        current_A = trial_A
                elif new_max < current_max + 5:
                    # Simulated annealing style: accept worse slightly
                    if random.random() < 0.1:
                        current_A = trial_A
                        current_max = new_max
                        
        return current_A, current_max

    # Run multiple rounds
    for i in range(20):
        if time.time() > deadline:
            break
        A = get_initial_set()
        # Randomize initial set slightly
        for _ in range(5):
            idx = random.randint(1, len(A)-1)
            A[idx] += random.randint(-10, 10)
            A.sort()
            if not is_valid_set(A):
                A = get_initial_set() # Reset if broken
                
        A, m = local_search(A, max_iter=10000)
        if m < best_max_val:
            best_max_val = m
            best_set = A
            # If we find something very good, we can stop or continue
            if best_max_val < 50:
                break

    if best_set is None:
        # Fallback
        A = get_initial_set()
        return A

    return best_set
