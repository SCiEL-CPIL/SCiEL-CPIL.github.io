import time
import random

def compute_objective(A):
    s = set()
    p = set()
    n = len(A)
    for i in range(n):
        ai = A[i]
        for j in range(n):
            aj = A[j]
            s.add(ai + aj)
            p.add(ai * aj)
    return max(len(s), len(p))

def construct():
    deadline = time.time() + 85
    
    # Generate smooth number candidates (2^a * 3^b * 5^c * 7^d <= 10000)
    candidates = set()
    for a in range(14):
        p2 = 2 ** a
        if p2 > 10000:
            break
        for b in range(9):
            p23 = p2 * (3 ** b)
            if p23 > 10000:
                break
            for c in range(6):
                p235 = p23 * (5 ** c)
                if p235 > 10000:
                    break
                for d in range(5):
                    val = p235 * (7 ** d)
                    if val <= 10000:
                        candidates.add(val)
    
    candidates = sorted(candidates)
    n_cand = len(candidates)
    
    # Greedy initialization: build set one element at a time minimizing objective
    A = [1]
    for _ in range(19):
        best_val = None
        best_score = float('inf')
        for c in candidates:
            if c in A:
                continue
            trial = A + [c]
            score = compute_objective(trial)
            if score < best_score:
                best_score = score
                best_val = c
        if best_val is not None:
            A.append(best_val)
        else:
            A.append(len(A) + 1)
    
    best_A = list(A)
    best_obj = compute_objective(best_A)
    
    # Local search with random swaps and perturbations
    random.seed(12345)
    no_improve = 0
    while time.time() < deadline and no_improve < 60:
        improved = False
        for _ in range(500):
            if time.time() > deadline:
                break
            i = random.randint(0, 19)
            new_val = candidates[random.randint(0, n_cand - 1)]
            if new_val in A:
                continue
            old_val = A[i]
            A[i] = new_val
            obj = compute_objective(A)
            if obj < best_obj:
                best_obj = obj
                best_A = list(A)
                improved = True
                no_improve = 0
            else:
                A[i] = old_val
        
        if not improved:
            no_improve += 1
            # Perturb current best to escape local minimum
            if random.random() < 0.6:
                A = list(best_A)
                for _ in range(random.randint(1, 4)):
                    i = random.randint(0, 19)
                    new_val = candidates[random.randint(0, n_cand - 1)]
                    if new_val not in A:
                        A[i] = new_val
                best_obj = compute_objective(A)
                best_A = list(A)
    
    return best_A
