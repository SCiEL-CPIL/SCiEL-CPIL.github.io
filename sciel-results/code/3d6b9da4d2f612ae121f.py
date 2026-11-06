import time
import random

def construct():
    start = time.time()
    deadline = start + 80
    
    def compute_mult(A):
        mult = [0] * 257
        for i, a in enumerate(A):
            for b in A[i:]:
                s = a + b
                if 128 <= s <= 256:
                    mult[s] += 1
        return mult
    
    def is_feasible(A):
        mult = compute_mult(A)
        for s in range(128, 257):
            if mult[s] == 0:
                return False
        return True
    
    def peak_val(A):
        mult = compute_mult(A)
        return max(mult[s] for s in range(128, 257))
    
    # Initial: full set, greedily remove
    A = list(range(129))
    random.shuffle(A)
    for x in A[:]:
        if time.time() > deadline:
            break
        A_test = [a for a in A if a != x]
        if is_feasible(A_test):
            A = A_test
    
    best_A = A[:]
    best_peak = peak_val(A)
    
    all_candidates = list(range(129))
    
    while time.time() < deadline:
        improved = False
        for _ in range(2000):
            if time.time() > deadline:
                break
            
            A_set = set(A)
            
            # Try swap
            if len(A) > 1:
                idx = random.randint(0, len(A) - 1)
                x = A[idx]
                candidates = [c for c in all_candidates if c not in A_set]
                if candidates:
                    y = random.choice(candidates)
                    A_new = A[:]
                    A_new[idx] = y
                    if is_feasible(A_new):
                        p = peak_val(A_new)
                        if p < best_peak:
                            best_peak = p
                            best_A = A_new[:]
                            A = A_new
                            improved = True
                            break
            
            # Try remove
            if len(A) > 1:
                idx = random.randint(0, len(A) - 1)
                A_new = A[:idx] + A[idx+1:]
                if is_feasible(A_new):
                    p = peak_val(A_new)
                    if p < best_peak:
                        best_peak = p
                        best_A = A_new[:]
                        A = A_new
                        improved = True
                        break
        
        # Try add if no improvement
        if not improved:
            A_set = set(A)
            candidates = [c for c in all_candidates if c not in A_set]
            if candidates:
                y = random.choice(candidates)
                A_new = A + [y]
                if is_feasible(A_new):
                    p = peak_val(A_new)
                    if p < best_peak:
                        best_peak = p
                        best_A = A_new[:]
                        A = A_new
    
    return best_A
