import random
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    LCM = 2329089562800
    multipliers = [LCM // (i + 1) for i in range(30)]
    
    best_abs = None
    best_coeffs = None
    
    random.seed(42)
    
    def update_best(coeffs, S):
        nonlocal best_abs, best_coeffs
        if S != 0 and (best_abs is None or abs(S) < best_abs):
            best_abs = abs(S)
            best_coeffs = list(coeffs)
    
    for restart in range(2000):
        if time.time() > deadline:
            break
        
        if restart == 0:
            coeffs = [0] * 28 + [1, -1]
        elif restart % 3 == 0:
            coeffs = [random.choice([-1, 0, 1]) for _ in range(30)]
        else:
            if best_coeffs is not None:
                coeffs = list(best_coeffs)
                for _ in range(random.randint(1, 3)):
                    idx = random.randint(0, 29)
                    coeffs[idx] = random.choice([-1, 0, 1])
            else:
                coeffs = [random.choice([-1, 0, 1]) for _ in range(30)]
        
        S = sum(c * m for c, m in zip(coeffs, multipliers))
        update_best(coeffs, S)
        
        for _ in range(100):
            if time.time() > deadline:
                break
            
            improved = False
            order = list(range(30))
            random.shuffle(order)
            for i in order:
                if time.time() > deadline:
                    break
                old_c = coeffs[i]
                for new_c in [-1, 0, 1]:
                    if new_c == old_c:
                        continue
                    delta = (new_c - old_c) * multipliers[i]
                    new_S = S + delta
                    if new_S == 0:
                        continue
                    if abs(new_S) < abs(S):
                        S = new_S
                        coeffs[i] = new_c
                        improved = True
                        update_best(coeffs, S)
                        break
                if improved:
                    break
            
            if not improved:
                if random.random() < 0.3:
                    for _ in range(random.randint(1, 2)):
                        idx = random.randint(0, 29)
                        coeffs[idx] = random.choice([-1, 0, 1])
                    S = sum(c * m for c, m in zip(coeffs, multipliers))
                    update_best(coeffs, S)
                else:
                    break
        
        if time.time() > deadline:
            break
    
    if best_coeffs is None:
        best_coeffs = [0] * 28 + [1, -1]
    
    return best_coeffs
# EVOLVE-BLOCK-END
