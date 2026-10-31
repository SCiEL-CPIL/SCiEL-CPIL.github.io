import time
import random
import math

def calculate_score(U):
    if not U or 0 not in U:
        return 0.0
    U_set = set(U)
    sum_set = set()
    diff_set = set()
    for a in U_set:
        for b in U_set:
            sum_set.add(a + b)
            diff_set.add(a - b)
    if not sum_set or not diff_set:
        return 0.0
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 1.0:
        return 1.0 # log(1)=0, score=1
    max_u = max(U_set)
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
    return 1.0 + math.log(ratio) / denom

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Initial seeds
    candidates = [
        [0, 1, 3],
        [0, 1, 3, 7],
        [0, 1, 3, 7, 13],
        [0, 1, 3, 7, 15],
        [0, 1, 4, 10, 12],
        list(range(10)) # interval
    ]
    
    best_U = candidates[0]
    best_score = calculate_score(best_U)
    
    # Hill Climbing
    # We will explore adding, removing, swapping elements
    # Limit size to 20, max value to 200 to keep search space manageable
    MAX_SIZE = 20
    MAX_VAL = 200
    
    while time.time() < deadline:
        current_U = best_U[:]
        improved = False
        current_score = best_score
        
        # Try adding an element
        if len(current_U) < MAX_SIZE:
            for v in range(1, MAX_VAL + 1):
                if v not in current_U:
                    new_U = current_U + [v]
                    new_U.sort()
                    s = calculate_score(new_U)
                    if s > current_score:
                        current_U = new_U
                        current_score = s
                        improved = True
                        break # restart loop with new best
        
        # Try removing an element
        for i, v in enumerate(current_U):
            if v == 0:
                continue
            new_U = current_U[:i] + current_U[i+1:]
            s = calculate_score(new_U)
            if s > current_score:
                current_U = new_U
                current_score = s
                improved = True
                break
        
        # Try swapping an element
        for i, v in enumerate(current_U):
            if v == 0:
                continue
            for new_v in range(1, MAX_VAL + 1):
                if new_v in current_U:
                    continue
                new_U = current_U[:]
                new_U[i] = new_v
                new_U.sort()
                s = calculate_score(new_U)
                if s > current_score:
                    current_U = new_U
                    current_score = s
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            best_U = current_U
            best_score = current_score
        else:
            # If no improvement, try a random restart or perturbation
            # Simple random walk to escape local minima
            if random.random() < 0.1:
                # Perturb best_U
                perturbed = best_U[:]
                if random.random() < 0.5 and len(perturbed) < MAX_SIZE:
                    v = random.randint(1, MAX_VAL)
                    if v not in perturbed:
                        perturbed.append(v)
                elif random.random() < 0.5 and len(perturbed) > 1:
                    idx = random.randint(1, len(perturbed)-1)
                    perturbed.pop(idx)
                else:
                    idx = random.randint(1, len(perturbed)-1)
                    v = random.randint(1, MAX_VAL)
                    if v not in perturbed:
                        perturbed[idx] = v
                        perturbed.sort()
                
                s = calculate_score(perturbed)
                if s > best_score:
                    best_U = perturbed
                    best_score = s
                else:
                    # Accept with some probability (Simulated Annealing lite)
                    if random.random() < 0.1:
                        best_U = perturbed
                        best_score = s
            else:
                # If stuck, break to try other seeds or just stop
                # For simplicity, we just continue checking time
                pass

    return best_U

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initial seeds
    candidates = [
        [0, 1, 3],
        [0, 1, 3, 7],
        [0, 1, 3, 7, 13],
        [0, 1, 3, 7, 15],
        [0, 1, 4, 10, 12],
        [0, 2, 5, 11, 14],
        [0, 1, 4, 10, 18, 26]
    ]
    
    best_U = candidates[0]
    best_score = calculate_score(best_U)
    
    MAX_SIZE = 25
    MAX_VAL = 300
    
    while time.time() < deadline:
        current_U = best_U[:]
        current_score = best_score
        improved = False
        
        # 1. Try adding
        if len(current_U) < MAX_SIZE:
            # Sample a few values to add to speed up
            for v in random.sample(range(1, MAX_VAL + 1), min(50, MAX_VAL)):
                if v not in current_U:
                    new_U = current_U + [v]
                    new_U.sort()
                    s = calculate_score(new_U)
                    if s > current_score:
                        current_U = new_U
                        current_score = s
                        improved = True
                        break
        
        # 2. Try removing
        for i in range(len(current_U) - 1, 0, -1):
            new_U = current_U[:i] + current_U[i+1:]
            s = calculate_score(new_U)
            if s > current_score:
                current_U = new_U
                current_score = s
                improved = True
                break
        
        # 3. Try swapping
        for i in range(len(current_U) - 1, 0, -1):
            v = current_U[i]
            # Try a few random replacements
            for new_v in random.sample(range(1, MAX_VAL + 1), min(20, MAX_VAL)):
                if new_v in current_U:
                    continue
                new_U = current_U[:]
                new_U[i] = new_v
                new_U.sort()
                s = calculate_score(new_U)
                if s > current_score:
                    current_U = new_U
                    current_score = s
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            best_U = current_U
            best_score = current_score
        else:
            # Stuck. Try a random restart from a new seed or perturbation
            if random.random() < 0.2:
                # Perturb
                perturbed = best_U[:]
                op = random.choice(['add', 'remove', 'swap'])
                if op == 'add' and len(perturbed) < MAX_SIZE:
                    v = random.randint(1, MAX_VAL)
                    if v not in perturbed:
                        perturbed.append(v)
                elif op == 'remove' and len(perturbed) > 1:
                    idx = random.randint(1, len(perturbed)-1)
                    perturbed.pop(idx)
                elif op == 'swap' and len(perturbed) > 1:
                    idx = random.randint(1, len(perturbed)-1)
                    v = random.randint(1, MAX_VAL)
                    if v not in perturbed:
                        perturbed[idx] = v
                        perturbed.sort()
                
                s = calculate_score(perturbed)
                if s > best_score:
                    best_U = perturbed
                    best_score = s
                elif random.random() < 0.05:
                    best_U = perturbed
                    best_score = s
            else:
                # If we've been stuck for a while, we might just break to save time
                # But we don't track stuck duration easily. Just continue.
                pass

    return best_U
# EVOLVE-BLOCK-END
