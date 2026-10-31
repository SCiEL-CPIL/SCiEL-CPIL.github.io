import time
import random
import math
import sys

def _score_set(U):
    if not U or 0 not in U:
        return 0.0
    U_set = set(U)
    if len(U_set) != len(U):
        return 0.0
    if min(U_set) < 0 or max(U_set) < 0:
        return 0.0
        
    max_u = max(U_set)
    if max_u == 0:
        return 1.0 # log(1)/log(1) is undefined, but limit is 1? 
                   # Actually if U={0}, |U-U|=1, |U+U|=1. ratio 1. log(1)=0.
                   # Denominator log(1)=0. 0/0.
                   # Limit as max->0? 
                   # For U={0,1}, max=1, denom log(3).
                   # Let's handle max_u=0 carefully.
                   # If U={0}, score is 1.0 effectively.
                   
    sums = set()
    diffs = set()
    # Optimization: sort U
    U_list = sorted(U_set)
    for i in range(len(U_list)):
        for j in range(len(U_list)):
            sums.add(U_list[i] + U_list[j])
            diffs.add(U_list[i] - U_list[j])
    
    size_sums = len(sums)
    size_diffs = len(diffs)
    
    if size_sums == 0:
        return 0.0
        
    ratio = size_diffs / size_sums
    if ratio <= 1.0:
        # log(ratio) <= 0, score <= 1. 
        # We want > 1.
        # But we should return the actual score to be safe, though we aim for > seed.
        pass
        
    denom_arg = 2 * max_u + 1
    if denom_arg <= 1:
        return 1.0
        
    try:
        score = 1.0 + math.log(ratio) / math.log(denom_arg)
    except Exception:
        return 0.0
        
    return score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_set = [0, 1, 3]
    best_score = _score_set(best_set)
    
    # If we have very little time, just return seed
    if time.time() > deadline - 5:
        return best_set

    # Search for better sets
    # We will try sizes N from 3 to 12
    # And max_val up to 100
    
    # Helper to generate a random set of size N containing 0
    def gen_random_set(N, max_val):
        if N <= 1:
            return [0]
        # Pick N-1 distinct random numbers from 1..max_val
        while True:
            try:
                candidates = random.sample(range(1, max_val + 1), N - 1)
                s = [0] + candidates
                s.sort()
                return s
            except ValueError:
                # Not enough numbers
                return None

    # Local search improvement
    def improve_set(U, max_val, max_iters=1000):
        U = sorted(set(U))
        current_score = _score_set(U)
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random move
            move_type = random.choice(['add', 'remove', 'shift'])
            
            if move_type == 'add' and len(U) < 15:
                # Add a random number
                new_val = random.randint(1, max_val)
                if new_val in U:
                    continue
                new_U = sorted(U + [new_val])
            elif move_type == 'remove' and len(U) > 2:
                # Remove a random non-zero element
                idx = random.randint(1, len(U) - 1)
                new_U = U[:idx] + U[idx+1:]
            elif move_type == 'shift' and len(U) > 1:
                # Shift a random non-zero element
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                delta = random.choice([-2, -1, 1, 2, 3])
                new_val = old_val + delta
                if new_val < 0:
                    continue
                if new_val in U and new_val != old_val:
                    continue
                # Replace
                new_U = U[:idx] + [new_val] + U[idx+1:]
                new_U = sorted(set(new_U))
            else:
                continue
                
            if not new_U or 0 not in new_U:
                continue
                
            new_score = _score_set(new_U)
            
            if new_score > current_score:
                U = new_U
                current_score = new_score
            else:
                # Simulated annealing style: accept worse with small prob?
                # For now, greedy is safer and faster.
                # Let's add a small probability of accepting worse to escape local minima
                if random.random() < 0.05 and new_score > 0.9:
                     U = new_U
                     current_score = new_score

        return U, current_score

    # Run multiple restarts with different sizes and max_vals
    restarts = 0
    max_restarts = 200
    
    while time.time() < deadline and restarts < max_restarts:
        N = random.randint(3, 10)
        max_val = random.choice([20, 50, 100, 200])
        
        # Ensure we can fit N elements in max_val
        if N > max_val + 1:
            max_val = N + 1
            
        init_set = gen_random_set(N, max_val)
        if not init_set:
            continue
            
        improved_set, score = improve_set(init_set, max_val, max_iters=200)
        
        if score > best_score:
            best_score = score
            best_set = improved_set
            
        restarts += 1
        
        # Periodically check time
        if restarts % 10 == 0:
             if time.time() > deadline:
                 break

    return best_set

# EVOLVE-BLOCK-START
def construct():
    return _construct_impl()

def _construct_impl():
    import time
    import random
    import math

    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin

    def _score_set(U):
        if not U or 0 not in U:
            return 0.0
        U_set = set(U)
        if len(U_set) != len(U):
            return 0.0
        if min(U_set) < 0:
            return 0.0
            
        max_u = max(U_set)
        if max_u == 0:
            return 1.0
            
        sums = set()
        diffs = set()
        U_list = sorted(U_set)
        for i in range(len(U_list)):
            for j in range(len(U_list)):
                sums.add(U_list[i] + U_list[j])
                diffs.add(U_list[i] - U_list[j])
        
        size_sums = len(sums)
        size_diffs = len(diffs)
        
        if size_sums == 0:
            return 0.0
            
        ratio = size_diffs / size_sums
        denom_arg = 2 * max_u + 1
        if denom_arg <= 1:
            return 1.0
            
        try:
            score = 1.0 + math.log(ratio) / math.log(denom_arg)
        except Exception:
            return 0.0
            
        return score

    best_set = [0, 1, 3]
    best_score = _score_set(best_set)
    
    if time.time() > deadline - 5:
        return best_set

    def gen_random_set(N, max_val):
        if N <= 1:
            return [0]
        if N > max_val + 1:
            return None
        while True:
            try:
                candidates = random.sample(range(1, max_val + 1), N - 1)
                s = [0] + candidates
                s.sort()
                return s
            except ValueError:
                return None

    def improve_set(U, max_val, max_iters=1000):
        U = sorted(set(U))
        current_score = _score_set(U)
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            move_type = random.choice(['add', 'remove', 'shift'])
            
            if move_type == 'add' and len(U) < 15:
                new_val = random.randint(1, max_val)
                if new_val in U:
                    continue
                new_U = sorted(U + [new_val])
            elif move_type == 'remove' and len(U) > 2:
                idx = random.randint(1, len(U) - 1)
                new_U = U[:idx] + U[idx+1:]
            elif move_type == 'shift' and len(U) > 1:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                delta = random.choice([-2, -1, 1, 2, 3])
                new_val = old_val + delta
                if new_val < 0:
                    continue
                if new_val in U and new_val != old_val:
                    continue
                new_U = U[:idx] + [new_val] + U[idx+1:]
                new_U = sorted(set(new_U))
            else:
                continue
                
            if not new_U or 0 not in new_U:
                continue
                
            new_score = _score_set(new_U)
            
            if new_score > current_score:
                U = new_U
                current_score = new_score
            else:
                if random.random() < 0.05 and new_score > 0.9:
                     U = new_U
                     current_score = new_score

        return U, current_score

    restarts = 0
    max_restarts = 200
    
    while time.time() < deadline and restarts < max_restarts:
        N = random.randint(3, 10)
        max_val = random.choice([20, 50, 100, 200])
        
        if N > max_val + 1:
            max_val = N + 1
            
        init_set = gen_random_set(N, max_val)
        if not init_set:
            continue
            
        improved_set, score = improve_set(init_set, max_val, max_iters=200)
        
        if score > best_score:
            best_score = score
            best_set = improved_set
            
        restarts += 1
        
        if restarts % 10 == 0:
             if time.time() > deadline:
                 break

    return best_set
# EVOLVE-BLOCK-END
