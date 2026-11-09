import random
import numpy as np

def construct():
    n = 128
    target_min, target_max = 128, 256
    
    # Helper to compute representation counts and max rep
    def get_reps(A_set):
        reps = np.zeros(target_max - target_min + 1, dtype=np.int32)
        A_list = sorted(A_set)
        for i in range(len(A_list)):
            a = A_list[i]
            for j in range(i, len(A_list)):
                b = A_list[j]
                s = a + b
                if target_min <= s <= target_max:
                    reps[s - target_min] += 1
        return reps

    def is_valid(A_set):
        reps = get_reps(A_set)
        return np.all(reps > 0)

    def max_rep(A_set):
        reps = get_reps(A_set)
        return np.max(reps)

    # 1. Greedy Construction of a valid set
    random.seed(42)
    A = set()
    # We iterate through targets. If a target is not covered, we add elements to cover it.
    # To minimize size, we prefer adding 1 element if possible.
    for t in range(target_min, target_max + 1):
        # Check if t is already covered
        covered = False
        for a in A:
            b = t - a
            if b >= a and b in A:
                covered = True
                break
        if covered:
            continue
        
        # Not covered. We need to add elements.
        # Option 1: Add one element x such that t-x is in A.
        # This requires t-x in A and x in [0, 128].
        # So x = t - a for some a in A.
        # We want to pick x that doesn't increase max rep too much.
        candidates = []
        for a in A:
            x = t - a
            if 0 <= x <= 128 and x not in A:
                candidates.append(x)
        
        if candidates:
            # Pick a random candidate to avoid bias
            x_add = random.choice(candidates)
            A.add(x_add)
        else:
            # Option 2: Add two elements a, b such that a+b=t.
            # We need a, b in [0, 128].
            # To keep set small and spread out, we can pick a random valid pair.
            # Constraints: 0 <= a <= 128, 0 <= b <= 128, a+b=t.
            # This implies max(0, t-128) <= a <= min(128, t).
            lo = max(0, t - 128)
            hi = min(128, t)
            if lo <= hi:
                a = random.randint(lo, hi)
                b = t - a
                A.add(a)
                A.add(b)
            else:
                # Should not happen for t in [128, 256]
                # But if it does, add 0 and t? No, t can be > 128.
                # If t > 256, impossible. But t <= 256.
                # If t=256, lo=128, hi=128. a=128, b=128.
                pass

    # Initial state is valid.
    current_A = set(A)
    current_max = max_rep(current_A)
    best_A = set(current_A)
    best_max = current_max
    
    # 2. Simulated Annealing
    # Temperature schedule
    T = 5.0
    cooling = 0.995
    min_T = 0.01
    
    # Precompute neighbors? No, just random moves.
    
    while T > min_T:
        # Generate a move
        move_type = random.random()
        
        if move_type < 0.6:
            # Swap: Remove x, Add y
            if not current_A:
                T *= cooling
                continue
            x = random.choice(list(current_A))
            # Pick y from [0, 128] \ current_A
            candidates_y = [y for y in range(129) if y not in current_A]
            if not candidates_y:
                T *= cooling
                continue
            y = random.choice(candidates_y)
            
            new_A = set(current_A)
            new_A.remove(x)
            new_A.add(y)
            
            if not is_valid(new_A):
                # Invalid move, reject
                T *= cooling
                continue
            
            new_max = max_rep(new_A)
            delta = new_max - current_max
            
            # Accept if delta < 0 or with probability exp(-delta/T)
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_A = new_A
                current_max = new_max
                if current_max < best_max:
                    best_max = current_max
                    best_A = set(current_A)
                    
        elif move_type < 0.8:
            # Add element
            candidates_y = [y for y in range(129) if y not in current_A]
            if not candidates_y:
                T *= cooling
                continue
            y = random.choice(candidates_y)
            new_A = set(current_A)
            new_A.add(y)
            
            # Adding an element cannot break validity, but might increase max rep
            new_max = max_rep(new_A)
            delta = new_max - current_max
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_A = new_A
                current_max = new_max
                if current_max < best_max:
                    best_max = current_max
                    best_A = set(current_A)
                    
        else:
            # Remove element
            if not current_A:
                T *= cooling
                continue
            x = random.choice(list(current_A))
            new_A = set(current_A)
            new_A.remove(x)
            
            if not is_valid(new_A):
                T *= cooling
                continue
            
            new_max = max_rep(new_A)
            delta = new_max - current_max
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_A = new_A
                current_max = new_max
                if current_max < best_max:
                    best_max = current_max
                    best_A = set(current_A)
        
        T *= cooling

    return sorted(best_A)
