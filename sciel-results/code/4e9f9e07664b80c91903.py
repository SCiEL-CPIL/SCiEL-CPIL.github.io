import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    def get_diffs(s):
        diffs = []
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                diffs.append(s[j] - s[i])
        return diffs

    def calc_energy(s, target_diffs):
        """
        Returns a tuple (penalty, max_val)
        penalty > 0 if invalid (duplicates or missing target diffs)
        max_val is the largest element
        """
        diffs = get_diffs(s)
        # Check for duplicates
        if len(set(diffs)) != len(diffs):
            # Count duplicates roughly or just return a high penalty
            # To be more precise for SA, count excess
            d_set = set()
            dup_count = 0
            for d in diffs:
                if d in d_set:
                    dup_count += 1
                else:
                    d_set.add(d)
            
            # Check missing target diffs
            missing_count = 0
            for d in target_diffs:
                if d not in d_set:
                    missing_count += 1
            
            penalty = 10000 * dup_count + 10000 * missing_count
            return penalty, max(s)
        
        # If no duplicates, check missing
        d_set = set(diffs)
        missing_count = 0
        for d in target_diffs:
            if d not in d_set:
                missing_count += 1
        
        penalty = 10000 * missing_count
        return penalty, max(s)

    def valid_solution(s, target_diffs):
        diffs = get_diffs(s)
        if len(set(diffs)) != len(diffs):
            return False
        d_set = set(diffs)
        for d in target_diffs:
            if d not in d_set:
                return False
        return True

    target_diffs = list(range(1, 31))
    best_solution = None
    best_max_val = float('inf')
    
    # Try different sizes k
    # k=9 is minimal to cover 30 diffs (36 total)
    # k=10 gives 45 total
    sizes_to_try = [9, 10, 11]
    
    for k in sizes_to_try:
        if time.time() > deadline - 10:
            break
            
        # Run SA for this size
        # Initial state: Random set of size k, including 0
        # We can start with a greedy-like valid set if we have one, but random is fine with high temp
        # Let's use a simple random initialization
        for restart in range(5): # 5 restarts per size
            if time.time() > deadline - 5:
                break
                
            # Initialize
            s = [0]
            # Generate random distinct positive integers
            # Range up to 200 is a reasonable guess for k=9
            max_init = 200
            while len(s) < k:
                v = random.randint(1, max_init)
                if v not in s:
                    s.append(v)
            s.sort()
            
            curr_penalty, curr_max = calc_energy(s, target_diffs)
            curr_energy = curr_penalty + curr_max
            
            if curr_penalty == 0:
                if curr_max < best_max_val:
                    best_max_val = curr_max
                    best_solution = list(s)
            
            T = 100.0
            T_min = 0.01
            alpha = 0.95
            
            while T > T_min and time.time() < deadline - 5:
                # Pick a random element to move (not the first one, which is 0)
                i = random.randint(1, k - 1)
                # Random step
                step = random.randint(-10, 10)
                if step == 0:
                    continue
                new_val = s[i] + step
                if new_val < 0:
                    continue
                if new_val in s:
                    continue
                
                # Create new state
                new_s = list(s)
                new_s[i] = new_val
                new_s.sort()
                
                new_penalty, new_max = calc_energy(new_s, target_diffs)
                new_energy = new_penalty + new_max
                
                delta = new_energy - curr_energy
                
                if delta < 0 or random.random() < math.exp(-delta / T):
                    s = new_s
                    curr_penalty, curr_max = new_penalty, new_max
                    curr_energy = new_energy
                    
                    if curr_penalty == 0:
                        if curr_max < best_max_val:
                            best_max_val = curr_max
                            best_solution = list(s)
                
                T *= alpha
                # Occasionally restart temperature or do a larger jump
                if random.random() < 0.01:
                    T = max(T, 50.0)

    if best_solution is None:
        # Fallback to a simple greedy if SA fails (should not happen)
        a = [0]
        def diffs(s):
            return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
        for d in range(1, 31):
            if d in diffs(sorted(a)):
                continue
            for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
                trial = sorted(a + [x])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
            else:
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    if len(set(diffs(trial))) == len(diffs(trial)):
                        a = trial
                        break
                    y += 1
        return a

    return best_solution
