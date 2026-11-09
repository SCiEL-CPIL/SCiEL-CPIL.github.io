import time
import random
import sys
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    N = 32
    MAX_VAL = 512
    MAX_D = 255  # max possible d: 1 + 2*255 = 511 <= 512
    
    def time_ok():
        return time.time() < deadline

    def score_mask(mask):
        """Count distinct d > 0 such that a, a+d, a+2d all in mask."""
        count = 0
        for d in range(1, MAX_D + 1):
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                count += 1
        return count

    def list_to_mask(lst):
        m = 0
        for x in lst:
            m |= (1 << (x - 1))
        return m

    def mask_to_list(mask):
        return [i + 1 for i in range(MAX_VAL) if (mask >> i) & 1]

    # Fast delta: when replacing old_val with new_val
    def delta_replace(mask, old_val, new_val):
        """Return (delta, new_mask) when replacing old_val with new_val."""
        new_mask = mask & ~(1 << (old_val - 1)) | (1 << (new_val - 1))
        
        # Find d's that could be affected
        affected_ds = set()
        # d's involving old_val
        for d in range(1, MAX_D + 1):
            if old_val + 2 * d <= MAX_VAL:
                affected_ds.add(d)
            if old_val - d >= 1 and old_val + d <= MAX_VAL:
                affected_ds.add(d)
            if old_val - 2 * d >= 1:
                affected_ds.add(d)
        
        # d's involving new_val
        for d in range(1, MAX_D + 1):
            if new_val + 2 * d <= MAX_VAL:
                affected_ds.add(d)
            if new_val - d >= 1 and new_val + d <= MAX_VAL:
                affected_ds.add(d)
            if new_val - 2 * d >= 1:
                affected_ds.add(d)
        
        delta = 0
        for d in affected_ds:
            old_has = (mask & (mask >> d) & (mask >> (2 * d))) != 0
            new_has = (new_mask & (new_mask >> d) & (new_mask >> (2 * d))) != 0
            if old_has != new_has:
                delta += 1 if new_has else -1
        
        return delta, new_mask

    # Greedy construction for initial state
    def greedy_construct():
        mask = 0
        s_set = set()
        
        for _ in range(N):
            best_new = None
            best_s = -1
            # Sample candidates to speed up
            candidates = random.sample(range(1, MAX_VAL + 1), 150)
            for v in candidates:
                if v in s_set:
                    continue
                new_mask = mask | (1 << (v - 1))
                s = score_mask(new_mask)
                if s > best_s:
                    best_s = s
                    best_new = v
            
            if best_new is None:
                best_new = random.choice([v for v in range(1, MAX_VAL + 1) if v not in s_set])
            
            mask |= (1 << (best_new - 1))
            s_set.add(best_new)
        
        return mask, score_mask(mask), mask_to_list(mask)

    # Targeted move generation: find values that complete 3-APs
    def get_targeted_moves(mask, s_set, current_list):
        """Generate candidate new values that would complete 3-APs with existing elements."""
        candidates = set()
        lst_set = set(current_list)
        
        for i in range(len(current_list)):
            for j in range(len(current_list)):
                if i == j:
                    continue
                a = current_list[i]
                b = current_list[j]
                # If a and b are endpoints of a 3-AP, middle = (a+b)/2
                if (a + b) % 2 == 0:
                    mid = (a + b) // 2
                    if 1 <= mid <= MAX_VAL and mid not in lst_set:
                        candidates.add(mid)
                # If a and b are first two terms, third = 2*b - a
                third = 2 * b - a
                if 1 <= third <= MAX_VAL and third not in lst_set:
                    candidates.add(third)
                # If a and b are last two terms, first = 2*a - b
                first = 2 * a - b
                if 1 <= first <= MAX_VAL and first not in lst_set:
                    candidates.add(first)
        
        return list(candidates)

    # Simulated Annealing with targeted moves
    def simulated_annealing(init_mask, init_score, init_list):
        current_mask = init_mask
        current_score = init_score
        current_list = init_list[:]
        current_set = set(current_list)
        
        best_mask = current_mask
        best_score = current_score
        best_list = current_list[:]
        
        # SA Parameters
        T = 10.0  # Initial temperature
        cooling_rate = 0.9995
        min_T = 0.01
        
        iteration = 0
        while time_ok() and T > min_T:
            iteration += 1
            
            # Mix of random and targeted moves
            use_targeted = (iteration % 3 == 0) and len(current_set) > 0
            
            if use_targeted:
                targeted = get_targeted_moves(current_mask, current_set, current_list)
                if targeted:
                    new_val = random.choice(targeted)
                else:
                    new_val = random.randint(1, MAX_VAL)
            else:
                new_val = random.randint(1, MAX_VAL)
            
            # Pick a random element to replace
            idx = random.randint(0, N - 1)
            old_val = current_list[idx]
            
            if new_val == old_val or new_val in current_set:
                continue
            
            delta, new_mask = delta_replace(current_mask, old_val, new_val)
            new_score = current_score + delta
            
            # Acceptance criterion
            if delta > 0 or random.random() < math.exp(delta / T):
                # Accept move
                current_mask = new_mask
                current_score = new_score
                current_set.discard(old_val)
                current_set.add(new_val)
                current_list[idx] = new_val
                
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
                    best_list = current_list[:]
            
            T *= cooling_rate
            
            # Check time more frequently
            if not time_ok():
                break
        
        return best_mask, best_score, best_list

    # Final polish pass: local search with targeted moves only
    def polish_pass(mask, score, lst):
        s_set = set(lst)
        current_mask = mask
        current_score = score
        current_list = lst[:]
        current_set = s_set
        
        best_mask = current_mask
        best_score = current_score
        best_list = current_list[:]
        
        # Multiple passes of targeted local search
        for _ in range(5):
            if not time_ok():
                break
            
            improved = False
            for _ in range(N):
                if not time_ok():
                    break
                
                targeted = get_targeted_moves(current_mask, current_set, current_list)
                if not targeted:
                    break
                
                idx = random.randint(0, N - 1)
                old_val = current_list[idx]
                
                best_delta = 0
                best_new = old_val
                best_new_mask = current_mask
                
                for new_val in targeted:
                    if new_val == old_val or new_val in current_set:
                        continue
                    delta, new_mask = delta_replace(current_mask, old_val, new_val)
                    if delta > best_delta:
                        best_delta = delta
                        best_new = new_val
                        best_new_mask = new_mask
                
                if best_new != old_val:
                    current_set.discard(old_val)
                    current_set.add(best_new)
                    current_list[idx] = best_new
                    current_mask = best_new_mask
                    current_score += best_delta
                    improved = True
            
            if not improved:
                break
        
        if current_score > best_score:
            best_score = current_score
            best_mask = current_mask
            best_list = current_list[:]
        
        return best_mask, best_score, best_list

    random.seed(42)
    
    best_mask = 0
    best_score = 0
    best_list = []
    
    # Run multiple restarts of SA to explore different basins
    num_restarts = 12
    for i in range(num_restarts):
        if not time_ok():
            break
            
        # Use different initializations
        if i == 0:
            # Greedy start
            mask, score, lst = greedy_construct()
        elif i == 1:
            # Interval start
            lst = list(range(1, 33))
            mask = list_to_mask(lst)
            score = score_mask(mask)
        elif i == 2:
            # Arithmetic progression start
            lst = [1 + i * 16 for i in range(N)]
            mask = list_to_mask(lst)
            score = score_mask(mask)
        else:
            # Random start
            lst = random.sample(range(1, MAX_VAL + 1), N)
            mask = list_to_mask(lst)
            score = score_mask(mask)
        
        # Run SA with targeted moves
        m, s, l = simulated_annealing(mask, score, lst)
        
        # Apply polish pass
        m, s, l = polish_pass(m, s, l)
        
        if s > best_score:
            best_score = s
            best_mask = m
            best_list = l[:]
            
        print(f"SA Restart {i}: {s}", file=sys.stderr)
    
    print(f"Final Score: {best_score}", file=sys.stderr)
    return best_list
