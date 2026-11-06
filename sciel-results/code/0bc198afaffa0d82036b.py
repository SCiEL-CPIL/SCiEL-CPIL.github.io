import random
import time
import sys

def construct():
    N = 128
    TARGETS = list(range(128, 257))
    n_targets = len(TARGETS)
    
    # State: A is a set of integers
    # mult[S] is the number of pairs {a,b} in A with a<=b such that a+b=S
    
    def calc_mult(A_set):
        mult = [0] * 257
        A_list = sorted(A_set)
        for i, a in enumerate(A_list):
            # a+a
            mult[2*a] += 1
            # a+b for b in A_list[i+1:]
            for b in A_list[i+1:]:
                mult[a+b] += 1
        return mult

    def evaluate(A_set):
        mult = calc_mult(A_set)
        uncovered = 0
        peak = 0
        sum_sq = 0
        for S in TARGETS:
            m = mult[S]
            if m == 0:
                uncovered += 1
            else:
                if m > peak:
                    peak = m
                sum_sq += m * m
        # Objective: minimize uncovered, then peak, then sum_sq
        return uncovered, peak, sum_sq

    # Initial A: Random subset or structured
    # Let's try a random subset of size 25
    # Or start with the seed and shrink it?
    # Seed is range(64, 129), size 65.
    # Let's start with a random set of size 30.
    
    best_A = None
    best_score = (float('inf'), float('inf'), float('inf'))
    
    # Try several random restarts
    start_time = time.time()
    max_time = 50.0 # Safety margin
    
    # Heuristic initialization:
    # We need sums to cover 128..256.
    # If A is roughly uniform in [0, 128], sums are distributed.
    
    for restart in range(20):
        # Random size between 20 and 40
        size = random.randint(20, 40)
        A = set(random.sample(range(N+1), size))
        
        # Local search
        A_list = list(A)
        # Sort for deterministic processing in some steps? Not strictly necessary for set ops.
        
        cur_uncovered, cur_peak, cur_sum_sq = evaluate(A)
        cur_score = (cur_uncovered, cur_peak, cur_sum_sq)
        
        if cur_score < best_score:
            best_score = cur_score
            best_A = set(A)
            
        # Simulated Annealing or Hill Climbing
        T = 1.0
        step = 0
        max_steps = 2000
        no_improve = 0
        max_no_improve = 100
        
        while step < max_steps and (time.time() - start_time) < max_time:
            # Choose a move
            # 1. Add a random element
            # 2. Remove a random element
            # 3. Swap a random element with another
            
            move_type = random.choice(['add', 'remove', 'swap'])
            
            new_A = set(A)
            
            if move_type == 'add':
                x = random.randint(0, N)
                if x not in new_A:
                    new_A.add(x)
                else:
                    continue
            elif move_type == 'remove':
                if len(new_A) <= 10: # Don't go too small
                    continue
                x = random.choice(list(new_A))
                new_A.remove(x)
            elif move_type == 'swap':
                if len(new_A) < 2:
                    continue
                x = random.choice(list(new_A))
                y = random.randint(0, N)
                if y == x or y in new_A:
                    # Try another y or skip
                    if random.random() < 0.5:
                        continue
                    else:
                        # Force swap even if y in A? No, y must be outside for a swap to change the set
                        # If y in A, it's a no-op or we need to pick a different y
                        # Let's just pick a y not in A
                        while y in new_A or y == x:
                            y = random.randint(0, N)
                new_A.remove(x)
                new_A.add(y)
            
            new_uncovered, new_peak, new_sum_sq = evaluate(new_A)
            new_score = (new_uncovered, new_peak, new_sum_sq)
            
            # Acceptance criterion
            # If better, accept.
            # If worse, accept with prob exp(-delta/T)
            
            # Delta calculation:
            # We want to minimize.
            # Score components: U, P, Q
            # Let's define a scalar cost for SA:
            # Cost = 1000*U + P + 0.01*Q
            old_cost = 1000*cur_uncovered + cur_peak + 0.01*cur_sum_sq
            new_cost = 1000*new_uncovered + new_peak + 0.01*new_sum_sq
            delta = new_cost - old_cost
            
            if delta < 0 or random.random() < pow(2.71828, -delta/T):
                A = new_A
                cur_uncovered, cur_peak, cur_sum_sq = new_uncovered, new_peak, new_sum_sq
                cur_score = (cur_uncovered, cur_peak, cur_sum_sq)
                
                if cur_score < best_score:
                    best_score = cur_score
                    best_A = set(A)
                    no_improve = 0
                else:
                    no_improve += 1
                    
                T *= 0.995
            else:
                no_improve += 1
            
            if no_improve > max_no_improve:
                break
            step += 1
            
        if (time.time() - start_time) > max_time:
            break
            
    # Final polish: try to remove elements if coverage is maintained and peak doesn't increase
    if best_A:
        A = set(best_A)
        improved = True
        while improved and (time.time() - start_time) < max_time:
            improved = False
            cur_uncovered, cur_peak, cur_sum_sq = evaluate(A)
            for x in list(A):
                if len(A) <= 15: break
                A.remove(x)
                new_uncovered, new_peak, new_sum_sq = evaluate(A)
                # Accept if coverage is same and peak is same or lower, and sum_sq is lower
                if new_uncovered == cur_uncovered and new_peak <= cur_peak and new_sum_sq < cur_sum_sq:
                    cur_uncovered, cur_peak, cur_sum_sq = new_uncovered, new_peak, new_sum_sq
                    improved = True
                else:
                    A.add(x)
            
            if (cur_uncovered, cur_peak, cur_sum_sq) < best_score:
                best_score = (cur_uncovered, cur_peak, cur_sum_sq)
                best_A = set(A)

    return list(best_A)
