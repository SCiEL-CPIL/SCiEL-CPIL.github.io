import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s margin
    
    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1  # 129
    
    def calc_reps(A):
        reps = [0] * RANGE_SIZE
        A_list = list(A)
        n = len(A_list)
        for i in range(n):
            ai = A_list[i]
            for j in range(i, n):
                aj = A_list[j]
                s = ai + aj
                if N_LOW <= s <= N_HIGH:
                    reps[s - N_LOW] += 1
        return reps

    def get_peak(reps):
        return max(reps)

    def is_valid(reps):
        return all(r > 0 for r in reps)

    best_A = list(range(129))
    best_peak = 1000  # Initialize high
    
    # We will run multiple restarts
    # Each restart tries to find a better peak
    
    max_restarts = 20
    restart_count = 0
    
    while restart_count < max_restarts and time.time() < deadline:
        # Initialize A
        # Strategy: Start with full set, or random set?
        # Full set is safer for coverage.
        # Random set might be faster if size is right.
        # Let's try random set of size 25-35.
        
        if restart_count == 0:
            A = set(range(129))
        else:
            # Random subset
            size = random.randint(20, 40)
            A = set(random.sample(range(129), size))
            
        reps = calc_reps(A)
        if not is_valid(reps):
            # If random init is invalid, try to fix by adding elements?
            # Or just skip this restart?
            # Better to start with full set if random fails often.
            # But full set is slow to shrink? No, shrink is fast.
            # Let's stick to full set for first few restarts, then random.
            if restart_count < 5:
                A = set(range(129))
                reps = calc_reps(A)
            else:
                continue # Skip this restart if invalid and we are in random phase
        
        current_peak = get_peak(reps)
        
        if current_peak < best_peak:
            best_peak = current_peak
            best_A = list(A)
            
        # Local search
        stuck_count = 0
        while time.time() < deadline:
            improved = False
            
            # 1. Try single removals
            A_list = list(A)
            random.shuffle(A_list)
            
            for x in A_list:
                if time.time() > deadline:
                    break
                
                # Tentatively remove x
                new_reps = reps[:]
                A_minus_x = A - {x}
                
                valid_removal = True
                for a in A_minus_x:
                    s = a + x
                    if N_LOW <= s <= N_HIGH:
                        new_reps[s - N_LOW] -= 1
                        if new_reps[s - N_LOW] < 0:
                            valid_removal = False
                            break
                if not valid_removal:
                    continue
                    
                s = 2 * x
                if N_LOW <= s <= N_HIGH:
                    new_reps[s - N_LOW] -= 1
                    if new_reps[s - N_LOW] < 0:
                        valid_removal = False
                if not valid_removal:
                    continue
                    
                if not is_valid(new_reps):
                    continue
                    
                new_peak = get_peak(new_reps)
                
                if new_peak < current_peak:
                    A = A_minus_x
                    reps = new_reps
                    current_peak = new_peak
                    improved = True
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_A = list(A)
                elif new_peak == current_peak and len(A) > 15:
                    # Accept sideways to reduce size
                    A = A_minus_x
                    reps = new_reps
                    improved = True
                    # Don't update best_peak as it's not better, but size reduction helps
                    pass

            if not improved:
                # 2. Try swaps
                # Limit swaps to avoid long stalls
                for _ in range(50):
                    if time.time() > deadline:
                        break
                    
                    if len(A) == 0:
                        break
                        
                    x = random.choice(list(A))
                    available = [i for i in range(129) if i not in A]
                    if not available:
                        break
                    y = random.choice(available)
                    
                    # Tentatively swap x with y
                    new_reps = reps[:]
                    A_minus_x = A - {x}
                    
                    valid_swap = True
                    for a in A_minus_x:
                        s = a + x
                        if N_LOW <= s <= N_HIGH:
                            new_reps[s - N_LOW] -= 1
                            if new_reps[s - N_LOW] < 0:
                                valid_swap = False
                                break
                    if not valid_swap:
                        continue
                        
                    s = 2 * x
                    if N_LOW <= s <= N_HIGH:
                        new_reps[s - N_LOW] -= 1
                        if new_reps[s - N_LOW] < 0:
                            valid_swap = False
                    if not valid_swap:
                        continue
                        
                    A_with_y = A_minus_x | {y}
                    for a in A_with_y:
                        s = a + y
                        if N_LOW <= s <= N_HIGH:
                            new_reps[s - N_LOW] += 1
                    
                    s = 2 * y
                    if N_LOW <= s <= N_HIGH:
                        new_reps[s - N_LOW] += 1
                    
                    if not is_valid(new_reps):
                        continue
                    
                    new_peak = get_peak(new_reps)
                    
                    if new_peak <= current_peak:
                        A = A_with_y
                        reps = new_reps
                        if new_peak < current_peak:
                            current_peak = new_peak
                            if current_peak < best_peak:
                                best_peak = current_peak
                                best_A = list(A)
                        improved = True
                        break # Break swap loop, restart removal

            if not improved:
                stuck_count += 1
                if stuck_count > 10:
                    break # Restart with new random init
            
        restart_count += 1

    return best_A

