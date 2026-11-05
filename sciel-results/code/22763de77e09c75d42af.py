import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    def calc_score(mask):
        """
        Count distinct d > 0 such that there exists a with a, a+d, a+2d in the set.
        Uses bitmask representation where bit i is set if i is in the set.
        d ranges from 1 to 255 (since max value is 512, 2d <= 511 => d <= 255).
        """
        s = 0
        m = mask
        # We can optimize by checking up to 255
        for d in range(1, 256):
            if m & (m >> d) & (m >> (2 * d)):
                s += 1
        return s

    best_mask = 0
    best_score = -1

    def update_best(mask, score):
        nonlocal best_mask, best_score
        if score > best_score:
            best_score = score
            best_mask = mask

    # Helper to generate initial masks
    def init_contiguous():
        mask = 0
        for i in range(1, 33):
            mask |= (1 << i)
        return mask

    def init_random():
        mask = 0
        elems = random.sample(range(1, 513), 32)
        for i in elems:
            mask |= (1 << i)
        return mask

    def init_ap():
        # Arithmetic progression
        start = random.randint(1, 10)
        diff = random.randint(1, 10)
        mask = 0
        for i in range(32):
            val = start + i * diff
            if val <= 512:
                mask |= (1 << val)
        # If we didn't get 32 distinct (shouldn't happen with diff>=1 and start small enough), fill rest
        # But with start<=10, diff<=10, max val = 10 + 31*10 = 320 < 512, so always 32 distinct
        return mask

    def init_interleaved():
        # Two interleaved APs or similar structure
        mask = 0
        base = random.randint(1, 5)
        step1 = random.randint(2, 5)
        step2 = random.randint(6, 10)
        for i in range(16):
            v1 = base + i * step1
            v2 = base + i * step2
            if v1 <= 512:
                mask |= (1 << v1)
            if v2 <= 512 and not (mask & (1 << v2)):
                mask |= (1 << v2)
        # Ensure 32 elements
        count = bin(mask).count('1')
        if count < 32:
            available = [i for i in range(1, 513) if not (mask & (1 << i))]
            random.shuffle(available)
            for i in available[:32 - count]:
                mask |= (1 << i)
        elif count > 32:
            # Remove extras
            bits = [i for i in range(1, 513) if (mask & (1 << i))]
            random.shuffle(bits)
            for i in bits[count-32:]:
                mask &= ~(1 << i)
        return mask

    def init_geometric():
        mask = 0
        base = random.randint(1, 3)
        ratio = random.randint(2, 3)
        # This might not give 32 elements or stay in range easily, so mix with random
        # Let's just do a mix of geometric-like and random
        mask = init_random()
        return mask

    initializers = [init_contiguous, init_random, init_ap, init_interleaved, init_geometric]

    # Main loop: multiple restarts of Simulated Annealing
    # We will run as many restarts as possible, or one long SA if time permits.
    # Given the landscape, multiple diverse restarts are good.
    # Let's allocate time per restart.
    
    num_restarts_planned = 10
    # But we can also just run SA until time is up, restarting if stuck.
    
    # Let's do a hybrid: 
    # 1. Evaluate all initializers, pick the best one as starting point for a long SA.
    # 2. Or run independent SAs.
    
    # Strategy: Run independent SAs for a fixed fraction of time, then one long SA on the best found.
    
    # Phase 1: Quick exploration with multiple short SAs
    time_per_short_sa = 5.0
    max_short_sas = int((deadline - start_time) // (time_per_short_sa + 1))
    
    for r in range(max_short_sas):
        if time.time() > deadline - 5:  # Leave 5s for final long SA and cleanup
            break
            
        init_func = initializers[r % len(initializers)]
        current_mask = init_func()
        current_score = calc_score(current_mask)
        update_best(current_mask, current_score)
        
        # SA parameters
        T = 1.0
        cool_rate = 0.999
        min_T = 0.01
        
        end_time = time.time() + time_per_short_sa
        
        while time.time() < end_time:
            if T < min_T:
                break
                
            # Generate move: Swap a random element with a random non-element
            x = random.randint(1, 512)
            y = random.randint(1, 512)
            
            if x == y:
                continue
                
            bx = 1 << x
            by = 1 << y
            
            # Check if x is in mask and y is not
            if (current_mask & bx) and not (current_mask & by):
                new_mask = current_mask ^ bx ^ by
                new_score = calc_score(new_mask)
                delta = new_score - current_score
                
                if delta >= 0 or random.random() < math.exp(delta / T):
                    current_mask = new_mask
                    current_score = new_score
                    update_best(current_mask, current_score)
            
            T *= cool_rate

    # Phase 2: Long SA starting from the best mask found so far
    if time.time() < deadline - 2:
        current_mask = best_mask
        current_score = best_score
        
        # Maybe perturb slightly to escape local optimum?
        # Or just continue SA from where it left off (T was likely low).
        # Let's restart T to be higher to allow larger moves.
        T = 0.5
        cool_rate = 0.9999
        min_T = 0.001
        
        while time.time() < deadline - 1:
            if T < min_T:
                break
                
            # Generate move: Swap a random element with a random non-element
            x = random.randint(1, 512)
            y = random.randint(1, 512)
            
            if x == y:
                continue
                
            bx = 1 << x
            by = 1 << y
            
            if (current_mask & bx) and not (current_mask & by):
                new_mask = current_mask ^ bx ^ by
                new_score = calc_score(new_mask)
                delta = new_score - current_score
                
                if delta >= 0 or random.random() < math.exp(delta / T):
                    current_mask = new_mask
                    current_score = new_score
                    update_best(current_mask, current_score)
            
            T *= cool_rate

    # Convert best_mask to list
    result = []
    for i in range(1, 513):
        if best_mask & (1 << i):
            result.append(i)
    
    # Verify length
    if len(result) != 32:
        # Fallback
        return list(range(1, 33))
        
    return result
