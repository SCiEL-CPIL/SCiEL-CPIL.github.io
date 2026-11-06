import random
import time
import math

def construct():
    start_time = time.time()
    # Safety margin: 145 seconds
    deadline = start_time + 145.0 

    # Precompute shift masks for each d: for a given d, we need to check
    # if there exists a such that bits a, a+d, a+2d are all set.
    # Precompute for each d a mask that has bit i set iff i, i+d, i+2d are valid positions.
    # Actually, let's precompute the "valid start" mask for each d.
    # For difference d, valid starts a satisfy 1 <= a and a+2d <= 512, so a <= 512-2d.
    # The check is: (mask & (mask >> d) & (mask >> 2*d)) & valid_starts_mask[d] != 0
    # But we can just do the original check since the shifts naturally handle it.
    # Let me think of a faster approach:
    # For each d, define M_d = mask & (mask >> d) & (mask >> 2*d)
    # We need M_d != 0.
    # This is already O(255) big-int ops per score call.
    # Alternative: precompute for each d the list of (a, a+d, a+2d) triples and use
    # a different representation. But bitmask is already quite fast.
    # Let's try: for each d, precompute a "progression mask" P_d where bit a is set
    # iff a+2d <= 512. Then check: (mask & (mask >> d) & (mask >> 2*d) & P_d) != 0
    # This doesn't help much since the shifts already limit valid a.
    
    # Actually, a key optimization: we can precompute for each d the mask of valid
    # starting positions, and use a single AND chain. Let's just keep the current
    # approach but make it slightly faster by using local variables.
    
    def calc_score(mask):
        """Count distinct d > 0 where a, a+d, a+2d all in set (bitmask)."""
        s = 0
        m = mask
        for d in range(1, 256):
            if m & (m >> d) & (m >> (2 * d)):
                s += 1
        return s

    def get_elements(mask):
        """Convert bitmask to list of elements."""
        elems = []
        m = mask
        i = 1
        while m:
            if m & 1:
                elems.append(i)
            m >>= 1
            i += 1
        return elems

    def greedy_coverage_init():
        """Greedy set-cover style: build a 32-element set maximizing covered differences.
        Universe: differences d in 1..255.
        Each element x 'covers' a difference d if there exists y in current set such that
        {x, y, z} forms a 3-AP with difference d, where z is also determined.
        Actually, a 3-AP {a, a+d, a+2d} covers difference d.
        Strategy: start with a seed, then greedily add elements that create new 3-APs.
        """
        # Start with a small structured seed
        seed = [1, 2, 3]  # covers d=1
        mask = 0
        for s in seed:
            mask |= (1 << s)
        
        used = set(seed)
        # Precompute: for each candidate element x and each d, what other elements are needed?
        # A 3-AP with difference d containing x could be:
        #   (x-2d, x-d, x), (x-d, x, x+d), (x, x+d, x+2d)
        # So adding x covers d if at least one of these triples is fully in the set after adding x.
        
        for _ in range(29):
            # Score each candidate: how many new d's would be covered?
            best_elem = -1
            best_gain = -1
            best_new_d_count = -1
            
            for x in range(1, 513):
                if x in used:
                    continue
                bx = 1 << x
                new_mask = mask | bx
                # Check which d's are newly covered
                gain = 0
                for d in range(1, 256):
                    # Was d covered before?
                    old_covered = (mask & (mask >> d) & (mask >> (2 * d))) != 0
                    new_covered = (new_mask & (new_mask >> d) & (new_mask >> (2 * d))) != 0
                    if new_covered and not old_covered:
                        gain += 1
                if gain > best_gain or (gain == best_gain and x < best_elem and best_elem != -1):
                    best_gain = gain
                    best_elem = x
                    best_new_d_count = gain
            
            if best_elem == -1 or best_gain == 0:
                # No improvement possible with greedy; pick random
                candidates = [x for x in range(1, 513) if x not in used]
                if candidates:
                    best_elem = random.choice(candidates)
                    best_gain = 0
            
            if best_elem != -1:
                mask |= (1 << best_elem)
                used.add(best_elem)
        
        return mask

    def count_bits(mask):
        """Count set bits."""
        c = 0
        m = mask
        while m:
            c += 1
            m &= m - 1
        return c

    def random_mask():
        """Generate a random 32-element bitmask."""
        elems = random.sample(range(1, 513), 32)
        mask = 0
        for e in elems:
            mask |= (1 << e)
        return mask

    def structured_mask(type_id):
        """Generate a structured initial mask."""
        mask = 0
        if type_id == 0:
            # Contiguous
            for i in range(1, 33):
                mask |= (1 << i)
        elif type_id == 1:
            # Sparse AP
            for i in range(32):
                val = 1 + i * 15
                if val <= 512:
                    mask |= (1 << val)
        elif type_id == 2:
            # Two interleaved APs
            for i in range(16):
                mask |= (1 << (1 + i * 32))
                mask |= (1 << (2 + i * 32))
        elif type_id == 3:
            # Random but sorted
            elems = sorted(random.sample(range(1, 513), 32))
            for e in elems:
                mask |= (1 << e)
        elif type_id == 4:
            # Blocky
            # 4 blocks of 8
            for b in range(4):
                start = 1 + b * 128
                for i in range(8):
                    mask |= (1 << (start + i))
        elif type_id == 5:
            # Geometric-ish
            vals = set()
            base = 1
            while len(vals) < 32:
                vals.add(base)
                base = int(base * 1.1) + 1
                if base > 512:
                    break
            if len(vals) < 32:
                remaining = 32 - len(vals)
                pool = [x for x in range(1, 513) if x not in vals]
                vals.update(random.sample(pool, remaining))
            for v in vals:
                if v <= 512:
                    mask |= (1 << v)
        elif type_id == 6:
            # Greedy coverage initialization
            mask = greedy_coverage_init()
            return mask
        elif type_id == 7:
            # Greedy coverage with different seed
            seed = random.sample(range(1, 513), 3)
            mask = 0
            for s in seed:
                mask |= (1 << s)
            used = set(seed)
            for _ in range(29):
                best_elem = -1
                best_gain = -1
                for x in range(1, 513):
                    if x in used:
                        continue
                    bx = 1 << x
                    new_mask = mask | bx
                    gain = 0
                    for d in range(1, 256):
                        old_covered = (mask & (mask >> d) & (mask >> (2 * d))) != 0
                        new_covered = (new_mask & (new_mask >> d) & (new_mask >> (2 * d))) != 0
                        if new_covered and not old_covered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_elem = x
                if best_elem == -1 or best_gain == 0:
                    candidates = [x for x in range(1, 513) if x not in used]
                    if candidates:
                        best_elem = random.choice(candidates)
                if best_elem != -1:
                    mask |= (1 << best_elem)
                    used.add(best_elem)
            return mask
        return mask

    best_mask = 0
    best_score = -1

    # Generate diverse initializations
    inits = []
    inits.append(structured_mask(0))
    inits.append(structured_mask(1))
    inits.append(structured_mask(2))
    inits.append(structured_mask(3))
    inits.append(structured_mask(4))
    inits.append(structured_mask(5))
    inits.append(structured_mask(6))  # Greedy coverage
    inits.append(structured_mask(7))  # Greedy coverage with random seed
    # Add a few random ones
    for _ in range(3):
        inits.append(random_mask())

    # Simulated Annealing parameters
    # We will run SA for a fixed number of iterations or time per restart
    # Total time budget is ~145s. 10 restarts => ~14s each.
    num_restarts = len(inits)
    time_per_restart = (deadline - start_time) / max(num_restarts, 1)
    
    # To be safe, let's use a fixed number of iterations per restart and check time
    # Or better, run until time limit for each restart.
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break

        current_mask = inits[r]
        current_score = calc_score(current_mask)

        if current_score > best_score:
            best_score = current_score
            best_mask = current_mask

        # SA Parameters
        T = 1.0
        cool_rate = 0.9998  # Slow cooling
        min_T = 0.001
        restart_deadline = start_time + (r + 1) * time_per_restart
        
        # If a restart takes too long, we skip to next
        # But we want to use all time. So we just run until global deadline or local quota.
        
        no_improve = 0
        max_no_improve = 5000
        
        # Maintain lists of set and unset elements for efficient move generation
        set_elems = get_elements(current_mask)
        unset_elems = [x for x in range(1, 513) if x not in set(set_elems)]
        
        while time.time() < restart_deadline - 0.5:
            if T < min_T:
                break

            # Choose move type: 1-swap (70%) or 2-swap (30%)
            if random.random() < 0.7:
                # 1-swap: remove one set element, add one unset element
                if not set_elems or not unset_elems:
                    break
                idx_x = random.randint(0, len(set_elems) - 1)
                idx_y = random.randint(0, len(unset_elems) - 1)
                x = set_elems[idx_x]
                y = unset_elems[idx_y]
                bx = 1 << x
                by = 1 << y
                new_mask = current_mask ^ bx ^ by
                new_score = calc_score(new_mask)
                delta = new_score - current_score

                if delta >= 0 or (T > 0.01 and random.random() < math.exp(delta / T if delta / T > -50 else -50)):
                    current_mask = new_mask
                    current_score = new_score
                    no_improve = 0
                    # Update lists
                    set_elems[idx_x] = y
                    unset_elems[idx_y] = x
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
                else:
                    no_improve += 1
            else:
                # 2-swap: remove 2 set elements, add 2 unset elements
                if len(set_elems) < 2 or len(unset_elems) < 2:
                    break
                idx_x1 = random.randint(0, len(set_elems) - 1)
                idx_x2 = random.randint(0, len(set_elems) - 1)
                if idx_x1 == idx_x2:
                    idx_x2 = (idx_x2 + 1) % len(set_elems)
                idx_y1 = random.randint(0, len(unset_elems) - 1)
                idx_y2 = random.randint(0, len(unset_elems) - 1)
                if idx_y1 == idx_y2:
                    idx_y2 = (idx_y2 + 1) % len(unset_elems)
                x1, x2 = set_elems[idx_x1], set_elems[idx_x2]
                y1, y2 = unset_elems[idx_y1], unset_elems[idx_y2]
                b1, b2, b3, b4 = 1 << x1, 1 << x2, 1 << y1, 1 << y2
                new_mask = current_mask ^ b1 ^ b2 ^ b3 ^ b4
                new_score = calc_score(new_mask)
                delta = new_score - current_score

                if delta >= 0 or (T > 0.01 and random.random() < math.exp(delta / T if delta / T > -50 else -50)):
                    current_mask = new_mask
                    current_score = new_score
                    no_improve = 0
                    # Update lists
                    set_elems[idx_x1] = y1
                    set_elems[idx_x2] = y2
                    unset_elems[idx_y1] = x1
                    unset_elems[idx_y2] = x2
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
                else:
                    no_improve += 1

            T *= cool_rate

            if no_improve > max_no_improve:
                # Kick: random 2-swap perturbation and reheat
                if len(set_elems) >= 2 and len(unset_elems) >= 2:
                    idx_x1 = random.randint(0, len(set_elems) - 1)
                    idx_x2 = random.randint(0, len(set_elems) - 1)
                    if idx_x1 == idx_x2:
                        idx_x2 = (idx_x2 + 1) % len(set_elems)
                    idx_y1 = random.randint(0, len(unset_elems) - 1)
                    idx_y2 = random.randint(0, len(unset_elems) - 1)
                    if idx_y1 == idx_y2:
                        idx_y2 = (idx_y2 + 1) % len(unset_elems)
                    x1, x2 = set_elems[idx_x1], set_elems[idx_x2]
                    y1, y2 = unset_elems[idx_y1], unset_elems[idx_y2]
                    b1, b2, b3, b4 = 1 << x1, 1 << x2, 1 << y1, 1 << y2
                    current_mask ^= b1 ^ b2 ^ b3 ^ b4
                    current_score = calc_score(current_mask)
                    set_elems[idx_x1] = y1
                    set_elems[idx_x2] = y2
                    unset_elems[idx_y1] = x1
                    unset_elems[idx_y2] = x2
                    T = 0.5
                    no_improve = 0

    # Final Polish: Hill Climbing on the best solution found
    if time.time() < deadline - 2 and best_mask != 0:
        current_mask = best_mask
        current_score = best_score
        T = 0.01
        cool_rate = 0.9995
        min_T = 0.0001
        
        # Maintain element lists for efficient moves
        set_elems = get_elements(current_mask)
        unset_elems = [x for x in range(1, 513) if x not in set(set_elems)]
        
        while time.time() < deadline - 0.5:
            if T < min_T:
                break
                
            # 1-swap move
            if set_elems and unset_elems:
                idx_x = random.randint(0, len(set_elems) - 1)
                idx_y = random.randint(0, len(unset_elems) - 1)
                x = set_elems[idx_x]
                y = unset_elems[idx_y]
                bx = 1 << x
                by = 1 << y
                new_mask = current_mask ^ bx ^ by
                new_score = calc_score(new_mask)
                delta = new_score - current_score
                
                if delta >= 0:
                    current_mask = new_mask
                    current_score = new_score
                    set_elems[idx_x] = y
                    unset_elems[idx_y] = x
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
                else:
                    if T > 0.001 and random.random() < math.exp(delta / T if delta / T > -50 else -50):
                        current_mask = new_mask
                        current_score = new_score
                        set_elems[idx_x] = y
                        unset_elems[idx_y] = x
                        
                T *= cool_rate

    result = get_elements(best_mask)
    
    # Ensure validity
    if len(result) != 32:
        return list(range(1, 33))
        
    # Ensure distinctness and range
    if len(set(result)) != 32 or any(x < 1 or x > 512 for x in result):
        return list(range(1, 33))
        
    return result
