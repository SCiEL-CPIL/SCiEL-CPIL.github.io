import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

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
        for i in range(1, 513):
            if mask & (1 << i):
                elems.append(i)
        return elems

    def count_bits(mask):
        """Count set bits."""
        return bin(mask).count('1')

    def get_set_and_unset(mask):
        """Return (set_list, unset_list) for a mask."""
        set_list = []
        unset_list = []
        for i in range(1, 513):
            if mask & (1 << i):
                set_list.append(i)
            else:
                unset_list.append(i)
        return set_list, unset_list

    def greedy_construct():
        """Greedy construction: place elements to maximize progression coverage."""
        mask = 0
        set_list = []
        
        # Start with a few seed elements
        seeds = [1, 3, 7, 15, 31, 63, 127, 255]
        for s in seeds:
            if s <= 512:
                mask |= (1 << s)
                set_list.append(s)
        
        remaining = 32 - len(set_list)
        
        # Greedily add elements that create the most new progressions
        candidates = list(range(1, 513))
        for c in seeds:
            if c in candidates:
                candidates.remove(c)
        
        for _ in range(remaining):
            if time.time() > deadline - 30:
                break
            best_gain = -1
            best_elem = None
            current_score = calc_score(mask)
            
            # Sample candidates to keep it fast
            sample_size = min(50, len(candidates))
            sampled = random.sample(candidates, sample_size) if len(candidates) > sample_size else candidates
            
            for elem in sampled:
                new_mask = mask | (1 << elem)
                new_score = calc_score(new_mask)
                gain = new_score - current_score
                if gain > best_gain:
                    best_gain = gain
                    best_elem = elem
            
            if best_elem is not None:
                mask |= (1 << best_elem)
                set_list.append(best_elem)
                if best_elem in candidates:
                    candidates.remove(best_elem)
        
        return mask, set_list

    def local_search_polish(mask, set_list, unset_list, max_iterations=5000):
        """Polish a solution using local search."""
        current_mask = mask
        current_score = calc_score(mask)
        
        for _ in range(max_iterations):
            if time.time() > deadline - 5:
                break
            
            improved = False
            for i in range(len(set_list)):
                if time.time() > deadline - 5:
                    break
                x = set_list[i]
                for j in range(len(unset_list)):
                    y = unset_list[j]
                    new_mask = current_mask ^ (1 << x) ^ (1 << y)
                    new_score = calc_score(new_mask)
                    if new_score > current_score:
                        current_mask = new_mask
                        current_score = new_score
                        set_list[i] = y
                        unset_list[j] = x
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return current_mask, current_score

    best_mask = 0
    best_score = -1

    # Initial configurations - diverse structured and random starts
    inits = []

    # Init 1: contiguous block 1..32
    mask1 = 0
    for i in range(1, 33):
        mask1 |= (1 << i)
    inits.append(mask1)

    # Init 2: random 32 elements
    elems = random.sample(range(1, 513), 32)
    mask2 = 0
    for e in elems:
        mask2 |= (1 << e)
    inits.append(mask2)

    # Init 3: arithmetic progression with step 16
    mask3 = 0
    for i in range(32):
        val = 1 + i * 16
        if val <= 512:
            mask3 |= (1 << val)
    inits.append(mask3)

    # Init 4: two interleaved APs
    mask4 = 0
    for i in range(16):
        mask4 |= (1 << (1 + i * 32))
        mask4 |= (1 << (2 + i * 32))
    inits.append(mask4)

    # Init 5: geometric-ish spread
    vals = sorted(random.sample(range(1, 513), 32))
    mask5 = 0
    for v in vals:
        mask5 |= (1 << v)
    inits.append(mask5)

    # Init 6: structured - every other number in a range
    mask6 = 0
    for i in range(1, 65, 2):
        mask6 |= (1 << i)
    inits.append(mask6)

    # Init 7: another random
    elems7 = random.sample(range(1, 513), 32)
    mask7 = 0
    for e in elems7:
        mask7 |= (1 << e)
    inits.append(mask7)

    # Init 8: spread out with gaps
    mask8 = 0
    for i in range(32):
        val = 1 + i * 15
        if val <= 512:
            mask8 |= (1 << val)
    inits.append(mask8)

    # Init 9: greedy construction
    greedy_mask, _ = greedy_construct()
    inits.append(greedy_mask)

    # Run simulated annealing for each init
    num_restarts = len(inits)
    elapsed = time.time() - start_time
    remaining = deadline - time.time()
    time_per_restart = remaining / max(num_restarts, 1)

    for r in range(num_restarts):
        if time.time() > deadline - 2:
            break

        current_mask = inits[r]
        current_score = calc_score(current_mask)

        if current_score > best_score:
            best_score = current_score
            best_mask = current_mask

        # Simulated annealing parameters
        T = 3.0
        cool_rate = 0.9995
        min_T = 0.05

        restart_deadline = start_time + (r + 1) * time_per_restart

        no_improve = 0
        max_no_improve = 5000

        # Cache set/unset lists, update incrementally
        set_list, unset_list = get_set_and_unset(current_mask)

        while time.time() < restart_deadline - 0.5:
            if T < min_T:
                break

            if not set_list or not unset_list:
                break

            # Generate move: swap a random element with a random non-element
            x = set_list[random.randrange(len(set_list))]
            y = unset_list[random.randrange(len(unset_list))]

            bx = 1 << x
            by = 1 << y

            new_mask = current_mask ^ bx ^ by
            new_score = calc_score(new_mask)
            delta = new_score - current_score

            if delta >= 0:
                accept = True
            else:
                threshold = math.exp(delta / T)
                accept = random.random() < threshold

            if accept:
                current_mask = new_mask
                current_score = new_score
                no_improve = 0

                # Update set/unset lists
                set_list.remove(x)
                set_list.append(y)
                unset_list.remove(y)
                unset_list.append(x)

                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
            else:
                no_improve += 1

            T *= cool_rate

            if no_improve > max_no_improve:
                break

            # Occasionally try a "kick" - swap 2 elements at once
            if random.random() < 0.05 and len(set_list) >= 2 and len(unset_list) >= 2:
                i1 = random.randrange(len(set_list))
                i2 = random.randrange(len(set_list))
                if i1 == i2:
                    continue
                j1 = random.randrange(len(unset_list))
                j2 = random.randrange(len(unset_list))
                if j1 == j2:
                    continue

                x1 = set_list[i1]
                x2 = set_list[i2]
                y1 = unset_list[j1]
                y2 = unset_list[j2]

                new_mask = current_mask
                new_mask ^= (1 << x1) ^ (1 << x2) ^ (1 << y1) ^ (1 << y2)

                if count_bits(new_mask) == 32:
                    new_score = calc_score(new_mask)
                    delta = new_score - current_score

                    if delta >= 0 or random.random() < math.exp(delta / T):
                        current_mask = new_mask
                        current_score = new_score
                        no_improve = 0

                        # Update set/unset lists
                        set_list[i1] = y1
                        set_list[i2] = y2
                        unset_list[j1] = x1
                        unset_list[j2] = x2

                        if current_score > best_score:
                            best_score = current_score
                            best_mask = current_mask

    # Final: deterministic local search polish from the best found
    if time.time() < deadline - 5 and best_mask != 0:
        current_mask = best_mask
        current_score = best_score
        set_list, unset_list = get_set_and_unset(current_mask)

        # Phase 1: hill climbing - only accept improvements
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for i in range(len(set_list)):
                if time.time() > deadline - 3:
                    break
                x = set_list[i]
                for j in range(len(unset_list)):
                    y = unset_list[j]
                    bx = 1 << x
                    by = 1 << y
                    new_mask = current_mask ^ bx ^ by
                    new_score = calc_score(new_mask)
                    if new_score > current_score:
                        current_mask = new_mask
                        current_score = new_score
                        set_list[i] = y
                        unset_list[j] = x
                        if current_score > best_score:
                            best_score = current_score
                            best_mask = current_mask
                        improved = True
                        break
                if improved:
                    break

        # Phase 2: low-temperature SA to escape local optima
        if time.time() < deadline - 1:
            T = 0.1
            cool_rate = 0.998
            min_T = 0.005
            no_improve = 0
            max_no_improve = 8000

            while time.time() < deadline - 1:
                if T < min_T:
                    break

                if not set_list or not unset_list:
                    break

                i = random.randrange(len(set_list))
                j = random.randrange(len(unset_list))
                x = set_list[i]
                y = unset_list[j]

                bx = 1 << x
                by = 1 << y
                new_mask = current_mask ^ bx ^ by
                new_score = calc_score(new_mask)
                delta = new_score - current_score

                if delta >= 0:
                    accept = True
                else:
                    threshold = math.exp(delta / T)
                    accept = random.random() < threshold

                if accept:
                    current_mask = new_mask
                    current_score = new_score
                    no_improve = 0
                    set_list[i] = y
                    unset_list[j] = x

                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
                else:
                    no_improve += 1

                T *= cool_rate

                if no_improve > max_no_improve:
                    break

    # Convert best_mask to list
    result = get_elements(best_mask)

    if len(result) != 32:
        return list(range(1, 33))

    return result
