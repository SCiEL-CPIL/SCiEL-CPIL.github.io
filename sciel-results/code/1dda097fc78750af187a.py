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
        c = 0
        m = mask
        while m:
            c += 1
            m &= m - 1
        return c

    best_mask = 0
    best_score = -1

    # Strategy: multiple simulated annealing restarts with different inits
    # Use bitmasks for fast scoring

    # Initial configurations
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

    # Init 3: arithmetic progression with step 16: 1, 17, 33, ...
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
    mask5 = 0
    vals = sorted(random.sample(range(1, 513), 32))
    for v in vals:
        mask5 |= (1 << v)
    inits.append(mask5)

    # Init 6: structured - every other number in a range
    mask6 = 0
    for i in range(1, 65, 2):
        mask6 |= (1 << i)
    inits.append(mask6)

    # Run simulated annealing for each init
    num_restarts = len(inits)
    time_per_restart = (deadline - start_time) / num_restarts

    for r in range(num_restarts):
        if time.time() > deadline - 2:
            break

        current_mask = inits[r]
        current_score = calc_score(current_mask)

        if current_score > best_score:
            best_score = current_score
            best_mask = current_mask

        # Simulated annealing parameters
        T = 1.0
        cool_rate = 0.9995
        min_T = 0.005

        restart_deadline = start_time + (r + 1) * time_per_restart

        no_improve = 0
        max_no_improve = 2000

        while time.time() < restart_deadline - 0.5:
            if T < min_T:
                break

            # Generate move: swap a random element with a random non-element
            # Pick random x in 1..512, random y in 1..512
            # If x in mask and y not in mask: remove x, add y
            x = random.randint(1, 512)
            y = random.randint(1, 512)

            if x == y:
                continue

            bx = 1 << x
            by = 1 << y

            # x must be in mask, y must NOT be in mask
            if (current_mask & bx) and not (current_mask & by):
                new_mask = current_mask ^ bx ^ by
                new_score = calc_score(new_mask)
                delta = new_score - current_score

                # Accept if better, or with probability exp(-delta/T) for worse
                if delta >= 0:
                    accept = True
                else:
                    # delta is negative, so -delta/T is positive
                    # Accept with probability exp(delta/T) where delta < 0
                    threshold = math.exp(delta / T)
                    accept = random.random() < threshold

                if accept:
                    current_mask = new_mask
                    current_score = new_score
                    no_improve = 0

                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
                else:
                    no_improve += 1

                T *= cool_rate

                if no_improve > max_no_improve:
                    break

            # Occasionally try a "kick" - swap 2 elements at once
            if random.random() < 0.05:
                # Pick 2 elements to remove, 2 to add
                # Find set bits and unset bits
                set_list = []
                unset_list = []
                for i in range(1, 513):
                    if current_mask & (1 << i):
                        set_list.append(i)
                    else:
                        unset_list.append(i)

                if len(set_list) >= 2 and len(unset_list) >= 2:
                    x1, x2 = random.sample(set_list, 2)
                    y1, y2 = random.sample(unset_list, 2)

                    new_mask = current_mask
                    new_mask ^= (1 << x1) ^ (1 << x2) ^ (1 << y1) ^ (1 << y2)

                    if count_bits(new_mask) == 32:
                        new_score = calc_score(new_mask)
                        delta = new_score - current_score

                        if delta >= 0 or random.random() < math.exp(delta / T):
                            current_mask = new_mask
                            current_score = new_score
                            no_improve = 0

                            if current_score > best_score:
                                best_score = current_score
                                best_mask = current_mask

    # Final: try a quick local search from the best found
    if time.time() < deadline - 5 and best_mask != 0:
        current_mask = best_mask
        current_score = best_score
        T = 0.05
        cool_rate = 0.999
        min_T = 0.001

        while time.time() < deadline - 1:
            if T < min_T:
                break

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

                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask

                T *= cool_rate

    # Convert best_mask to list
    result = get_elements(best_mask)

    # Verify length
    if len(result) != 32:
        # Fallback
        return list(range(1, 33))

    return result
