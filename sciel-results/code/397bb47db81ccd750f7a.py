import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

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
            # Every other number in a range
            for i in range(1, 65, 2):
                mask |= (1 << i)
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
    inits.append(structured_mask(6))
    for _ in range(4):
        inits.append(random_mask())

    num_restarts = len(inits)
    time_per_restart = (deadline - start_time) / max(num_restarts, 1)

    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break

        current_mask = inits[r]
        current_score = calc_score(current_mask)

        if current_score > best_score:
            best_score = current_score
            best_mask = current_mask

        # SA Parameters
        T = 1.5
        cool_rate = 0.9998
        min_T = 0.001
        restart_deadline = start_time + (r + 1) * time_per_restart

        no_improve = 0
        max_no_improve = 5000

        while time.time() < restart_deadline - 0.5:
            if T < min_T:
                break

            # Generate move: swap a random element with a random non-element
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

                if delta >= 0:
                    accept = True
                else:
                    if delta / T < -50:
                        accept = False
                    else:
                        accept = random.random() < math.exp(delta / T)

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
                    # Kick: random perturbation with 2-swap
                    kick_success = False
                    for _ in range(20):
                        x1 = random.randint(1, 512)
                        x2 = random.randint(1, 512)
                        y1 = random.randint(1, 512)
                        y2 = random.randint(1, 512)
                        if x1 == x2 or y1 == y2 or x1 == y1 or x1 == y2 or x2 == y1 or x2 == y2:
                            continue
                        bx1, bx2, by1, by2 = 1 << x1, 1 << x2, 1 << y1, 1 << y2
                        if (current_mask & bx1) and (current_mask & bx2) and not (current_mask & by1) and not (current_mask & by2):
                            new_mask = current_mask ^ bx1 ^ bx2 ^ by1 ^ by2
                            current_mask = new_mask
                            current_score = calc_score(current_mask)
                            T = 0.5
                            no_improve = 0
                            kick_success = True
                            break
                    if not kick_success:
                        break

            # Occasionally try a 2-swap move in the main loop
            if random.random() < 0.1:
                x1 = random.randint(1, 512)
                x2 = random.randint(1, 512)
                y1 = random.randint(1, 512)
                y2 = random.randint(1, 512)
                if x1 == x2 or y1 == y2 or x1 == y1 or x1 == y2 or x2 == y1 or x2 == y2:
                    continue
                bx1, bx2, by1, by2 = 1 << x1, 1 << x2, 1 << y1, 1 << y2
                if (current_mask & bx1) and (current_mask & bx2) and not (current_mask & by1) and not (current_mask & by2):
                    new_mask = current_mask ^ bx1 ^ bx2 ^ by1 ^ by2
                    new_score = calc_score(new_mask)
                    delta = new_score - current_score
                    if delta >= 0 or random.random() < math.exp(delta / T):
                        current_mask = new_mask
                        current_score = new_score
                        no_improve = 0
                        if current_score > best_score:
                            best_score = current_score
                            best_mask = current_mask

    # Final Polish: Hill Climbing on the best solution found
    if time.time() < deadline - 2 and best_mask != 0:
        current_mask = best_mask
        current_score = best_score
        T = 0.01
        cool_rate = 0.9995
        min_T = 0.0001

        while time.time() < deadline - 0.5:
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

                if delta >= 0:
                    current_mask = new_mask
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
                else:
                    if random.random() < math.exp(delta / T):
                        current_mask = new_mask
                        current_score = new_score

                T *= cool_rate

    result = get_elements(best_mask)

    # Ensure validity
    if len(result) != 32:
        return list(range(1, 33))

    if len(set(result)) != 32 or any(x < 1 or x > 512 for x in result):
        return list(range(1, 33))

    return result
