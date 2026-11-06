import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin before 60s limit

    n = 256
    max_sum = 2 * n  # 512

    # Precompute squares up to max_sum
    squares = set()
    k = 1
    while k * k <= max_sum:
        squares.add(k * k)
        k += 1

    # Filter out numbers where 2*a is a square (diagonal constraint)
    valid_numbers = [a for a in range(1, n + 1) if 2 * a not in squares]
    num_valid = len(valid_numbers)
    val_to_idx = {v: i for i, v in enumerate(valid_numbers)}

    # Build conflict adjacency bitsets
    adj_bits = [0] * num_valid
    for i in range(num_valid):
        a = valid_numbers[i]
        mask = 0
        for s in squares:
            b = s - a
            if b != a and b in val_to_idx:
                mask |= (1 << val_to_idx[b])
        adj_bits[i] = mask

    # Helper: count bits in a mask
    def popcount(x):
        return bin(x).count('1')

    # Greedy initial solution: minimum degree ordering
    def greedy_initial():
        remaining = (1 << num_valid) - 1
        selected = 0
        while remaining:
            # Find vertex with minimum degree in remaining subgraph
            best_v = -1
            best_deg = num_valid + 1
            temp = remaining
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                deg = popcount(adj_bits[v] & remaining)
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
                temp ^= lsb
            if best_v >= 0:
                selected |= (1 << best_v)
                remaining &= ~(1 << best_v)
                remaining &= ~adj_bits[best_v]
        return selected

    # Mod-3 class 1 baseline
    def mod3_solution():
        mask = 0
        for v in valid_numbers:
            if v % 3 == 1:
                idx = val_to_idx[v]
                mask |= (1 << idx)
        # Verify validity
        temp = mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            if (mask & adj_bits[idx]) != 0:
                return 0  # Invalid
            temp ^= lsb
        return mask

    # Get initial best
    best_mask = greedy_initial()
    best_size = popcount(best_mask)

    mod3_mask = mod3_solution()
    if popcount(mod3_mask) > best_size:
        best_mask = mod3_mask
        best_size = popcount(mod3_mask)

    # Try several random greedy orders
    random.seed(12345)
    for trial in range(20):
        order = list(range(num_valid))
        random.shuffle(order)
        mask = 0
        for idx in order:
            if (mask & adj_bits[idx]) == 0:
                mask |= (1 << idx)
        sz = popcount(mask)
        if sz > best_size:
            best_size = sz
            best_mask = mask

    # Local search with add/remove/swap/perturbation
    def count_conflicts(mask):
        """Count number of conflict edges in the induced subgraph on mask."""
        count = 0
        temp = mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            count += popcount(mask & adj_bits[idx])
            temp ^= lsb
        return count // 2

    current_mask = best_mask
    current_conflicts = count_conflicts(current_mask)  # Should be 0
    current_size = popcount(current_mask)

    T = 30.0
    T_min = 0.05
    alpha = 0.997
    M = 200  # Penalty for conflicts

    # Precompute for speed
    all_indices = list(range(num_valid))

    while time.time() - start_time < deadline - 3:
        # Choose move type
        r = random.random()
        if r < 0.30:
            # Add a random vertex not in set
            not_in_set = ~current_mask & ((1 << num_valid) - 1)
            if not_in_set == 0:
                continue
            lsb = not_in_set & -not_in_set
            idx = lsb.bit_length() - 1
            bit = 1 << idx
            if current_mask & bit:
                continue
            delta_size = 1
            delta_conf = popcount(current_mask & adj_bits[idx])
        elif r < 0.55:
            # Remove a random vertex in set
            if current_mask == 0:
                continue
            lsb = current_mask & -current_mask
            idx = lsb.bit_length() - 1
            bit = 1 << idx
            delta_size = -1
            delta_conf = -popcount(current_mask & adj_bits[idx])
        elif r < 0.80:
            # Swap: remove one in set, add one not in set
            in_set = current_mask
            not_in_set = ~current_mask & ((1 << num_valid) - 1)
            if in_set == 0 or not_in_set == 0:
                continue
            # Random removal
            lsb_r = in_set & -in_set
            # Pick a random bit from in_set
            rand_val = random.getrandbits(num_valid)
            in_set_bits = in_set & rand_val
            if in_set_bits == 0:
                in_set_bits = in_set
            lsb_r = in_set_bits & -in_set_bits
            idx_r = lsb_r.bit_length() - 1
            bit_r = 1 << idx_r

            # Random addition
            rand_val2 = random.getrandbits(num_valid)
            not_in_set_bits = not_in_set & rand_val2
            if not_in_set_bits == 0:
                not_in_set_bits = not_in_set
            lsb_a = not_in_set_bits & -not_in_set_bits
            idx_a = lsb_a.bit_length() - 1
            bit_a = 1 << idx_a

            if idx_r == idx_a:
                continue

            # Compute delta
            # Remove idx_r, add idx_a
            # Conflicts: removing idx_r removes its neighbors in set
            # Adding idx_a adds its neighbors in set (minus idx_r if they conflict)
            old_conf_r = popcount(current_mask & adj_bits[idx_r])
            # After removal, set is current_mask without idx_r
            new_set_for_a = (current_mask & ~bit_r)
            new_conf_a = popcount(new_set_for_a & adj_bits[idx_a])

            delta_size = 0
            delta_conf = -old_conf_r + new_conf_a
        else:
            # Perturbation: randomly flip 2-5 bits
            num_flips = random.randint(2, 5)
            delta_size = 0
            delta_conf = 0
            new_mask = current_mask
            for _ in range(num_flips):
                rand_val = random.getrandbits(num_valid)
                lsb = rand_val & -rand_val
                if lsb == 0:
                    continue
                idx = lsb.bit_length() - 1
                bit = 1 << idx
                if new_mask & bit:
                    # Remove
                    delta_size -= 1
                    delta_conf -= popcount(new_mask & adj_bits[idx])
                    new_mask ^= bit
                else:
                    # Add
                    delta_size += 1
                    delta_conf += popcount(new_mask & adj_bits[idx])
                    new_mask ^= bit

            # Check if it's a valid perturbation (not same as current)
            if new_mask == current_mask:
                continue

            delta_cost = -delta_size + M * delta_conf
            if delta_cost < 0:
                accept = True
            else:
                if T > T_min:
                    accept = random.random() < math.exp(-delta_cost / T)
                else:
                    accept = False

            if accept:
                current_mask = new_mask
                current_size += delta_size
                current_conflicts += delta_conf
                if current_conflicts == 0 and current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask

            T *= alpha
            continue  # Skip the normal accept logic below

        # Normal move accept logic (for add/remove/swap)
        delta_cost = -delta_size + M * delta_conf
        if delta_cost < 0:
            accept = True
        else:
            if T > T_min:
                accept = random.random() < math.exp(-delta_cost / T)
            else:
                accept = False

        if accept:
            if r < 0.30:
                # Add
                current_mask |= bit
            elif r < 0.55:
                # Remove
                current_mask &= ~bit
            elif r < 0.80:
                # Swap
                current_mask &= ~bit_r
                current_mask |= bit_a

            current_size += delta_size
            current_conflicts += delta_conf

            # Track best valid solution
            if current_conflicts == 0 and current_size > best_size:
                best_size = current_size
                best_mask = current_mask

        T *= alpha

    # Final verification and cleanup
    # Verify best_mask is valid
    temp = best_mask
    valid = True
    while temp:
        lsb = temp & -temp
        idx = lsb.bit_length() - 1
        if (best_mask & adj_bits[idx]) != 0:
            valid = False
            break
        temp ^= lsb

    if not valid:
        # Fallback to a known valid solution
        best_mask = 1 << val_to_idx[1] if 1 in val_to_idx else 0
        best_size = popcount(best_mask)

    # Convert to list
    result = []
    temp = best_mask
    while temp:
        lsb = temp & -temp
        idx = lsb.bit_length() - 1
        result.append(valid_numbers[idx])
        temp ^= lsb

    if not result:
        result = [1]

    return result
