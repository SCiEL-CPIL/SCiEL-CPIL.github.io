import time
import random

P = {'n': 128}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    n = P['n']
    max_val = n

    # Precompute pairs_sum[x] = list of (a,b) with a<b, a+b=x
    pairs_sum = [[] for _ in range(2 * max_val + 1)]
    for a in range(1, max_val + 1):
        for b in range(a + 1, max_val + 1):
            s = a + b
            if s <= 2 * max_val:
                pairs_sum[s].append((a, b))

    def is_valid_addition(x, mask):
        """
        Check if adding x to set represented by mask is valid.
        mask: bitset where bit i-1 is 1 if i is in S.
        """
        if mask & (1 << (x - 1)):
            return True

        # Case 1: x acts as a member 'a' in a triple (a,b,c)
        # Need b,c in S distinct, b!=x, c!=x such that x+b in S, x+c in S, b+c in S
        # U_x = { y in S | x+y in S } - these are potential b's
        U_x = mask & (mask >> x)

        if U_x:
            # For each pair b,c in U_x with b!=c, check if b+c in S
            # We need to check all pairs in U_x
            m = U_x
            # Collect all bits in U_x
            bits = []
            while m:
                lsb = m & -m
                bits.append(lsb)
                m ^= lsb

            # Check all pairs
            for i in range(len(bits)):
                b_bit = bits[i]
                b_val = b_bit.bit_length()  # This gives position+1, but we need value
                # Actually bit_length() of (1<<k) is k+1, so value = k+1 where k is index
                # Wait: if bit is at position k (0-indexed), value is k+1
                # b_bit.bit_length() gives k+1 for a single bit at position k
                # So b_val = b_bit.bit_length() is correct for the value
                b_val = b_bit.bit_length()
                for j in range(i + 1, len(bits)):
                    c_bit = bits[j]
                    c_val = c_bit.bit_length()
                    # Check if b+c in S
                    if b_val + c_val <= max_val and (mask >> (b_val + c_val - 1)) & 1:
                        return False

        # Case 2: x acts as a sum a+b = x, with a,b in S
        # Need c in S \ {a,b} such that a+c in S and b+c in S
        for a, b in pairs_sum[x]:
            if not (mask & (1 << (a - 1))):
                continue
            if not (mask & (1 << (b - 1))):
                continue

            # V_a = { c in S | a+c in S }
            # V_b = { c in S | b+c in S }
            # We need c in V_a ∩ V_b, c != a, c != b
            # V_a: for each c in S, check if a+c in S
            # Using bitsets: V_a_mask = mask & (mask >> a) gives bits c where c in S and a+c in S
            V_a = mask & (mask >> a)
            V_b = mask & (mask >> b)

            common = V_a & V_b
            # Remove a and b from common
            a_bit = 1 << (a - 1)
            b_bit = 1 << (b - 1)
            if common & ~(a_bit | b_bit):
                return False

        return True

    def elements_of(mask):
        return [i for i in range(1, max_val + 1) if (mask & (1 << (i - 1)))]

    def unused_of(mask):
        return [i for i in range(1, max_val + 1) if not (mask & (1 << (i - 1)))]

    # ---- Multiple initial constructions ----
    best_mask = 0
    best_size = 0

    # Try different K values
    for K in [64, 60, 56, 52, 48, 68, 72]:
        if time.time() > deadline - 30:
            break
        alt_mask = 0
        for v in range(K, max_val + 1):
            alt_mask |= (1 << (v - 1))

        # Greedy add lower numbers
        for v in range(K - 1, 0, -1):
            if is_valid_addition(v, alt_mask):
                alt_mask |= (1 << (v - 1))

        alt_size = bin(alt_mask).count('1')
        if alt_size > best_size:
            best_size = alt_size
            best_mask = alt_mask

    # Also try starting with just upper half and adding greedily in different orders
    for _ in range(3):
        if time.time() > deadline - 25:
            break
        alt_mask = 0
        for v in range(65, max_val + 1):
            alt_mask |= (1 << (v - 1))

        # Random order for lower numbers
        lower = list(range(1, 65))
        random.shuffle(lower)
        for v in lower:
            if is_valid_addition(v, alt_mask):
                alt_mask |= (1 << (v - 1))

        alt_size = bin(alt_mask).count('1')
        if alt_size > best_size:
            best_size = alt_size
            best_mask = alt_mask

    current_mask = best_mask
    current_size = best_size
    current_elements = elements_of(current_mask)
    unused_elements = set(unused_of(current_mask))

    no_progress_count = 0
    max_no_progress = 100

    while time.time() < deadline - 3.0:
        # 1. Greedy Add - try all unused elements, pick first valid
        if unused_elements:
            added = False
            candidates = list(unused_elements)
            random.shuffle(candidates)
            for x in candidates:
                if is_valid_addition(x, current_mask):
                    current_mask |= (1 << (x - 1))
                    current_size += 1
                    unused_elements.remove(x)
                    if current_size > best_size:
                        best_size = current_size
                        best_mask = current_mask
                    added = True
                    no_progress_count = 0
                    break

            if added:
                continue

        # 2. Multi-swap strategies
        found_swap = False

        # 2a. 1-out, 2-in Swap
        if current_elements and len(unused_elements) >= 2:
            x_out = random.choice(current_elements)
            mask_trial = current_mask & ~(1 << (x_out - 1))

            valid_adds = []
            for x in unused_elements:
                if is_valid_addition(x, mask_trial):
                    valid_adds.append(x)

            if len(valid_adds) >= 2:
                random.shuffle(valid_adds)
                max_checks = 50
                checks = 0
                for i in range(len(valid_adds)):
                    if checks > max_checks:
                        break
                    x1 = valid_adds[i]
                    mask_1 = mask_trial | (1 << (x1 - 1))
                    for j in range(i + 1, len(valid_adds)):
                        if checks > max_checks:
                            break
                        checks += 1
                        x2 = valid_adds[j]
                        if is_valid_addition(x2, mask_1):
                            current_mask = mask_1 | (1 << (x2 - 1))
                            current_size += 1
                            current_elements.remove(x_out)
                            current_elements.append(x1)
                            current_elements.append(x2)
                            unused_elements.remove(x1)
                            unused_elements.remove(x2)
                            unused_elements.add(x_out)

                            if current_size > best_size:
                                best_size = current_size
                                best_mask = current_mask

                            found_swap = True
                            no_progress_count = 0
                            break
                    if found_swap:
                        break

        # 2b. 2-out, 3-in Swap
        if not found_swap and current_elements and len(unused_elements) >= 3:
            if len(current_elements) >= 2:
                out_indices = random.sample(range(len(current_elements)), 2)
                x_out1 = current_elements[out_indices[0]]
                x_out2 = current_elements[out_indices[1]]

                mask_trial = current_mask & ~(1 << (x_out1 - 1)) & ~(1 << (x_out2 - 1))

                valid_adds = []
                for x in unused_elements:
                    if is_valid_addition(x, mask_trial):
                        valid_adds.append(x)

                if len(valid_adds) >= 3:
                    random.shuffle(valid_adds)
                    max_checks = 30
                    checks = 0
                    for i in range(len(valid_adds)):
                        if checks > max_checks:
                            break
                        x1 = valid_adds[i]
                        mask_1 = mask_trial | (1 << (x1 - 1))
                        for j in range(i + 1, len(valid_adds)):
                            if checks > max_checks:
                                break
                            x2 = valid_adds[j]
                            mask_2 = mask_1 | (1 << (x2 - 1))
                            for k in range(j + 1, len(valid_adds)):
                                if checks > max_checks:
                                    break
                                checks += 1
                                x3 = valid_adds[k]
                                if is_valid_addition(x3, mask_2):
                                    current_mask = mask_2 | (1 << (x3 - 1))
                                    current_size += 1
                                    current_elements.remove(x_out1)
                                    current_elements.remove(x_out2)
                                    current_elements.append(x1)
                                    current_elements.append(x2)
                                    current_elements.append(x3)
                                    unused_elements.remove(x1)
                                    unused_elements.remove(x2)
                                    unused_elements.remove(x3)
                                    unused_elements.add(x_out1)
                                    unused_elements.add(x_out2)

                                    if current_size > best_size:
                                        best_size = current_size
                                        best_mask = current_mask

                                    found_swap = True
                                    no_progress_count = 0
                                    break
                            if found_swap:
                                break
                        if found_swap:
                            break

        # 2c. 1-in, 1-out swap (size-neutral, but may enable future adds)
        if not found_swap and current_elements and unused_elements:
            x_out = random.choice(current_elements)
            mask_trial = current_mask & ~(1 << (x_out - 1))

            valid_adds = []
            for x in unused_elements:
                if x != x_out and is_valid_addition(x, mask_trial):
                    valid_adds.append(x)

            if valid_adds:
                x_in = random.choice(valid_adds)
                current_mask = mask_trial | (1 << (x_in - 1))
                current_elements.remove(x_out)
                current_elements.append(x_in)
                unused_elements.remove(x_in)
                unused_elements.add(x_out)
                found_swap = True
                no_progress_count = 0

        if found_swap:
            continue

        # 3. Perturbation - remove 1-3 random elements
        if current_elements:
            num_remove = random.randint(1, 3)
            num_remove = min(num_remove, len(current_elements))
            for _ in range(num_remove):
                if current_elements:
                    x_out = random.choice(current_elements)
                    current_mask &= ~(1 << (x_out - 1))
                    current_size -= 1
                    current_elements.remove(x_out)
                    unused_elements.add(x_out)
            no_progress_count += 1

        if no_progress_count > max_no_progress:
            # Restart from best with some randomization
            current_mask = best_mask
            current_size = best_size
            current_elements = elements_of(current_mask)
            unused_elements = set(unused_of(current_mask))
            no_progress_count = 0

            # Try a few random removals to escape local optimum
            for _ in range(random.randint(1, 5)):
                if current_elements:
                    x_out = random.choice(current_elements)
                    current_mask &= ~(1 << (x_out - 1))
                    current_size -= 1
                    current_elements.remove(x_out)
                    unused_elements.add(x_out)

    # Final polish - multiple passes
    if time.time() < deadline - 1.0:
        final_mask = best_mask
        final_unused = set(unused_of(final_mask))

        for pass_num in range(10):
            if time.time() > deadline - 1.0:
                break
            changed = True
            while changed and time.time() < deadline - 1.0:
                changed = False
                if pass_num % 2 == 0:
                    candidates = sorted(final_unused)
                else:
                    candidates = list(final_unused)
                    random.shuffle(candidates)
                for x in candidates:
                    if x not in final_unused:
                        continue
                    if is_valid_addition(x, final_mask):
                        final_mask |= (1 << (x - 1))
                        final_unused.remove(x)
                        changed = True

        if bin(final_mask).count('1') > best_size:
            best_mask = final_mask
            best_size = bin(final_mask).count('1')

    # Convert best_mask to list of ints
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(idx + 1)
        m ^= lsb

    return sorted(result)
