import time

def construct():
    start_time = time.time()
    deadline = start_time + 48.0  # 48s margin (evaluator kills at 90s, but keep safe)

    # State: (values_list, sums_set, last_val, suffix_sums_list)
    # suffix_sums: for values [v0,...,v_{n-1}], suffix_sums[i] = sum(values[i:])
    # So suffix_sums[0] = total sum, suffix_sums[n-1] = v_{n-1}
    # When appending x, new suffix sums are:
    #   x, x + suffix_sums[n-1], x + suffix_sums[n-2], ..., x + suffix_sums[0]
    # i.e., [x] + [x + s for s in reversed(suffix_sums)]
    # The new block sums are exactly these values.

    beam = [([], set(), 0, [])]  # (values, sums, last_val, suffix_sums)
    best_seq = []
    max_len = 0
    W = 100  # Beam width

    while True:
        if time.time() > deadline:
            break

        next_beam = []
        for values, sums, last_val, suffix_sums in beam:
            n = len(values)
            # Precompute nothing extra; suffix_sums is already available.
            # For each candidate x > last_val:
            #   new_block_sums = [x] + [x + suffix_sums[n-1-i] for i in range(n)]
            #   Check none are in sums.
            #   If valid, new_sums = sums | set(new_block_sums)
            #   new_suffix_sums = [x + suffix_sums[i] for i in range(n)][::-1] + [x]
            #   Wait, let me re-derive.
            #   values = [v0, v1, ..., v_{n-1}]
            #   suffix_sums[i] = v_i + v_{i+1} + ... + v_{n-1}
            #   After appending x: values' = [v0,...,v_{n-1}, x]
            #   suffix_sums'[i] for i < n: v_i + ... + v_{n-1} + x = suffix_sums[i] + x
            #   suffix_sums'[n] = x
            #   So new_suffix_sums = [s + x for s in suffix_sums] + [x]
            #   The new consecutive block sums ending at x are:
            #     x (block [x])
            #     x + v_{n-1} (block [v_{n-1}, x])
            #     x + v_{n-2} + v_{n-1}
            #     ...
            #     x + v_0 + ... + v_{n-1}
            #   These are: x, x + suffix_sums[n-1], x + suffix_sums[n-2], ..., x + suffix_sums[0]
            #   i.e., [x + suffix_sums[i] for i in range(n-1, -1, -1)] = [x + s for s in reversed(suffix_sums)]
            #   Plus x itself.
            #   So new_block_sums = [x] + [x + s for s in reversed(suffix_sums)]

            for x in range(last_val + 1, 257):
                # Fast check: is x in sums?
                if x in sums:
                    continue

                # Check all new block sums
                valid = True
                # new_block_sums = [x] + [x + s for s in reversed(suffix_sums)]
                # Check each
                # To avoid building the list if invalid, check on the fly
                # But we need the list to add to sums if valid.
                # Let's build it and check simultaneously.
                new_sums_list = [x]
                for s in reversed(suffix_sums):
                    ns = x + s
                    if ns in sums:
                        valid = False
                        break
                    new_sums_list.append(ns)

                if not valid:
                    continue

                # Valid: create new state
                new_values = values + [x]
                new_sums = sums.union(new_sums_list)
                # new_suffix_sums = [s + x for s in suffix_sums] + [x]
                new_suffix_sums = [s + x for s in suffix_sums] + [x]
                next_beam.append((new_values, new_sums, x, new_suffix_sums))

        if not next_beam:
            break

        # Heuristic: prefer states with smaller total sum and smaller last value
        # Total sum = suffix_sums[0] if n > 0 else 0
        # Actually, for a state, sum(values) = suffix_sums[0] if values non-empty
        def heuristic(state):
            vals, sums_set, last, suffix = state
            total = suffix[0] if suffix else 0
            # Prefer smaller total, then smaller last
            return (total, last)

        next_beam.sort(key=heuristic)

        if len(next_beam) > W:
            next_beam = next_beam[:W]

        # Update best sequence
        cur_len = len(next_beam[0][0])
        if cur_len > max_len:
            max_len = cur_len
            best_seq = next_beam[0][0]

        beam = next_beam

    return best_seq
