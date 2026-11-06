import time
import random
import bisect

N = 256

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s limit

    def is_valid_addition(s_set, s_sorted, candidate):
        """Check if adding candidate to s_sorted (sorted list) keeps validity."""
        pos = bisect.bisect_left(s_sorted, candidate)
        n = len(s_sorted)

        # candidate as 'a': for each b > candidate, no c >= b in set with (b+c) % candidate == 0
        for i in range(pos, n):
            b = s_sorted[i]
            target = (-b) % candidate
            # Check if any c in s_sorted[i:] has c % candidate == target
            for k in range(i, n):
                if s_sorted[k] % candidate == target:
                    return False

        # candidate as 'b': for each a < candidate, no c >= candidate in set with (candidate+c) % a == 0
        for i in range(pos):
            a = s_sorted[i]
            # Check c = candidate: (candidate + candidate) % a
            if (2 * candidate) % a == 0:
                return False
            # Check c in s_sorted[pos:]
            for k in range(pos, n):
                if (candidate + s_sorted[k]) % a == 0:
                    return False

        # candidate as 'c': for each a < b <= candidate in set, (b + candidate) % a != 0
        for i in range(n):
            a = s_sorted[i]
            for j in range(i + 1, n):
                b = s_sorted[j]
                if b > candidate:
                    break
                if (b + candidate) % a == 0:
                    return False

        return True

    def conflict_count(elem, s_sorted, s_set):
        """Count how many valid triples would be violated if elem were in the set."""
        count = 0
        pos = bisect.bisect_left(s_sorted, elem)
        n = len(s_sorted)

        # elem as 'a'
        for i in range(pos, n):
            b = s_sorted[i]
            target = (-b) % elem
            for k in range(i, n):
                if s_sorted[k] % elem == target:
                    count += 1

        # elem as 'b'
        for i in range(pos):
            a = s_sorted[i]
            if (2 * elem) % a == 0:
                count += 1
            for k in range(pos, n):
                if (elem + s_sorted[k]) % a == 0:
                    count += 1

        # elem as 'c'
        for i in range(n):
            a = s_sorted[i]
            for j in range(i + 1, n):
                b = s_sorted[j]
                if b > elem:
                    break
                if (b + elem) % a == 0:
                    count += 1

        return count

    best_set = []

    # Try different starting points
    for min_start in [128, 144, 160, 171, 180, 192, 200, 210, 220]:
        if time.time() - start_time > deadline:
            break

        # Start with all elements from min_start to N
        s_sorted = list(range(min_start, N + 1))
        s_set = set(s_sorted)

        # Repair: iteratively remove the element with the highest conflict count
        changed = True
        while changed:
            changed = False
            if len(s_sorted) <= 1:
                break
            # Find the element with the most conflicts
            max_conf = 0
            max_idx = -1
            for i, a in enumerate(s_sorted):
                # Temporarily consider a's conflicts with the rest
                # a as 'a': count pairs (b,c) with b,c > a, b<=c, (b+c)%a==0
                pos = bisect.bisect_left(s_sorted, a)
                n = len(s_sorted)
                conf = 0
                # a as 'a'
                for j in range(pos, n):
                    b = s_sorted[j]
                    if b == a:
                        continue
                    target = (-b) % a
                    for k in range(j, n):
                        if s_sorted[k] % a == target:
                            conf += 1
                # a as 'b': pairs (x, a, c) with x < a <= c
                for j in range(pos):
                    x = s_sorted[j]
                    if (2 * a) % x == 0:
                        conf += 1
                    for k in range(pos, n):
                        if (a + s_sorted[k]) % x == 0:
                            conf += 1
                # a as 'c': pairs (x, y, a) with x < y <= a
                for j in range(n):
                    x = s_sorted[j]
                    for k in range(j + 1, n):
                        y = s_sorted[k]
                        if y > a:
                            break
                        if (y + a) % x == 0:
                            conf += 1

                if conf > max_conf:
                    max_conf = conf
                    max_idx = i

            if max_idx >= 0 and max_conf > 0:
                s_sorted.pop(max_idx)
                s_set.discard(s_sorted[max_idx] if max_idx < len(s_sorted) else list(s_set)[0])
                # Recompute s_set
                s_set = set(s_sorted)
                changed = True
            else:
                break

        # Now greedily add elements from 1 to min_start-1
        for candidate in range(1, min_start):
            if candidate in s_set:
                continue
            if is_valid_addition(s_set, s_sorted, candidate):
                pos = bisect.bisect_left(s_sorted, candidate)
                s_sorted.insert(pos, candidate)
                s_set.add(candidate)

        if len(s_sorted) > len(best_set):
            best_set = s_sorted[:]

    # WalkSAT-style MaxSAT local search with dynamic clause weighting
    s_set = set(best_set)
    n_set = len(best_set)

    # Precompute all forbidden triples (a, b, c) where a < b <= c and a | (b+c)
    # We'll use this to quickly check violations
    # For efficiency, store for each 'a' the set of (b, c) pairs that violate
    # Actually, let's just compute violations on the fly but cache per-element

    # Build a list of all potential triples for clause weighting
    # Clause: NOT(x_a AND x_b AND x_c) for each violating triple
    # In MaxSAT, we want to maximize sum of x_i subject to no clause being violated
    # WalkSAT: flip one variable at a time, choosing based on clause weights

    # For efficiency, precompute for each element i, which other elements form violations
    # violation_map[i] = list of (j, k) such that {i,j,k} forms a violation with i being 'a'
    # But we need to be careful about roles. Let's just compute full violation check.

    def get_violations(s_sorted, s_set):
        """Return list of violating triples (a, b, c) in the set."""
        violations = []
        n = len(s_sorted)
        for i in range(n):
            a = s_sorted[i]
            for j in range(i + 1, n):
                b = s_sorted[j]
                for k in range(j, n):
                    c = s_sorted[k]
                    if (b + c) % a == 0:
                        violations.append((a, b, c))
        return violations

    def violation_count_fast(s_sorted, s_set):
        """Count number of violating triples."""
        count = 0
        n = len(s_sorted)
        for i in range(n):
            a = s_sorted[i]
            # For 'a' role: find pairs (b,c) with b,c > a, b<=c, (b+c)%a==0
            for j in range(i + 1, n):
                b = s_sorted[j]
                target = (-b) % a
                # Count c in s_sorted[j:] with c % a == target
                for k in range(j, n):
                    if s_sorted[k] % a == target:
                        count += 1
        return count

    # WalkSAT parameters
    p_flip = 0.4  # probability of flipping a random variable vs. best one
    clause_weight_decay = 0.95
    max_steps = 5000

    # Dynamic clause weights
    # We'll track weight per element based on how many violations it participates in
    elem_weight = {x: 1.0 for x in range(1, N + 1)}

    current_set = best_set[:]
    current_set_set = set(current_set)
    current_size = len(current_set)

    for step in range(max_steps):
        if time.time() - start_time > deadline:
            break

        if current_size <= 1:
            break

        # Compute current violations and their involvement
        violations = get_violations(current_set, current_set_set)
        if not violations:
            # No violations, try to add elements
            others = [x for x in range(1, N + 1) if x not in current_set_set]
            random.shuffle(others)
            improved = False
            for candidate in others[:10]:
                if is_valid_addition(current_set_set, current_set, candidate):
                    pos = bisect.bisect_left(current_set, candidate)
                    current_set.insert(pos, candidate)
                    current_set_set.add(candidate)
                    current_size += 1
                    if current_size > n_set:
                        best_set = current_set[:]
                        n_set = current_size
                    improved = True
                    break
            if not improved:
                # Try removing an element to allow addition
                idx = random.randint(0, current_size - 1)
                removed = current_set.pop(idx)
                current_set_set.discard(removed)
                current_size -= 1
                elem_weight[removed] *= 1.2
                continue

        # Update clause weights based on violations
        # Elements in more violations get higher weight
        involved = {}
        for (a, b, c) in violations:
            for x in (a, b, c):
                involved[x] = involved.get(x, 0) + 1

        # Decay old weights and boost involved elements
        for x in elem_weight:
            elem_weight[x] *= clause_weight_decay
        for x, cnt in involved.items():
            elem_weight[x] += cnt * 0.5

        # Choose which element to flip
        # Option 1: Remove a heavily involved element
        # Option 2: Add a random element not in set
        if random.random() < p_flip:
            # Random flip: remove a random element
            idx = random.randint(0, current_size - 1)
            removed = current_set.pop(idx)
            current_set_set.discard(removed)
            current_size -= 1
        else:
            # Greedy: remove the element with highest weight that is in a violation
            if involved:
                # Find the element with max weight among involved
                max_w = -1
                max_elem = None
                for x in involved:
                    if x in current_set_set and elem_weight[x] > max_w:
                        max_w = elem_weight[x]
                        max_elem = x
                if max_elem is not None:
                    idx = current_set.index(max_elem)
                    current_set.pop(idx)
                    current_set_set.discard(max_elem)
                    current_size -= 1
                else:
                    idx = random.randint(0, current_size - 1)
                    removed = current_set.pop(idx)
                    current_set_set.discard(removed)
                    current_size -= 1
            else:
                idx = random.randint(0, current_size - 1)
                removed = current_set.pop(idx)
                current_set_set.discard(removed)
                current_size -= 1

        # After removal, try to add back or add new elements
        others = [x for x in range(1, N + 1) if x not in current_set_set]
        random.shuffle(others)

        # Try to add up to 2 elements
        added_count = 0
        for candidate in others[:15]:
            if added_count >= 2:
                break
            if is_valid_addition(current_set_set, current_set, candidate):
                pos = bisect.bisect_left(current_set, candidate)
                current_set.insert(pos, candidate)
                current_set_set.add(candidate)
                current_size += 1
                added_count += 1

        # Update best
        if current_size > n_set:
            best_set = current_set[:]
            n_set = current_size

    return best_set
