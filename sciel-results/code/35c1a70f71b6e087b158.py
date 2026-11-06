import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s limit

    base = [1, 2, 5, 11]
    base_set = set(base)
    max_val = 284
    all_candidates = list(range(1, max_val + 1))

    def get_sums(s):
        """Return a set of all pairwise sums a+b where a<=b from sorted list s."""
        sums = set()
        n = len(s)
        for i in range(n):
            a = s[i]
            for j in range(i, n):
                sums.add(a + s[j])
        return sums

    def can_add(s, sums, x):
        """Check if x can be added to s without creating duplicate sums."""
        for y in s:
            if x + y in sums:
                return False
        if 2 * x in sums:
            return False
        return True

    def greedy_fill(s, sums, candidates, bias='random', time_limit=None):
        s = sorted(s)
        sums = get_sums(s)
        s_set = set(s)
        avail = [c for c in candidates if c not in s_set]

        if bias == 'random':
            random.shuffle(avail)
        elif bias == 'low':
            avail.sort()
        elif bias == 'high':
            avail.sort(reverse=True)
        else:
            random.shuffle(avail)

        for x in avail:
            if time_limit is not None and time.time() > time_limit:
                break
            if can_add(s, sums, x):
                s.append(x)
                s.sort()
                sums = get_sums(s)
        return s, sums

    def try_add_two(s, sums, x, y):
        """Try to add both x and y to s (neither in s). Return (new_s, new_sums) or (None, None)."""
        if x in s or y in s or x == y:
            return None, None
        # Check x addable
        okx = True
        for z in s:
            if x + z in sums:
                okx = False
                break
        if not okx or 2 * x in sums:
            return None, None
        # Check y addable to s
        oky = True
        for z in s:
            if y + z in sums:
                oky = False
                break
        if not oky or 2 * y in sums:
            return None, None
        # Check x+y not in sums (new cross sum)
        if x + y in sums:
            return None, None
        new_s = s + [x, y]
        new_s.sort()
        new_sums = get_sums(new_s)
        return new_s, new_sums

    def two_opt_swap(current_set, current_sums):
        """Remove 2 non-base elements, then try to add 2 different elements."""
        if time.time() > deadline - 5:
            return current_set, current_sums
        non_base = [x for x in current_set if x not in base_set]
        if len(non_base) < 2:
            return current_set, current_sums
        a, b = random.sample(non_base, 2)
        rem = {a, b}
        s = [x for x in current_set if x not in rem]
        sums = get_sums(s)
        s_set = set(s)
        avail = [c for c in all_candidates if c not in s_set]
        random.shuffle(avail)
        # Try to add two elements
        added = []
        for x in avail:
            if len(added) == 2:
                break
            if can_add(s, sums, x):
                s.append(x)
                s.sort()
                sums = get_sums(s)
                added.append(x)
        return s, sums

    def kick_and_refill(current_set, current_sums, kick_size):
        if time.time() > deadline - 5:
            return current_set, current_sums
        non_base = [x for x in current_set if x not in base_set]
        if len(non_base) < kick_size:
            new_set, new_sums = greedy_fill(current_set, current_sums, all_candidates, bias='random')
            return new_set, new_sums
        to_remove = random.sample(non_base, kick_size)
        remove_set = set(to_remove)
        new_set = [x for x in current_set if x not in remove_set]
        new_sums = get_sums(new_set)
        best_new = new_set
        best_new_sums = new_sums
        for bias in ['random', 'low', 'high']:
            if time.time() > deadline - 5:
                break
            trial_set, trial_sums = greedy_fill(new_set[:], new_sums, all_candidates, bias=bias)
            if len(trial_set) > len(best_new):
                best_new = trial_set
                best_new_sums = trial_sums
        return best_new, best_new_sums

    def backtracking_extend(s, sums, target_size, max_depth=5):
        if time.time() > deadline - 5:
            return s, sums
        s = sorted(s)
        sums = get_sums(s)
        if len(s) >= target_size:
            return s, sums
        s_set = set(s)
        available = [c for c in all_candidates if c not in s_set]
        # MRV: score by how many others it blocks
        scored = []
        for x in available:
            if can_add(s, sums, x):
                block_count = 0
                for y in available:
                    if y == x:
                        continue
                    if x + y in sums:
                        block_count += 1
                scored.append((block_count, x))
        if not scored:
            return s, sums
        scored.sort()
        max_branch = min(len(scored), max(1, max_depth * 4))
        candidates_to_try = [x for _, x in scored[:max_branch]]
        random.shuffle(candidates_to_try)
        for x in candidates_to_try:
            if time.time() > deadline - 5:
                break
            if not can_add(s, sums, x):
                continue
            new_s = s + [x]
            new_s.sort()
            new_sums = get_sums(new_s)
            if len(new_s) >= target_size:
                return new_s, new_sums
            if max_depth > 1:
                ext_s, ext_sums = backtracking_extend(new_s, new_sums, target_size, max_depth - 1)
                if len(ext_s) > len(s):
                    return ext_s, ext_sums
        return s, sums

    def simulated_annealing_step(current_set, current_sums, kick_size, temperature):
        if time.time() > deadline - 5:
            return current_set, current_sums
        r = random.random()
        if r < 0.35:
            new_set, new_sums = kick_and_refill(current_set, current_sums, kick_size)
        elif r < 0.75:
            new_set, new_sums = two_opt_swap(current_set, current_sums)
        else:
            # kick 1 then try to add 2 specific
            new_set, new_sums = kick_and_refill(current_set, current_sums, 1)
        if len(new_set) >= len(current_set):
            return new_set, new_sums
        else:
            delta = len(current_set) - len(new_set)
            prob = pow(2.718281828, -delta / max(temperature, 0.1))
            if random.random() < prob:
                return new_set, new_sums
            else:
                return current_set, current_sums

    # Initial Solution
    best_set = list(base)
    best_sums = get_sums(best_set)
    best_size = len(best_set)

    biases = ['random', 'low', 'high', 'random', 'random', 'random', 'random', 'random']
    for bias in biases:
        if time.time() > deadline - 40:
            break
        s, sums = greedy_fill(list(base), get_sums(list(base)), all_candidates, bias=bias)
        if len(s) > best_size:
            best_set = s
            best_sums = sums
            best_size = len(s)

    current_set = best_set[:]
    current_sums = best_sums
    current_size = best_size

    no_improve_count = 0
    max_no_improve = 40
    temperature = 5.0
    cooling_rate = 0.998

    while time.time() < deadline - 30:
        r = random.random()
        if r < 0.25:
            kick_size = 1
        elif r < 0.6:
            kick_size = 2
        elif r < 0.85:
            kick_size = 3
        else:
            kick_size = 4

        new_set, new_sums = simulated_annealing_step(current_set, current_sums, kick_size, temperature)

        if len(new_set) >= best_size - 1 and time.time() < deadline - 15:
            target = max(best_size + 1, len(new_set) + 1)
            ext_set, ext_sums = backtracking_extend(new_set, new_sums, target, max_depth=5)
            if len(ext_set) > len(new_set):
                new_set, new_sums = ext_set, ext_sums

        if len(new_set) > best_size:
            best_set = new_set
            best_sums = new_sums
            best_size = len(new_set)
            current_set = new_set
            current_sums = new_sums
            current_size = best_size
            no_improve_count = 0
        elif len(new_set) == best_size:
            if new_set != current_set:
                current_set = new_set
                current_sums = new_sums
                current_size = len(new_set)
            no_improve_count += 1
        else:
            current_set = new_set
            current_sums = new_sums
            current_size = len(new_set)
            no_improve_count += 1

        temperature *= cooling_rate

        if no_improve_count > max_no_improve:
            current_set = best_set[:]
            current_sums = best_sums
            current_size = best_size
            no_improve_count = 0
            temperature = max(temperature, 3.0)

    # Final polish: try to extend the best set
    if time.time() < deadline - 10:
        target = best_size + 1
        ext_set, ext_sums = backtracking_extend(best_set, best_sums, target, max_depth=7)
        if len(ext_set) > best_size:
            best_set = ext_set
            best_sums = ext_sums
            best_size = len(ext_set)

    # Another round of greedy tries with the best set as seed
    if time.time() < deadline - 5:
        for _ in range(5):
            s, sums = greedy_fill(best_set[:], best_sums, all_candidates, bias='random')
            if len(s) > best_size:
                best_set = s
                best_sums = sums
                best_size = len(s)

    return sorted(best_set)

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
