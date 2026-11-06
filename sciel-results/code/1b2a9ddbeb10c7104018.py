import time
import random
import sys


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit

    base = [1, 2, 5, 11]
    base_set = set(base)
    best_size = len(base)
    best_solution = base[:]

    def compute_sums(lst):
        """Compute the set of all pairwise sums a+b where a<=b from lst."""
        s = set()
        n = len(lst)
        for i in range(n):
            a = lst[i]
            for j in range(i, n):
                s.add(a + lst[j])
        return s

    def can_add(lst, sums, x):
        """Check if x can be added to lst with given sums."""
        for y in lst:
            if x + y in sums:
                return False
        if 2 * x in sums:
            return False
        return True

    def add_to(lst, sums, x):
        """Add x to lst and update sums in-place. Assumes can_add returned True."""
        lst.append(x)
        for y in lst[:-1]:
            sums.add(x + y)
        sums.add(2 * x)

    def try_2opt_swap(lst, sums, used, candidates, deadline):
        """Try swapping a non-base element with a candidate. Returns improved list or None."""
        non_base = [x for x in lst if x not in base_set]
        if not non_base:
            return None
        # Shuffle non-base elements for randomization
        random.shuffle(non_base)
        for old in non_base:
            if time.time() > deadline:
                return None
            # Remove old from working copies
            lst2 = [x for x in lst if x != old]
            used2 = used - {old}
            # Recompute sums without old
            sums2 = compute_sums(lst2)
            # Try adding each candidate
            for new_x in candidates:
                if time.time() > deadline:
                    return None
                if new_x in used2:
                    continue
                if can_add(lst2, sums2, new_x):
                    lst2.append(new_x)
                    lst2.sort()
                    return lst2
        return None

    def greedy_expand(lst, candidates, deadline=None):
        """Greedily expand lst by adding candidates in order if valid."""
        cur = lst[:]
        sums = compute_sums(cur)
        used = set(cur)
        for x in candidates:
            if deadline is not None and time.time() > deadline:
                break
            if x in used:
                continue
            if can_add(cur, sums, x):
                add_to(cur, sums, x)
                used.add(x)
        cur.sort()
        return cur

    def polish(lst, deadline=None):
        """Try to add any remaining candidate in ascending order."""
        cur = lst[:]
        sums = compute_sums(cur)
        used = set(cur)
        for x in range(1, 285):
            if deadline is not None and time.time() > deadline:
                break
            if x in used:
                continue
            if can_add(cur, sums, x):
                add_to(cur, sums, x)
                used.add(x)
        cur.sort()
        return cur

    def kick(lst, k=3):
        """Remove k random non-base elements, return the reduced list."""
        non_base = [x for x in lst if x not in base_set]
        if len(non_base) < k:
            return lst[:]
        to_remove = set(random.sample(non_base, k))
        return [x for x in lst if x not in to_remove]

    all_cands = [x for x in range(1, 285) if x not in base_set]

    # Phase 1: randomized greedy constructions from scratch
    for trial in range(500):
        if time.time() > deadline - 30.0:
            break
        cands = all_cands[:]
        random.shuffle(cands)
        built = greedy_expand(base[:], cands, deadline=deadline - 30.0)
        built = polish(built, deadline=deadline - 30.0)
        if len(built) > best_size:
            best_size = len(built)
            best_solution = built[:]

    # Phase 2: 2-opt swap local search + kick-and-regrow
    cur = best_solution[:]
    cur_size = len(cur)
    no_improve = 0
    max_no_improve = 40

    while time.time() < deadline - 5.0:
        r = random.random()

        if r < 0.5:
            # 2-opt swap: try to replace one non-base element with a better candidate
            sums = compute_sums(cur)
            used = set(cur)
            candidates = [x for x in range(1, 285) if x not in used]
            random.shuffle(candidates)
            # Limit candidates to speed up
            candidates = candidates[:100]
            improved = try_2opt_swap(cur, sums, used, candidates, deadline - 5.0)
            if improved is not None:
                # Polish the result
                improved = polish(improved, deadline=deadline - 5.0)
                if len(improved) > cur_size:
                    cur = improved
                    cur_size = len(cur)
                    no_improve = 0
                    if cur_size > best_size:
                        best_size = cur_size
                        best_solution = cur[:]
                    continue
            no_improve += 1

        elif r < 0.8:
            # Kick and regrow
            kick_size = random.choice([2, 3, 4])
            reduced = kick(cur, kick_size)
            cands = [x for x in range(1, 285) if x not in set(reduced)]
            random.shuffle(cands)
            grown = greedy_expand(reduced, cands, deadline=deadline - 5.0)
            grown = polish(grown, deadline=deadline - 5.0)
            if len(grown) > cur_size:
                cur = grown
                cur_size = len(grown)
                no_improve = 0
                if cur_size > best_size:
                    best_size = cur_size
                    best_solution = cur[:]
            else:
                no_improve += 1
        else:
            # Larger kick
            kick_size = random.choice([4, 6, 8])
            reduced = kick(cur, kick_size)
            cands = [x for x in range(1, 285) if x not in set(reduced)]
            random.shuffle(cands)
            grown = greedy_expand(reduced, cands, deadline=deadline - 5.0)
            grown = polish(grown, deadline=deadline - 5.0)
            if len(grown) > cur_size:
                cur = grown
                cur_size = len(grown)
                no_improve = 0
                if cur_size > best_size:
                    best_size = cur_size
                    best_solution = cur[:]
            else:
                no_improve += 1

        if no_improve > max_no_improve:
            # Fresh random start
            cands = all_cands[:]
            random.shuffle(cands)
            cur = greedy_expand(base[:], cands, deadline=deadline - 5.0)
            cur = polish(cur, deadline=deadline - 5.0)
            cur_size = len(cur)
            if cur_size > best_size:
                best_size = cur_size
                best_solution = cur[:]
            no_improve = 0

    # Final polish on best solution
    best_solution = polish(best_solution, deadline=deadline)

    return best_solution


if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res)
