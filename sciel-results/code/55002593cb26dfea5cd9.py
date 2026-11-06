import time
import random

MASK = (1 << 201) - 1          # bits 0..200
TARGET = 200
TARGET_BIT = 1 << 200


def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    def can_add(reachable, x):
        """Does adding x to a set with reachable-sum bitset avoid creating sum 200?"""
        new_bits = (reachable << x) & MASK
        return not (new_bits & TARGET_BIT), new_bits

    def greedy(order):
        """Greedy: include numbers in given order if it doesn't create sum 200."""
        reachable = 1  # bit 0
        selected = []
        for x in order:
            ok, new_bits = can_add(reachable, x)
            if ok:
                reachable |= new_bits
                selected.append(x)
        return selected, reachable

    def recompute(selected):
        """Recompute reachable bitset from selected set."""
        reachable = 1
        for x in selected:
            reachable |= (reachable << x) & MASK
        return reachable

    def local_improve(selected, reachable, deadline):
        """Local search: try to add excluded elements or do 1-for-2 swaps."""
        selected_set = set(selected)
        excluded = [x for x in range(1, 101) if x not in selected_set]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False

            # 1) Add any excluded element that doesn't create sum 200
            for x in excluded[:]:
                ok, new_bits = can_add(reachable, x)
                if ok:
                    selected.append(x)
                    selected_set.add(x)
                    excluded.remove(x)
                    reachable |= new_bits
                    improved = True
                    break
            if improved:
                continue

            # 2) 1-for-2 swap: remove y, add x1,x2 (net +1) if valid
            for y in selected[:]:
                if time.time() > deadline:
                    return selected, reachable
                # Compute reachable without y
                reachable_no_y = 1
                for z in selected:
                    if z != y:
                        reachable_no_y |= (reachable_no_y << z) & MASK
                
                found = False
                for i, x1 in enumerate(excluded):
                    b1 = (reachable_no_y << x1) & MASK
                    if b1 & TARGET_BIT:
                        continue
                    for x2 in excluded[i + 1:]:
                        b2 = (b1 << x2) & MASK
                        if b2 & TARGET_BIT:
                            continue
                        # Valid swap found
                        selected.remove(y)
                        selected_set.discard(y)
                        selected.extend([x1, x2])
                        selected_set.update([x1, x2])
                        excluded.remove(x1)
                        excluded.remove(x2)
                        reachable = b2
                        found = True
                        break
                    if found:
                        break
                if found:
                    improved = True
                    break
        
        return selected, reachable

    all_numbers = list(range(1, 101))
    best = []
    best_reach = 1

    # Strategy 1: Start from [66..100] (provably no 200 subset since min pair sum > 200)
    # then greedily add 65..1 in descending order
    base_order = list(range(100, 65, -1)) + list(range(65, 0, -1))
    sel, reach = greedy(base_order)
    if len(sel) > len(best):
        best, best_reach = sel, reach

    # Strategy 2: Pure descending
    sel, reach = greedy(all_numbers[::-1])
    if len(sel) > len(best):
        best, best_reach = sel, reach

    # Strategy 3: Pure ascending
    sel, reach = greedy(all_numbers)
    if len(sel) > len(best):
        best, best_reach = sel, reach

    # Strategy 4: Random shuffles to escape greedy-order bias
    rng = random.Random(12345)
    stall = 0
    while time.time() < deadline - 20:
        order = all_numbers[:]
        rng.shuffle(order)
        sel, reach = greedy(order)
        if len(sel) > len(best):
            best, best_reach = sel, reach
            stall = 0
        else:
            stall += 1
        if stall > 300:
            break

    # Final local improvement on the best found
    if time.time() < deadline - 5:
        best, _ = local_improve(list(best), best_reach, deadline)

    if not best:
        return [100]
    return best
