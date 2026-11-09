"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 80.0

    def compute_reachable(s):
        r = {0}
        for x in s:
            r |= {v + x for v in r if v + x <= 200}
        return r

    # Greedy: iterate from high to low
    selected = set()
    reachable = {0}
    for x in range(100, 0, -1):
        new = {s + x for s in reachable if s + x <= 200}
        if 200 not in new:
            selected.add(x)
            reachable |= new

    # Try to add excluded elements (multiple passes)
    excluded = [x for x in range(1, 101) if x not in selected]
    for _ in range(3):
        if time.time() > deadline - 5.0:
            break
        changed = False
        for x in excluded:
            if time.time() > deadline - 5.0:
                break
            if x in selected:
                continue
            new_reach = {s + x for s in reachable if s + x <= 200}
            if 200 not in new_reach:
                selected.add(x)
                reachable |= new_reach
                changed = True
        if not changed:
            break

    # Try 1-for-2 swaps: remove one, add two excluded
    if time.time() < deadline - 15.0:
        excluded_list = [x for x in range(1, 101) if x not in selected]
        for rem in list(selected):
            if time.time() > deadline - 5.0:
                break
            temp_reach = compute_reachable(selected - {rem})
            added = []
            for add in excluded_list:
                if time.time() > deadline - 5.0:
                    break
                if add in selected:
                    continue
                new_reach = {s + add for s in temp_reach if s + add <= 200}
                if 200 not in new_reach:
                    temp_reach = new_reach
                    added.append(add)
            if len(added) >= 2:
                selected.discard(rem)
                for a in added:
                    selected.add(a)
                reachable = compute_reachable(selected)

    # Try 1-for-1 swaps: remove one, add one excluded (net zero, but may enable further adds)
    if time.time() < deadline - 10.0:
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            excluded_list = [x for x in range(1, 101) if x not in selected]
            for rem in list(selected):
                if time.time() > deadline - 5.0:
                    break
                temp_reach = compute_reachable(selected - {rem})
                # Try to add as many excluded as possible after removing rem
                added = []
                for add in excluded_list:
                    if time.time() > deadline - 5.0:
                        break
                    if add in selected:
                        continue
                    new_reach = {s + add for s in temp_reach if s + add <= 200}
                    if 200 not in new_reach:
                        temp_reach = new_reach
                        added.append(add)
                if len(added) > 1:
                    selected.discard(rem)
                    for a in added:
                        selected.add(a)
                    reachable = compute_reachable(selected)
                    improved = True

    # Final exhaustive add pass
    if time.time() < deadline - 5.0:
        excluded_list = [x for x in range(1, 101) if x not in selected]
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            for x in excluded_list:
                if time.time() > deadline - 5.0:
                    break
                if x in selected:
                    continue
                new_reach = {s + x for s in reachable if s + x <= 200}
                if 200 not in new_reach:
                    selected.add(x)
                    reachable |= new_reach
                    changed = True

    return list(selected)
# EVOLVE-BLOCK-END
