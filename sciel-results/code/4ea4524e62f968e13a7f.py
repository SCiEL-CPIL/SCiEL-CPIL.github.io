import time

def construct():
    deadline = time.time() + 85.0

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

    # Multiple passes to add excluded elements
    excluded = [x for x in range(1, 101) if x not in selected]
    for _ in range(5):
        if time.time() > deadline - 10.0:
            break
        changed = False
        for x in excluded[:]:
            if time.time() > deadline - 10.0:
                break
            if x in selected:
                continue
            new_reach = {s + x for s in reachable if s + x <= 200}
            if 200 not in new_reach:
                selected.add(x)
                reachable |= new_reach
                excluded.remove(x)
                changed = True
        if not changed:
            break

    # 1-for-2 swaps
    if time.time() < deadline - 20.0:
        excluded_list = [x for x in range(1, 101) if x not in selected]
        improved = True
        while improved and time.time() < deadline - 20.0:
            improved = False
            for rem in list(selected):
                if time.time() > deadline - 10.0:
                    break
                temp_reach = compute_reachable(selected - {rem})
                added = []
                for add in excluded_list:
                    if time.time() > deadline - 10.0:
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
                    excluded_list = [x for x in range(1, 101) if x not in selected]
                    reachable = compute_reachable(selected)
                    improved = True
                    break

    # 2-for-3 swaps
    if time.time() < deadline - 15.0:
        excluded_list = [x for x in range(1, 101) if x not in selected]
        improved = True
        while improved and time.time() < deadline - 15.0:
            improved = False
            sel_list = list(selected)
            for i in range(len(sel_list)):
                if time.time() > deadline - 10.0:
                    break
                for j in range(i + 1, len(sel_list)):
                    if time.time() > deadline - 10.0:
                        break
                    rem1, rem2 = sel_list[i], sel_list[j]
                    temp_reach = compute_reachable(selected - {rem1, rem2})
                    added = []
                    for add in excluded_list:
                        if time.time() > deadline - 10.0:
                            break
                        if add in selected:
                            continue
                        new_reach = {s + add for s in temp_reach if s + add <= 200}
                        if 200 not in new_reach:
                            temp_reach = new_reach
                            added.append(add)
                    if len(added) >= 3:
                        selected.discard(rem1)
                        selected.discard(rem2)
                        for a in added:
                            selected.add(a)
                        excluded_list = [x for x in range(1, 101) if x not in selected]
                        reachable = compute_reachable(selected)
                        improved = True
                        break
                if improved:
                    break

    # Final add pass
    if time.time() < deadline - 5.0:
        excluded_list = [x for x in range(1, 101) if x not in selected]
        for x in excluded_list:
            if time.time() > deadline - 5.0:
                break
            new_reach = {s + x for s in reachable if s + x <= 200}
            if 200 not in new_reach:
                selected.add(x)
                reachable |= new_reach

    return list(selected)
