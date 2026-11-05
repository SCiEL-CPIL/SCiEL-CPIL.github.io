def construct():
    from math import gcd
    import random
    
    n = 256
    candidates = list(range(2, n + 1))
    
    def has_conflict(a, b):
        if a == b:
            return False
        g = gcd(a, b)
        return (a // g) * b <= n
    
    # Build conflict adjacency
    conflicts = {x: set() for x in candidates}
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a, b = candidates[i], candidates[j]
            if has_conflict(a, b):
                conflicts[a].add(b)
                conflicts[b].add(a)
    
    def reciprocal_sum(s):
        return sum(1.0 / x for x in s)
    
    def local_search(selected, current_sum):
        improved = True
        while improved:
            improved = False
            # Try adding one element
            for y in candidates:
                if y not in selected and not (conflicts[y] & selected):
                    selected.add(y)
                    current_sum += 1.0 / y
                    improved = True
                    break
            # Try single swap: remove x, add y
            for x in list(selected):
                remaining = selected - {x}
                for y in candidates:
                    if y in selected:
                        continue
                    if not (conflicts[y] & remaining):
                        delta = -1.0 / x + 1.0 / y
                        if delta > 1e-12:
                            selected = remaining | {y}
                            current_sum += delta
                            improved = True
                            break
                if improved:
                    break
            # Try double swap: remove x1,x2, add y1,y2
            if len(selected) >= 2:
                for i in range(len(selected)):
                    for j in range(i + 1, len(selected)):
                        x1 = list(selected)[i]
                        x2 = list(selected)[j]
                        remaining = selected - {x1, x2}
                        for y1 in candidates:
                            if y1 in selected:
                                continue
                            if conflicts[y1] & remaining:
                                continue
                            for y2 in candidates:
                                if y2 in selected or y2 == y1:
                                    continue
                                if conflicts[y2] & remaining or conflicts[y2] & {y1}:
                                    continue
                                delta = -1.0 / x1 - 1.0 / x2 + 1.0 / y1 + 1.0 / y2
                                if delta > 1e-12:
                                    selected = remaining | {y1, y2}
                                    current_sum += delta
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
        return selected, current_sum
    
    # Greedy initialization
    best_selected = set()
    best_sum = 0.0
    
    # Primary greedy: sort by 1/x descending
    s = set()
    for x in sorted(candidates, key=lambda v: 1.0 / v, reverse=True):
        if not (conflicts[x] & s):
            s.add(x)
    s, s_sum = local_search(s, reciprocal_sum(s))
    best_selected = s
    best_sum = s_sum
    
    # Random restarts
    random.seed(42)
    for trial in range(50):
        s = set()
        available = list(candidates)
        random.shuffle(available)
        for x in available:
            if not (conflicts[x] & s):
                s.add(x)
        s, s_sum = local_search(s, reciprocal_sum(s))
        if s_sum > best_sum:
            best_sum = s_sum
            best_selected = s
    
    return list(best_selected)
