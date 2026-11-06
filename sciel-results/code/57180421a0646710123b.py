def _is_smooth(x):
    while x % 2 == 0:
        x //= 2
    while x % 3 == 0:
        x //= 3
    while x % 5 == 0:
        x //= 5
    return x == 1

def _find_antichain(n, max_k=6):
    # Generate all 5-smooth numbers < n
    smooths = [x for x in range(1, n) if _is_smooth(x)]
    smooths_set = set(smooths)
    
    for k in range(1, max_k + 1):
        if k == 1:
            if _is_smooth(n):
                return [n]
            continue
        if k == 2:
            for a in smooths:
                b = n - a
                if b <= a:
                    continue
                if b not in smooths_set:
                    continue
                if b % a == 0:
                    continue
                return [a, b]
            continue
        if k == 3:
            for i in range(len(smooths)):
                a = smooths[i]
                for j in range(i + 1, len(smooths)):
                    b = smooths[j]
                    c = n - a - b
                    if c <= b:
                        continue
                    if c not in smooths_set:
                        continue
                    if b % a == 0 or c % a == 0 or c % b == 0:
                        continue
                    return [a, b, c]
            continue
        if k == 4:
            for i in range(len(smooths)):
                a = smooths[i]
                for j in range(i + 1, len(smooths)):
                    b = smooths[j]
                    for l in range(j + 1, len(smooths)):
                        c = smooths[l]
                        d = n - a - b - c
                        if d <= c:
                            continue
                        if d not in smooths_set:
                            continue
                        if b % a == 0 or c % a == 0 or d % a == 0:
                            continue
                        if c % b == 0 or d % b == 0:
                            continue
                        if d % c == 0:
                            continue
                        return [a, b, c, d]
            continue
        if k == 5:
            for i in range(len(smooths)):
                a = smooths[i]
                for j in range(i + 1, len(smooths)):
                    b = smooths[j]
                    for l in range(j + 1, len(smooths)):
                        c = smooths[l]
                        for m in range(l + 1, len(smooths)):
                            d = smooths[m]
                            e = n - a - b - c - d
                            if e <= d:
                                continue
                            if e not in smooths_set:
                                continue
                            # Check all pairs
                            terms = [a, b, c, d, e]
                            valid = True
                            for p in range(5):
                                for q in range(p + 1, 5):
                                    if terms[q] % terms[p] == 0:
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if valid:
                                return terms
            continue
        if k == 6:
            for i in range(len(smooths)):
                a = smooths[i]
                for j in range(i + 1, len(smooths)):
                    b = smooths[j]
                    for l in range(j + 1, len(smooths)):
                        c = smooths[l]
                        for m in range(l + 1, len(smooths)):
                            d = smooths[m]
                            for o in range(m + 1, len(smooths)):
                                e = smooths[o]
                                f = n - a - b - c - d - e
                                if f <= e:
                                    continue
                                if f not in smooths_set:
                                    continue
                                terms = [a, b, c, d, e, f]
                                valid = True
                                for p in range(6):
                                    for q in range(p + 1, 6):
                                        if terms[q] % terms[p] == 0:
                                            valid = False
                                            break
                                    if not valid:
                                        break
                                if valid:
                                    return terms
            continue
    return None

def _precompute_solutions():
    solutions = []
    for n in range(1, 129):
        sol = _find_antichain(n)
        if sol is None:
            raise RuntimeError(f"No antichain representation found for n={n}")
        solutions.append(sol)
    return solutions

_PRECOMPUTED = _precompute_solutions()

def construct():
    return _PRECOMPUTED
