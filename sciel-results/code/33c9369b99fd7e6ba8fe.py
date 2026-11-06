import time

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s kill

    # Start with the "safe" upper set: if min(S) > 2*max(S)/3, then for any a<b<=c in S,
    # a >= min(S) > 2*max(S)/3, and b+c <= 2*max(S) < 3*a, so b+c < 3a.
    # Since b+c > a (as b>a), the only multiple of a in (a, 3a) is 2a.
    # But b+c >= (a+1) + (a+1) = 2a+2 > 2a, so b+c cannot be 2a either.
    # Thus a cannot divide b+c.
    # K = floor(2*max(S)/3) + 1. For max(S)=256, K = floor(512/3)+1 = 170+1 = 171.
    K = 171
    S = set(range(K, n + 1))

    # Greedy add smaller numbers from K-1 down to 1
    for x in range(K - 1, 0, -1):
        if time.time() > deadline - 10:
            break
        valid = True
        lst = list(S)
        m = len(lst)
        for i in range(m):
            b = lst[i]
            for j in range(i, m):
                c = lst[j]
                if (b + c) % x == 0:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            S.add(x)

    # Local search: try adding elements not in S, with conflict-based removal
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        current_size = len(S)
        for a in range(1, n + 1):
            if time.time() > deadline - 5:
                break
            if a in S:
                continue

            # Check conflicts: which elements in S would need to be removed to add a?
            conflicts = set()
            lst = list(S)
            m = len(lst)
            for i in range(m):
                b = lst[i]
                for j in range(i, m):
                    c = lst[j]
                    # Check triple (a, b, c) with a < b <= c
                    if a < b:
                        if (b + c) % a == 0:
                            conflicts.add(b)
                            conflicts.add(c)
                    # Check triple (b, a, c) with b < a <= c
                    elif b < a <= c:
                        if (a + c) % b == 0:
                            conflicts.add(b)
                            conflicts.add(c)
                    # Check triple (b, c, a) with b < c < a
                    elif b < c < a:
                        if (c + a) % b == 0:
                            conflicts.add(b)
                            conflicts.add(c)

            if len(conflicts) == 0:
                S.add(a)
                improved = True
            elif len(conflicts) < len(S) * 0.4:
                new_S = S - conflicts
                new_S.add(a)
                if len(new_S) > len(S):
                    S = new_S
                    improved = True

    # Second pass: try removing one element and adding two others
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        for rem in list(S):
            if time.time() > deadline - 3:
                break
            S_without = S - {rem}
            for add1 in range(1, n + 1):
                if add1 in S_without:
                    continue
                # Check if add1 can be added to S_without
                conflicts1 = set()
                lst = list(S_without)
                m = len(lst)
                for i in range(m):
                    b = lst[i]
                    for j in range(i, m):
                        c = lst[j]
                        if add1 < b:
                            if (b + c) % add1 == 0:
                                conflicts1.add(b)
                                conflicts1.add(c)
                        elif b < add1 <= c:
                            if (add1 + c) % b == 0:
                                conflicts1.add(b)
                                conflicts1.add(c)
                        elif b < c < add1:
                            if (c + add1) % b == 0:
                                conflicts1.add(b)
                                conflicts1.add(c)
                if len(conflicts1) == 0:
                    S_with1 = S_without | {add1}
                    for add2 in range(add1 + 1, n + 1):
                        if add2 in S_with1:
                            continue
                        # Check if add2 can be added to S_with1
                        conflicts2 = set()
                        lst2 = list(S_with1)
                        m2 = len(lst2)
                        for i in range(m2):
                            b = lst2[i]
                            for j in range(i, m2):
                                c = lst2[j]
                                if add2 < b:
                                    if (b + c) % add2 == 0:
                                        conflicts2.add(b)
                                        conflicts2.add(c)
                                elif b < add2 <= c:
                                    if (add2 + c) % b == 0:
                                        conflicts2.add(b)
                                        conflicts2.add(c)
                                elif b < c < add2:
                                    if (c + add2) % b == 0:
                                        conflicts2.add(b)
                                        conflicts2.add(c)
                        if len(conflicts2) == 0:
                            new_S = S_with1 | {add2}
                            if len(new_S) > len(S):
                                S = new_S
                                improved = True
                                break
                    if improved:
                        break
                if time.time() > deadline - 3:
                    break
            if time.time() > deadline - 3:
                break

    return list(S)
