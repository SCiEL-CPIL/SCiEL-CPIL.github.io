def construct():
    import time
    start = time.time()
    DEADLINE = 85
    n = 128

    aps = []
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 3 + 1):
            aps.append((a, a+d, a+2*d, a+3*d))

    pos_aps = [[] for _ in range(n + 1)]
    for idx, (a, b, c, d) in enumerate(aps):
        for p in (a, b, c, d):
            pos_aps[p].append(idx)

    colors = [-1] * (n + 1)

    def check(pos, color):
        for ap_idx in pos_aps[pos]:
            a, b, c, d = aps[ap_idx]
            others = [p for p in (a, b, c, d) if p != pos]
            if all(colors[p] != -1 for p in others):
                if len({colors[p] for p in others} | {color}) < 3:
                    return False
        return True

    def bt(pos):
        if time.time() - start > DEADLINE - 3:
            return False
        if pos > n:
            return True
        for c in range(3):  # Try 3 colors first
            if check(pos, c):
                colors[pos] = c
                if bt(pos + 1):
                    return True
                colors[pos] = -1
        return False

    if bt(1):
        return colors[1:]

    # Fallback to original greedy
    result = []
    for x in range(1, 129):
        for color in range(128):
            ok = True
            for d in range(1, (x-1)//3 + 1):
                if x-1-3*d >= 0:
                    if len({color, result[x-2-d], result[x-1-2*d], result[x-3-3*d]}) < 3:
                        ok = False
                        break
            if ok:
                result.append(color)
                break
    return result
