"""Rewritten program for 4-coloring complete graphs without monochromatic triangles."""
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 135

    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)

    n0 = 32
    colors0 = []
    for i in range(n0):
        for j in range(i + 1, n0):
            colors0.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)

    best_n = n0
    best_colors = colors0[:]

    def try_extend(colors, n, max_target):
        nonlocal best_n, best_colors
        current_colors = colors[:]
        current_n = n
        for target in range(current_n + 1, max_target + 1):
            if time.time() > deadline - 5:
                break
            new_v = target - 1
            edges = [None] * new_v

            def bt(pos):
                if time.time() > deadline - 5:
                    return False
                if pos == new_v:
                    for i in range(new_v):
                        for j in range(i + 1, new_v):
                            idx = i * (2 * current_n - i - 1) // 2 + (j - i - 1)
                            if edges[i] == current_colors[idx] == edges[j]:
                                return False
                    return True
                for c in range(4):
                    edges[pos] = c
                    ok = True
                    for j in range(pos):
                        idx = j * (2 * current_n - j - 1) // 2 + (pos - j - 1)
                        if current_colors[idx] == c == edges[j]:
                            ok = False
                            break
                    if ok and bt(pos + 1):
                        return True
                    edges[pos] = None
                return False

            if bt(0):
                current_colors.extend(edges)
                current_n = target
                if current_n > best_n:
                    best_n = current_n
                    best_colors = current_colors[:]
            else:
                break
        return current_n, current_colors

    try_extend(colors0, n0, 50)

    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
