"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _check_validity(colors, n=16):
    """Check every K4 spans >= 5 distinct colours."""
    ec = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            ec[(i, j)] = colors[idx]
            idx += 1
    for a in range(n - 3):
        for b in range(a + 1, n - 2):
            for c in range(b + 1, n - 1):
                for d in range(c + 1, n):
                    if len({ec[(a,b)], ec[(a,c)], ec[(a,d)],
                            ec[(b,c)], ec[(b,d)], ec[(c,d)]}) < 5:
                        return False
    return True

def construct():
    t0 = time.time()
    deadline = t0 + 110

    best = None
    best_n = 999

    # Search linear colourings (alpha*i + beta*j) % p for p in 13..19
    for p in range(13, 20):
        for alpha in range(1, p):
            for beta in range(1, p):
                if time.time() > deadline:
                    break
                colors = [(alpha * i + beta * j) % p
                          for i in range(16) for j in range(i + 1, 16)]
                used = len(set(colors))
                if used >= best_n:
                    continue
                if _check_validity(colors):
                    best = colors
                    best_n = used
                    if best_n <= 13:
                        return colors
        if time.time() > deadline:
            break

    if best is not None:
        return best
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
# EVOLVE-BLOCK-END
