import time
import random

def construct():
    start = time.time()
    deadline = start + 80.0
    N = 128
    rng = random.Random(987654321)

    def time_left():
        return deadline - time.time()

    def can_add(mem, inS, x):
        # x is NOT in S. Return True if adding x creates no forbidden triple.
        # A forbidden triple must contain x in some role.
        # Case 1: x is a member. Distinct a,b in S with x+a, x+b, a+b all in S.
        #   (a+b may equal x, which is now in S.)
        for i in range(len(mem)):
            a = mem[i]
            xa = x + a
            if xa > N or not inS[xa]:
                continue
            for j in range(i + 1, len(mem)):
                b = mem[j]
                xb = x + b
                ab = a + b
                if xb > N or not inS[xb]:
                    continue
                # ab must be in S or ab == x (x is now in S)
                if ab > N:
                    continue
                if inS[ab] or ab == x:
                    return False
        # Case 2: x is a sum, x = a+b for distinct a,b in S.
        #   Need distinct c in S, c != a, c != b, such that a+c and b+c are in S
        #   (or equal to x, which is now in S).
        for i in range(len(mem)):
            a = mem[i]
            b = x - a
            if b <= 0 or b > N or b == a:
                continue
            if not inS[b]:
                continue
            for c in mem:
                if c == a or c == b:
                    continue
                ac = a + c
                bc = b + c
                ac_ok = (ac <= N and inS[ac]) or ac == x
                bc_ok = (bc <= N and inS[bc]) or bc == x
                if ac_ok and bc_ok:
                    return False
        return True

    def full_valid(mem):
        s = set(mem)
        m = sorted(mem)
        n = len(m)
        for i in range(n):
            a = m[i]
            for j in range(i + 1, n):
                b = m[j]
                ab = a + b
                if ab not in s:
                    continue
                for k in range(j + 1, n):
                    c = m[k]
                    if (a + c) in s and (b + c) in s:
                        return False
        return True

    def local_search(mem, inS):
        # Repeatedly try to add any missing element; if stuck, do swaps.
        for _ in range(80):
            if time_left() < 4.0:
                break
            added = False
            for x in range(1, N + 1):
                if time_left() < 4.0:
                    break
                if inS[x]:
                    continue
                if can_add(mem, inS, x):
                    mem.append(x)
                    inS[x] = 1
                    added = True
            if added:
                continue
            # No direct add possible; try swaps to escape local optimum.
            sw = False
            for _ in range(80):
                if time_left() < 4.0:
                    break
                rm = rng.choice(mem)
                add = rng.randint(1, N)
                if inS[add]:
                    continue
                inS[rm] = 0
                mem.remove(rm)
                if can_add(mem, inS, add):
                    mem.append(add)
                    inS[add] = 1
                    sw = True
                    break
                else:
                    mem.append(rm)
                    inS[rm] = 1
            if not sw:
                break
        return mem

    def solve_restart(order):
        mem = [i for i in range(N // 2 + 1, N + 1)]
        inS = bytearray(N + 1)
        for i in mem:
            inS[i] = 1
        # Greedy add of small numbers in the given order.
        for x in order:
            if time_left() < 4.0:
                break
            if inS[x]:
                continue
            if can_add(mem, inS, x):
                mem.append(x)
                inS[x] = 1
        # Local search.
        local_search(mem, inS)
        return mem

    base = [i for i in range(N // 2 + 1, N + 1)]
    best = base
    best_size = len(base)

    small = list(range(1, N // 2 + 1))
    orders = [list(small), list(reversed(small))]
    for _ in range(20):
        o = list(small)
        rng.shuffle(o)
        orders.append(o)

    for o in orders:
        if time_left() < 8.0:
            break
        mem = solve_restart(o)
        if len(mem) > best_size and full_valid(mem):
            best = mem
            best_size = len(mem)

    # Use remaining time to further improve the best set via local search.
    if time_left() > 10.0:
        inS = bytearray(N + 1)
        for i in best:
            inS[i] = 1
        mem = list(best)
        local_search(mem, inS)
        if len(mem) > best_size and full_valid(mem):
            best = mem
            best_size = len(mem)

    if not full_valid(best):
        best = base

    return [int(x) for x in best]
