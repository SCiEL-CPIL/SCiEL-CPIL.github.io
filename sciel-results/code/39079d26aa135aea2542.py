import time
import random

P = {'n': 128}


def construct():
    n = P['n']
    deadline = time.time() + 85.0

    def bits_to_list(s):
        return [i + 1 for i in range(n) if (s >> i) & 1]

    def is_valid_full(s):
        elems = bits_to_list(s)
        L = len(elems)
        for i in range(L):
            a = elems[i]
            for j in range(i + 1, L):
                b = elems[j]
                ab = a + b
                if ab > n:
                    break
                if not ((s >> (ab - 1)) & 1):
                    continue
                for k in range(j + 1, L):
                    c = elems[k]
                    ac = a + c
                    bc = b + c
                    if ac > n or bc > n:
                        break
                    if ((s >> (ac - 1)) & 1) and ((s >> (bc - 1)) & 1):
                        return False
        return True

    class SetRep:
        def __init__(self, n):
            self.n = n
            self.in_set = [False] * (n + 1)
            self.k = [0] * (n + 1)  # k[x] = |{a in S : a+x <= n and a+x in S}|
            self.size = 0

        def can_add(self, x):
            return self.k[x] <= 1

        def add(self, v):
            if self.in_set[v]:
                return True
            if not self.can_add(v):
                return False
            self.in_set[v] = True
            self.size += 1
            for x in range(1, self.n + 1):
                vx = v + x
                if vx <= self.n and self.in_set[vx]:
                    self.k[x] += 1
                vx2 = v - x
                if vx2 >= 1 and self.in_set[vx2]:
                    self.k[x] += 1
            return True

        def remove(self, v):
            if not self.in_set[v]:
                return
            for x in range(1, self.n + 1):
                vx = v + x
                if vx <= self.n and self.in_set[vx]:
                    self.k[x] -= 1
                vx2 = v - x
                if vx2 >= 1 and self.in_set[vx2]:
                    self.k[x] -= 1
            self.in_set[v] = False
            self.size -= 1

        def to_bitmask(self):
            s = 0
            for v in range(1, self.n + 1):
                if self.in_set[v]:
                    s |= (1 << (v - 1))
            return s

        def to_list(self):
            return [v for v in range(1, self.n + 1) if self.in_set[v]]

        def copy(self):
            r = SetRep(self.n)
            r.in_set = self.in_set[:]
            r.k = self.k[:]
            r.size = self.size
            return r

    # Build initial set: {64, ..., 128}
    rep = SetRep(n)
    for v in range(64, 129):
        rep.in_set[v] = True
        rep.size += 1

    # Compute initial k[x]
    for x in range(1, n + 1):
        cnt = 0
        for a in range(1, n + 1):
            if rep.in_set[a] and (a + x <= n) and rep.in_set[a + x]:
                cnt += 1
        rep.k[x] = cnt

    best_size = rep.size
    best_rep = rep.copy()

    # Greedy: add small numbers 1..63 ascending
    for v in range(1, 64):
        if time.time() > deadline - 5:
            break
        rep.add(v)

    if rep.size > best_size:
        best_size = rep.size
        best_rep = rep.copy()

    # Multiple random-order greedy passes
    rng = random.Random(42)
    for trial in range(500):
        if time.time() > deadline - 12:
            break
        r = best_rep.copy()
        remaining = [v for v in range(1, 64) if not r.in_set[v]]
        rng.shuffle(remaining)
        for v in remaining:
            if time.time() > deadline - 12:
                break
            r.add(v)
        if r.size > best_size:
            best_size = r.size
            best_rep = r.copy()

    # Local search: 2-for-1 swaps
    if time.time() < deadline - 15:
        r = best_rep.copy()
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            in_list = r.to_list()
            out_list = [v for v in range(1, n + 1) if not r.in_set[v]]
            for a in in_list:
                if time.time() > deadline - 15:
                    break
                r.remove(a)
                added = []
                for b in out_list:
                    if b == a:
                        continue
                    if r.can_add(b):
                        r.add(b)
                        added.append(b)
                        if len(added) >= 2:
                            break
                if len(added) >= 2:
                    if r.size > best_size:
                        best_size = r.size
                        best_rep = r.copy()
                    improved = True
                else:
                    for b in added:
                        r.remove(b)
                    r.add(a)

    # Additional random 2-for-1
    if time.time() < deadline - 10:
        for trial in range(200):
            if time.time() > deadline - 8:
                break
            r = best_rep.copy()
            in_list = r.to_list()
            rng.shuffle(in_list)
            for a in in_list[:30]:
                if time.time() > deadline - 8:
                    break
                r.remove(a)
                out_list = [v for v in range(1, n + 1) if not r.in_set[v] and v != a]
                rng.shuffle(out_list)
                added = []
                for b in out_list:
                    if r.can_add(b):
                        r.add(b)
                        added.append(b)
                        if len(added) >= 2:
                            break
                if len(added) >= 2:
                    if r.size > best_size:
                        best_size = r.size
                        best_rep = r.copy()
                else:
                    for b in added:
                        r.remove(b)
                    r.add(a)

    # Final safety
    best_mask = best_rep.to_bitmask()
    if not is_valid_full(best_mask):
        s = 0
        for x in range(64, 129):
            s |= (1 << (x - 1))
        result = [int(v) for v in bits_to_list(s)]
    else:
        result = [int(v) for v in best_rep.to_list()]

    return result
