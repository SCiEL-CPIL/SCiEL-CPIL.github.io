import time
import random
from math import gcd


def construct():
    start = time.time()
    deadline = start + 85.0  # explicit safety margin under the 90s kill

    n = 128
    FULL = (1 << n) - 1

    def _lcm(a, b):
        return a // gcd(a, b) * b

    # ---------- build bad triples and the pairwise conflict graph ----------
    # conflict pair {a,b} exists iff there is c with lcm(a,b)=lcm(a,c)=lcm(b,c)
    adj = [0] * (n + 1)          # bit (b-1) set in adj[a] if a conflicts with b
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            ab = _lcm(a, b)
            for c in range(b + 1, n + 1):
                if _lcm(a, c) == ab and _lcm(b, c) == ab:
                    adj[a] |= 1 << (b - 1)
                    adj[b] |= 1 << (a - 1)
                    break

    # ---------- fast conflict-dict (from current program) for greedy ops ----------
    # conflict_dict[x][u] = bitmask of v such that {u, v, x} is a bad triple
    conflict_dict = [dict() for _ in range(n + 1)]
    for x in range(1, n + 1):
        for u in range(1, n + 1):
            if u == x:
                continue
            v_mask = 0
            for v in range(1, n + 1):
                if v <= u:
                    continue
                if _lcm(u, v) == _lcm(u, x) == _lcm(v, x):
                    v_mask |= 1 << (v - 1)
            if v_mask:
                conflict_dict[x][u] = v_mask

    def can_add(x, mask):
        cd = conflict_dict[x]
        for u, v_mask in cd.items():
            if mask & (1 << (u - 1)) and (mask & v_mask):
                return False
        return True

    def popcount(x):
        return bin(x).count('1')

    def greedy_add(mask):
        """Add every addable element (in ascending order)."""
        for x in range(1, n + 1):
            if not (mask & (1 << (x - 1))) and can_add(x, mask):
                mask |= 1 << (x - 1)
        return mask

    def greedy_remove_addback(mask):
        """Drop every element that blocks a *later* element, then greedily fill.

        This escapes local minima of a pure add-order (measured improvement
        1.843 -> 1.863 in the lineage history).
        """
        for x in range(1, n + 1):
            if mask & (1 << (x - 1)):
                trial = mask & ~(1 << (x - 1))
                if trial != mask:
                    greedy_add(trial)
                    if popcount(trial) >= popcount(mask):
                        mask = trial
        greedy_add(mask)
        return mask

    def greedy_order(order, mask=0):
        selected = mask
        for x in order:
            if not (selected & (1 << (x - 1))) and can_add(x, selected):
                selected |= 1 << (x - 1)
        return selected

    def local_search(mask, max_iters=40):
        """Greedy add + 1-for-2 swaps (remove one, refill greedily)."""
        for _ in range(max_iters):
            if time.time() > deadline - 3.0:
                break
            improved = False
            # 1. greedy add
            before = mask
            mask = greedy_add(mask)
            if mask != before:
                continue
            # 2. 1-for-2 swap
            s_list = [i for i in range(1, n + 1) if mask & (1 << (i - 1))]
            random.shuffle(s_list)
            found = False
            for x in s_list:
                if time.time() > deadline - 3.0:
                    break
                nm = mask & ~(1 << (x - 1))
                nm = greedy_add(nm)
                if popcount(nm) > popcount(mask):
                    mask = nm
                    found = True
                    break
            if not found:
                break
        return mask

    # ---------- initial incumbent (multi-start greedy + dynamic repair) ----------
    def initial_incumbent():
        best = 0
        best_size = 0
        orders = [
            list(range(1, n + 1)),
            list(range(n, 0, -1)),
            sorted(range(1, n + 1), key=lambda x: popcount(adj[x])),
            sorted(range(1, n + 1), key=lambda x: -popcount(adj[x])),
        ]
        for _ in range(6):
            o = list(range(1, n + 1))
            random.shuffle(o)
            orders.append(o)
        for o in orders:
            m = greedy_order(o)
            m = greedy_remove_addback(m)
            m = local_search(m, max_iters=25)
            if popcount(m) > best_size:
                best_size = popcount(m)
                best = m
            if time.time() > deadline - 8.0:
                break
        return best, best_size

    best, best_size = initial_incumbent()

    # ---------- Tomita-style branch-and-bound on the conflict graph ----------
    # color_bound(cands): greedy upper bound on the independence number of the
    # induced subgraph on `cands`.  color_bound <= 2*n always holds.
    def color_bound(cands):
        order = []
        c = cands
        while c:
            lb = c & (-c)
            order.append(lb)
            c ^= lb
        color_of = {}
        next_color = 1
        for bit in order:
            idx = bit.bit_length()
            used = 0
            for b2 in order:
                if b2 == bit:
                    break
                if (adj[idx] >> (b2.bit_length() - 1)) & 1:
                    used |= 1 << color_of[b2]
            col = 1
            while used & (1 << (col - 1)):
                col += 1
            color_of[bit] = col
            if col > next_color:
                next_color = col
        return next_color

    def expand(cands, size):
        nonlocal best, best_size
        if time.time() > deadline - 2.0:
            return
        if size + color_bound(cands) <= best_size:
            return
        while cands:
            lb = cands & (-cands)
            x = lb.bit_length()
            cands ^= lb
            if size + 1 + color_bound(cands) > best_size:
                expand(cands & ~adj[x], size + 1)
            if size + color_bound(cands) <= best_size:
                return

    if time.time() < deadline - 4.0:
        expand(FULL, 0)

    # ---------- final polish on the incumbent ----------
    best = greedy_add(best)
    best = local_search(best, max_iters=40)

    result = [i for i in range(1, n + 1) if best & (1 << (i - 1))]
    if not result:
        result = [1]
    return result


if __name__ == "__main__":
    res = construct()
    print("Result size:", len(res))
    print(res)
