"""Improved solver: bitset reachability + greedy + local search, with deadline guard."""
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80.0  # safety margin under 90s kill
    TARGET = 200
    MAXV = 100
    universe = list(range(1, MAXV + 1))

    # reachability as a bitset: bit s set means sum s is reachable (0..TARGET)
    # We only care up to TARGET.
    def add_to_reach(reach, x):
        # returns new reach after adding x, and whether TARGET becomes reachable
        shifted = (reach << x)
        # mask to bits 0..TARGET
        mask = (1 << (TARGET + 1)) - 1
        shifted &= mask
        new_reach = reach | shifted
        return new_reach, (new_reach >> TARGET) & 1

    # Greedy: try elements in an order; start with descending
    def greedy(order):
        reach = 1  # sum 0
        chosen = []
        for x in order:
            new_reach, hits = add_to_reach(reach, x)
            if not hits:
                reach = new_reach
                chosen.append(x)
        return chosen, reach

    # Try several orders, keep best
    best, best_reach = greedy(list(range(MAXV, 0, -1))), None
    best, best_reach = greedy(list(range(MAXV, 0, -1)))
    # also ascending, and a "large then fill small" heuristic
    cand_orders = [
        list(range(MAXV, 0, -1)),
        list(range(MAXV, 0, -1)) + list(range(1, MAXV + 1)),  # same
    ]
    for order in cand_orders:
        c, r = greedy(order)
        if len(c) > len(best):
            best, best_reach = c, r

    chosen = set(best)
    reach = best_reach

    def can_add(x):
        if x in chosen:
            return False
        nr, hits = add_to_reach(reach, x)
        return not hits

    # Local search: try adding any excluded element
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in range(1, MAXV + 1):
            if x not in chosen and can_add(x):
                chosen.add(x)
                reach, _ = add_to_reach(reach, x)
                improved = True
        # 2-opt: remove one, try to add two
        for a in list(chosen):
            if time.time() >= deadline:
                break
            # compute reach without a (recompute from scratch is safe but slow)
            # recompute reach from chosen - {a}
            rem = set(chosen)
            rem.discard(a)
            rr = 1
            ok = True
            for y in sorted(rem):
                nr, hits = add_to_reach(rr, y)
                if hits:
                    ok = False
                    break
                rr = nr
            if not ok:
                continue
            # try to add two excluded elements
            added = []
            for x in range(1, MAXV + 1):
                if x == a or x in rem:
                    continue
                nr, hits = add_to_reach(rr, x)
                if not hits:
                    # tentatively
                    rr2, _ = add_to_reach(rr, x)
                    # now try a second
                    for z in range(x + 1, MAXV + 1):
                        if z == a or z in rem:
                            continue
                        nr3, hits3 = add_to_reach(rr2, z)
                        if not hits3:
                            # accept: remove a, add x and z
                            chosen.discard(a)
                            chosen.add(x)
                            chosen.add(z)
                            # recompute reach
                            rr = 1
                            for y in sorted(chosen):
                                rr, _ = add_to_reach(rr, y)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    result = list(chosen)
    if not result:
        return [100]
    return result
# EVOLVE-BLOCK-END
