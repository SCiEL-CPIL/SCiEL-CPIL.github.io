import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 80
    N = 512
    B = 512

    def lcm(a, b):
        return a // gcd(a, b) * b

    # compat[i] = bitmask of j with lcm(i,j) <= B
    compat = [0] * (N + 1)
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            if lcm(i, j) <= B:
                mask |= (1 << j)
        compat[i] = mask

    def elapsed_ok():
        return time.time() < deadline

    def greedy_fill(clique, mask):
        """Add compatible vertices greedily (in order 1..N)."""
        for v in range(1, N + 1):
            if mask & (1 << v):
                continue
            if (mask & ~compat[v]) == 0:
                clique.append(v)
                mask |= (1 << v)
        return mask

    def initial_clique(order):
        chosen = []
        mask = 0
        for v in order:
            if (mask & ~compat[v]) == 0:
                chosen.append(v)
                mask |= (1 << v)
        mask = greedy_fill(chosen, mask)
        return chosen, mask

    best_clique = None
    best_mask = 0
    best_size = 0

    # Try several orderings
    orders = []
    orders.append(list(range(1, N + 1)))
    orders.append(list(range(N, 0, -1)))
    deg_order = sorted(range(1, N + 1), key=lambda x: bin(compat[x]).count('1'), reverse=True)
    orders.append(deg_order)
    orders.append(sorted(deg_order, reverse=True))

    for order in orders:
        if not elapsed_ok():
            break
        clique, mask = initial_clique(order)
        size = bin(mask).count('1')
        if size > best_size:
            best_size = size
            best_clique = list(clique)
            best_mask = mask

    # Local search on best
    def local_search(clique, mask):
        nonlocal best_size, best_clique, best_mask
        improved = True
        while improved and elapsed_ok():
            improved = False
            cur_size = len(clique)
            # Try adding (already done by greedy_fill, but re-check)
            # Try swaps: remove v, add w
            for i in range(len(clique)):
                v = clique[i]
                base_mask = mask & ~(1 << v)
                for w in range(1, N + 1):
                    if mask & (1 << w):
                        continue
                    if (base_mask & ~compat[w]) == 0:
                        # feasible swap
                        new_clique = clique[:i] + clique[i+1:] + [w]
                        new_mask = base_mask | (1 << w)
                        new_mask = greedy_fill(new_clique, new_mask)
                        new_size = bin(new_mask).count('1')
                        if new_size > cur_size:
                            clique = new_clique
                            mask = new_mask
                            cur_size = new_size
                            if new_size > best_size:
                                best_size = new_size
                                best_clique = list(clique)
                                best_mask = mask
                            improved = True
                            break
                if improved:
                    break
        return mask

    local_search(list(best_clique), best_mask)

    result = [v for v in range(1, N + 1) if best_mask & (1 << v)]
    return result
