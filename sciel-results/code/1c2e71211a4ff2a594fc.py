import time
import random
import sys
from collections import deque

def construct():
    t0 = time.time()
    DEADLINE = t0 + 85.0

    N = 256

    # Precompute conflict masks: cm[i] has bit j set iff (i+1)*(j+1) % ((i+1)+(j+1)) == 0
    cm = [0] * N
    for i in range(N):
        a = i + 1
        m = 0
        for j in range(N):
            if i == j:
                continue
            b = j + 1
            if (a * b) % (a + b) == 0:
                m |= (1 << j)
        cm[i] = m

    # Precompute conflict count per vertex
    conf_count = [0] * N
    for i in range(N):
        c = cm[i]
        cnt = 0
        while c:
            c &= c - 1
            cnt += 1
        conf_count[i] = cnt

    # Sort vertices by (conflict_count, value) for greedy ordering
    vertices_by_conf = sorted(range(N), key=lambda i: (conf_count[i], i + 1))
    vertices_by_conf_desc = sorted(range(N), key=lambda i: (-conf_count[i], i + 1))

    # v2 classification
    def v2(n):
        c = 0
        while n % 2 == 0 and n > 0:
            c += 1
            n //= 2
        return c

    v2_class = [v2(i + 1) for i in range(N)]
    v2_groups = {}
    for i in range(N):
        v = v2_class[i]
        if v not in v2_groups:
            v2_groups[v] = []
        v2_groups[v].append(i)

    def popcount(x):
        return x.bit_count()

    def is_valid(m):
        for i in range(N):
            if (m >> i) & 1:
                if cm[i] & m:
                    return False
        return True

    # Greedy construction from a given ordering
    def greedy(order):
        m = 0
        for e in order:
            if not (cm[e] & m):
                m |= (1 << e)
        return m

    # Multi-start greedy with various orderings
    def multi_greedy(n_starts):
        best_m = 0
        best_sz = 0
        for s in range(n_starts):
            if time.time() > DEADLINE:
                break
            # Various orderings
            if s % 5 == 0:
                order = vertices_by_conf[:]
            elif s % 5 == 1:
                order = vertices_by_conf_desc[:]
            elif s % 5 == 2:
                order = list(range(N))
            elif s % 5 == 3:
                order = list(range(N - 1, -1, -1))
            else:
                order = list(range(N))
                random.shuffle(order)
            m = greedy(order)
            sz = popcount(m)
            if sz > best_sz:
                best_m, best_sz = m, sz
        return best_m, best_sz

    # Local search with swap and 2-swap moves
    def local_search(m, max_iters, swap_ratio=0.5, two_swap_ratio=0.1):
        best = m
        bestsz = popcount(m)
        no_improve = 0
        max_no_improve = 3000

        for it in range(max_iters):
            if time.time() > DEADLINE:
                break
            if no_improve > max_no_improve:
                break

            r = random.random()
            if r < swap_ratio:
                # Try adding a single element
                # Pick a random element not in m
                for _ in range(20):
                    e = random.randrange(N)
                    if (m >> e) & 1:
                        continue
                    if not (cm[e] & m):
                        m |= (1 << e)
                        sz = popcount(m)
                        if sz > bestsz:
                            best, bestsz = m, sz
                            no_improve = 0
                        else:
                            no_improve += 1
                        break
                    else:
                        # e conflicts with m; try to swap: remove a conflicting element, add e
                        conf = cm[e] & m
                        if conf:
                            # Pick a random conflicting element to remove
                            tmp = conf
                            c = (tmp & -tmp).bit_length() - 1
                            # Also try to find a better removal candidate
                            removals = []
                            while tmp:
                                cc = (tmp & -tmp).bit_length() - 1
                                tmp &= tmp - 1
                                removals.append(cc)
                            random.shuffle(removals)
                            for c in removals[:5]:
                                trial = (m & ~(1 << c)) | (1 << e)
                                if cm[e] & (trial & ~(1 << e)):
                                    continue
                                sz = popcount(trial)
                                if sz > popcount(m):
                                    m = trial
                                    if sz > bestsz:
                                        best, bestsz = m, sz
                                        no_improve = 0
                                    else:
                                        no_improve += 1
                                    break
                                elif sz == popcount(m) and random.random() < 0.3:
                                    # Accept neutral swaps to escape plateaus
                                    m = trial
                                    no_improve += 1
                                    break
            elif r < swap_ratio + two_swap_ratio:
                # 2-swap: remove two elements, add two
                # Pick two random elements not in m
                adds = []
                for _ in range(30):
                    e = random.randrange(N)
                    if not ((m >> e) & 1) and e not in adds:
                        adds.append(e)
                    if len(adds) == 2:
                        break
                if len(adds) < 2:
                    no_improve += 1
                    continue
                e1, e2 = adds
                # Check if both can be added
                if (cm[e1] & m) and (cm[e2] & m):
                    # Find elements to remove
                    conf1 = cm[e1] & m
                    conf2 = cm[e2] & m
                    conf_union = conf1 | conf2
                    if not conf_union:
                        # Shouldn't happen
                        no_improve += 1
                        continue
                    # Try removing one from conf1 and one from conf2
                    removals1 = []
                    tmp = conf1
                    while tmp:
                        c = (tmp & -tmp).bit_length() - 1
                        tmp &= tmp - 1
                        removals1.append(c)
                    removals2 = []
                    tmp = conf2
                    while tmp:
                        c = (tmp & -tmp).bit_length() - 1
                        tmp &= tmp - 1
                        removals2.append(c)
                    random.shuffle(removals1)
                    random.shuffle(removals2)
                    for c1 in removals1[:5]:
                        for c2 in removals2[:5]:
                            if c1 == c2:
                                continue
                            trial = (m & ~(1 << c1) & ~(1 << c2)) | (1 << e1) | (1 << e2)
                            # Check validity
                            valid = True
                            for x in [e1, e2]:
                                if cm[x] & trial:
                                    valid = False
                                    break
                            if valid:
                                sz = popcount(trial)
                                if sz > popcount(m):
                                    m = trial
                                    if sz > bestsz:
                                        best, bestsz = m, sz
                                        no_improve = 0
                                    else:
                                        no_improve += 1
                                    break
                        if no_improve == 0 or popcount(m) != popcount(best):
                            if no_improve == 0:
                                break
                else:
                    no_improve += 1
            else:
                # Try removing an element and greedily re-adding
                in_m = [i for i in range(N) if (m >> i) & 1]
                if in_m:
                    c = random.choice(in_m)
                    trial = m & ~(1 << c)
                    # Greedily add elements
                    for e in vertices_by_conf[:50]:
                        if (trial >> e) & 1:
                            continue
                        if not (cm[e] & trial):
                            trial |= (1 << e)
                    sz = popcount(trial)
                    if sz > popcount(m):
                        m = trial
                        if sz > bestsz:
                            best, bestsz = m, sz
                            no_improve = 0
                        else:
                            no_improve += 1
                    else:
                        no_improve += 1
        return best, bestsz

    # Systematic polish: try all single additions and swaps
    def polish(m, max_iters=200):
        best = m
        bestsz = popcount(m)
        for _ in range(max_iters):
            if time.time() > DEADLINE:
                break
            improved = False
            # Try adding each missing element
            for e in range(N):
                if (m >> e) & 1:
                    continue
                if not (cm[e] & m):
                    m |= (1 << e)
                    sz = popcount(m)
                    if sz > bestsz:
                        best, bestsz = m, sz
                        improved = True
            if not improved:
                # Try single-element swaps
                for c in range(N):
                    if not ((m >> c) & 1):
                        continue
                    trial = m & ~(1 << c)
                    for e in range(N):
                        if (trial >> e) & 1:
                            continue
                        if not (cm[e] & trial):
                            newm = trial | (1 << e)
                            sz = popcount(newm)
                            if sz > bestsz:
                                m = newm
                                best, bestsz = m, sz
                                improved = True
                                break
                    if improved:
                        break
            if not improved:
                break
        return best, bestsz

    # Simulated annealing
    def simulated_annealing(m, max_iters, T0=1.0, cooling=0.9995):
        best = m
        bestsz = popcount(m)
        T = T0
        for it in range(max_iters):
            if time.time() > DEADLINE:
                break
            T *= cooling
            # Generate a neighbor
            r = random.random()
            if r < 0.4:
                # Add an element
                for _ in range(10):
                    e = random.randrange(N)
                    if (m >> e) & 1:
                        continue
                    if not (cm[e] & m):
                        m |= (1 << e)
                        break
                    else:
                        break
            elif r < 0.8:
                # Swap: remove one, add one
                in_m = [i for i in range(N) if (m >> i) & 1]
                if not in_m:
                    continue
                c = random.choice(in_m)
                for _ in range(10):
                    e = random.randrange(N)
                    if (m >> e) & 1:
                        continue
                    trial = (m & ~(1 << c)) | (1 << e)
                    if not (cm[e] & (trial & ~(1 << e))):
                        m = trial
                        break
                    else:
                        continue
            else:
                # Remove an element
                in_m = [i for i in range(N) if (m >> i) & 1]
                if in_m:
                    c = random.choice(in_m)
                    m &= ~(1 << c)

            sz = popcount(m)
            if sz > bestsz:
                best, bestsz = m, sz
        return best, bestsz

    # Track best overall
    best_m = 0
    best_sz = 0

    def consider(m):
        nonlocal best_m, best_sz
        if not is_valid(m):
            return
        sz = popcount(m)
        if sz > best_sz:
            best_m, best_sz = m, sz

    # Phase 1: Multi-start greedy
    g_m, g_sz = multi_greedy(50)
    consider(g_m)

    # Phase 2: Local search from best
    ls_m, ls_sz = local_search(best_m, 100000)
    consider(ls_m)

    # Phase 3: More local search with different parameters
    if time.time() < DEADLINE - 20:
        ls_m, ls_sz = local_search(best_m, 50000, swap_ratio=0.6, two_swap_ratio=0.15)
        consider(ls_m)

    # Phase 4: Simulated annealing from best
    if time.time() < DEADLINE - 15:
        sa_m, sa_sz = simulated_annealing(best_m, 50000)
        consider(sa_m)

    # Phase 5: Local search again
    if time.time() < DEADLINE - 10:
        ls_m, ls_sz = local_search(best_m, 80000)
        consider(ls_m)

    # Phase 6: Polish
    if time.time() < DEADLINE - 5:
        p_m, p_sz = polish(best_m, 300)
        consider(p_m)

    # Phase 7: Final local search
    if time.time() < DEADLINE - 3:
        ls_m, ls_sz = local_search(best_m, 30000)
        consider(ls_m)

    # Phase 8: Final polish
    if time.time() < DEADLINE - 1:
        p_m, p_sz = polish(best_m, 100)
        consider(p_m)

    # Fallback
    if best_m == 0:
        # All odds
        best_m = 0
        for i in range(N):
            if (i + 1) % 2 == 1:
                best_m |= (1 << i)

    # Final validation
    if not is_valid(best_m):
        best_m = 0
        for i in range(N):
            if (i + 1) % 2 == 1:
                best_m |= (1 << i)

    result = sorted([i + 1 for i in range(N) if (best_m >> i) & 1])
    return result
