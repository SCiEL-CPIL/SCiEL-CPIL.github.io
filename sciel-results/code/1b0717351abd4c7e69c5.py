import time
import random
from math import lcm
from functools import reduce

UNIVERSE = list(range(1, 25))
L = reduce(lcm, UNIVERSE)
N = 24

def subset_sum_check(target, weights):
    if target < 0:
        return False
    if target == 0:
        return True
    if not weights:
        return False
    if sum(weights) < target:
        return False
    if min(weights) > target:
        return False
    ws = sorted(weights, reverse=True)
    n = len(ws)
    if n <= 20:
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + ws[i]
        def dfs(idx, rem):
            if rem == 0:
                return True
            if idx == n or rem < 0:
                return False
            if rem > prefix[n] - prefix[idx]:
                return False
            if dfs(idx + 1, rem - ws[idx]):
                return True
            if dfs(idx + 1, rem):
                return True
            return False
        return dfs(0, target)
    else:
        ach = {0}
        for w in ws:
            ns = set()
            for s in ach:
                v = s + w
                if v <= target:
                    ns.add(v)
            ach.update(ns)
            if target in ach:
                return True
            if len(ach) > 100000:
                break
        return target in ach

def is_valid(S):
    if not S:
        return True
    for x in S:
        if subset_sum_check(L // x, [L // y for y in S if y != x]):
            return False
    return True

def construct():
    deadline = time.time() + 85
    best = list(range(13, 25))
    if not is_valid(best):
        best = [24]
    best_size = len(best)

    random.seed(42)
    orders = [sorted(UNIVERSE, reverse=True), sorted(UNIVERSE), list(reversed(UNIVERSE))]
    for _ in range(15):
        p = UNIVERSE[:]
        random.shuffle(p)
        orders.append(p)

    for order in orders:
        if time.time() > deadline:
            break
        S = []
        for x in order:
            if time.time() > deadline:
                break
            if is_valid(S + [x]):
                S = S + [x]
        if len(S) > best_size:
            best, best_size = S, len(S)

    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in UNIVERSE:
            if time.time() > deadline:
                break
            if x not in best and is_valid(best + [x]):
                best, best_size, improved = best + [x], len(best) + 1, True

    # 2-swap local search: remove one, add two
    for _ in range(50):
        if time.time() > deadline:
            break
        for i in range(len(best)):
            if time.time() > deadline:
                break
            rem = best[i]
            base = [x for x in best if x != rem]
            found = False
            for x1 in UNIVERSE:
                if time.time() > deadline:
                    break
                if x1 in base:
                    continue
                for x2 in UNIVERSE:
                    if time.time() > deadline:
                        break
                    if x2 in base or x2 == x1:
                        continue
                    cand = base + [x1, x2]
                    if len(cand) > best_size and is_valid(cand):
                        best, best_size = cand, len(cand)
                        found = True
                        break
                if found:
                    break
            if found:
                break

    # Precompute weights and use memoized subset sum for B&B
    eo = list(range(24, 0, -1))
    ew = [L // e for e in eo]
    ei = {e: i for i, e in enumerate(eo)}
    
    # Memoize subset sum checks
    ss_memo = {}
    def ss_check(target, weights_tuple):
        key = (target, weights_tuple)
        if key in ss_memo:
            return ss_memo[key]
        result = subset_sum_check(target, list(weights_tuple))
        ss_memo[key] = result
        return result
    
    bm = sum(1 << ei[x] for x in best)
    bb = best_size

    def bb_search(pos, mask, sz, sw_tuple, se_tuple):
        nonlocal bm, bb
        if time.time() > deadline:
            return
        if sz + (N - pos) <= bb:
            return
        if pos == N:
            if sz > bb:
                bb, bm = sz, mask
            return
        e, w = eo[pos], ew[pos]
        ok = True
        if sw_tuple and sum(sw_tuple) >= w and min(sw_tuple) <= w:
            if ss_check(w, sw_tuple):
                ok = False
        if ok:
            for x in se_tuple:
                tx = L // x
                if sum(sw_tuple) + w < tx:
                    continue
                if ss_check(tx, sw_tuple + (w,)):
                    ok = False
                    break
        if ok:
            bb_search(pos + 1, mask | (1 << pos), sz + 1, sw_tuple + (w,), se_tuple + (e,))
        bb_search(pos + 1, mask, sz, sw_tuple, se_tuple)

    bb_search(0, 0, 0, (), ())

    if bb > best_size:
        best = [eo[i] for i in range(N) if bm & (1 << i)]
    return best
