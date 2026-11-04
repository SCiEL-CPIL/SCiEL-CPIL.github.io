import time
from math import log, sqrt
import random
import numpy as np

_START = time.time()
DEADLINE = _START + 118  # safety margin under 120s

def _time_left():
    return DEADLINE - time.time()

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    sums = set()
    diffs = set()
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    sum_count = len(sums)
    diff_count = len(diffs)
    if sum_count == 0:
        return -1
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1 + log(ratio) / log(D)

# Fast incremental scorer: maintain sum/diff multisets as we add/remove elements.
class FastScorer:
    def __init__(self):
        self.U = []            # sorted list
        self.sums = {}         # value -> count
        self.diffs = {}        # value -> count (symmetric, store both signs)
        self.sum_count = 0
        self.diff_count = 0
        self._build()

    def _build(self):
        self.U = []
        self.sums = {}
        self.diffs = {}
        self.sum_count = 0
        self.diff_count = 0

    def _add_sum(self, v):
        c = self.sums.get(v, 0)
        if c == 0:
            self.sum_count += 1
        self.sums[v] = c + 1

    def _rem_sum(self, v):
        c = self.sums.get(v, 0)
        if c == 1:
            del self.sums[v]
            self.sum_count -= 1
        else:
            self.sums[v] = c - 1

    def _add_diff(self, v):
        if v == 0:
            return
        for s in (v, -v):
            c = self.diffs.get(s, 0)
            if c == 0:
                self.diff_count += 1
            self.diffs[s] = c + 1

    def _rem_diff(self, v):
        if v == 0:
            return
        for s in (v, -v):
            c = self.diffs.get(s, 0)
            if c == 1:
                del self.diffs[s]
                self.diff_count -= 1
            else:
                self.diffs[s] = c - 1

    def add(self, x):
        # add x to U (assumes x not already present and x>0)
        for ui in self.U:
            self._add_sum(ui + x)
            self._add_diff(x - ui)
        self._add_sum(2 * x)
        self.U.append(x)
        self.U.sort()

    def remove(self, x):
        # remove x from U (assumes x present, x>0)
        for ui in self.U:
            if ui == x:
                continue
            self._rem_sum(ui + x)
            self._rem_diff(x - ui)
        self._rem_sum(2 * x)
        self.U.remove(x)

    def score(self):
        if self.sum_count == 0 or self.diff_count == 0:
            return -1
        ratio = self.diff_count / self.sum_count
        if ratio <= 1:
            return -1
        max_u = self.U[-1] if self.U else 0
        D = 2 * max_u + 1
        if D <= 1:
            return -1
        return 1 + log(ratio) / log(D)

def build_scorer(U):
    sc = FastScorer()
    for x in sorted(U):
        if x > 0:
            sc.add(x)
    return sc

def greedy_collision_construct(max_val, target_n, seed_rng):
    """
    Greedy: start with 0,1. Repeatedly add the candidate x that maximizes
    the resulting score, but penalize candidates that create 'collisions'
    (sums/diffs that already exist, which inflate |U+U| without helping |U-U|).
    We approximate the marginal collision penalty cheaply.
    """
    rng = seed_rng
    U = [0, 1]
    sc = build_scorer(U)
    base_score = sc.score()

    for _ in range(target_n - 2):
        if _time_left() < 25:
            break
        # candidate pool: small integers near current scale
        cur_max = max(U)
        lo = 1
        hi = max_val
        # focus candidates in a band that keeps max small relative to diff spread
        # prefer x not too large (to keep D small) but new
        cand_lo = max(1, cur_max // 4)
        cand_hi = min(hi, max(cur_max * 2, cur_max + 20))
        # build a candidate list, sample if too large
        if cand_hi - cand_lo > 400:
            cands = [cand_lo] + rng.sample(range(cand_lo + 1, cand_hi + 1), 300) + [cand_hi]
        else:
            cands = list(range(cand_lo, cand_hi + 1))
        U_set = set(U)
        best_x = None
        best_score = base_score
        for x in cands:
            if x in U_set:
                continue
            if x <= 0:
                continue
            # cheap collision estimate: count how many new sums are duplicates
            # and how many new diffs are duplicates
            dup_sums = 0
            new_sums = 0
            dup_diffs = 0
            new_diffs = 0
            for ui in U:
                s = ui + x
                if s in sc.sums:
                    dup_sums += 1
                else:
                    new_sums += 1
                d = x - ui
                if d != 0:
                    if d in sc.diffs or -d in sc.diffs:
                        dup_diffs += 1
                    else:
                        new_diffs += 1
            s2 = 2 * x
            if s2 in sc.sums:
                dup_sums += 1
            else:
                new_sums += 1
            # net effect: sum_count += new_sums, diff_count += new_diffs
            # ratio change
            new_sum_count = sc.sum_count + new_sums
            new_diff_count = sc.diff_count + new_diffs
            if new_sum_count == 0:
                continue
            new_ratio = new_diff_count / new_sum_count
            if new_ratio <= 1:
                continue
            new_max = max(cur_max, x)
            D = 2 * new_max + 1
            if D <= 1:
                continue
            new_score = 1 + log(new_ratio) / log(D)
            if new_score > best_score:
                best_score = new_score
                best_x = x
        if best_x is None:
            break
        sc.add(best_x)
        U.append(best_x)
        base_score = sc.score()
    return U

def local_search(U, deadline, max_n=30, max_val=2000, rng=None):
    if rng is None:
        rng = random
    U = sorted(set(U))
    sc = build_scorer(U)
    current_score = sc.score()
    best_score = current_score
    best_set = list(U)
    no_improve = 0
    max_no_improve = 60
    while _time_left() > 8 and no_improve < max_no_improve:
        improved = False
        move = rng.random()
        cur_max = max(U)
        if move < 0.45:
            # ADD
            if len(U) < max_n:
                lo = max(1, cur_max // 5)
                hi = min(max_val, max(cur_max * 2, cur_max + 30))
                if hi <= lo:
                    hi = lo + 1
                cands = list(range(lo, hi + 1))
                if len(cands) > 200:
                    cands = rng.sample(cands, 200)
                rng.shuffle(cands)
                for x in cands:
                    if x in U or x <= 0:
                        continue
                    sc.add(x)
                    ns = sc.score()
                    sc.remove(x)
                    if ns > current_score:
                        current_score = ns
                        U.append(x)
                        U.sort()
                        improved = True
                        break
        elif move < 0.65:
            # REMOVE
            if len(U) > 3:
                removable = [x for x in U if x != 0]
                rng.shuffle(removable)
                for x in removable:
                    sc.remove(x)
                    ns = sc.score()
                    if ns > current_score:
                        U = [y for y in U if y != x]
                        current_score = ns
                        improved = True
                        break
                    else:
                        sc.add(x)
        else:
            # REPLACE
            if len(U) > 2:
                idx = rng.randint(1, len(U) - 1)
                old = U[idx]
                lo = max(1, old // 2)
                hi = min(max_val, old * 2 + 20)
                cands = list(range(lo, hi + 1))
                if len(cands) > 150:
                    cands = rng.sample(cands, 150)
                rng.shuffle(cands)
                for x in cands:
                    if x == old or x in U:
                        continue
                    sc.remove(old)
                    sc.add(x)
                    ns = sc.score()
                    if ns > current_score:
                        U[idx] = x
                        U.sort()
                        current_score = ns
                        improved = True
                        break
                    else:
                        sc.remove(x)
                        sc.add(old)
        if improved:
            no_improve = 0
            if current_score > best_score:
                best_score = current_score
                best_set = list(U)
        else:
            no_improve += 1
    return best_score, best_set

def final_polish(U, deadline):
    U = sorted(set(U))
    sc = build_scorer(U)
    current_score = sc.score()
    improved = True
    iters = 0
    while improved and _time_left() > 2 and iters < 400:
        improved = False
        iters += 1
        n = len(U)
        # replace each with nearby
        for i in range(1, n):
            if _time_left() < 2:
                break
            old = U[i]
            for delta in range(-12, 13):
                if delta == 0:
                    continue
                nv = old + delta
                if nv <= 0 or nv in U:
                    continue
                sc.remove(old)
                sc.add(nv)
                ns = sc.score()
                if ns > current_score:
                    U[i] = nv
                    U.sort()
                    current_score = ns
                    improved = True
                    break
                else:
                    sc.remove(nv)
                    sc.add(old)
            if improved:
                break
        # add
        if not improved and n < 28:
            cur_max = max(U)
            for nv in range(max(1, cur_max - 12), cur_max + 13):
                if nv <= 0 or nv in U:
                    continue
                sc.add(nv)
                ns = sc.score()
                if ns > current_score:
                    U.append(nv)
                    U.sort()
                    current_score = ns
                    improved = True
                    break
                else:
                    sc.remove(nv)
        # remove
        if not improved and n > 3:
            for i in range(1, n):
                x = U[i]
                sc.remove(x)
                ns = sc.score()
                if ns > current_score:
                    U = [y for y in U if y != x]
                    current_score = ns
                    improved = True
                    break
                else:
                    sc.add(x)
    return U, current_score

def geometric_set(b, m):
    U = [0]
    val = 1.0
    for k in range(1, m + 1):
        U.append(int(val))
        val *= b
    return sorted(set(U))

def construct():
    rng = random.Random(12345)
    best_set = [0, 1, 4, 9, 11]
    best_score = score_set(best_set)

    seeds = []

    # 1. Geometric seeds
    for b in [1.5, 1.7, 1.8, 2.0, 2.2, 2.5, 3.0]:
        for m in range(4, 14):
            U = geometric_set(b, m)
            if len(U) < 2:
                continue
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U
            if s > 1.0:
                seeds.append(U)

    # 2. Greedy collision construction (the core new strategy)
    for trial in range(4):
        if _time_left() < 40:
            break
        g = greedy_collision_construct(1500, 18, rng)
        if len(g) >= 2:
            s = score_set(g)
            if s > best_score:
                best_score = s
                best_set = g
            if s > 1.0:
                seeds.append(g)

    # 3. Digit-subset seeds (base expansion)
    for base, digits in [(2, [0, 1]), (3, [0, 1]), (10, [0, 1])]:
        if _time_left() < 35:
            break
        for length in [7, 8, 9, 10]:
            if _time_left() < 30:
                break
            for _ in range(15):
                U = {0}
                for _ in range(rng.randint(3, 9)):
                    digs = [rng.choice(digits) for _ in range(length)]
                    val = 0
                    for d in reversed(digs):
                        val = val * base + d
                    if val > 0:
                        U.add(val)
                if len(U) >= 2:
                    Us = sorted(U)
                    s = score_set(Us)
                    if s > best_score:
                        best_score = s
                        best_set = Us
                    if s > 1.0:
                        seeds.append(Us)

    if best_set not in seeds:
        seeds.insert(0, best_set)
    else:
        seeds.remove(best_set)
        seeds.insert(0, best_set)

    # 4. Local search over multiple seeds
    restarts = 0
    while _time_left() > 15:
        restarts += 1
        base = seeds[restarts % len(seeds)]
        U = list(base)
        ls, lset = local_search(U, DEADLINE, max_n=28, max_val=1500, rng=rng)
        if ls > best_score:
            best_score = ls
            best_set = lset
            if len(seeds) < 8:
                seeds.append(lset)
            else:
                seeds[-1] = lset

    # 5. Final high-precision polish
    if _time_left() > 3:
        ps, psc = final_polish(list(best_set), DEADLINE)
        if psc > best_score:
            best_score = psc
            best_set = ps

    return best_set

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
