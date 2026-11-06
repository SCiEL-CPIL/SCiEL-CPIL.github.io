"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import log

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # explicit safety margin before the 90s kill

    N = 256

    # ---- Structural idea: precompute the 3-AP conflict table ----------------
    # Two distinct numbers x,y CONFLICT if some third z in [1,N] makes {x,y,z}
    # a nonconstant 3-AP, i.e. one of them is the midpoint of the other two.
    # A set S is 3-AP-free  <=>  no pair in S conflicts.
    conflicts = [set() for _ in range(N + 1)]
    for x in range(1, N + 1):
        cx = conflicts[x]
        for y in range(x + 1, N + 1):
            # x midpoint: z = 2x - y
            z = 2 * x - y
            if 1 <= z <= N and z != x and z != y:
                cx.add(y)
                conflicts[y].add(x)
                continue
            # y midpoint: z = 2y - x
            z = 2 * y - x
            if 1 <= z <= N and z != x and z != y:
                cx.add(y)
                conflicts[y].add(x)
                continue
            # (x+y)/2 midpoint
            if (x + y) % 2 == 0:
                z = (x + y) // 2
                if 1 <= z <= N and z != x and z != y:
                    cx.add(y)
                    conflicts[y].add(x)

    def is_valid_addition(selected_set, x):
        """O(|conflicts|) check via the precomputed table."""
        for a in conflicts[x]:
            if a in selected_set:
                return False
        return True

    # ---- Greedy + swap hill-climb (fast, always valid, nonempty) -----------
    def greedy(order):
        S = set()
        for x in order:
            if is_valid_addition(S, x):
                S.add(x)
        return S

    def try_swap(S, remove, add):
        if remove == add or remove not in S or add in S:
            return None
        new_set = S - {remove}
        for a in conflicts[add]:
            if a in new_set:
                return None
        return new_set | {add}

    def hill_climb(S, tlimit):
        cur = set(S)
        cur_score = sum(1.0 / x for x in cur)
        improved = True
        while improved:
            improved = False
            if time.time() > tlimit:
                break
            for remove in list(cur):
                if time.time() > tlimit:
                    break
                for add in range(1, N + 1):
                    if add in cur:
                        continue
                    ns = try_swap(cur, remove, add)
                    if ns is not None:
                        ns_score = cur_score - 1.0 / remove + 1.0 / add
                        if ns_score > cur_score + 1e-12:
                            cur = ns
                            cur_score = ns_score
                            improved = True
                            break
                if improved:
                    break
        return cur, cur_score

    best_set = greedy(range(1, N + 1))
    best_set, best_score = hill_climb(best_set, deadline - 30.0)

    # ---- Refinement: branch-and-bound over undecided elements ---------------
    # Prunes an optimistic upper bound (current sum + sum of reciprocals of all
    # remaining undecided elements) against the best known score.
    def branch_and_bound(initial, tlimit):
        best = [set(initial)]
        best_score = [sum(1.0 / x for x in best[0])]
        base_score = best_score[0]
        undecided = [x for x in range(1, N + 1) if x not in best[0]]
        undecided.sort()  # small (large-reciprocal) elements first -> good bounds

        def dfs(idx, current, cur_score, suffix):
            if time.time() > tlimit:
                return
            if cur_score + suffix[idx] <= best_score[0] + 1e-12:
                return  # optimistic bound cannot beat best
            if idx == len(undecided):
                if cur_score > best_score[0] + 1e-12:
                    best_score[0] = cur_score
                    best[0] = set(current)
                return
            x = undecided[idx]
            # branch 1: include x (if it does not conflict with current)
            ok = True
            for a in conflicts[x]:
                if a in current:
                    ok = False
                    break
            if ok:
                current.add(x)
                dfs(idx + 1, current, cur_score + 1.0 / x, suffix)
                current.remove(x)
            # branch 2: exclude x
            dfs(idx + 1, current, cur_score, suffix)

        m = len(undecided)
        suffix = [0.0] * (m + 1)
        for i in range(m - 1, -1, -1):
            suffix[i] = suffix[i + 1] + 1.0 / undecided[i]
        dfs(0, set(), 0.0, suffix)
        return best[0], best_score[0]

    if time.time() < deadline - 3.0:
        bb_set, bb_score = branch_and_bound(best_set, deadline - 3.0)
        if bb_score > best_score + 1e-12:
            best_set, best_score = bb_set, bb_score

    # ---- Final verification (recompute from the conflict table) ------------
    def is_valid(S):
        for x in S:
            for a in conflicts[x]:
                if a in S:
                    return False
        return True

    if not is_valid(best_set):
        best_set = greedy(range(1, N + 1))
    if not best_set:
        best_set = {1}

    return sorted(best_set)
# EVOLVE-BLOCK-END
