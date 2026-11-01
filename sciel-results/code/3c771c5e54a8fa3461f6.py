"""Constraint-driven exact-branching construction for C(24,7,3)."""
import itertools
import random
import time
import sys


def _new_block_4(v, k, t, core, avail, uncovered, deadline, rng, max_nodes=4000):
    """
    Choose `k - len(core)` points from `avail` to append to `core`, maximizing the
    number of NEW t-sets (t-sets containing the core) that become covered.
    Uses a bounded depth-(k-len(core)) DFS with an upper-bound prune and a time/node guard.
    """
    need = k - len(core)
    if need <= 0:
        return list(core)
    core_set = set(core)

    def gain_of(choice):
        # number of uncovered t-sets that contain core + choice
        cnt = 0
        for S in itertools.combinations(choice, t - len(core)):
            cand = core + list(S)
            if len(cand) == t:
                tup = tuple(sorted(cand))
                if tup in uncovered:
                    cnt += 1
        return cnt

    def upper_bound(choice):
        # optimistic: every (core+choice) t-subset that is uncovered counts
        cnt = 0
        for S in itertools.combinations(choice, t - len(core)):
            cand = core + list(S)
            if len(cand) == t:
                tup = tuple(sorted(cand))
                if tup in uncovered:
                    cnt += 1
        return cnt

    best_choice = None
    best_gain = -1
    nodes = [0]
    # Precompute avail as a list
    avail = list(avail)

    def dfs(depth, choice, cur_gain):
        if time.time() > deadline:
            return
        nodes[0] += 1
        if nodes[0] > max_nodes:
            return
        if depth == need:
            g = gain_of(choice)
            if g > best_gain:
                best_gain = g
                best_choice = list(choice)
            return
        # upper bound prune
        ub = cur_gain + upper_bound(choice)
        if ub <= best_gain:
            return
        for x in avail:
            if x in choice:
                continue
            new_choice = choice + [x]
            # incremental gain for partial choice: count uncovered t-sets fully inside core+new_choice
            inc = 0
            for S in itertools.combinations(new_choice, t - len(core)):
                cand = core + list(S)
                if len(cand) == t:
                    tup = tuple(sorted(cand))
                    if tup in uncovered:
                        inc += 1
            dfs(depth + 1, new_choice, cur_gain + inc)

    dfs(0, [], 0)
    if best_choice is None:
        # fallback: random pick
        best_choice = list(core) + list(rng.sample(avail, need))
    return best_choice


def _greedy_branching(v, k, t, rng, time_limit, max_nodes=4000):
    deadline = time.time() + time_limit
    all_tsets = list(itertools.combinations(range(v), t))
    uncovered = set(all_tsets)
    blocks = []

    while uncovered and time.time() < deadline:
        first_uncovered = min(uncovered)
        core = list(first_uncovered)
        avail = [x for x in range(v) if x not in core]
        block = _new_block_4(v, k, t, core, avail, uncovered, deadline, rng, max_nodes=max_nodes)
        block.sort()
        blocks.append(tuple(block))
        covered = set(itertools.combinations(block, t))
        uncovered -= covered
    return [list(b) for b in blocks]


def _build_tset_index(blocks, v, t):
    tset_to_blocks = {}
    for i, b in enumerate(blocks):
        for ts in itertools.combinations(b, t):
            if ts not in tset_to_blocks:
                tset_to_blocks[ts] = []
            tset_to_blocks[ts].append(i)
    return tset_to_blocks


def _local_search(blocks, v, k, t, rng, time_limit):
    deadline = time.time() + time_limit
    blocks = [list(b) for b in blocks]
    if not blocks:
        return blocks

    tset_to_blocks = _build_tset_index(blocks, v, t)

    def rebuild_index():
        nonlocal tset_to_blocks
        tset_to_blocks = _build_tset_index(blocks, v, t)

    # Phase 1: remove redundant blocks
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(blocks)):
            if time.time() >= deadline:
                break
            redundant = True
            for ts in itertools.combinations(blocks[i], t):
                covering = tset_to_blocks.get(ts, [])
                if covering == [i]:
                    redundant = False
                    break
            if redundant:
                for ts in itertools.combinations(blocks[i], t):
                    if ts in tset_to_blocks and i in tset_to_blocks[ts]:
                        tset_to_blocks[ts].remove(i)
                blocks.pop(i)
                rebuild_index()
                changed = True
                break

    # Phase 2: bounded swaps to improve coverage (try to find replacement covering critical t-sets with more extra)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(blocks)
        if n == 0:
            break
        for idx in range(n):
            if time.time() >= deadline:
                break
            old_block = blocks[idx]
            critical = []
            for ts in itertools.combinations(old_block, t):
                covering = tset_to_blocks.get(ts, [])
                if covering == [idx]:
                    critical.append(ts)
            if not critical:
                for ts in itertools.combinations(old_block, t):
                    if ts in tset_to_blocks and idx in tset_to_blocks[ts]:
                        tset_to_blocks[ts].remove(idx)
                blocks.pop(idx)
                rebuild_index()
                improved = True
                break
            best_new_block = None
            best_extra = -1
            for _ in range(25):
                if time.time() >= deadline:
                    break
                cand = sorted(rng.sample(range(v), k))
                cand_set = set(cand)
                ok = True
                for ts in critical:
                    if not set(ts).issubset(cand_set):
                        ok = False
                        break
                if not ok:
                    continue
                extra = 0
                for ts in itertools.combinations(cand, t):
                    covering = tset_to_blocks.get(ts, [])
                    others = [c for c in covering if c != idx]
                    if not others:
                        extra += 1
                if extra > best_extra:
                    best_extra = extra
                    best_new_block = cand
            if best_new_block is not None and best_extra > 0:
                for ts in itertools.combinations(old_block, t):
                    if ts in tset_to_blocks and idx in tset_to_blocks[ts]:
                        tset_to_blocks[ts].remove(idx)
                blocks[idx] = best_new_block
                for ts in itertools.combinations(best_new_block, t):
                    if ts not in tset_to_blocks:
                        tset_to_blocks[ts] = []
                    if idx not in tset_to_blocks[ts]:
                        tset_to_blocks[ts].append(idx)
                improved = True
                break
    return blocks


def _verify_covering(blocks, v, t):
    tset_to_blocks = _build_tset_index(blocks, v, t)
    for ts in itertools.combinations(range(v), t):
        if ts not in tset_to_blocks or not tset_to_blocks[ts]:
            return False
    return True


def construct(v, k, t):
    start_time = time.time()
    total_budget = 140.0
    best_blocks = None
    best_len = float('inf')

    # Phase 1: multiple constraint-driven branching greedy runs with varied node budgets/seeds
    greedy_budget = total_budget * 0.55
    ls_budget = total_budget * 0.40
    margin = total_budget * 0.05

    configs = [
        (42, 6000),
        (137, 3000),
        (274, 9000),
        (411, 2000),
        (548, 5000),
        (685, 4000),
    ]
    per_greedy_time = greedy_budget / len(configs)

    for seed, max_nodes in configs:
        elapsed = time.time() - start_time
        if elapsed > total_budget - ls_budget - margin:
            break
        rng = random.Random(seed)
        blocks = _greedy_branching(v, k, t, rng, per_greedy_time, max_nodes=max_nodes)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks

    # A couple more quick runs if time remains
    remaining_for_more = total_budget - ls_budget - margin - (time.time() - start_time)
    if remaining_for_more > 5.0:
        for seed in [722, 859]:
            if time.time() - start_time > total_budget - ls_budget - margin:
                break
            rng = random.Random(seed)
            blocks = _greedy_branching(v, k, t, rng, 4.0, max_nodes=2500)
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks

    if best_blocks is None:
        blocks = _greedy_branching(v, k, t, random.Random(1), 5.0, max_nodes=3000)
        best_blocks = blocks
        best_len = len(blocks)

    # Phase 2: local search refinement on the best
    remaining = total_budget - (time.time() - start_time) - margin
    if remaining > 10.0 and best_blocks is not None:
        rng = random.Random(12345)
        refined = _local_search(best_blocks, v, k, t, rng, remaining)
        if _verify_covering(refined, v, t):
            if len(refined) < best_len:
                best_blocks = refined
                best_len = len(refined)

    if time.time() - start_time > total_budget - 5.0:
        return best_blocks
    return best_blocks
