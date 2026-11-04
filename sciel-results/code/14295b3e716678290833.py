import itertools
import time

def construct(v, k, t):
    start = time.time()
    DEADLINE = 110.0

    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {s: i for i, s in enumerate(all_tsubs)}

    # Precompute: for each (t-1)-subset S, map point x (not in S) -> t-subset index of S ∪ {x}
    t1_to_tsub = {}
    for S in itertools.combinations(range(v), t - 1):
        S_set = set(S)
        mapping = {}
        for x in range(v):
            if x not in S_set:
                new_tsub = tuple(sorted(S + (x,)))
                mapping[x] = tsub_to_idx[new_tsub]
        t1_to_tsub[S] = mapping

    cov = [0] * n_tsubs
    uncovered = set(range(n_tsubs))
    blocks = []

    def now_ok():
        return (time.time() - start) < DEADLINE

    while uncovered and now_ok():
        seed_idx = min(uncovered)
        seed = all_tsubs[seed_idx]
        block = list(seed)
        block_set = set(block)

        while len(block) < k and now_ok():
            t1_subs = list(itertools.combinations(block, t - 1))

            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for S in t1_subs:
                    idx = t1_to_tsub[S][x]
                    if idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x

            if best_x < 0 or best_gain <= 0:
                break

            block.append(best_x)
            block_set.add(best_x)

        block.sort()
        blocks.append(block)

        for s in itertools.combinations(block, t):
            idx = tsub_to_idx[s]
            cov[idx] += 1
            if cov[idx] == 1:
                uncovered.discard(idx)

        if not now_ok():
            break

    # Post-pass: remove redundant blocks
    if now_ok():
        changed = True
        while changed and now_ok():
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                if not now_ok():
                    break
                redundant = True
                for s in itertools.combinations(blocks[i], t):
                    idx = tsub_to_idx[s]
                    if cov[idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    for s in itertools.combinations(blocks[i], t):
                        idx = tsub_to_idx[s]
                        cov[idx] -= 1
                    blocks.pop(i)
                    changed = True
                    break

    return [list(b) for b in blocks]
