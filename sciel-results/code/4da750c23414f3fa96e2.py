import itertools
import time
import random
import numpy as np

V = 19
K = 7
T = 4

# Precomputation: coverage matrix  (N_BLOCKS x N_SUBSETS) bool
subsets = list(itertools.combinations(range(V), T))
subset_to_idx = {s: i for i, s in enumerate(subsets)}
N_SUBSETS = len(subsets)

all_blocks = []
for block in itertools.combinations(range(V), K):
    all_blocks.append(block)
N_BLOCKS = len(all_blocks)

# Build coverage matrix
cov = np.zeros((N_BLOCKS, N_SUBSETS), dtype=bool)
for bi, block in enumerate(all_blocks):
    for sub in itertools.combinations(block, T):
        cov[bi, subset_to_idx[sub]] = True

# Per-subset list of blocks that contain it (for repair)
blocks_containing = [[] for _ in range(N_SUBSETS)]
for si in range(N_SUBSETS):
    idxs = np.nonzero(cov[:, si])[0]
    blocks_containing[si] = idxs.tolist()

# Per-block count of subsets covered (all 35)
block_gain = cov.sum(axis=1)  # = 35 for every block, but keep for clarity


def construct(v, k, t):
    if v != 19 or k != 7 or t != 4:
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        all_k_subsets = list(itertools.combinations(range(v), k))
        random.shuffle(all_k_subsets)
        for b in all_k_subsets:
            if not uncovered:
                break
            covered = set(itertools.combinations(b, t))
            if uncovered & covered:
                blocks.append(list(b))
                uncovered -= covered
        return blocks

    start = time.time()
    deadline = start + 118

    # ---------- Phase 1: Packing-first greedy core ----------
    # covered_mask: which subsets are already covered
    covered_mask = np.zeros(N_SUBSETS, dtype=bool)
    chosen = []
    chosen_set = set()

    while True:
        if time.time() > deadline - 70:
            break
        # gain per block = number of uncovered subsets it covers
        gains = cov[:, ~covered_mask].sum(axis=1)
        # skip already-chosen
        for c in chosen_set:
            gains[c] = -1
        best = int(np.argmax(gains))
        if gains[best] <= 0:
            break
        chosen.append(best)
        chosen_set.add(best)
        covered_mask |= cov[best]
        if covered_mask.all():
            break

    # ---------- Phase 2: Targeted repair for uncovered subsets ----------
    if not covered_mask.all():
        uncov = np.nonzero(~covered_mask)[0]
        # For each uncovered subset, pick a block covering it (least loaded)
        for si in uncov:
            if time.time() > deadline - 60:
                break
            cands = blocks_containing[si]
            # choose candidate that adds the most NEW coverage
            best_c = cands[0]
            best_new = -1
            # limit scan for speed
            scan = cands if len(cands) <= 40 else random.sample(cands, 40)
            for c in scan:
                if c in chosen_set:
                    continue
                new = int((cov[c] & ~covered_mask).sum())
                if new > best_new:
                    best_new = new
                    best_c = c
            chosen.append(best_c)
            chosen_set.add(best_c)
            covered_mask |= cov[best_c]

    # ---------- Phase 3: Local search polish ----------
    # maintain count of how many chosen blocks cover each subset
    def build_counts(chosen_list):
        cnt = np.zeros(N_SUBSETS, dtype=np.int32)
        for c in chosen_list:
            cnt += cov[c].astype(np.int32)
        return cnt

    counts = build_counts(chosen)
    best_len = len(chosen)
    best_chosen = chosen[:]

    def is_redundant(i):
        c = chosen[i]
        return bool((counts[cov[c]] > 1).all())

    no_improve = 0
    max_no_improve = 25
    while time.time() < deadline - 3:
        # --- prune redundant ---
        removed = False
        order = list(range(len(chosen)))
        random.shuffle(order)
        for i in order:
            if time.time() > deadline - 3:
                break
            if is_redundant(i):
                c = chosen.pop(i)
                counts[cov[c]] -= 1
                removed = True
                if len(chosen) < best_len:
                    best_len = len(chosen)
                    best_chosen = chosen[:]
                break
        if removed:
            no_improve = 0
            continue

        # --- 1-for-1 swap ---
        if len(chosen) == 0:
            break
        found = False
        for _ in range(4):
            if time.time() > deadline - 3:
                break
            i = random.randrange(len(chosen))
            c = chosen[i]
            # subsets uniquely covered by c
            uniq = np.nonzero(cov[c] & (counts == 1))[0]
            if len(uniq) == 0:
                chosen.pop(i)
                counts[cov[c]] -= 1
                if len(chosen) < best_len:
                    best_len = len(chosen)
                    best_chosen = chosen[:]
                found = True
                no_improve = 0
                break
            if len(uniq) <= 3:
                # find a block covering all uniq
                cand = blocks_containing[uniq[0]]
                for b in uniq[1:]:
                    cand = [x for x in cand if x in set(blocks_containing[b])]
                    if not cand:
                        break
                if cand:
                    cand = [x for x in cand if x != c]
                    if cand:
                        j = random.choice(cand)
                        chosen[i] = j
                        counts[cov[c]] -= 1
                        counts[cov[j]] += 1
                        found = True
                        no_improve = 0
                        break
        if found:
            continue

        # --- perturbation ---
        no_improve += 1
        if no_improve > max_no_improve:
            n_rm = random.randint(4, 9)
            to_rm = random.sample(range(len(chosen)), min(n_rm, len(chosen)))
            to_rm.sort(reverse=True)
            for i in to_rm:
                c = chosen.pop(i)
                counts[cov[c]] -= 1
            # greedy re-fill
            if time.time() < deadline - 5:
                cm = (counts > 0)
                while True:
                    if time.time() > deadline - 5:
                        break
                    gains = cov[:, ~cm].sum(axis=1)
                    best = int(np.argmax(gains))
                    if gains[best] <= 0:
                        break
                    chosen.append(best)
                    counts[cov[best]] += 1
                    cm |= cov[best]
                    if cm.all():
                        break
            no_improve = 0
            max_no_improve = min(120, max_no_improve * 2)

    # final prune
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(chosen) - 1, -1, -1):
            if is_redundant(i):
                c = chosen.pop(i)
                counts[cov[c]] -= 1
                changed = True
                if len(chosen) < best_len:
                    best_len = len(chosen)
                    best_chosen = chosen[:]

    if len(chosen) < best_len:
        best_chosen = chosen[:]

    return [list(all_blocks[i]) for i in best_chosen]
