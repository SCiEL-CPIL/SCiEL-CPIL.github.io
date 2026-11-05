import itertools
import time
import random
import sys
import math

def construct(v, k, t):
    start = time.time()
    deadline = start + 140.0

    def timed():
        return time.time() > deadline

    # Precompute all t-subsets and all k-blocks
    all_tsubs = list(itertools.combinations(range(v), t))
    n_t = len(all_tsubs)
    tsub_id = {ts: i for i, ts in enumerate(all_tsubs)}

    all_blocks = list(itertools.combinations(range(v), k))
    n_b = len(all_blocks)

    # Incidence: for each block, which t-subsets does it cover
    block_covers = [set() for _ in range(n_b)]
    for i, b in enumerate(all_blocks):
        for ts in itertools.combinations(b, t):
            block_covers[i].add(tsub_id[ts])

    # Reverse: for each t-subset, which blocks cover it
    tsub_blocks = [[] for _ in range(n_t)]
    for i in range(n_b):
        for j in block_covers[i]:
            tsub_blocks[j].append(i)

    # --- Greedy initialization ---
    def greedy_init():
        remaining = set(range(n_t))
        chosen = []
        # Precompute for each t-subset: sorted list of blocks covering it
        while remaining and not timed():
            # Pick a random uncovered t-subset
            u = random.choice(tuple(remaining))
            cands = [b for b in tsub_blocks[u] if all(b not in chosen for b in [b])]
            if not cands:
                break
            # Among candidates, pick the one covering most remaining
            best_b, best_gain = -1, -1
            for b in cands:
                gain = sum(1 for j in block_covers[b] if j in remaining)
                if gain > best_gain:
                    best_gain = gain
                    best_b = b
            if best_b == -1:
                break
            chosen.append(best_b)
            remaining -= block_covers[best_b]
        return chosen, remaining

    # --- Local search: remove redundant, then swap ---
    def local_search(chosen):
        chosen_set = set(chosen)
        # Count coverage
        counts = [0] * n_t
        for b in chosen_set:
            for j in block_covers[b]:
                counts[j] += 1

        improved = True
        while improved and not timed():
            improved = False
            # Removal pass
            for b in list(chosen_set):
                # Check if removing b leaves any t-subset uncovered
                removable = True
                for j in block_covers[b]:
                    if counts[j] <= 1:
                        removable = False
                        break
                if removable:
                    chosen_set.discard(b)
                    for j in block_covers[b]:
                        counts[j] -= 1
                    improved = True
                    break

            if improved:
                continue

            # Swap pass
            for b in list(chosen_set):
                lost = [j for j in block_covers[b] if counts[j] == 1]
                if not lost:
                    continue
                if len(lost) > 4:
                    continue
                # Find a replacement block covering all 'lost'
                min_list = min((tsub_blocks[j] for j in lost), key=len)
                if not min_list:
                    continue
                found = False
                for nb in min_list:
                    if nb in chosen_set:
                        continue
                    if all(j in block_covers[nb] for j in lost):
                        # Do the swap
                        chosen_set.discard(b)
                        chosen_set.add(nb)
                        for j in block_covers[b]:
                            counts[j] -= 1
                        for j in block_covers[nb]:
                            counts[j] += 1
                        improved = True
                        found = True
                        break
                if found:
                    break

        return list(chosen_set)

    # --- Algorithm X style backtracking for repair ---
    def repair_gaps(chosen, remaining):
        """Try to cover remaining t-subsets using few blocks via backtracking."""
        if not remaining or timed():
            return chosen, remaining

        chosen_set = set(chosen)
        rem = set(remaining)

        # Sort remaining by number of covering blocks (most constrained first)
        rem_sorted = sorted(rem, key=lambda j: len(tsub_blocks[j]))

        def backtrack(rem_set):
            if timed():
                return False
            if not rem_set:
                return True

            # Most constrained element
            u = min(rem_set, key=lambda j: len(tsub_blocks[j]))
            cands = [b for b in tsub_blocks[u] if b not in chosen_set]
            if not cands:
                return False

            # Limit branching
            cands = cands[:30]

            for b in cands:
                chosen_set.add(b)
                new_rem = rem_set - block_covers[b]
                if backtrack(new_rem):
                    return True
                chosen_set.discard(b)

            return False

        if backtrack(rem):
            return list(chosen_set), set()
        return chosen, rem

    # --- Main: multiple restarts ---
    best = None
    best_len = float('inf')

    for restart in range(15):
        if timed():
            break

        chosen, remaining = greedy_init()
        if not remaining:
            chosen = local_search(chosen)
        else:
            chosen, remaining = repair_gaps(chosen, remaining)
            if remaining:
                # Fallback: greedy fill for what's left
                rem = set(remaining)
                while rem and not timed():
                    u = random.choice(tuple(rem))
                    cands = [b for b in tsub_blocks[u] if b not in set(chosen)]
                    if not cands:
                        break
                    best_b = max(cands, key=lambda b: sum(1 for j in block_covers[b] if j in rem))
                    chosen.append(best_b)
                    rem -= block_covers[best_b]
                chosen = local_search(chosen)

        if len(chosen) < best_len:
            best_len = len(chosen)
            best = chosen

        # Perturb best and re-optimize
        if best and not timed():
            perturbed = list(best)
            # Remove 2-3 random blocks
            n_remove = random.randint(2, min(3, len(perturbed)))
            for _ in range(n_remove):
                if perturbed:
                    perturbed.pop(random.randrange(len(perturbed)))
            # Check coverage and repair
            counts = [0] * n_t
            for b in perturbed:
                for j in block_covers[b]:
                    counts[j] += 1
            missing = [j for j in range(n_t) if counts[j] == 0]
            if missing:
                perturbed, _ = repair_gaps(perturbed, set(missing))
            perturbed = local_search(perturbed)
            if len(perturbed) < best_len:
                best_len = len(perturbed)
                best = perturbed

    if best is None:
        # Emergency fallback: pure greedy
        chosen, _ = greedy_init()
        best = chosen

    # Ensure we return actual blocks
    return [list(all_blocks[i]) for i in best]
