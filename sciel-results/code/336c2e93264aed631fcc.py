"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
# EVOLVE-BLOCK-START
def construct():
    _t0 = time.time()
    _deadline = _t0 + 140.0  # safety margin before 150s kill

    def _elapsed():
        return time.time() - _t0

    def _sum_counts(chosen):
        """Compute sum counts for a given set (sorted list)."""
        counts = {}
        for i, a in enumerate(chosen):
            for b in chosen[i:]:
                s = a + b
                counts[s] = counts.get(s, 0) + 1
        return counts

    def _is_valid_insertion(chosen, counts, x):
        """Check if x can be added to chosen without violating B2[2]."""
        if x in _chosen_set:
            return False
        new_sums = [x + y for y in chosen] + [2 * x]
        for s in new_sums:
            if counts.get(s, 0) + 1 > 2:
                return False
        return True

    # Phase 1: Greedy baseline (fast, deterministic)
    chosen, counts = [], {}
    _chosen_set = set()
    for x in range(1, 1001):
        new_sums = [x + y for y in chosen] + [2 * x]
        ok = True
        for s in new_sums:
            if counts.get(s, 0) + 1 > 2:
                ok = False
                break
        if ok:
            chosen.append(x)
            _chosen_set.add(x)
            for s in new_sums:
                counts[s] = counts.get(s, 0) + 1

    best_size = len(chosen)
    best_chosen = list(chosen)

    if _elapsed() > _deadline - 5.0:
        return best_chosen

    # Phase 2: Local search - try to add elements not in baseline
    # Build a list of candidates that weren't selected
    baseline_set = set(chosen)
    candidates = [x for x in range(1, 1001) if x not in baseline_set]
    random.shuffle(candidates)

    # Try adding candidates one by one
    for x in candidates:
        if _elapsed() > _deadline - 10.0:
            break
        if _is_valid_insertion(chosen, counts, x):
            # Add x
            new_sums = [x + y for y in chosen] + [2 * x]
            for s in new_sums:
                counts[s] = counts.get(s, 0) + 1
            chosen.append(x)
            _chosen_set.add(x)
            chosen.sort()
            if len(chosen) > best_size:
                best_size = len(chosen)
                best_chosen = list(chosen)

    if _elapsed() > _deadline - 5.0:
        return best_chosen

    # Phase 3: Swap-based local search
    # Try swapping elements: remove one, add another
    for _iteration in range(50):
        if _elapsed() > _deadline - 10.0:
            break

        # Try random swaps
        for _trial in range(200):
            if _elapsed() > _deadline - 10.0:
                break

            if len(chosen) < 2:
                break

            # Pick a random element to remove
            idx = random.randint(0, len(chosen) - 1)
            removed = chosen[idx]

            # Remove it
            chosen.pop(idx)
            _chosen_set.discard(removed)

            # Recompute counts after removal
            # Subtract contributions of removed
            for y in chosen:
                s = removed + y
                if s in counts:
                    counts[s] -= 1
                    if counts[s] == 0:
                        del counts[s]
            if 2 * removed in counts:
                counts[2 * removed] -= 1
                if counts[2 * removed] == 0:
                    del counts[2 * removed]

            # Try to add a random candidate
            if candidates:
                x = random.choice(candidates)
                if x not in _chosen_set and _is_valid_insertion(chosen, counts, x):
                    # Accept the swap
                    new_sums = [x + y for y in chosen] + [2 * x]
                    for s in new_sums:
                        counts[s] = counts.get(s, 0) + 1
                    chosen.append(x)
                    _chosen_set.add(x)
                    chosen.sort()
                    if len(chosen) > best_size:
                        best_size = len(chosen)
                        best_chosen = list(chosen)
                else:
                    # Reject: put removed back
                    chosen.append(removed)
                    _chosen_set.add(removed)
                    for y in chosen:
                        s = removed + y
                        counts[s] = counts.get(s, 0) + 1
                    s2 = 2 * removed
                    counts[s2] = counts.get(s2, 0) + 1
                    chosen.sort()
            else:
                # No candidates, put removed back
                chosen.append(removed)
                _chosen_set.add(removed)
                for y in chosen:
                    s = removed + y
                    counts[s] = counts.get(s, 0) + 1
                s2 = 2 * removed
                counts[s2] = counts.get(s2, 0) + 1
                chosen.sort()

    if _elapsed() > _deadline - 5.0:
        return best_chosen

    # Phase 4: Second pass - try to add more from remaining candidates
    remaining = [x for x in candidates if x not in _chosen_set]
    random.shuffle(remaining)
    for x in remaining:
        if _elapsed() > _deadline - 5.0:
            break
        if _is_valid_insertion(chosen, counts, x):
            new_sums = [x + y for y in chosen] + [2 * x]
            for s in new_sums:
                counts[s] = counts.get(s, 0) + 1
            chosen.append(x)
            _chosen_set.add(x)
            chosen.sort()
            if len(chosen) > best_size:
                best_size = len(chosen)
                best_chosen = list(chosen)

    return best_chosen
# EVOLVE-BLOCK-END
