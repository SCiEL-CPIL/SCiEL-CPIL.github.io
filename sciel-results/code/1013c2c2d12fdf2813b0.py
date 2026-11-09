N = 256

import random
import time


def construct():
    start_time = time.time()
    deadline = start_time + 27.0  # 3s buffer before 30s limit

    def greedy_randomized():
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)

        selected = []
        sums = set()

        for x in candidates:
            added_sums = [x + a for a in selected] + [2 * x]
            # Check internal consistency of new sums
            if len(set(added_sums)) != len(added_sums):
                continue
            # Check against existing sums
            if sums.intersection(added_sums):
                continue

            selected.append(x)
            sums.update(added_sums)

        return selected

    def check_maximal(selected, sums):
        """Check if set is inclusion-maximal. Returns True if maximal."""
        sel_set = set(selected)
        for x in range(1, N + 1):
            if x in sel_set:
                continue
            added_sums = [x + a for a in selected] + [2 * x]
            if len(set(added_sums)) != len(added_sums):
                continue
            if not sums.intersection(added_sums):
                return False  # x can be added, not maximal
        return True

    best_set = None
    best_size = float('inf')

    # Phase 1: Many fast greedy restarts
    while time.time() < deadline - 10.0:
        current_set = greedy_randomized()
        if len(current_set) < best_size:
            best_size = len(current_set)
            best_set = current_set

    if best_set is None:
        # Deterministic ascending greedy as fallback
        selected, sums = [], set()
        for x in range(1, N + 1):
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        best_set = selected
        best_size = len(best_set)

    # Phase 2: Local search - try to remove elements while maintaining maximality
    # A valid Sidon set must remain Sidon after removal. Maximality may break.
    # We only keep removals if the result is still maximal.
    def try_remove(best):
        """Try to remove one element and check if result is still maximal."""
        selected = list(best)
        sel_set = set(selected)
        # Rebuild sums
        sums = set()
        for i, a in enumerate(selected):
            for b in selected[i:]:
                sums.add(a + b)

        # Try removing each element
        for idx in range(len(selected) - 1, -1, -1):
            x = selected[idx]
            new_selected = selected[:idx] + selected[idx+1:]
            if not new_selected:
                continue

            # Rebuild sums after removal
            new_sums = set()
            for i, a in enumerate(new_selected):
                for b in new_selected[i:]:
                    new_sums.add(a + b)

            # Check if new set is still maximal
            if check_maximal(new_selected, new_sums):
                return new_selected

        return None

    # Phase 3: Iterative removal with randomization
    while time.time() < deadline - 2.0:
        if not best_set:
            break
        result = try_remove(best_set)
        if result is not None and len(result) < best_size:
            best_size = len(result)
            best_set = result
        else:
            # Try randomized removals
            for _ in range(20):
                if time.time() >= deadline - 2.0:
                    break
                idx = random.randint(0, len(best_set) - 1)
                new_selected = best_set[:idx] + best_set[idx+1:]
                if not new_selected:
                    continue
                new_sums = set()
                for i, a in enumerate(new_selected):
                    for b in new_selected[i:]:
                        new_sums.add(a + b)
                if check_maximal(new_selected, new_sums):
                    best_set = new_selected
                    best_size = len(new_selected)
                    break
            else:
                # If no removal works, do a fresh greedy and try again
                if time.time() < deadline - 2.0:
                    current_set = greedy_randomized()
                    if len(current_set) < best_size:
                        best_size = len(current_set)
                        best_set = current_set

    # Final verification pass
    def verify_final(selected):
        sel_set = set(selected)
        if len(sel_set) != len(selected):
            return False
        if any(x < 1 or x > N for x in selected):
            return False
        sums = set()
        for i, a in enumerate(selected):
            for b in selected[i:]:
                s = a + b
                if s in sums:
                    return False
                sums.add(s)
        # Check maximality
        for x in range(1, N + 1):
            if x in sel_set:
                continue
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) != len(added):
                continue
            if not sums.intersection(added):
                return False
        return True

    if not verify_final(best_set):
        # Fallback: deterministic greedy
        selected, sums = [], set()
        for x in range(1, N + 1):
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        best_set = selected

    return best_set
