"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin

    target_diffs = set(range(1, 31))
    best_solution = None
    best_max_val = float('inf')

    def all_diffs(S):
        """Return list of all positive differences in sorted set S."""
        return [S[j] - S[i] for i in range(len(S)) for j in range(i + 1, len(S))]

    def is_sidon(S):
        """Check if S is a Sidon set (all positive differences distinct)."""
        d = all_diffs(S)
        return len(d) == len(set(d))

    def covers_all(S):
        """Check if S covers all differences 1..30."""
        d = set(all_diffs(S))
        return target_diffs.issubset(d)

    def objective(S):
        """Objective: maximize element (lower is better). Returns inf if invalid."""
        if not is_sidon(S):
            return float('inf')
        if not covers_all(S):
            return float('inf')
        return max(S)

    # Phase 1: Randomized constructive search to find a valid solution quickly
    def randomized_construct(max_iters=2000):
        nonlocal best_solution, best_max_val
        for it in range(max_iters):
            if time.time() > deadline:
                break
            S = [0]
            used_diffs = set()
            for step in range(30):
                if time.time() > deadline:
                    break
                missing = target_diffs - used_diffs
                if not missing:
                    break
                # Pick a random missing diff
                d = random.choice(list(missing))
                # Find all valid x such that x - y = d for some y in S,
                # and x > max(S), and x doesn't create duplicate diffs
                candidates = set()
                for y in S:
                    x = y + d
                    if x > max(S):
                        candidates.add(x)
                # Also try other x values that might cover multiple missing diffs
                for _ in range(5):
                    x = max(S) + random.randint(1, 15)
                    candidates.add(x)

                valid_cands = []
                for x in sorted(candidates):
                    new_diffs = [x - y for y in S]
                    if any(nd in used_diffs for nd in new_diffs):
                        continue
                    if len(set(new_diffs)) != len(new_diffs):
                        continue
                    valid_cands.append(x)

                if not valid_cands:
                    break

                # Choose: prefer smaller x to keep diameter small, but add randomness
                if random.random() < 0.7:
                    x = min(valid_cands)
                else:
                    x = random.choice(valid_cands[:min(5, len(valid_cands))])

                S.append(x)
                for y in S[:-1]:
                    used_diffs.add(x - y)

            if covers_all(S) and is_sidon(S):
                mx = max(S)
                if mx < best_max_val:
                    best_max_val = mx
                    best_solution = sorted(S)
                    if best_max_val <= 50:  # good enough, stop
                        break
        return best_solution

    # Phase 2: Local search to minimize max element
    def local_search(S, max_iters=5000):
        nonlocal best_solution, best_max_val
        S = sorted(S)
        cur_obj = objective(S)
        if cur_obj == float('inf'):
            return S

        for it in range(max_iters):
            if time.time() > deadline:
                break
            improved = False

            # Try removing an element (if set has > minimum size)
            if len(S) > 8:
                for i in range(len(S)):
                    if i == 0:
                        continue  # must keep 0
                    trial = S[:i] + S[i+1:]
                    if len(trial) < 8:
                        continue
                    obj = objective(trial)
                    if obj < cur_obj:
                        S = trial
                        cur_obj = obj
                        improved = True
                        break

            if improved:
                continue

            # Try moving an element to a smaller position
            for i in range(1, len(S)):
                orig = S[i]
                # Try smaller positions
                for new_val in range(S[i-1] + 1, orig):
                    if new_val in S:
                        continue
                    trial = sorted(S[:i] + [new_val] + S[i+1:])
                    obj = objective(trial)
                    if obj < cur_obj:
                        S = trial
                        cur_obj = obj
                        improved = True
                        break
                if improved:
                    break

            if not improved:
                # Try inserting a new element and removing another (swap)
                found_swap = False
                for i in range(1, len(S)):
                    for new_val in range(S[i-1] + 1, S[i] + 20):
                        if new_val in S or new_val == S[i]:
                            continue
                        trial = sorted(S[:i] + [new_val] + S[i+1:])
                        obj = objective(trial)
                        if obj < cur_obj:
                            S = trial
                            cur_obj = obj
                            found_swap = True
                            break
                    if found_swap:
                        break
                if found_swap:
                    continue

            # No improvement; try a random perturbation
            if random.random() < 0.3:
                i = random.randint(1, len(S) - 1)
                delta = random.randint(-5, 5)
                new_val = S[i] + delta
                if new_val > S[i-1] and (i == len(S)-1 or new_val < S[i+1]) and new_val not in S and new_val > 0:
                    trial = sorted(S[:i] + [new_val] + S[i+1:])
                    if objective(trial) != float('inf'):
                        S = trial
                        cur_obj = objective(S)

        mx = max(S)
        if mx < best_max_val:
            best_max_val = mx
            best_solution = list(S)
        return S

    # Phase 3: Targeted backtracking from a good starting point
    def backtrack_from(S_start, time_budget):
        nonlocal best_solution, best_max_val
        S = sorted(S_start)
        used_diffs = set(all_diffs(S))
        missing = target_diffs - used_diffs
        if not missing:
            mx = max(S)
            if mx < best_max_val:
                best_max_val = mx
                best_solution = list(S)
            return

        def bt(S, used_diffs, missing, depth):
            nonlocal best_solution, best_max_val
            if time.time() > time_budget:
                return False
            if not missing:
                mx = max(S)
                if mx < best_max_val:
                    best_max_val = mx
                    best_solution = list(S)
                return True

            n = len(S)
            remaining = len(missing)
            # Lower bound on elements needed
            t = 0
            total = 0
            while total < remaining:
                t += 1
                total += n + t - 1
            upper_bound = best_max_val - 1
            if upper_bound <= max(S):
                return False
            if upper_bound - max(S) < t:
                return False

            # MRV
            best_d = None
            best_options = None
            best_count = float('inf')
            for d in sorted(missing):
                options = set()
                for y in S:
                    x = y + d
                    if x > max(S) and x <= upper_bound:
                        new_diffs = [x - y2 for y2 in S]
                        if all(nd not in used_diffs for nd in new_diffs) and len(set(new_diffs)) == len(new_diffs):
                            options.add(x)
                if len(options) == 0:
                    return False
                if len(options) < best_count:
                    best_count = len(options)
                    best_d = d
                    best_options = options
                    if best_count == 1:
                        break

            if best_d is None:
                return False

            for x in sorted(best_options):
                if time.time() > time_budget:
                    return False
                new_diffs = [x - y for y in S]
                if any(nd in used_diffs for nd in new_diffs):
                    continue
                if len(set(new_diffs)) != len(new_diffs):
                    continue

                S.append(x)
                added = set()
                for nd in new_diffs:
                    used_diffs.add(nd)
                    added.add(nd)
                    if nd in missing:
                        missing.discard(nd)

                if bt(S, used_diffs, missing, depth + 1):
                    S.pop()
                    for d in added:
                        used_diffs.discard(d)
                    return True

                S.pop()
                for d in added:
                    used_diffs.discard(d)
            return False

        bt(S, used_diffs, missing, 0)

    # Execute phases
    # Phase 1: Get a valid solution
    randomized_construct(3000)

    if best_solution is None:
        # Emergency greedy fallback
        S = [0]
        used_diffs = set()
        for _ in range(50):
            missing = target_diffs - used_diffs
            if not missing:
                break
            d = min(missing)
            found = False
            for x in range(max(S) + 1, max(S) + 200):
                new_diffs = [x - y for y in S]
                if all(nd not in used_diffs for nd in new_diffs) and len(set(new_diffs)) == len(new_diffs):
                    S.append(x)
                    for y in S[:-1]:
                        used_diffs.add(x - y)
                    found = True
                    break
            if not found:
                break
        if covers_all(S) and is_sidon(S):
            best_solution = S
            best_max_val = max(S)

    if best_solution is None:
        # Last resort: return a known valid set (may not be optimal but valid)
        # A Sidon set covering 1..30: use quadratic residues or known construction
        # Simple: [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324, 361, 400, 441, 484, 529, 576, 625, 676, 729, 784, 841, 900, 961]
        # But these are squares - differences are 2k+1, not all 1..30.
        # Just return [0..30] won't work (not Sidon).
        # Return a valid but large set
        S = [0]
        for i in range(1, 31):
            S.append(i * i)  # squares are Sidon
        # Check if covers 1..30... squares differences: 1,3,5,7,...,59 - covers odd only
        # Need a better fallback
        # Use a known Sidon set that covers 1..30
        # Let's just construct one greedily with large gaps
        S = [0]
        used = set()
        for target in range(1, 31):
            if target in used:
                continue
            placed = False
            for x in range(max(S) + 1, max(S) + 500):
                nds = [x - y for y in S]
                if all(nd not in used for nd in nds) and len(set(nds)) == len(nds):
                    S.append(x)
                    for y in S[:-1]:
                        used.add(x - y)
                    placed = True
                    break
            if not placed:
                # Force place
                x = max(S) + 100
                while any((x - y) in used or (x - y) in [x - y2 for y2 in S] for y in S):
                    x += 1
                S.append(x)
                for y in S[:-1]:
                    used.add(x - y)
        return S

    # Phase 2: Local search
    S_ls = local_search(best_solution, max_iters=8000)

    # Phase 3: Backtrack from local search result
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        backtrack_from(S_ls, deadline - 2)

    if best_solution is not None:
        return best_solution

    return S_ls
# EVOLVE-BLOCK-END
