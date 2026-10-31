import time
import math
import random
from collections import Counter

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def compute_score(U):
        """Compute the score for a set U of nonnegative integers containing 0."""
        if not U or 0 not in U:
            return 0.0
        max_u = max(U)
        if max_u == 0:
            return 0.0
        # Compute sumset and diffset
        sumset = set()
        diffset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
                diffset.add(a - b)
        size_sum = len(sumset)
        size_diff = len(diffset)
        if size_sum == 0:
            return 0.0
        ratio = size_diff / size_sum
        if ratio <= 1.0:
            return 1.0
        denom = math.log(2 * max_u + 1)
        if denom <= 0:
            return 1.0
        return 1.0 + math.log(ratio) / denom

    def compute_sum_diff(U):
        """Compute sumset and diffset sizes efficiently."""
        sumset = set()
        diffset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
                diffset.add(a - b)
        return len(sumset), len(diffset)

    def marginal_gain(U, new_elem):
        """Compute the marginal improvement in score if we add new_elem to U."""
        if new_elem in U:
            return 0.0
        U_new = sorted(U + [new_elem])
        if 0 not in U_new:
            return 0.0
        old_score = compute_score(U)
        new_score = compute_score(U_new)
        return new_score - old_score

    best_score = 0.0
    best_set = [0, 1, 3]

    # Phase 1: Generate diverse structured candidates
    candidates = []

    # Basic small sets
    basic_sets = [
        [0, 1, 3],
        [0, 1, 2, 4],
        [0, 1, 3, 4],
        [0, 1, 2, 5],
        [0, 1, 4, 6],
        [0, 1, 2, 3, 7],
        [0, 1, 3, 7],
        [0, 1, 2, 5, 10],
        [0, 1, 3, 9],
        [0, 1, 4, 13],
        [0, 1, 5, 9],
        [0, 1, 2, 4, 7],
        [0, 1, 3, 6, 9],
        [0, 1, 2, 5, 8],
        [0, 1, 4, 8, 12],
        [0, 1, 3, 7, 11],
        [0, 1, 2, 4, 8, 12],
        [0, 1, 3, 6, 10, 15],
        [0, 1, 5, 10, 15],
        [0, 1, 2, 6, 10, 14],
    ]
    candidates.extend(basic_sets)

    # Arithmetic progressions
    for max_u in range(5, 60):
        for n in range(3, min(12, max_u + 1)):
            if max_u % (n - 1) == 0:
                step = max_u // (n - 1)
                if step >= 1:
                    ap = [i * step for i in range(n)]
                    candidates.append(ap)

    # Geometric sequences
    for ratio in [2, 3, 4]:
        for max_u in range(10, 100):
            geo = [0, 1]
            while geo[-1] * ratio <= max_u:
                geo.append(geo[-1] * ratio)
            if len(geo) >= 3:
                candidates.append(list(geo))

    # Fibonacci-like
    fib = [0, 1]
    while fib[-1] <= 80:
        fib.append(fib[-1] + fib[-2])
    for n in range(3, len(fib)):
        candidates.append(fib[:n])

    # Mixed patterns: small AP + spread elements
    for max_u in range(10, 80):
        for n_ap in [3, 4, 5]:
            for step in [1, 2, 3]:
                ap_part = [i * step for i in range(n_ap)]
                if ap_part[-1] < max_u:
                    # Add some spread elements
                    remaining = max_u - ap_part[-1]
                    extra = []
                    v = ap_part[-1] + step
                    while v <= max_u and len(extra) < 3:
                        extra.append(v)
                        v += max(1, remaining // 4)
                    if extra:
                        cand = sorted(ap_part + extra + [max_u])
                        candidates.append(cand)

    # Random structured sets
    random.seed(42)
    for _ in range(300):
        max_u = random.randint(5, 60)
        n = random.randint(3, min(10, max_u))
        inner = list(range(1, max_u))
        if len(inner) >= n - 1:
            selected = random.sample(inner, n - 1)
            cand = sorted([0] + selected)
            candidates.append(cand)

    # Evaluate all candidates
    for cand in candidates:
        cand = sorted(set(cand))
        if not cand or 0 not in cand or max(cand) == 0:
            continue
        s = compute_score(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # Phase 2: Greedy marginal gain search from best set
    if time.time() < deadline - 30:
        current = list(best_set)
        current_score = best_score
        max_elements = 15
        max_val = max(current)

        improved = True
        while improved and time.time() < deadline - 30:
            improved = False

            # Try adding elements
            if len(current) < max_elements:
                best_gain = -float('inf')
                best_elem = None
                # Search range: from 1 to max_val + some margin
                search_upper = max_val + 10
                for elem in range(1, search_upper + 1):
                    if elem in current:
                        continue
                    gain = marginal_gain(current, elem)
                    if gain > best_gain:
                        best_gain = gain
                        best_elem = elem

                if best_elem is not None and best_gain > 1e-10:
                    current = sorted(current + [best_elem])
                    current_score = compute_score(current)
                    if current_score > best_score:
                        best_score = current_score
                        best_set = list(current)
                    improved = True
                    max_val = max(current)
                    continue

            # Try replacing elements
            if len(current) >= 3:
                best_replace_gain = -float('inf')
                best_replace = None
                for i in range(len(current)):
                    if current[i] == 0:
                        continue
                    old_val = current[i]
                    # Try nearby values
                    for new_val in range(max(1, old_val - 3), old_val + 4):
                        if new_val == old_val or new_val in current:
                            continue
                        cand = list(current)
                        cand[i] = new_val
                        cand = sorted(set(cand))
                        if 0 not in cand or len(cand) < len(current) - 1:
                            continue
                        s = compute_score(cand)
                        gain = s - current_score
                        if gain > best_replace_gain:
                            best_replace_gain = gain
                            best_replace = (i, new_val, cand)

                if best_replace is not None and best_replace_gain > 1e-10:
                    i, new_val, cand = best_replace
                    current = cand
                    current_score = compute_score(current)
                    if current_score > best_score:
                        best_score = current_score
                        best_set = list(current)
                    improved = True
                    max_val = max(current)
                    continue

            # Try removing elements
            if len(current) > 3:
                best_remove_gain = -float('inf')
                best_remove_idx = None
                for i in range(len(current)):
                    if current[i] == 0:
                        continue
                    cand = [x for j, x in enumerate(current) if j != i]
                    s = compute_score(cand)
                    gain = s - current_score
                    if gain > best_remove_gain:
                        best_remove_gain = gain
                        best_remove_idx = i

                if best_remove_idx is not None and best_remove_gain > 1e-10:
                    current = [x for j, x in enumerate(current) if j != best_remove_idx]
                    current_score = compute_score(current)
                    if current_score > best_score:
                        best_score = current_score
                        best_set = list(current)
                    improved = True
                    max_val = max(current)
                    continue

    # Phase 3: Hill climbing with random restarts
    if time.time() < deadline - 20:
        current = list(best_set)
        current_score = best_score
        no_improve_count = 0
        max_no_improve = 100

        while time.time() < deadline - 20 and no_improve_count < max_no_improve:
            improved = False
            op = random.randint(0, 2)

            if op == 0 and len(current) > 2:
                # Remove random element
                idx = random.randint(1, len(current) - 1)
                neighbor = [x for j, x in enumerate(current) if j != idx]
            elif op == 1 and len(current) < 15:
                # Add random element
                max_cur = max(current)
                new_val = random.randint(1, max_cur + 8)
                if new_val not in current:
                    neighbor = sorted(current + [new_val])
                else:
                    neighbor = list(current)
            else:
                # Modify random element
                if len(current) > 1:
                    idx = random.randint(1, len(current) - 1)
                    old_val = current[idx]
                    new_val = max(1, old_val + random.randint(-3, 3))
                    if new_val not in current or new_val == old_val:
                        neighbor = list(current)
                        neighbor[idx] = new_val
                        neighbor = sorted(set(neighbor))
                    else:
                        neighbor = list(current)

            if not neighbor or 0 not in neighbor:
                no_improve_count += 1
                continue

            new_score = compute_score(neighbor)
            if new_score > current_score:
                current = neighbor
                current_score = new_score
                no_improve_count = 0
                if current_score > best_score:
                    best_score = current_score
                    best_set = list(current)
            else:
                no_improve_count += 1
                # Occasionally accept worse moves
                if random.random() < 0.02 and new_score > current_score - 0.01:
                    current = neighbor
                    current_score = new_score

    # Phase 4: Final polish - try all single-element modifications
    if time.time() < deadline - 5:
        current = list(best_set)
        current_score = best_score
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False

            # Try adding each element from 1 to max+5
            max_cur = max(current)
            for elem in range(1, max_cur + 6):
                if elem in current or len(current) >= 15:
                    continue
                cand = sorted(current + [elem])
                s = compute_score(cand)
                if s > current_score:
                    current = cand
                    current_score = s
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_set = list(current)
                    break

            # Try removing each element
            if len(current) > 2:
                for i in range(len(current)):
                    if current[i] == 0:
                        continue
                    cand = [x for j, x in enumerate(current) if j != i]
                    s = compute_score(cand)
                    if s > current_score:
                        current = cand
                        current_score = s
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_set = list(current)
                        break

            # Try replacing each element with nearby values
            if len(current) >= 3:
                max_cur = max(current)
                for i in range(len(current)):
                    if current[i] == 0:
                        continue
                    old_val = current[i]
                    for new_val in range(max(1, old_val - 2), old_val + 3):
                        if new_val == old_val or new_val in current:
                            continue
                        cand = list(current)
                        cand[i] = new_val
                        cand = sorted(set(cand))
                        if 0 not in cand:
                            continue
                        s = compute_score(cand)
                        if s > current_score:
                            current = cand
                            current_score = s
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_set = list(current)
                            break
                    if improved:
                        break

    return best_set
# EVOLVE-BLOCK-END
