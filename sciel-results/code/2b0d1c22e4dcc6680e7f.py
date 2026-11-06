import time
import random
from itertools import combinations

def _is_sidon(subset):
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_size(A):
    n = len(A)
    # Check from largest down
    for k in range(n, 2, -1):
        for combo in combinations(A, k):
            if _is_sidon(combo):
                return k
    # Check size 3
    for combo in combinations(A, 3):
        if _is_sidon(combo):
            return 3
    return 2

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin (evaluator kills at 90s)

    best_set = list(range(12))
    best_score = _max_sidon_size(best_set)  # 4

    # Key structural idea: pairs {2k, 2k+1} for k=0..5
    # Within any triple, the two from same pair + one from another form a 3-AP
    # This should make every 3-subset non-Sidon => score 2
    candidates = []

    # Primary candidate: consecutive pairs
    cand = [2*k for k in range(6)] + [2*k+1 for k in range(6)]
    candidates.append(cand)

    # Variants: different pair spacings
    for base in range(0, 10):
        for gap in [3, 4, 5, 6, 7, 8]:
            cand = []
            for k in range(6):
                cand.append(base + gap*k)
                cand.append(base + gap*k + 1)
            if len(set(cand)) == 12 and all(0 <= x <= 128 for x in cand):
                candidates.append(cand)

    # Evaluate candidates
    for cand in candidates:
        if time.time() > deadline:
            break
        if len(set(cand)) != 12:
            continue
        if any(x < 0 or x > 128 for x in cand):
            continue
        score = _max_sidon_size(cand)
        if score < best_score:
            best_score = score
            best_set = cand[:]
            if best_score == 2:
                return best_set

    # Local search: try to improve by swapping elements
    random.seed(12345)
    current = best_set[:]
    current_score = best_score

    while time.time() < deadline and current_score > 2:
        improved = False
        # Try random swaps
        for _ in range(500):
            if time.time() > deadline:
                break
            i = random.randint(0, 11)
            new_val = random.randint(0, 128)
            if new_val in current:
                continue
            old_val = current[i]
            current[i] = new_val
            sz = _max_sidon_size(current)
            if sz < current_score:
                current_score = sz
                improved = True
                if sz == 2:
                    break
            else:
                current[i] = old_val

        if current_score == 2:
            break

        if not improved:
            # Try a few random restarts with pair structure
            for _ in range(10):
                if time.time() > deadline:
                    break
                # Generate a pair-structured random set
                pairs = []
                used = set()
                for _ in range(6):
                    a = random.randint(0, 127)
                    b = a + 1
                    if a not in used and b not in used and a >= 0 and b <= 128:
                        pairs.append((a, b))
                        used.add(a)
                        used.add(b)
                if len(pairs) == 6:
                    cand = [x for pair in pairs for x in pair]
                    sz = _max_sidon_size(cand)
                    if sz < current_score:
                        current = cand
                        current_score = sz
                        improved = True
                        if sz == 2:
                            break

            if not improved:
                break

    if current_score < best_score:
        best_set = current[:]
        best_score = current_score

    return best_set
