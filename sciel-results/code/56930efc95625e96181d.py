"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np

_start_time = time.time()
_DEADLINE = 105  # safety margin before 150s limit

def _elapsed():
    return time.time() - _start_time

def _score_set(U):
    """Compute score = 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)."""
    if not U or 0 not in U:
        return -1.0
    n = len(U)
    max_u = max(U)
    if max_u == 0:
        return -1.0
    arr = np.array(U, dtype=np.int64)
    sums = arr[:, None] + arr[None, :]
    diffs = arr[:, None] - arr[None, :]
    num_sums = len(np.unique(sums))
    num_diffs = len(np.unique(diffs))
    if num_sums == 0:
        return -1.0
    ratio = num_diffs / num_sums
    if ratio <= 1:
        return -1.0
    log_denom = math.log(2 * max_u + 1)
    if log_denom <= 0:
        return -1.0
    return 1.0 + math.log(ratio) / log_denom

def _golomb_seed(n, max_val):
    """Greedy Golomb ruler: pick n elements with distinct pairwise differences."""
    U = [0]
    diffs = set()
    for _ in range(1, n):
        best_v = None
        for v in range(1, max_val + 1):
            if v in U:
                continue
            ok = True
            for a in U:
                d = abs(v - a)
                if d in diffs:
                    ok = False
                    break
            if ok:
                best_v = v
                break
        if best_v is None:
            break
        U.append(best_v)
        for a in U:
            diffs.add(abs(best_v - a))
    return U

def _local_search(initial, max_steps=2000):
    """Local search: add/remove/swap elements to improve score."""
    best_U = list(initial)
    best_score = _score_set(best_U)
    current_U = list(best_U)
    current_score = best_score
    stall_count = 0

    for step in range(max_steps):
        if _elapsed() > _DEADLINE:
            break
        max_val = max(current_U) if current_U else 0
        improved = False
        cur_set = set(current_U)

        # Try adding a new element
        candidates = []
        for delta in range(1, 50):
            candidates.append(max_val + delta)
        for delta in range(1, 30):
            candidates.append(max_val * 2 + delta)
        for delta in range(1, 15):
            candidates.append(max_val * 3 + delta)
        # Some interior values (sparse sampling)
        if max_val > 10:
            step_size = max(1, max_val // 30)
            for v in range(1, max_val + 1, step_size):
                if v not in cur_set:
                    candidates.append(v)
        else:
            for v in range(1, max_val + 1):
                if v not in cur_set:
                    candidates.append(v)
        random.shuffle(candidates)

        for new_val in candidates[:20]:
            if new_val <= 0 or new_val in cur_set:
                continue
            new_U = list(current_U) + [new_val]
            new_U.sort()
            new_score = _score_set(new_U)
            if new_score > current_score + 1e-10:
                current_U = new_U
                current_score = new_score
                cur_set = set(current_U)
                if current_score > best_score:
                    best_U = list(current_U)
                    best_score = current_score
                improved = True
                stall_count = 0
                break

        if not improved:
            # Try removing an element
            if len(current_U) > 2:
                remove_candidates = list(range(1, len(current_U)))
                random.shuffle(remove_candidates)
                for idx in remove_candidates[:10]:
                    new_U = [x for i, x in enumerate(current_U) if i != idx]
                    new_score = _score_set(new_U)
                    if new_score > current_score + 1e-10:
                        current_U = new_U
                        current_score = new_score
                        cur_set = set(current_U)
                        if current_score > best_score:
                            best_U = list(current_U)
                            best_score = current_score
                        improved = True
                        stall_count = 0
                        break

        if not improved:
            stall_count += 1
            if stall_count > 30:
                # Perturb: replace a random element with a new value
                if len(current_U) > 2:
                    idx = random.randint(1, len(current_U) - 1)
                    max_val = max(current_U)
                    new_val = random.randint(1, max_val * 3)
                    if new_val not in cur_set:
                        current_U[idx] = new_val
                        current_U.sort()
                        cur_set = set(current_U)
                        current_score = _score_set(current_U)
                        stall_count = 0
                else:
                    break
            elif step > 100:
                break

    return best_U, best_score

def construct():
    global _start_time
    _start_time = time.time()

    best_U = [0, 1, 3]
    best_score = _score_set(best_U)

    # Try geometric progressions
    for base in range(2, 30):
        if _elapsed() > _DEADLINE - 60:
            break
        for n in range(2, 12):
            if _elapsed() > _DEADLINE - 60:
                break
            U = [0]
            val = 1
            for i in range(1, n):
                val *= base
                U.append(val)
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_U = U[:]

    # Try small sets {0,1,a,b}, {0,1,a,b,c}
    for a in range(2, 25):
        if _elapsed() > _DEADLINE - 40:
            break
        for b in range(a + 1, 150):
            if _elapsed() > _DEADLINE - 40:
                break
            U = [0, 1, a, b]
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_U = U[:]

    for a in range(2, 15):
        if _elapsed() > _DEADLINE - 25:
            break
        for b in range(a + 1, 80):
            if _elapsed() > _DEADLINE - 25:
                break
            for c in range(b + 1, 300):
                if _elapsed() > _DEADLINE - 25:
                    break
                U = [0, 1, a, b, c]
                s = _score_set(U)
                if s > best_score:
                    best_score = s
                    best_U = U[:]

    # Try Golomb rulers as seeds
    for n in range(4, 10):
        if _elapsed() > _DEADLINE - 20:
            break
        for max_val in [n * n, n * n + n, n * n * 2]:
            if _elapsed() > _DEADLINE - 20:
                break
            U = _golomb_seed(n, max_val)
            if len(U) >= n - 1:
                s = _score_set(U)
                if s > best_score:
                    best_score = s
                    best_U = U[:]

    # Local search from best (more steps thanks to faster scoring)
    if _elapsed() < _DEADLINE - 5:
        U, score = _local_search(best_U, max_steps=3000)
        if score > best_score:
            best_score = score
            best_U = U

    # Local search from Golomb seeds
    for n in [5, 6, 7, 8]:
        if _elapsed() > _DEADLINE - 5:
            break
        U = _golomb_seed(n, n * n + n)
        if len(U) >= 4:
            U2, score2 = _local_search(U, max_steps=1500)
            if score2 > best_score:
                best_score = score2
                best_U = U2

    # Local search from geometric seeds
    for base in [2, 3, 4, 5, 7]:
        if _elapsed() > _DEADLINE - 5:
            break
        U = [0]
        val = 1
        for i in range(1, 6):
            val *= base
            U.append(val)
        U2, score2 = _local_search(U, max_steps=1500)
        if score2 > best_score:
            best_score = score2
            best_U = U2

    # Final polish: one more local search pass on the best
    if _elapsed() < _DEADLINE - 2:
        U, score = _local_search(best_U, max_steps=1000)
        if score > best_score:
            best_score = score
            best_U = U

    return sorted(best_U)
