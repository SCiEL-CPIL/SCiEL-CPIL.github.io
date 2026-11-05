import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin under 150s wall-clock

    total_words = q ** n
    ball_size = 1 + n * (q - 1)  # radius=1

    # ---- Precompute balls (each row = indices within distance <=1) ----
    balls = np.zeros((total_words, ball_size), dtype=np.int32)
    for i in range(total_words):
        tmp = i
        word = [0] * n
        for k in range(n - 1, -1, -1):
            word[k] = tmp % q
            tmp //= q
        balls[i, 0] = i
        idx = 1
        for pos in range(n):
            orig = word[pos]
            for nv in range(q):
                if nv == orig:
                    continue
                word[pos] = nv
                new_idx = 0
                for k in range(n):
                    new_idx = new_idx * q + word[k]
                balls[i, idx] = new_idx
                idx += 1
            word[pos] = orig

    # ---- Vectorized greedy set cover ----
    def greedy_cover(max_time):
        code = []
        cover_count = np.zeros(total_words, dtype=np.int32)
        covered = np.zeros(total_words, dtype=bool)
        while not covered.all():
            if time.time() > max_time:
                break
            gains = np.sum(1 - covered[balls], axis=1)
            best_idx = int(np.argmax(gains))
            if gains[best_idx] == 0:
                break
            code.append(best_idx)
            cover_count[balls[best_idx]] += 1
            covered[balls[best_idx]] = True
        return code, cover_count, covered

    # ---- Local search: batched deletions + 1-exchange swaps + SA kicks ----
    def local_search(code, cover_count, covered, max_time):
        code = list(code)
        cover_count = cover_count.copy()
        covered = covered.copy()

        def is_redundant(idx):
            return np.all(cover_count[balls[idx]] > 1)

        def remove_idx(idx):
            for j in balls[idx]:
                cover_count[j] -= 1
                if cover_count[j] == 0:
                    covered[j] = False

        def add_idx(idx):
            cover_count[balls[idx]] += 1
            covered[balls[idx]] = True

        improved = True
        while improved and time.time() < max_time:
            improved = False

            # --- Batched deletion pass ---
            to_remove = [c for c in code if is_redundant(c)]
            if to_remove:
                for c in to_remove:
                    code.remove(c)
                    remove_idx(c)
                improved = True
                continue

            # --- 1-exchange swap pass ---
            random.shuffle(code)
            for c in code:
                if time.time() > max_time:
                    break
                if is_redundant(c):
                    continue
                crit = [i for i in balls[c] if cover_count[i] == 1]
                if not crit:
                    continue
                # candidates covering the first critical point, not already in code
                first = crit[0]
                cands = balls[first]
                # score each candidate: covers all crit? then redundancy it creates
                best_w = -1
                best_score = -1
                for w in cands:
                    if w == c:
                        continue
                    if w in code:
                        continue
                    # check covers all critical points
                    ok = True
                    for k in crit:
                        if not np.any(balls[w] == k):
                            ok = False
                            break
                    if not ok:
                        continue
                    # redundancy created = number of points with count==1 that w covers
                    score = int(np.sum(cover_count[balls[w]] == 1))
                    if score > best_score:
                        best_score = score
                        best_w = w
                if best_w != -1:
                    code.remove(c)
                    remove_idx(c)
                    code.append(best_w)
                    add_idx(best_w)
                    improved = True
                    break

        # --- Simulated annealing kick phase (escape local minima) ---
        if time.time() < max_time:
            temp = 6.0
            best_local = len(code)
            stall = 0
            while time.time() < max_time and temp > 0.5:
                old_len = len(code)
                # kick: remove a random codeword
                kick_pos = random.randint(0, len(code) - 1)
                kick_idx = code[kick_pos]
                old_code = code.copy()
                old_cc = cover_count.copy()
                old_cov = covered.copy()
                code.pop(kick_pos)
                remove_idx(kick_idx)
                # greedily repair
                while not covered.all() and time.time() < max_time:
                    gains = np.sum(1 - covered[balls], axis=1)
                    best_idx = int(np.argmax(gains))
                    if gains[best_idx] == 0:
                        break
                    code.append(best_idx)
                    add_idx(best_idx)
                new_len = len(code)
                delta = new_len - old_len
                # accept if improvement, or with prob exp(-delta/temp)
                if delta <= 0 or random.random() < np.exp(-delta / temp):
                    if new_len < best_local:
                        best_local = new_len
                        stall = 0
                    else:
                        stall += 1
                else:
                    # revert
                    code = old_code
                    cover_count = old_cc
                    covered = old_cov
                    stall += 1
                temp *= 0.93
                if stall > 12:
                    break

        # final deletion polish
        for _ in range(50):
            if time.time() > max_time:
                break
            to_remove = [c for c in code if is_redundant(c)]
            if not to_remove:
                break
            for c in to_remove:
                code.remove(c)
                remove_idx(c)

        return code, cover_count, covered

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # ---- Multi-restart driver with global best tracking ----
    best_code = None
    best_len = float('inf')

    # First restart gets a generous greedy budget
    random.seed(42)
    code, cc, cov = greedy_cover(deadline - 30.0)
    code, cc, cov = local_search(code, cc, cov, deadline - 8.0)
    if len(code) < best_len:
        best_len = len(code)
        best_code = code.copy()

    # Additional restarts if time permits
    restart = 1
    while time.time() < deadline - 20.0:
        random.seed(100 + restart)
        per = (deadline - 20.0 - (time.time() - start_time)) / max(1, 4 - restart)
        code, cc, cov = greedy_cover(start_time + (time.time() - start_time) + per * 0.35)
        code, cc, cov = local_search(code, cc, cov, deadline - 8.0)
        if len(code) < best_len:
            best_len = len(code)
            best_code = code.copy()
        if best_len <= 190:
            break
        restart += 1

    if best_code is None:
        code, _, _ = greedy_cover(deadline - 2.0)
        best_code = code
        best_len = len(code)

    # Final deletion polish on the best
    if time.time() < deadline - 3.0:
        cover_count = np.zeros(total_words, dtype=np.int32)
        covered = np.zeros(total_words, dtype=bool)
        for idx in best_code:
            cover_count[balls[idx]] += 1
            covered[balls[idx]] = True
        while time.time() < deadline - 3.0:
            to_remove = [c for c in best_code if np.all(cover_count[balls[c]] > 1)]
            if not to_remove:
                break
            for c in to_remove:
                best_code.remove(c)
                for j in balls[c]:
                    cover_count[j] -= 1
                    if cover_count[j] == 0:
                        covered[j] = False

    return [idx_to_word(idx) for idx in best_code]
