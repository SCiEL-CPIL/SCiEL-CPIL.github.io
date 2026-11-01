import math
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin below 150s limit

    # ---------- scoring ----------
    def compute_score_direct(u):
        u = sorted(set(u))
        if not u or 0 not in u:
            return -1.0
        max_u = max(u)
        m = 2 * max_u + 1
        if m <= 1:
            return 1.0
        sumset = set()
        diffset = set()
        for a in u:
            for b in u:
                sumset.add(a + b)
                diffset.add(a - b)
        r = len(diffset) / len(sumset)
        return 1 + math.log(r) / math.log(m)

    def compute_score(u, fast=True):
        u = sorted(set(u))
        if not u or 0 not in u:
            return -1.0
        max_u = max(u)
        m = 2 * max_u + 1
        if m <= 1:
            return 1.0
        n = len(u)
        if not fast or n <= 40:
            sumset = set()
            diffset = set()
            for i in range(n):
                for j in range(i, n):
                    s = u[i] + u[j]
                    sumset.add(s)
                    if i != j:
                        d = u[i] - u[j]
                        diffset.add(d)
                        diffset.add(-d)
            diffset.add(0)
            size_sum = len(sumset)
            size_diff = len(diffset)
        else:
            # FFT-based computation
            size = 2 * max_u + 1
            fft_size = 1
            while fft_size < size:
                fft_size *= 2
            f = np.zeros(fft_size, dtype=np.float64)
            for uu in u:
                f[uu] = 1.0
            F = np.fft.fft(f)
            conv = np.fft.ifft(F * F).real
            threshold = 0.5
            size_sum = int(np.sum(conv[:size] > threshold))
            g = np.zeros(fft_size, dtype=np.float64)
            for uu in u:
                if uu == 0:
                    g[0] = 1.0
                else:
                    g[fft_size - uu] = 1.0
            G = np.fft.fft(g)
            D = np.fft.ifft(F * G).real
            diff_nonzero = 0
            for k in range(0, max_u + 1):
                if abs(D[k]) > threshold:
                    diff_nonzero += 1
            for k in range(fft_size - max_u, fft_size):
                if abs(D[k]) > threshold:
                    diff_nonzero += 1
            size_diff = diff_nonzero
        if size_sum == 0:
            return 1.0
        r = size_diff / size_sum
        return 1 + math.log(r) / math.log(m)

    def is_golomb(u):
        u = sorted(u)
        diffs = set()
        for i in range(len(u)):
            for j in range(i + 1, len(u)):
                d = u[j] - u[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    # ---------- deterministic Golomb-ruler core (floor) ----------
    def try_extend(base_u, max_n, time_limit=20):
        start_time = time.time()
        best_u = list(base_u)
        best_score = compute_score(best_u, fast=False)
        for target_n in range(len(base_u) + 1, max_n + 1):
            if time.time() - start_time > time_limit:
                break
            improved = True
            while improved:
                improved = False
                if time.time() - start_time > time_limit:
                    break
                current_max = max(best_u)
                best_candidate = None
                best_candidate_score = best_score
                search_hi = current_max + target_n * target_n
                for new_mark in range(current_max + 1, search_hi + 1):
                    candidate = sorted(best_u + [new_mark])
                    if not is_golomb(candidate):
                        continue
                    s = compute_score(candidate, fast=False)
                    if s > best_candidate_score:
                        best_candidate_score = s
                        best_candidate = candidate
                    if time.time() - start_time > time_limit:
                        break
                if best_candidate is not None:
                    best_u = best_candidate
                    best_score = best_candidate_score
                    improved = True
        return best_u, best_score

    def greedy_golomb(n, time_limit=10):
        start_time = time.time()
        u = [0]
        for i in range(1, n):
            if time.time() - start_time > time_limit:
                break
            pos = 0
            found = False
            while not found:
                pos += 1
                candidate = sorted(u + [pos])
                if is_golomb(candidate):
                    u = candidate
                    found = True
                if pos > n * n * 20:
                    break
        return u

    # Build a deterministic baseline
    base_candidates = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
    ]
    ext_u, _ = try_extend([0, 1, 4, 10, 12, 17], 9, time_limit=15)
    if ext_u and len(ext_u) > 6:
        base_candidates.append(ext_u)
    greedy_u = greedy_golomb(9, time_limit=8)
    if greedy_u and len(greedy_u) >= 6:
        base_candidates.append(greedy_u)

    best_set = [0, 1, 3]
    best_score = 0.0
    for u in base_candidates:
        u = sorted(set(u))
        if 0 not in u:
            u = [0] + u
        s = compute_score(u, fast=False)
        if s > best_score:
            best_score = s
            best_set = u

    # ---------- simulated-annealing hill climber (refinement) ----------
    random.seed(12345)

    def hill_climb(start_u, time_budget):
        start_time = time.time()
        current = sorted(set(start_u))
        current_score = compute_score(current, fast=True)
        local_best = list(current)
        local_best_score = current_score
        T = 0.02
        cool = 0.9995
        while time.time() - start_time < time_budget and time.time() < deadline:
            neighbor = list(current)
            move = random.choice(['add', 'remove', 'shift'])
            max_val = max(neighbor)
            if move == 'add':
                pool = [i for i in range(1, max_val + 6) if i not in neighbor]
                if pool:
                    neighbor.append(random.choice(pool))
            elif move == 'remove':
                if len(neighbor) > 1:
                    rem = random.choice([x for x in neighbor if x != 0])
                    neighbor.remove(rem)
            else:
                if len(neighbor) > 1:
                    idx = random.choice([i for i, x in enumerate(neighbor) if x != 0])
                    old_val = neighbor[idx]
                    new_val = random.randint(1, max_val + 6)
                    if new_val not in neighbor:
                        neighbor[idx] = new_val
            if neighbor == current:
                continue
            neighbor.sort()
            new_score = compute_score(neighbor, fast=True)
            if new_score > current_score:
                current = neighbor
                current_score = new_score
                if current_score > local_best_score:
                    local_best = list(current)
                    local_best_score = current_score
            else:
                if random.random() < math.exp((new_score - current_score) / max(T, 1e-6)):
                    current = neighbor
                    current_score = new_score
            T *= cool
        return local_best, local_best_score

    # Run a few hill-climb attempts from different starts
    attempts = [
        list(best_set),
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
    ]
    remaining = deadline - time.time()
    if remaining > 5:
        per = max(1.0, remaining * 0.8 / len(attempts))
        for st in attempts:
            if time.time() >= deadline - 3:
                break
            cand, cs = hill_climb(st, per)
            if cs > best_score:
                best_score = cs
                best_set = cand

    # Final polish: a short deterministic local search on the best set
    if time.time() < deadline - 3:
        polish_budget = min(8.0, deadline - time.time() - 3)
        cur = list(best_set)
        cur_score = compute_score(cur, fast=True)
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            max_val = max(cur)
            # try removing each element
            for i in range(len(cur)):
                if cur[i] == 0:
                    continue
                trial = cur[:i] + cur[i+1:]
                s = compute_score(trial, fast=True)
                if s > cur_score + 1e-12:
                    cur = trial
                    cur_score = s
                    improved = True
                    break
            # try adding near existing values
            if not improved:
                for i in range(len(cur)):
                    if cur[i] == 0:
                        continue
                    for dv in (1, 2, -1, -2):
                        v = cur[i] + dv
                        if v > 0 and v not in cur:
                            trial = sorted(cur + [v])
                            s = compute_score(trial, fast=True)
                            if s > cur_score + 1e-12:
                                cur = trial
                                cur_score = s
                                improved = True
                                break
                    if improved:
                        break
        if cur_score > best_score:
            best_score = cur_score
            best_set = cur

    # Final exact verification
    final_score = compute_score(best_set, fast=False)
    if final_score < 1.0:
        best_set = [0, 1, 3]
    return best_set
