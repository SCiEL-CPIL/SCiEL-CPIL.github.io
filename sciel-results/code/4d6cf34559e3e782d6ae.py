"""Improved covering-code search for K_3(6,1) with vectorized coverage + simulated annealing."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    total_words = q ** n

    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Precompute ball membership matrix: B[c, w] = 1 iff w within radius of c
    B = np.zeros((total_words, total_words), dtype=np.int32)
    for c in range(total_words):
        B[c, c] = 1
        if radius >= 1:
            w = idx_to_word(c)
            for pos in range(n):
                orig = w[pos]
                for nv in range(q):
                    if nv != orig:
                        w[pos] = nv
                        B[c, word_to_idx(w)] = 1
                        w[pos] = orig
    # neighbor list per codeword (self + radius-1 neighbors)
    nb_list = [np.nonzero(B[c] > 0)[0] for c in range(total_words)]

    def coverage_from(x):
        return B @ x

    def is_valid(x):
        return np.all(coverage_from(x) >= 1)

    def remove_redundant(x, max_time=None):
        """Remove every codeword whose ball is double-covered; repeat to fixpoint."""
        if max_time is None:
            max_time = deadline
        changed = True
        while changed and time.time() < max_time:
            changed = False
            cov = coverage_from(x)
            # Find all removable codewords in one pass
            removable = []
            for i in range(total_words):
                if x[i] and np.all(cov[nb_list[i]] >= 2):
                    removable.append(i)
            # Remove all at once, then recompute coverage once
            if removable:
                for i in removable:
                    x[i] = 0
                cov = coverage_from(x)
                changed = True
                if time.time() > max_time:
                    return x
        return x

    def greedy_cover(x, seed=0):
        """Greedy set cover starting from current inclusion x."""
        rng = random.Random(seed)
        cov = coverage_from(x)
        uncovered = np.nonzero(cov == 0)[0]
        guard = 0
        while len(uncovered) > 0 and time.time() < deadline and guard < 5000:
            guard += 1
            best = -1
            best_count = 0
            # Only consider candidates that cover at least one uncovered word
            candidates = []
            for cand in range(total_words):
                if x[cand]:
                    continue
                cnt = np.sum(cov[nb_list[cand]] == 0)
                if cnt > 0:
                    candidates.append((cnt, cand))
            if not candidates:
                break
            # Sort by coverage count descending, with random tie-breaking
            candidates.sort(key=lambda t: (-t[0], rng.random()))
            best_count, best = candidates[0]
            x[best] = 1
            cov[nb_list[best]] += 1
            uncovered = np.nonzero(cov == 0)[0]
        return x

    def simulated_annealing(x, seed=0, steps=8000, t0=1.0, t1=1e-3):
        """SA over binary inclusion; objective = #codewords + deficit^2."""
        rng = random.Random(seed)
        cov = coverage_from(x)
        weight = int(np.sum(x))
        deficit = int(np.sum(cov == 0))
        obj = weight + deficit * deficit
        best_x = x.copy()
        best_obj = obj
        n_steps = max(1, steps)
        included_indices = np.nonzero(x)[0].tolist()
        excluded_indices = np.nonzero(x == 0)[0].tolist()
        
        for it in range(n_steps):
            if time.time() > deadline:
                break
            temp = t0 * (t1 / t0) ** (it / n_steps)
            # choose a move: remove an included word, or add an excluded word
            if weight > 0 and (deficit == 0 or rng.random() < 0.5):
                if not included_indices:
                    continue
                i = rng.choice(included_indices)
                new_cov = cov - B[i]
                new_deficit = int(np.sum(new_cov == 0))
                new_weight = weight - 1
                new_obj = new_weight + new_deficit * new_deficit
                delta = new_obj - obj
                if delta <= 0 or rng.random() < np.exp(-delta / max(temp, 1e-9)):
                    x[i] = 0
                    cov = new_cov
                    weight = new_weight
                    deficit = new_deficit
                    obj = new_obj
                    included_indices.remove(i)
                    excluded_indices.append(i)
                    if obj < best_obj:
                        best_obj = obj
                        best_x = x.copy()
            else:
                if not excluded_indices:
                    continue
                i = rng.choice(excluded_indices)
                new_cov = cov + B[i]
                new_deficit = int(np.sum(new_cov == 0))
                new_weight = weight + 1
                new_obj = new_weight + new_deficit * new_deficit
                delta = new_obj - obj
                if delta <= 0 or rng.random() < np.exp(-delta / max(temp, 1e-9)):
                    x[i] = 1
                    cov = new_cov
                    weight = new_weight
                    deficit = new_deficit
                    obj = new_obj
                    excluded_indices.remove(i)
                    included_indices.append(i)
                    if obj < best_obj:
                        best_obj = obj
                        best_x = x.copy()
        # ensure validity of returned best
        if np.all(coverage_from(best_x) >= 1):
            return best_x
        return x

    best_x = None
    best_count = float('inf')

    def consider(x):
        nonlocal best_x, best_count
        x = remove_redundant(x)
        if not np.all(coverage_from(x) >= 1):
            return
        c = int(np.sum(x))
        if c < best_count:
            best_count = c
            best_x = x.copy()

    # --- tetracode construction (81 words) as a strong valid start ---
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    tet_x = np.zeros(total_words, dtype=np.int32)
    for ta in range(3):
        for tb in range(3):
            for h in tetracode:
                tet_x[word_to_idx(h + [ta, tb])] = 1
    consider(tet_x)

    # --- greedy from scratch with several seeds ---
    for seed in range(20):
        if time.time() > deadline - 5.0:
            break
        x = np.zeros(total_words, dtype=np.int32)
        x = greedy_cover(x, seed=seed)
        consider(x)

    # --- simulated annealing polishing from the best found ---
    if best_x is not None and time.time() < deadline - 10.0:
        for seed in range(15):
            if time.time() > deadline - 3.0:
                break
            x = best_x.copy()
            # Use longer runs with slower cooling
            x = simulated_annealing(x, seed=seed, steps=12000, t0=1.2, t1=1e-5)
            consider(x)

    # --- hybrid: random subset of tetracode then greedy ---
    rng = random.Random(12345)
    tet_indices = np.nonzero(tet_x)[0]
    for seed in range(12):
        if time.time() > deadline - 8.0:
            break
        k = rng.randint(50, 75)
        x = np.zeros(total_words, dtype=np.int32)
        x[rng.sample(tet_indices.tolist(), k)] = 1
        x = greedy_cover(x, seed=seed)
        consider(x)
        # Polish with SA
        if time.time() < deadline - 5.0:
            x = simulated_annealing(x, seed=seed + 100, steps=4000, t0=0.5, t1=1e-4)
            consider(x)

    # --- final polish on the overall best ---
    if best_x is not None and time.time() < deadline - 2.0:
        best_x = remove_redundant(best_x)
        if not np.all(coverage_from(best_x) >= 1):
            # fallback to tetracode
            best_x = tet_x.copy()

    if best_x is None:
        best_x = tet_x.copy()

    return [idx_to_word(int(i)) for i in np.nonzero(best_x)[0]]
# EVOLVE-BLOCK-END
