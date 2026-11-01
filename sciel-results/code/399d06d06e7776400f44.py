"""Improved covering-code search for K_3(6,1) with hypergraph 2-core peeling + SA polish."""
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

    # Precompute ball membership: B[c, w] = 1 iff w within radius of c
    # For n=6, q=3, radius=1: each ball has 1 + 6*2 = 13 words
    # Use a list of sets for faster membership and intersection
    balls = [None] * total_words
    ball_sets = [None] * total_words
    for c in range(total_words):
        s = set()
        w = idx_to_word(c)
        s.add(c)
        if radius >= 1:
            for pos in range(n):
                orig = w[pos]
                for nv in range(q):
                    if nv != orig:
                        w[pos] = nv
                        s.add(word_to_idx(w))
                        w[pos] = orig
        balls[c] = sorted(s)
        ball_sets[c] = s

    # Coverage count array
    def compute_coverage(x):
        """x is a boolean array of length total_words; returns coverage count array."""
        cov = np.zeros(total_words, dtype=np.int32)
        for i in range(total_words):
            if x[i]:
                for w in balls[i]:
                    cov[w] += 1
        return cov

    def is_valid(x):
        cov = compute_coverage(x)
        return np.all(cov >= 1)

    def hypergraph_core_peel(x, max_time=None):
        """
        Hypergraph 2-core peeling: iteratively remove codewords that are redundant.
        A codeword c is redundant if every word in its ball is covered by at least one
        other selected codeword. This is the 2-core of the hypergraph where vertices
        are codewords and hyperedges connect codewords sharing a covered word.
        We peel until a fixed point.
        """
        if max_time is None:
            max_time = deadline
        x = x.copy()
        cov = compute_coverage(x)
        changed = True
        while changed and time.time() < max_time:
            changed = False
            # Recompute coverage
            cov = compute_coverage(x)
            for i in range(total_words):
                if time.time() > max_time:
                    return x
                if x[i]:
                    # Check if removing i leaves any word uncovered
                    redundant = True
                    for w in balls[i]:
                        if cov[w] <= 1:
                            redundant = False
                            break
                    if redundant:
                        x[i] = 0
                        changed = True
                        cov = compute_coverage(x)
                        # Early exit check
                        if not np.all(cov >= 1):
                            x[i] = 1
                            cov = compute_coverage(x)
                            redundant = False
        return x

    def greedy_cover(x, seed=0):
        """Greedy set cover starting from current inclusion x."""
        rng = random.Random(seed)
        cov = compute_coverage(x)
        uncovered = set(np.nonzero(cov == 0)[0].tolist())
        guard = 0
        while len(uncovered) > 0 and time.time() < deadline and guard < 5000:
            guard += 1
            best = -1
            best_count = 0
            # Random tie-break
            candidates = list(range(total_words))
            rng.shuffle(candidates)
            for cand in candidates:
                if x[cand]:
                    continue
                cnt = sum(1 for w in balls[cand] if w in uncovered)
                if cnt > best_count:
                    best_count = cnt
                    best = cand
            if best < 0 or best_count == 0:
                break
            x[best] = 1
            for w in balls[best]:
                cov[w] += 1
                if w in uncovered:
                    uncovered.discard(w)
        return x

    def simulated_annealing(x, seed=0, steps=5000, t0=1.0, t1=1e-3):
        """SA over binary inclusion; objective = #codewords + deficit^2."""
        rng = random.Random(seed)
        cov = compute_coverage(x)
        weight = int(np.sum(x))
        deficit = int(np.sum(cov == 0))
        obj = weight + deficit * deficit
        best_x = x.copy()
        best_obj = obj
        n_steps = max(1, steps)
        for it in range(n_steps):
            if time.time() > deadline:
                break
            temp = t0 * (t1 / t0) ** (it / n_steps)
            if weight > 0 and (deficit == 0 or rng.random() < 0.5):
                i = rng.randrange(total_words)
                if x[i]:
                    new_cov = cov.copy()
                    for w in balls[i]:
                        new_cov[w] -= 1
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
                        if obj < best_obj:
                            best_obj = obj
                            best_x = x.copy()
            else:
                i = rng.randrange(total_words)
                if not x[i]:
                    new_cov = cov.copy()
                    for w in balls[i]:
                        new_cov[w] += 1
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
                        if obj < best_obj:
                            best_obj = obj
                            best_x = x.copy()
        if np.all(compute_coverage(best_x) >= 1):
            return best_x
        return x

    def local_search_swap(x, seed=0, max_iter=2000):
        """Try to swap one codeword out and one in, or just remove, to reduce size."""
        rng = random.Random(seed)
        x = x.copy()
        cov = compute_coverage(x)
        weight = int(np.sum(x))
        deficit = int(np.sum(cov == 0))
        obj = weight + deficit * deficit
        best_x = x.copy()
        best_obj = obj
        for it in range(max_iter):
            if time.time() > deadline:
                break
            if deficit > 0:
                # Need to add
                i = rng.randrange(total_words)
                if not x[i]:
                    new_cov = cov.copy()
                    for w in balls[i]:
                        new_cov[w] += 1
                    new_deficit = int(np.sum(new_cov == 0))
                    new_weight = weight + 1
                    new_obj = new_weight + new_deficit * new_deficit
                    delta = new_obj - obj
                    if delta <= 0:
                        x[i] = 1
                        cov = new_cov
                        weight = new_weight
                        deficit = new_deficit
                        obj = new_obj
                        if obj < best_obj:
                            best_obj = obj
                            best_x = x.copy()
            else:
                # Try to remove
                i = rng.randrange(total_words)
                if x[i]:
                    new_cov = cov.copy()
                    for w in balls[i]:
                        new_cov[w] -= 1
                    if np.all(new_cov >= 1):
                        x[i] = 0
                        cov = new_cov
                        weight -= 1
                        obj = weight
                        if obj < best_obj:
                            best_obj = obj
                            best_x = x.copy()
        return best_x

    best_x = None
    best_count = float('inf')

    def consider(x):
        nonlocal best_x, best_count
        # Apply hypergraph core peeling
        x = hypergraph_core_peel(x)
        if not np.all(compute_coverage(x) >= 1):
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
    for seed in range(15):
        if time.time() > deadline - 5.0:
            break
        x = np.zeros(total_words, dtype=np.int32)
        x = greedy_cover(x, seed=seed)
        consider(x)

    # --- simulated annealing polishing from the best found ---
    if best_x is not None and time.time() < deadline - 15.0:
        for seed in range(8):
            if time.time() > deadline - 5.0:
                break
            x = best_x.copy()
            x = simulated_annealing(x, seed=seed, steps=4000, t0=0.8, t1=1e-4)
            consider(x)

    # --- hybrid: random subset of tetracode then greedy ---
    rng = random.Random(12345)
    tet_indices = np.nonzero(tet_x)[0]
    for seed in range(10):
        if time.time() > deadline - 10.0:
            break
        k = rng.randint(40, 70)
        x = np.zeros(total_words, dtype=np.int32)
        x[rng.sample(tet_indices.tolist(), k)] = 1
        x = greedy_cover(x, seed=seed)
        consider(x)

    # --- local search swap polish ---
    if best_x is not None and time.time() < deadline - 5.0:
        for seed in range(5):
            if time.time() > deadline - 2.0:
                break
            x = best_x.copy()
            x = local_search_swap(x, seed=seed, max_iter=1500)
            consider(x)

    # --- final polish on the overall best ---
    if best_x is not None and time.time() < deadline - 2.0:
        best_x = hypergraph_core_peel(best_x)
        if not np.all(compute_coverage(best_x) >= 1):
            # fallback to tetracode
            best_x = tet_x.copy()

    if best_x is None:
        best_x = tet_x.copy()

    return [idx_to_word(int(i)) for i in np.nonzero(best_x)[0]]
# EVOLVE-BLOCK-END
