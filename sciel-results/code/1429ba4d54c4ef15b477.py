import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Handle the specific task parameters
    if q != 3 or n != 6 or radius != 1:
        return _generic_construct(q, n, radius)

    total_words = q ** n  # 729

    # Precompute all words and index mapping
    all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    # Precompute coverage: covers[c] = set of word indices within Hamming distance 1 of c
    covers = [None] * total_words
    for i, w in enumerate(all_words):
        cov = set([i])
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                cov.add(word_to_idx[tuple(new_w)])
        covers[i] = cov

    # Precompute distance-2 neighbors for independent set construction
    dist2_neighbors = [set() for _ in range(total_words)]
    for i in range(total_words):
        w1 = all_words[i]
        for j in range(i + 1, total_words):
            w2 = all_words[j]
            d = 0
            for k in range(n):
                if w1[k] != w2[k]:
                    d += 1
                    if d > 2:
                        break
            if d == 2:
                dist2_neighbors[i].add(j)
                dist2_neighbors[j].add(i)

    # Build initial cover via randomized greedy independent set on distance-2 graph
    # An independent set in the distance-2 graph means no two codewords are within
    # distance 2, which gives a well-spread initial cover.
    best_independent_set = []
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 40:
            break
        random.seed(restart * 12345 + 678)
        remaining = set(range(total_words))
        independent_set = []
        while remaining:
            # Sample candidates and pick lowest degree
            sample_size = min(50, len(remaining))
            candidates = random.sample(list(remaining), sample_size)
            best_v = None
            best_deg = float('inf')
            for v in candidates:
                deg = len(dist2_neighbors[v] & remaining)
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
            if best_v is not None:
                independent_set.append(best_v)
                remaining.remove(best_v)
                remaining -= dist2_neighbors[best_v]
        if len(independent_set) > len(best_independent_set):
            best_independent_set = independent_set

    # Fallback to tetracode direct sum if independent set is too large or time is tight
    if len(best_independent_set) > 81 or time.time() > deadline - 30:
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        tails = list(itertools.product(range(q), repeat=n - 4))
        initial_indices = []
        for h in tetracode:
            for tail in tails:
                cw = tuple(h + list(tail))
                initial_indices.append(word_to_idx[cw])
    else:
        initial_indices = best_independent_set

    active = set(initial_indices)

    # Initialize coverage counts
    cover_count = np.zeros(total_words, dtype=np.int32)
    for cw in active:
        cover_count[list(covers[cw])] += 1

    # Ensure full coverage: add codewords for any uncovered words
    uncovered = set(np.where(cover_count == 0)[0])
    while uncovered and time.time() < deadline:
        best_c = -1
        best_count = -1
        for c in range(total_words):
            if c in active:
                continue
            count = len(covers[c] & uncovered)
            if count > best_count:
                best_count = count
                best_c = c
        if best_c == -1 or best_count == 0:
            break
        active.add(best_c)
        for w in covers[best_c]:
            cover_count[w] += 1
            if cover_count[w] == 1:
                uncovered.discard(w)

    best_size = len(active)
    best_active = set(active)

    # Precompute covered_by: for each word w, which codewords cover it
    covered_by = [set() for _ in range(total_words)]
    for c in range(total_words):
        for w in covers[c]:
            covered_by[w].add(c)

    def is_removable(cw):
        """Check if codeword cw can be removed without losing coverage."""
        for idx in covers[cw]:
            if cover_count[idx] == 1:
                return False
        return True

    def greedy_remove():
        """Remove all redundant codewords."""
        nonlocal active, cover_count
        changed = True
        while changed:
            changed = False
            to_check = list(active)
            random.shuffle(to_check)
            for cw in to_check:
                if cw in active and is_removable(cw):
                    active.remove(cw)
                    cover_count[list(covers[cw])] -= 1
                    changed = True

    # Initial greedy removal
    greedy_remove()
    if len(active) < best_size:
        best_size = len(active)
        best_active = set(active)

    # Main local search loop
    steps = 0
    max_steps = 200000

    while time.time() < deadline and steps < max_steps:
        steps += 1

        # Periodic greedy removal
        if steps % 20 == 0:
            greedy_remove()
            if len(active) < best_size:
                best_size = len(active)
                best_active = set(active)
            if best_size <= 73:
                break

        if len(active) < 2:
            break

        active_list = list(active)

        # --- Move type 1: Swap (replace one codeword with another) ---
        # Pick a random active codeword to potentially replace
        u = random.choice(active_list)

        # Find words uniquely covered by u
        unique_words = set()
        for w in covers[u]:
            if cover_count[w] == 1:
                unique_words.add(w)

        if unique_words:
            # Candidate v must cover all unique_words
            # Intersect covered_by for each unique word
            candidate_set = covered_by[next(iter(unique_words))].copy()
            for w in list(unique_words):
                candidate_set &= covered_by[w]
            candidate_set.discard(u)
            candidate_set -= active  # Don't swap with existing codewords
        else:
            # u is fully redundant (shouldn't happen after greedy_remove, but be safe)
            candidate_set = set(range(total_words)) - active - {u}

        if candidate_set:
            # Pick a candidate, preferring ones that cover more currently under-covered words
            # Sample a few and pick the best
            sample_size = min(10, len(candidate_set))
            candidates = random.sample(list(candidate_set), sample_size)
            best_v = None
            best_score = -1
            for v in candidates:
                # Score: number of words in covers[v] that have low coverage
                score = 0
                for w in covers[v]:
                    if cover_count[w] <= 2:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_v = v

            if best_v is not None:
                # Verify the swap is valid
                valid = True
                for w in covers[u]:
                    if cover_count[w] == 1 and w not in covers[best_v]:
                        valid = False
                        break

                if valid:
                    # Perform swap
                    active.remove(u)
                    cover_count[list(covers[u])] -= 1
                    active.add(best_v)
                    cover_count[list(covers[best_v])] += 1

                    # Try to remove redundant codewords after swap
                    greedy_remove()

                    if len(active) < best_size:
                        best_size = len(active)
                        best_active = set(active)
                        if best_size <= 73:
                            break

        # --- Move type 2: 2-for-1 (replace two codewords with one) ---
        if steps % 50 == 0 and len(active) > 2:
            # Pick a random pair
            c1, c2 = random.sample(active_list, 2)
            union_cov = covers[c1] | covers[c2]

            # Find c3 that covers all of union_cov
            # Start from a word in union_cov and intersect covered_by
            sample_word = random.choice(list(union_cov))
            candidate_c3s = covered_by[sample_word]
            found_c3 = -1
            for c3 in candidate_c3s:
                if c3 == c1 or c3 == c2:
                    continue
                if union_cov.issubset(covers[c3]):
                    found_c3 = c3
                    break

            if found_c3 != -1:
                # Perform 2-for-1: remove c1, c2, add c3
                active.remove(c1)
                active.remove(c2)
                cover_count[list(covers[c1])] -= 1
                cover_count[list(covers[c2])] -= 1
                active.add(found_c3)
                cover_count[list(covers[found_c3])] += 1

                greedy_remove()

                if len(active) < best_size:
                    best_size = len(active)
                    best_active = set(active)
                    if best_size <= 73:
                        break

        # --- Move type 3: Random perturbation (add a random codeword, then remove redundants) ---
        if steps % 100 == 0 and len(active) < total_words:
            v = random.randint(0, total_words - 1)
            if v not in active:
                active.add(v)
                cover_count[list(covers[v])] += 1
                greedy_remove()
                if len(active) < best_size:
                    best_size = len(active)
                    best_active = set(active)

    # Convert best_active back to list of lists
    final_code = []
    for idx in best_active:
        final_code.append(list(all_words[idx]))

    return final_code


def _generic_construct(q, n, radius):
    """Fallback for non-(3,6,1) parameters."""
    start_time = time.time()
    deadline = start_time + 140.0

    space_size = q ** n
    all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    # Precompute balls
    balls = []
    for i, w in enumerate(all_words):
        ball = set([i])
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                ball.add(word_to_idx[tuple(new_w)])
        balls.append(ball)

    # Greedy set cover
    active = set()
    covered = set()
    remaining = set(range(space_size))

    while remaining and time.time() < deadline:
        best_c = -1
        best_count = -1
        for c in range(space_size):
            if c in active:
                continue
            count = len(balls[c] & remaining)
            if count > best_count:
                best_count = count
                best_c = c
        if best_c == -1 or best_count == 0:
            break
        active.add(best_c)
        covered |= balls[best_c]
        remaining -= balls[best_c]

    # Greedy removal
    cover_count = np.zeros(space_size, dtype=np.int32)
    for cw in active:
        cover_count[list(balls[cw])] += 1

    changed = True
    while changed and time.time() < deadline:
        changed = False
        for cw in list(active):
            removable = True
            for idx in balls[cw]:
                if cover_count[idx] == 1:
                    removable = False
                    break
            if removable:
                active.remove(cw)
                cover_count[list(balls[cw])] -= 1
                changed = True

    return [list(all_words[idx]) for idx in active]
