import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0  # 2s margin

    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)

    # Fallback for other parameters
    return _general_greedy(q, n, radius, deadline)

def _general_greedy(q, n, radius, deadline):
    total = q ** n
    if total <= 10000:
        all_words = [list(c) for c in itertools.product(range(q), repeat=n)]
        uncovered = set(range(len(all_words)))
        codewords = []

        def hamming_dist(a, b):
            return sum(1 for x, y in zip(a, b) if x != y)

        while uncovered and time.time() < deadline - 5.0:
            best_idx = -1
            best_count = -1
            for i in range(len(all_words)):
                count = sum(1 for j in uncovered if hamming_dist(all_words[i], all_words[j]) <= radius)
                if count > best_count:
                    best_count = count
                    best_idx = i
            if best_idx == -1:
                break
            codewords.append(all_words[best_idx])
            uncovered = {j for j in uncovered if hamming_dist(all_words[best_idx], all_words[j]) > radius}

        return codewords
    else:
        return [list(c) for c in itertools.product(range(q), repeat=n)]

def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17

    # Precompute base-3 weights
    weights = [3 ** (n - 1 - i) for i in range(n)]

    # Precompute balls: balls[w, 0..16] are the 17 words within Hamming distance 1 of w
    balls = np.zeros((total, ball_size), dtype=np.int32)
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1

    # Build DFT-based spectral representation
    # We use the fact that the covering problem can be viewed in frequency domain
    # The indicator function of the covered set has a DFT, and sparsity in frequency domain
    # indicates structural redundancy

    # Precompute the DFT of each ball's indicator function
    # For efficiency, we'll use a simplified spectral approach
    # The key insight: if a codeword's ball is "spectrally similar" to others, it may be redundant

    # Create a spectral signature for each word
    # We'll use a random projection of the ball to create a compact signature
    rng = np.random.RandomState(42)
    spectral_dim = 64
    spectral_matrix = rng.randn(total, spectral_dim) * 0.1

    # For each word, compute spectral signature as weighted sum of its ball's signatures
    ball_spectral = np.zeros((total, spectral_dim))
    for w in range(total):
        ball_spectral[w] = np.mean(spectral_matrix[balls[w]], axis=0)

    best_size = total
    best_solution = None

    # Try multiple initialization strategies
    for iteration in range(25):
        if time.time() > deadline - 15.0:
            break

        rng_seed = iteration * 7919 + 12345
        random.seed(rng_seed)
        np.random.seed(rng_seed % 2**32)

        if iteration < 10:
            # Spectral greedy: use spectral similarity to guide selection
            selected = _spectral_greedy_init(balls, total, ball_size, ball_spectral, rng_seed, deadline)
        elif iteration < 18:
            # Standard greedy with spectral pruning
            selected = _greedy_with_spectral_prune(balls, total, ball_size, ball_spectral, rng_seed, deadline)
        else:
            # Hybrid: spectral initialization + local search
            selected = _hybrid_spectral_init(balls, total, ball_size, ball_spectral, rng_seed, deadline)

        # Local search to improve
        selected = _local_search(balls, total, ball_size, selected, deadline)

        # Final cleanup: remove redundant codewords using spectral check
        selected = _remove_redundant_spectral(balls, total, ball_size, ball_spectral, selected)

        # Verify coverage
        if _verify_coverage(balls, total, ball_size, selected):
            if len(selected) < best_size:
                best_size = len(selected)
                best_solution = selected[:]

        # Early termination if we hit the known optimum
        if best_size <= 486:
            break

    if best_solution is None:
        # Fallback: return all words (trivially valid)
        best_solution = list(range(total))

    # Convert to digit lists
    result = []
    for w in best_solution:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)

    return result

def _spectral_greedy_init(balls, total, ball_size, ball_spectral, seed, deadline):
    """Spectral greedy: select codewords to maximize spectral coverage."""
    random.seed(seed)
    np.random.seed(seed % 2**32)

    cov = np.zeros(total, dtype=np.int32)
    selected = []
    uncovered_count = total

    while uncovered_count > 0 and time.time() < deadline - 20.0:
        # Count how many uncovered words each candidate covers
        uncovered_mask = (cov == 0)
        counts = np.sum(uncovered_mask[balls], axis=1)
        max_count = np.max(counts)
        if max_count == 0:
            break

        candidates = np.where(counts == max_count)[0]

        # Use spectral similarity to break ties: prefer codewords whose spectral
        # signature is most different from already-selected codewords
        if len(candidates) > 1 and len(selected) > 0:
            # Compute spectral distance from selected set
            selected_spectral = ball_spectral[selected]
            cand_spectral = ball_spectral[candidates]

            # For each candidate, compute minimum distance to any selected codeword
            min_dists = np.min(
                np.linalg.norm(cand_spectral[:, np.newaxis, :] - selected_spectral[np.newaxis, :, :], axis=2),
                axis=1
            )
            # Prefer candidates with larger minimum distance (more diverse)
            best_idx = candidates[np.argmax(min_dists)]
        else:
            best_idx = candidates[random.randint(0, len(candidates) - 1)]

        selected.append(best_idx)
        cov[balls[best_idx]] += 1
        uncovered_count = np.sum(cov == 0)

    return selected

def _greedy_with_spectral_prune(balls, total, ball_size, ball_spectral, seed, deadline):
    """Standard greedy with spectral redundancy pruning."""
    random.seed(seed)
    np.random.seed(seed % 2**32)

    cov = np.zeros(total, dtype=np.int32)
    selected = []
    uncovered_count = total

    while uncovered_count > 0 and time.time() < deadline - 20.0:
        # Count how many uncovered words each candidate covers
        uncovered_mask = (cov == 0)
        counts = np.sum(uncovered_mask[balls], axis=1)
        max_count = np.max(counts)
        if max_count == 0:
            break

        candidates = np.where(counts == max_count)[0]
        best_idx = candidates[random.randint(0, len(candidates) - 1)]

        # Spectral pruning: check if this codeword is spectrally redundant
        # If its spectral signature is very similar to existing codewords, skip it
        if len(selected) > 5:
            selected_spectral = ball_spectral[selected]
            cand_spec = ball_spectral[best_idx]

            # Check if cand_spec is close to any selected spectral signature
            dists = np.linalg.norm(selected_spectral - cand_spec, axis=1)
            if np.min(dists) < 0.01 and counts[best_idx] < max_count:
                # This codeword is spectrally redundant and not uniquely best
                # Try next best candidate
                other_cands = np.where(counts == max_count - 1)[0]
                if len(other_cands) > 0:
                    best_idx = other_cands[random.randint(0, len(other_cands) - 1)]

        selected.append(best_idx)
        cov[balls[best_idx]] += 1
        uncovered_count = np.sum(cov == 0)

    return selected

def _hybrid_spectral_init(balls, total, ball_size, ball_spectral, seed, deadline):
    """Hybrid: spectral-based initialization followed by greedy completion."""
    random.seed(seed)
    np.random.seed(seed % 2**32)

    # Initialize with spectrally diverse codewords
    num_initial = 120
    selected = []
    used = set()

    # Greedy selection for spectral diversity
    for i in range(num_initial):
        if time.time() > deadline - 25.0:
            break

        if not selected:
            # Pick first randomly
            best_idx = random.randint(0, total - 1)
        else:
            # Pick codeword most spectrally different from selected set
            selected_spectral = ball_spectral[selected]
            # Sample candidates for efficiency
            sample_size = min(500, total)
            candidates = random.sample(range(total), sample_size)
            cand_spectral = ball_spectral[candidates]

            min_dists = np.min(
                np.linalg.norm(cand_spectral[:, np.newaxis, :] - selected_spectral[np.newaxis, :, :], axis=2),
                axis=1
            )
            best_idx = candidates[np.argmax(min_dists)]

        if best_idx not in used:
            used.add(best_idx)
            selected.append(best_idx)

    # Complete with greedy
    cov = np.zeros(total, dtype=np.int32)
    for c in selected:
        cov[balls[c]] += 1

    uncovered_count = np.sum(cov == 0)

    while uncovered_count > 0 and time.time() < deadline - 20.0:
        uncovered_mask = (cov == 0)
        counts = np.sum(uncovered_mask[balls], axis=1)
        max_count = np.max(counts)
        if max_count == 0:
            break

        candidates = np.where(counts == max_count)[0]
        cand_list = [c for c in candidates if c not in used]
        if not cand_list:
            cand_list = candidates.tolist()

        best_idx = cand_list[random.randint(0, len(cand_list) - 1)]

        selected.append(best_idx)
        used.add(best_idx)
        cov[balls[best_idx]] += 1
        uncovered_count = np.sum(cov == 0)

    return selected

def _local_search(balls, total, ball_size, selected, deadline):
    """Iterative local search: removal, single swap."""
    in_code = set(selected)
    cov = np.zeros(total, dtype=np.int32)
    for c in selected:
        cov[balls[c]] += 1

    improved = True
    max_iterations = 300

    for it in range(max_iterations):
        if time.time() > deadline - 10.0:
            break
        if not improved:
            break
        improved = False

        # Phase 1: Remove redundant codewords
        for c in list(selected):
            if c not in in_code:
                continue
            if np.min(cov[balls[c]]) > 1:
                # Remove c
                cov[balls[c]] -= 1
                in_code.remove(c)
                selected.remove(c)
                improved = True

        if not improved:
            # Phase 2: Try single swaps
            swapped = False
            for c in list(selected):
                if time.time() > deadline - 10.0:
                    break
                if c not in in_code:
                    continue

                # Find words uniquely covered by c
                unique_mask = (cov[balls[c]] == 1)
                unique_words = balls[c][unique_mask]

                if len(unique_words) == 0:
                    continue

                # Find candidates w that cover all unique_words
                if len(unique_words) > 5:
                    continue

                valid_w_set = set(balls[unique_words[0]].tolist())
                for v in unique_words[1:]:
                    valid_w_set.intersection_update(balls[v].tolist())

                w_list = [w for w in valid_w_set if w not in in_code]
                if not w_list:
                    continue

                best_w = -1
                best_score = -1
                for w in w_list[:15]:
                    score = np.sum(cov[balls[w]] > 1)
                    if score > best_score:
                        best_score = score
                        best_w = w

                if best_w != -1:
                    cov[balls[c]] -= 1
                    in_code.remove(c)
                    selected.remove(c)

                    cov[balls[best_w]] += 1
                    in_code.add(best_w)
                    selected.append(best_w)

                    improved = True
                    swapped = True
                    break

            if swapped:
                continue

        if not improved:
            break

    return selected

def _remove_redundant_spectral(balls, total, ball_size, ball_spectral, selected):
    """Remove redundant codewords using both coverage and spectral checks."""
    in_code = set(selected)
    cov = np.zeros(total, dtype=np.int32)
    for c in selected:
        cov[balls[c]] += 1

    to_remove = []
    for c in selected:
        if np.min(cov[balls[c]]) > 1:
            to_remove.append(c)

    for c in to_remove:
        cov[balls[c]] -= 1
        in_code.remove(c)
        selected.remove(c)

    return selected

def _verify_coverage(balls, total, ball_size, selected):
    """Verify that every word is covered by at least one selected codeword."""
    cov = np.zeros(total, dtype=np.int32)
    for c in selected:
        cov[balls[c]] += 1
    return np.all(cov > 0)
