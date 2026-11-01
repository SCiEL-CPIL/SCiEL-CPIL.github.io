import time
import itertools
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1(deadline)

    # Fallback for other cases
    if q == 3 and n >= 4:
        # Direct sum with tetracode
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Generic fallback
    return [list(c) for c in itertools.product(range(q), repeat=n)]


def _construct_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561

    # Precompute neighbors: for each word, the 17 words within distance 1
    neighbors = np.zeros((total, 17), dtype=np.int32)
    for w in range(total):
        neighbors[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = 3 ** (n - 1 - pos)
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                neighbors[w, idx] = w + (new_digit - digit) * weight
                idx += 1

    # Build the seed: tetracode ⊗ F_3^4 (486 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    seed_words = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=4):
            word = h + list(tail)
            # Convert to integer
            w = 0
            for d in word:
                w = w * 3 + d
            seed_words.append(w)

    # Verify seed is a valid cover
    cover_count = np.zeros(total, dtype=np.int32)
    for w in seed_words:
        cover_count[neighbors[w]] += 1
    assert np.all(cover_count >= 1), "Seed is not a valid cover"

    best_size = len(seed_words)
    best_set = seed_words[:]

    def is_valid(cover_count):
        return np.all(cover_count >= 1)

    # Local search with 1<->1 and 2<->2 swaps
    current = seed_words[:]
    current_set = set(current)
    cover = cover_count.copy()

    # Precompute: for each word, its neighbor indices
    # We'll use neighbors matrix directly

    while time.time() < deadline - 3.0:
        # Try 1<->1 swaps: remove one codeword, add one new one
        found_swap = False
        # Sample a subset of codewords to try removing (to save time)
        try_count = min(200, len(current))
        for _ in range(try_count):
            if time.time() > deadline - 4.0:
                break
            idx = random.randrange(len(current))
            cw_remove = current[idx]
            ball = neighbors[cw_remove]
            
            # Which words become uncovered if we remove cw_remove?
            # Words that are ONLY covered by cw_remove
            # cover[ball] == 1 means only cw_remove covers them
            only_covered = ball[cover[ball] == 1]
            
            if len(only_covered) == 0:
                # Safe to remove without adding
                cover[ball] -= 1
                current.pop(idx)
                current_set.discard(cw_remove)
                found_swap = True
                if len(current) < best_size:
                    best_size = len(current)
                    best_set = current[:]
                continue
            
            # Need to add a word that covers all of only_covered
            # For each candidate new word, check if it covers all only_covered
            # A new word v covers only_covered if for each u in only_covered, u is in neighbors[v]
            # i.e., v is in neighbors[u] for all u in only_covered
            # So v must be in the intersection of neighbors[u] for u in only_covered
            # And v must not already be in current_set
            
            # Compute intersection of neighbors for all u in only_covered
            # Start with neighbors of first element
            if len(only_covered) > 10:
                # Too many constraints, skip (unlikely to find a single word)
                continue
            
            # Find common neighbors
            common = set(neighbors[only_covered[0]].tolist())
            for u in only_covered[1:]:
                common &= set(neighbors[u].tolist())
                if not common:
                    break
            
            if not common:
                continue
            
            # Remove current codewords from candidates
            common -= current_set
            if not common:
                continue
            
            # Pick a random candidate
            candidate = random.choice(list(common))
            
            # Verify: adding candidate covers all only_covered
            # (already ensured by construction)
            # Remove cw_remove, add candidate
            cover[ball] -= 1
            cover[neighbors[candidate]] += 1
            
            if is_valid(cover):
                current.pop(idx)
                current_set.discard(cw_remove)
                current.append(candidate)
                current_set.add(candidate)
                found_swap = True
                if len(current) < best_size:
                    best_size = len(current)
                    best_set = current[:]
            else:
                # Revert
                cover[neighbors[candidate]] -= 1
                cover[ball] += 1

        # Try 2<->2 swaps
        if time.time() < deadline - 5.0:
            try_count_2 = min(50, len(current))
            for _ in range(try_count_2):
                if time.time() > deadline - 5.0:
                    break
                i1 = random.randrange(len(current))
                i2 = random.randrange(len(current))
                if i1 == i2:
                    continue
                cw1 = current[i1]
                cw2 = current[i2]
                ball1 = neighbors[cw1]
                ball2 = neighbors[cw2]
                
                # Words only covered by cw1 or cw2
                # After removing both, words with cover==0 need to be covered by new words
                cover[ball1] -= 1
                cover[ball2] -= 1
                
                uncovered = np.where(cover == 0)[0]
                
                if len(uncovered) == 0:
                    # Safe to remove both without adding
                    current.pop(max(i1, i2))
                    current.pop(min(i1, i2))
                    current_set.discard(cw1)
                    current_set.discard(cw2)
                    found_swap = True
                    if len(current) < best_size:
                        best_size = len(current)
                        best_set = current[:]
                    continue
                
                if len(uncovered) > 20:
                    # Revert
                    cover[ball2] += 1
                    cover[ball1] += 1
                    continue
                
                # Need to add 2 words that cover all uncovered
                # For each pair of candidates, check coverage
                # Find candidates that cover at least one uncovered word
                # Then try pairs
                
                # Get all words that are neighbors of any uncovered word
                candidate_set = set()
                for u in uncovered:
                    candidate_set.update(neighbors[u].tolist())
                candidate_set -= current_set
                
                if len(candidate_set) < 2:
                    cover[ball2] += 1
                    cover[ball1] += 1
                    continue
                
                candidates = list(candidate_set)
                random.shuffle(candidates)
                
                # Try a few pairs
                found_pair = False
                max_pairs = min(20, len(candidates) * (len(candidates) - 1) // 2)
                for p in range(max_pairs):
                    if time.time() > deadline - 5.0:
                        break
                    a = random.choice(candidates)
                    b = random.choice(candidates)
                    if a == b:
                        continue
                    cover[neighbors[a]] += 1
                    cover[neighbors[b]] += 1
                    if is_valid(cover):
                        current.pop(max(i1, i2))
                        current.pop(min(i1, i2))
                        current_set.discard(cw1)
                        current_set.discard(cw2)
                        current.append(a)
                        current.append(b)
                        current_set.add(a)
                        current_set.add(b)
                        found_swap = True
                        found_pair = True
                        if len(current) < best_size:
                            best_size = len(current)
                            best_set = current[:]
                        break
                    else:
                        cover[neighbors[b]] -= 1
                        cover[neighbors[a]] -= 1
                
                if not found_pair:
                    cover[ball2] += 1
                    cover[ball1] += 1

        # If no swaps found, try a random perturbation: remove a few, add greedy
        if not found_swap and time.time() < deadline - 8.0 and len(current) > 486:
            orig_current = current[:]
            orig_set = current_set.copy()
            orig_cover = cover.copy()
            
            num_remove = min(random.randint(2, 4), len(current))
            for _ in range(num_remove):
                idx = random.randrange(len(current))
                cw = current.pop(idx)
                current_set.discard(cw)
                cover[neighbors[cw]] -= 1
            
            # Greedy repair
            for _ in range(num_remove + 1):
                if time.time() > deadline - 5.0:
                    break
                uncov = np.where(cover == 0)[0]
                if len(uncov) == 0:
                    break
                # Find best word to add
                u_mask = np.zeros(total, dtype=bool)
                u_mask[uncov] = True
                counts = u_mask[neighbors].sum(axis=1)
                counts[list(current_set)] = 0
                best_cw = int(np.argmax(counts))
                if counts[best_cw] == 0:
                    break
                current.append(best_cw)
                current_set.add(best_cw)
                cover[neighbors[best_cw]] += 1
            
            # Check if valid
            if is_valid(cover):
                if len(current) < best_size:
                    best_size = len(current)
                    best_set = current[:]
            else:
                # Revert
                current = orig_current
                current_set = orig_set
                cover = orig_cover

        # Early exit if we've hit the known optimum
        if best_size <= 486:
            break

    # Convert best_set to list of digit lists
    if best_set is None:
        return [[0] * n]

    result = []
    for w in best_set:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)

    return result
