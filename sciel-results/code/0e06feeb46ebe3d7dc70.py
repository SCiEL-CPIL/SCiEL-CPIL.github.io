import time
import random
import sys
from itertools import product

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin

    total_words = q ** n

    # For the specific problem q=3, n=7, radius=1, total_words=2187
    # We can use bitsets for efficiency

    # Precompute neighbors for each word
    # neighbors[i] is a set of indices within Hamming distance radius from word i
    # Also precompute the "covering" relationship: which words are covered by codeword i

    # Helper functions
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return w

    def word_to_idx(w):
        idx = 0
        for i in range(n-1, -1, -1):
            idx = idx * q + w[i]
        return idx

    # Generate all words
    all_words = [idx_to_word(i) for i in range(total_words)]

    # Precompute neighbors (words within Hamming distance radius)
    # For radius=1: self + n*(q-1) neighbors
    # For general radius: need to generate all combinations

    # Build coverage masks: cover_mask[i] is a bitset of all words covered by codeword i
    # Using Python integers as bitsets (arbitrary precision)

    # For efficiency with 2187 words, we can use Python's arbitrary precision integers
    # Each bit position corresponds to a word index

    cover_masks = [0] * total_words

    if radius == 1:
        for i in range(total_words):
            w = all_words[i]
            mask = 1 << i  # self
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]:
                        continue
                    new_w = w[:]
                    new_w[pos] = val
                    j = word_to_idx(new_w)
                    mask |= (1 << j)
            cover_masks[i] = mask
    else:
        # General case: compute Hamming distance
        for i in range(total_words):
            w = all_words[i]
            mask = 0
            for j in range(total_words):
                u = all_words[j]
                dist = sum(1 for a, b in zip(w, u) if a != b)
                if dist <= radius:
                    mask |= (1 << j)
            cover_masks[i] = mask

    # Greedy algorithm to build initial cover
    def greedy_cover():
        remaining = (1 << total_words) - 1
        chosen = []
        while remaining:
            best_idx = -1
            best_count = -1
            for i in range(total_words):
                if not (cover_masks[i] & remaining):
                    continue
                count = (cover_masks[i] & remaining).bit_count()
                if count > best_count:
                    best_count = count
                    best_idx = i
            if best_idx == -1:
                break
            chosen.append(best_idx)
            remaining &= ~cover_masks[best_idx]
        return chosen

    # Local search: try to remove redundant codewords
    def try_reduce(chosen):
        chosen_set = set(chosen)
        changed = True
        while changed:
            changed = False
            for i in list(chosen_set):
                # Check if i is redundant
                union_others = 0
                for j in chosen_set:
                    if j == i:
                        continue
                    union_others |= cover_masks[j]
                if (cover_masks[i] & ~union_others) == 0:
                    chosen_set.remove(i)
                    changed = True
        return list(chosen_set)

    # Local search: try swaps to reduce size
    def try_swap_improve(chosen):
        chosen_set = set(chosen)
        improved = True
        while improved:
            improved = False
            for i in list(chosen_set):
                # Try removing i
                if (cover_masks[i] & ~((1 << total_words) - 1)) == 0:
                    pass
                # Check if i is redundant
                union_others = 0
                for j in chosen_set:
                    if j == i:
                        continue
                    union_others |= cover_masks[j]
                if (cover_masks[i] & ~union_others) == 0:
                    chosen_set.remove(i)
                    improved = True
                    break
        return list(chosen_set)

    # More aggressive local search: try replacing one codeword with another
    def local_search(chosen, max_iters=100):
        chosen_set = set(chosen)
        best_size = len(chosen_set)
        best_set = set(chosen_set)

        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break

            # Try removing each codeword and see if we can re-add fewer
            improved = False
            for i in list(chosen_set):
                # Temporarily remove i
                chosen_set.remove(i)

                # Check coverage
                union = 0
                for j in chosen_set:
                    union |= cover_masks[j]

                # Find uncovered words
                uncovered = ~union & ((1 << total_words) - 1)

                if uncovered == 0:
                    # i was redundant, keep it removed
                    best_size = len(chosen_set)
                    best_set = set(chosen_set)
                    improved = True
                    break
                else:
                    # Try to find a single codeword that covers all uncovered
                    found = False
                    for k in range(total_words):
                        if k in chosen_set:
                            continue
                        if (cover_masks[k] & uncovered) == uncovered:
                            chosen_set.add(k)
                            best_size = len(chosen_set)
                            best_set = set(chosen_set)
                            found = True
                            improved = True
                            break
                    if not found:
                        # Restore i
                        chosen_set.add(i)

            if not improved:
                break

        return list(best_set)

    # Run multiple random restarts
    best_cover = None
    best_size = float('inf')

    # First, run greedy
    initial = greedy_cover()
    reduced = try_reduce(initial)
    if len(reduced) < best_size:
        best_size = len(reduced)
        best_cover = reduced

    # Run local search
    if time.time() < deadline - 10:
        ls_result = local_search(reduced, max_iters=50)
        if len(ls_result) < best_size:
            best_size = len(ls_result)
            best_cover = ls_result

    # Try random restarts with perturbation
    restart_count = 0
    while time.time() < deadline - 5:
        restart_count += 1

        # Perturb the current best
        if best_cover is None:
            candidate = greedy_cover()
        else:
            candidate = list(best_cover)
            # Remove a random codeword
            if len(candidate) > 1:
                idx_to_remove = random.randint(0, len(candidate) - 1)
                candidate.pop(idx_to_remove)

        # Greedily add back to restore coverage
        candidate_set = set(candidate)
        union = 0
        for j in candidate_set:
            union |= cover_masks[j]
        remaining = ~union & ((1 << total_words) - 1)

        while remaining:
            best_idx = -1
            best_count = -1
            for i in range(total_words):
                if i in candidate_set:
                    continue
                count = (cover_masks[i] & remaining).bit_count()
                if count > best_count:
                    best_count = count
                    best_idx = i
            if best_idx == -1:
                break
            candidate_set.add(best_idx)
            remaining &= ~cover_masks[best_idx]

        candidate = list(candidate_set)
        reduced = try_reduce(candidate)

        if len(reduced) < best_size:
            best_size = len(reduced)
            best_cover = reduced

        # Local search on the new candidate
        if time.time() < deadline - 5:
            ls_result = local_search(reduced, max_iters=20)
            if len(ls_result) < best_size:
                best_size = len(ls_result)
                best_cover = ls_result

    if best_cover is None:
        best_cover = list(range(total_words))

    return [idx_to_word(i) for i in best_cover]
