"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    # Deadline guard: keep a safety margin before the 120s evaluator limit.
    deadline = time.time() + 110.0

    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters: greedy set cover.
        return _greedy_set_cover(q, n, radius, deadline)

    # Ternary Hamming(7,4) code has 9 codewords of length 7, covering radius 1.
    # We need length 6, so we can't use it directly.
    # The tetracode (length 4, 9 words, radius 1) is perfect for n=4.
    # For n=6, we need a better construction.

    # Start with the direct sum construction: 9 * 3^(6-4) = 9*9 = 81 words.
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Convert to numpy for efficient operations
    code = np.array(initial_code)
    
    # Build coverage: which words are covered by the current code
    # For n=6, q=3, total words = 729
    # Each codeword covers 1 + n*(q-1) = 1 + 6*2 = 13 words
    
    # Precompute neighbors for each word (index 0..728)
    # A word is an integer 0..728 representing base-3 number
    # neighbors[i] = list of indices at Hamming distance 1 from i
    
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n-1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute all neighbors
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx_map = {w: i for i, w in enumerate(all_words)}
    
    neighbors = [[] for _ in range(q**n)]
    for i, w in enumerate(all_words):
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    nw = list(w)
                    nw[pos] = val
                    j = word_to_idx_map[tuple(nw)]
                    neighbors[i].append(j)

    # Current coverage set
    covered = np.zeros(q**n, dtype=bool)
    for cw in code:
        idx = word_to_idx(cw)
        covered[idx] = True
        for nb in neighbors[idx]:
            covered[nb] = True

    # Greedy local search: try to remove codewords that are redundant
    # A codeword is redundant if all words it covers are also covered by other codewords
    
    def is_redundant(cw_idx, code_set, covered):
        """Check if removing code_set[cw_idx] still leaves all its covered words covered."""
        idx = word_to_idx(code_set[cw_idx])
        # Words covered by this codeword: itself and its neighbors
        covered_by_this = set([idx] + neighbors[idx])
        # Check if each of these is covered by some other codeword
        for w_idx in covered_by_this:
            # Count how many codewords cover w_idx
            count = 0
            for other_idx in code_set:
                if other_idx == idx:
                    continue
                o_idx = word_to_idx(other_idx)
                # Check if w_idx is within distance 1 of o_idx
                # This is expensive; use precomputed neighbors
                if w_idx == o_idx or w_idx in neighbors[o_idx]:
                    count += 1
                    if count >= 1:
                        break
            if count == 0:
                return False
        return True

    # Faster redundancy check using coverage counts
    # coverage_count[i] = number of codewords that cover word i
    coverage_count = np.zeros(q**n, dtype=int)
    for cw in code:
        idx = word_to_idx(cw)
        coverage_count[idx] += 1
        for nb in neighbors[idx]:
            coverage_count[nb] += 1

    def try_remove(cw_idx, coverage_count):
        """Try to remove codeword at index cw_idx. Returns True if successful."""
        idx = word_to_idx(code[cw_idx])
        # Words that would become uncovered if we remove this codeword
        at_risk = [idx] + neighbors[idx]
        for w_idx in at_risk:
            if coverage_count[w_idx] <= 1:
                return False
        # Remove it
        coverage_count[idx] -= 1
        for nb in neighbors[idx]:
            coverage_count[nb] -= 1
        code = np.delete(code, cw_idx, axis=0)
        return True, code, coverage_count

    # Iterative removal
    improved = True
    while improved:
        if time.time() > deadline - 10:
            break
        improved = False
        # Try to remove each codeword
        for i in range(len(code)):
            if time.time() > deadline - 10:
                break
            # Check if removable
            idx = word_to_idx(code[i])
            at_risk = [idx] + neighbors[idx]
            removable = True
            for w_idx in at_risk:
                if coverage_count[w_idx] <= 1:
                    removable = False
                    break
            if removable:
                # Remove it
                coverage_count[idx] -= 1
                for nb in neighbors[idx]:
                    coverage_count[nb] -= 1
                code = np.delete(code, i, axis=0)
                improved = True
                break  # Restart scan after removal

    # Now try local search: swap codewords to reduce size
    # For each pair of codewords, try removing one and adding another that covers more
    # Actually, simpler: try to replace two codewords with one that covers the union
    
    # Better approach: iterative improvement
    # Try removing each codeword and then greedily re-covering uncovered words with minimal additions
    
    best_code = code.copy()
    best_size = len(code)
    
    for iteration in range(200):
        if time.time() > deadline - 15:
            break
        # Try removing each codeword
        for i in range(len(code)):
            if time.time() > deadline - 15:
                break
            idx = word_to_idx(code[i])
            at_risk = [idx] + neighbors[idx]
            removable = True
            for w_idx in at_risk:
                if coverage_count[w_idx] <= 1:
                    removable = False
                    break
            if not removable:
                continue
            
            # Tentatively remove
            old_coverage = coverage_count.copy()
            coverage_count[idx] -= 1
            for nb in neighbors[idx]:
                coverage_count[nb] -= 1
            new_code = np.delete(code, i, axis=0)
            
            # Check if still fully covered
            if not np.all(coverage_count >= 1):
                # Need to re-cover. Find uncovered words
                uncovered = np.where(coverage_count == 0)[0]
                # Greedily add codewords to cover them
                added = 0
                for u_idx in uncovered:
                    if time.time() > deadline - 15:
                        break
                    # Find codeword (from new_code or all_words) that covers u_idx and most other uncovered
                    # For simplicity, just add a random neighbor
                    # Actually, try to find a word that covers u_idx
                    # The word u_idx itself covers u_idx
                    candidate = idx_to_word(u_idx)
                    # Check if already in new_code
                    if not any(np.array_equal(c, candidate) for c in new_code):
                        new_code = np.vstack([new_code, [candidate]])
                        c_idx = word_to_idx(candidate)
                        coverage_count[c_idx] += 1
                        for nb in neighbors[c_idx]:
                            coverage_count[nb] += 1
                        added += 1
                    else:
                        # Find another word that covers u_idx
                        for nb in neighbors[u_idx]:
                            candidate = idx_to_word(nb)
                            if not any(np.array_equal(c, candidate) for c in new_code):
                                new_code = np.vstack([new_code, [candidate]])
                                c_idx = word_to_idx(candidate)
                                coverage_count[c_idx] += 1
                                for nb2 in neighbors[c_idx]:
                                    coverage_count[nb2] += 1
                                added += 1
                                break
                        else:
                            # u_idx itself must be added
                            candidate = idx_to_word(u_idx)
                            new_code = np.vstack([new_code, [candidate]])
                            c_idx = word_to_idx(candidate)
                            coverage_count[c_idx] += 1
                            for nb in neighbors[c_idx]:
                                coverage_count[nb] += 1
                            added += 1
                
                # If we added fewer than 1 (i.e., 0), we improved
                if added == 0:
                    code = new_code
                    best_code = code.copy()
                    best_size = len(code)
                else:
                    # Revert
                    code = best_code if len(best_code) < len(new_code) else code
                    coverage_count = old_coverage
                    # Recompute if we changed
                    if code is not best_code:
                        coverage_count = np.zeros(q**n, dtype=int)
                        for cw in code:
                            c_idx = word_to_idx(cw)
                            coverage_count[c_idx] += 1
                            for nb in neighbors[c_idx]:
                                coverage_count[nb] += 1
            else:
                # Successfully removed without adding
                code = new_code
                best_code = code.copy()
                best_size = len(code)

    # Return as list of lists
    return [list(cw) for cw in best_code]


def _greedy_set_cover(q, n, radius, deadline):
    """Greedy set cover for general parameters."""
    all_words = list(itertools.product(range(q), repeat=n))
    total = q ** n
    
    # Precompute neighbors within radius
    def hamming_dist(a, b):
        return sum(1 for x, y in zip(a, b) if x != y)
    
    # For each word, find all words within radius
    coverage = {}
    for i, w in enumerate(all_words):
        covered = set()
        for j, v in enumerate(all_words):
            if hamming_dist(w, v) <= radius:
                covered.add(j)
        coverage[i] = covered
    
    uncovered = set(range(total))
    code = []
    
    while uncovered and time.time() < deadline - 5:
        # Find word that covers most uncovered
        best_i = -1
        best_count = 0
        for i in range(total):
            count = len(uncovered & coverage[i])
            if count > best_count:
                best_count = count
                best_i = i
            if time.time() > deadline - 5:
                break
        if best_i == -1:
            break
        code.append(all_words[best_i])
        uncovered -= coverage[best_i]
    
    return code
# EVOLVE-BLOCK-END
