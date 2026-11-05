"""Ternary covering code K_3(7,1) via randomized greedy + local search."""
import time
import random
import itertools
import numpy as np

# Deadline guard: keep a safety margin before the 150s wall-clock kill.
_START = time.time()
DEADLINE = _START + 142.0  # 8s margin


def _elapsed():
    return time.time() - _START


def construct(q, n, radius):
    # For this task q=3, n=7, radius=1. We build a covering code by:
    # 1) Greedy randomized construction (multiple restarts).
    # 2) Local search: try to delete codewords that are redundant, and try swaps.
    # 3) Keep the best (fewest codewords) valid solution.

    if q != 3 or n != 7 or radius != 1:
        # Fallback: simple product construction
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            # n < 4, just use all words
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    q, n, radius = 3, 7, 1
    total_words = q ** n  # 2187

    # Precompute all words as indices 0..2186
    # Word index -> list of digits
    # We'll represent a word as an integer 0..2186 (base-3, 7 digits)
    
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return w

    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    # Precompute for each word index, its neighbors within Hamming distance <= 1
    # neighbor_indices[word_idx] = list of word indices within distance 1
    neighbor_indices = []
    for idx in range(total_words):
        w = idx_to_word(idx)
        nbrs = [idx]
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val == orig:
                    continue
                w[pos] = val
                nbrs.append(word_to_idx(w))
            w[pos] = orig
        neighbor_indices.append(nbrs)

    # We'll maintain a boolean array 'covered' of size total_words
    # and a list 'code' of chosen codeword indices.

    def is_covered(code_indices):
        """Check if all words are covered by the given code indices."""
        covered = np.zeros(total_words, dtype=bool)
        for c in code_indices:
            covered[neighbor_indices[c]] = True
        return covered.all()

    def greedy_randomized():
        """Randomized greedy: repeatedly pick a word that covers the most uncovered words."""
        covered = np.zeros(total_words, dtype=bool)
        code = []
        # To speed up, track uncovered count per candidate
        # We'll use a simpler approach: iterate, compute gain, pick best (with random tie-breaking)
        
        # Precompute neighbor sets as arrays for speed
        nbr_arrays = [np.array(nbrs) for nbrs in neighbor_indices]
        
        remaining = set(range(total_words))
        
        while remaining:
            if _elapsed() > DEADLINE:
                break
            best_gain = -1
            best_candidates = []
            # Sample a subset of candidates if too many
            sample = list(remaining)
            random.shuffle(sample)
            limit = min(len(sample), 200)
            for c in sample[:limit]:
                nbrs = nbr_arrays[c]
                gain = np.sum(~covered[nbrs])
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [c]
                elif gain == best_gain:
                    best_candidates.append(c)
            if not best_candidates:
                # Fallback: pick random
                c = random.choice(list(remaining))
            else:
                c = random.choice(best_candidates)
            code.append(c)
            covered[nbr_arrays[c]] = True
            # Update remaining: remove all covered
            # This is O(total_words) each time, but total_words=2187, and code size ~186, so ~400k ops
            # Actually we can just rebuild remaining
            remaining = set(i for i in range(total_words) if not covered[i])
        
        return code

    def local_search(code):
        """Try to delete redundant codewords and perform swaps."""
        if not code:
            return code
        code = list(code)
        covered = np.zeros(total_words, dtype=bool)
        for c in code:
            covered[neighbor_indices[c]] = True
        
        nbr_arrays = [np.array(nbrs) for nbrs in neighbor_indices]
        
        improved = True
        while improved:
            if _elapsed() > DEADLINE:
                break
            improved = False
            # Try deletion
            for i in range(len(code)):
                if _elapsed() > DEADLINE:
                    break
                c = code[i]
                # Check if removing c leaves all words covered
                # Temporarily remove c
                old_covered = covered.copy()
                covered[nbr_arrays[c]] = False
                # Recompute coverage from remaining codewords
                # This is expensive; instead, count how many words are only covered by c
                # Better: for each neighbor of c, check if it's covered by another codeword
                # We'll do a quick check: if any neighbor of c becomes uncovered, deletion fails
                # Actually, we need to check if ALL words are still covered.
                # Efficient: count for each word in nbrs[c] how many codewords cover it.
                # Precompute coverage counts.
                pass
            
            # Simpler deletion check:
            # For each codeword c, check if there exists a word w in ball(c) such that c is the ONLY codeword covering w.
            # If no such w, c is redundant.
            # We maintain cover_count[w] = number of codewords covering w.
            cover_count = np.zeros(total_words, dtype=int)
            for c in code:
                cover_count[nbr_arrays[c]] += 1
            
            new_code = []
            for c in code:
                if _elapsed() > DEADLINE:
                    break
                # Check if c is redundant: min cover_count in ball(c) >= 2
                min_count = np.min(cover_count[nbr_arrays[c]])
                if min_count >= 2:
                    # c is redundant, remove it
                    cover_count[nbr_arrays[c]] -= 1
                    improved = True
                else:
                    new_code.append(c)
            code = new_code
        
        # Try swaps: replace one codeword with another to enable further deletions
        # This is more complex; we'll do a simple version
        for _ in range(3):  # limited swaps
            if _elapsed() > DEADLINE:
                break
            cover_count = np.zeros(total_words, dtype=int)
            for c in code:
                cover_count[nbr_arrays[c]] += 1
            
            found_swap = False
            for i in range(len(code)):
                if _elapsed() > DEADLINE:
                    break
                c_old = code[i]
                # Try replacing c_old with a random new codeword
                for _ in range(5):
                    c_new = random.randint(0, total_words - 1)
                    if c_new == c_old:
                        continue
                    # Check if replacing c_old with c_new keeps coverage
                    # Words that were only covered by c_old must be covered by c_new
                    # And words covered by c_new are fine
                    # Check: for each w in ball(c_old), if cover_count[w] == 1, then w must be in ball(c_new)
                    # And after replacement, all words must be covered
                    # Simpler: simulate
                    cover_count[nbr_arrays[c_old]] -= 1
                    cover_count[nbr_arrays[c_new]] += 1
                    if np.min(cover_count) >= 1:
                        # Valid swap
                        code[i] = c_new
                        found_swap = True
                        improved = True
                        break
                    else:
                        cover_count[nbr_arrays[c_old]] += 1
                        cover_count[nbr_arrays[c_new]] -= 1
                if found_swap:
                    break
            
            if not found_swap:
                break
            
            # After swap, try deletions again
            cover_count = np.zeros(total_words, dtype=int)
            for c in code:
                cover_count[nbr_arrays[c]] += 1
            new_code = []
            for c in code:
                min_count = np.min(cover_count[nbr_arrays[c]])
                if min_count >= 2:
                    cover_count[nbr_arrays[c]] -= 1
                else:
                    new_code.append(c)
            code = new_code
        
        return code

    best_code = None
    best_len = float('inf')
    
    # Multiple restarts
    restarts = 0
    while _elapsed() < DEADLINE - 5:  # Leave 5s for final validation
        restarts += 1
        code = greedy_randomized()
        code = local_search(code)
        
        # Validate
        if is_covered(code):
            if len(code) < best_len:
                best_len = len(code)
                best_code = list(code)
        
        # If we've hit a good number, we can be more selective
        if best_len <= 180:
            break
    
    if best_code is None:
        # Fallback: use the tetracode product construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        best_code = [word_to_idx(h + list(tail)) for h in tetracode for tail in itertools.product(range(3), repeat=3)]
    
    # Final validation
    if not is_covered(best_code):
        # Should not happen, but fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        best_code = [word_to_idx(h + list(tail)) for h in tetracode for tail in itertools.product(range(3), repeat=3)]
    
    # Convert back to list of lists
    return [idx_to_word(c) for c in best_code]
