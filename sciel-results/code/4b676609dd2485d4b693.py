"""Iterated Local Search with repair for K_3(6,1)."""
import time
import random
import itertools
from collections import Counter

# Deadline guard: evaluator kills at 150s wall-clock; keep a safety margin.
_DEADLINE = 150.0
_START = None


def construct(q, n, radius):
    global _START
    if _START is None:
        _START = time.time()
    if q != 3 or n != 6 or radius != 1:
        # Fallback: direct sum construction (works for any q,n,r=1 if n>=4 and q=3)
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        # Generic greedy fallback
        return _greedy_cover(q, n, radius)

    # Precompute all words as integers 0..728
    total = q ** n  # 729
    # For each codeword, precompute the set of words it covers (itself + all 1-error neighbors)
    # Represent each word as an integer in [0, total)
    # Precompute neighbors: for each word w and each position i and each symbol s != w[i], the neighbor
    
    def word_to_int(w):
        v = 0
        for x in w:
            v = v * q + x
        return v

    def int_to_word(v):
        w = []
        for _ in range(n):
            w.append(v % q)
            v //= q
        w.reverse()
        return w

    # Precompute coverage for each possible codeword index (0..728)
    # coverage[c] = frozenset of word-integers covered by codeword c
    # This is expensive to store as sets for all 729, but we can compute on the fly or precompute
    # Actually, let's precompute the list of covered words for each codeword index
    # Each codeword covers 1 + n*(q-1) = 1 + 6*2 = 13 words
    
    coverage = [None] * total
    for c in range(total):
        cw = int_to_word(c)
        covered = [c]
        for i in range(n):
            orig = cw[i]
            for s in range(q):
                if s == orig:
                    continue
                nw = cw[:]
                nw[i] = s
                covered.append(word_to_int(nw))
        coverage[c] = frozenset(covered)

    # Greedy initial solution
    def greedy_initial():
        covered = [False] * total
        uncovered_count = total
        codewords = []
        while uncovered_count > 0:
            best_c = -1
            best_gain = 0
            for c in range(total):
                if c in codewords_set:
                    continue
                gain = sum(1 for w in coverage[c] if not covered[w])
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            if best_c == -1:
                break
            codewords.append(best_c)
            codewords_set.add(best_c)
            for w in coverage[best_c]:
                if not covered[w]:
                    covered[w] = True
                    uncovered_count -= 1
        return codewords

    codewords_set = set()
    codewords = greedy_initial()
    codewords_set = set(codewords)
    best = list(codewords)
    best_len = len(best)

    # Local search: try to remove redundant codewords, then try swaps (remove one, add one)
    def is_removable(codewords_list, codewords_set_local, c):
        """Check if codeword c can be removed while maintaining coverage."""
        # Check if every word covered by c is also covered by some other codeword
        for w in coverage[c]:
            # Check if w is covered by any other codeword
            found = False
            for other in codewords_list:
                if other == c:
                    continue
                if w in coverage[other]:
                    found = True
                    break
            if not found:
                return False
        return True

    def try_remove_redundant(codewords_list, codewords_set_local):
        """Iteratively remove redundant codewords."""
        changed = True
        while changed:
            changed = False
            for c in list(codewords_list):
                if is_removable(codewords_list, codewords_set_local, c):
                    codewords_list.remove(c)
                    codewords_set_local.discard(c)
                    changed = True
        return codewords_list

    # Iterated Local Search
    rng = random.Random(42)
    iterations = 0
    while time.time() - _START < _DEADLINE - 5:  # 5s safety margin
        iterations += 1
        current = list(best)
        current_set = set(current)

        # Local search: remove redundants, then try 1-for-1 swaps
        # First pass: remove redundants
        current = try_remove_redundant(current, current_set)

        # Try to improve by swapping: remove one codeword, add another that covers more
        improved = True
        while improved and time.time() - _START < _DEADLINE - 5:
            improved = False
            for i, c in enumerate(current):
                # Try removing c
                if not is_removable(current, current_set, c):
                    continue
                # Can't just remove if it reduces coverage... wait, is_removable means coverage is maintained
                # So we can remove it
                new_current = [x for x in current if x != c]
                new_set = set(new_current)
                # Now try to see if we can keep the same size or improve by adding a different codeword
                # Actually, if we can remove c, we have a shorter code. That's an improvement.
                current = new_current
                current_set = new_set
                improved = True
                break

        # Now try 1-for-1 swaps: remove c, add d where d covers words not covered by others
        # More precisely: find c and d such that removing c and adding d maintains coverage and 
        # potentially allows further reductions. For now, let's just try to find any swap that
        # leads to a removable codeword after the swap.
        
        # Simpler approach: try all pairs (remove c, add d) where d is not in current
        # and check if the resulting set is valid and shorter or equal but with different structure
        # This is O(k^2 * total) which might be too slow. Let's be smarter.

        # For each codeword c in current, check if there's a d not in current such that
        # replacing c with d maintains coverage.
        # Coverage is maintained if: every word covered only by c (among current) is also covered by d.
        
        # Precompute for current: for each word w, how many codewords in current cover it
        # This is O(k * 13) per iteration
        cover_count = [0] * total
        for c in current:
            for w in coverage[c]:
                cover_count[w] += 1
        
        # Words covered only by one codeword
        for i, c in enumerate(current):
            # Find words that only c covers
            critical_words = [w for w in coverage[c] if cover_count[w] == 1]
            if not critical_words:
                # c is redundant, should have been removed
                continue
            # Try to find d not in current that covers all critical_words
            for d in range(total):
                if d in current_set:
                    continue
                if all(w in coverage[d] for w in critical_words):
                    # Swap c with d
                    current[i] = d
                    current_set.discard(c)
                    current_set.add(d)
                    improved = True
                    break
            if improved:
                break
        
        if len(current) < best_len:
            best = list(current)
            best_len = len(current)
            # Restart local search from this better solution
            continue

        # Perturbation: randomly remove a codeword and repair
        if len(current) > 0 and time.time() - _START < _DEADLINE - 5:
            # Pick a random codeword to remove
            idx = rng.randint(0, len(current) - 1)
            removed = current[idx]
            current = [x for x in current if x != removed]
            current_set = set(current)
            
            # Repair: greedily add codewords to restore coverage
            # Find uncovered words
            covered = [False] * total
            for c in current:
                for w in coverage[c]:
                    covered[w] = True
            
            uncovered = [w for w in range(total) if not covered[w]]
            while uncovered:
                # Find codeword that covers most uncovered words
                best_c = -1
                best_gain = 0
                for c in range(total):
                    if c in current_set:
                        continue
                    gain = sum(1 for w in uncovered if covered_by(c, w, coverage))
                    if gain > best_gain:
                        best_gain = gain
                        best_c = c
                if best_c == -1:
                    break
                current.append(best_c)
                current_set.add(best_c)
                for w in coverage[best_c]:
                    if not covered[w]:
                        covered[w] = True
                        # Remove from uncovered
                        if w in uncovered:
                            uncovered.remove(w)
            
            # Local search on repaired solution
            current = try_remove_redundant(current, current_set)
            
            if len(current) < best_len:
                best = list(current)
                best_len = len(current)

    # Final cleanup
    best_set = set(best)
    best = try_remove_redundant(best, best_set)
    
    # Convert to list of lists
    return [int_to_word(c) for c in best]


def covered_by(c, w, coverage):
    return w in coverage[c]


def _greedy_cover(q, n, radius):
    """Simple greedy covering code for general parameters."""
    total = q ** n
    covered = [False] * total
    codewords = []
    
    def word_to_int(w):
        v = 0
        for x in w:
            v = v * q + x
        return v
    
    def int_to_word(v):
        w = []
        for _ in range(n):
            w.append(v % q)
            v //= q
        w.reverse()
        return w
    
    # Precompute coverage
    coverage = [None] * total
    for c in range(total):
        cw = int_to_word(c)
        covered_set = [c]
        for i in range(n):
            for s in range(q):
                if s == cw[i]:
                    continue
                nw = cw[:]
                nw[i] = s
                covered_set.append(word_to_int(nw))
        coverage[c] = frozenset(covered_set)
    
    uncovered_count = total
    while uncovered_count > 0:
        best_c = -1
        best_gain = 0
        for c in range(total):
            if c in set(codewords):
                continue
            gain = sum(1 for w in coverage[c] if not covered[w])
            if gain > best_gain:
                best_gain = gain
                best_c = c
        if best_c == -1:
            break
        codewords.append(best_c)
        for w in coverage[best_c]:
            if not covered[w]:
                covered[w] = True
                uncovered_count -= 1
    
    return [int_to_word(c) for c in codewords]
