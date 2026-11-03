"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def construct(q, n, radius):
    # Deadline: evaluator kills at 150s; we guard at 145s.
    deadline = time.time() + 143.0

    # For this task: q=3, n=7, radius=1
    # The direct sum construction gives 9 * 3^(7-4) = 9 * 27 = 243 words.
    # We want to reduce this toward the best known 186.

    # Strategy: 
    # 1. Start with a greedy covering code (greedy set cover on the Hamming graph).
    # 2. Apply iterated local search: remove redundant words, perturb, and re-cover.
    # 3. Use the tetracode-based construction as an initial upper bound.

    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters: use the tetracode direct sum if possible
        if n >= 4:
            tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            # For very small n, just return all words
            return [list(x) for x in itertools.product(range(q), repeat=n)]

    # For q=3, n=7, radius=1: 3^7 = 2187 words in the space.
    # Each codeword covers itself + 6*(3-1) = 13 words within Hamming distance 1.
    # Lower bound: ceil(2187 / 13) = 169. Best known: 186.

    # Precompute all words as integers 0..2186 for fast operations.
    all_words = list(itertools.product(range(3), repeat=7))
    num_words = 3 ** 7  # 2187

    # Map from tuple to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    # For each word, precompute its "ball" (the set of words within Hamming distance 1).
    # We'll represent balls as bitsets using Python integers (2187 bits fits in ~274 bytes per int).
    # Actually, Python big ints handle this fine.
    def hamming_neighbors(w):
        """Yield all words within Hamming distance 1 of w (including w itself)."""
        yield w
        for pos in range(7):
            for val in range(3):
                if val != w[pos]:
                    new_list = list(w)
                    new_list[pos] = val
                    yield tuple(new_list)

    # Precompute ball membership as bitsets (integer where bit i is set if word i is in the ball)
    balls = []
    for w in all_words:
        mask = 0
        for nb in hamming_neighbors(w):
            idx = word_to_idx[nb]
            mask |= (1 << idx)
        balls.append(mask)

    # Greedy set cover: repeatedly pick the word whose ball covers the most uncovered words.
    def greedy_cover():
        uncovered = (1 << num_words) - 1  # All bits set
        chosen = []
        # We'll use a max-heap approach, but for simplicity and speed with 2187 elements,
        # we'll iterate and pick the best each time. This is O(num_words^2) which is ~4.8M ops, fine.
        
        # To speed up, maintain count of uncovered words covered by each candidate.
        # uncovered_count[cw] = number of words in ball(cw) that are currently uncovered.
        uncovered_count = [bin(balls[i] & uncovered).count('1') for i in range(num_words)]
        
        while uncovered:
            # Find the word with maximum uncovered_count
            best_idx = -1
            best_count = -1
            for i in range(num_words):
                if uncovered_count[i] > best_count:
                    best_count = uncovered_count[i]
                    best_idx = i
            
            if best_idx == -1 or best_count == 0:
                break
            
            chosen.append(best_idx)
            # Remove covered words
            uncovered &= ~balls[best_idx]
            
            # Update uncovered counts: for each candidate, subtract the newly covered words.
            # newly_covered = balls[best_idx] & old_uncovered
            # Actually we already updated uncovered, so newly_covered = ~uncovered & balls[best_idx]
            # But it's easier to just recompute or update incrementally.
            # Incremental update:
            newly_covered = balls[best_idx] & uncovered  # This is wrong because we already cleared them
            # Let me redo: before clearing, newly_covered = balls[best_idx] & uncovered_old
            # We cleared uncovered, so let's recompute. It's only 2187 iterations * up to 243 picks = ~530K ops.
        
        # Let me rewrite more carefully:
        return chosen

    def greedy_cover_v2():
        """Greedy set cover with proper incremental updates."""
        uncovered = (1 << num_words) - 1
        chosen = []
        
        # uncovered_count[i] = popcount(balls[i] & uncovered)
        uncovered_count = [0] * num_words
        for i in range(num_words):
            uncovered_count[i] = bin(balls[i] & uncovered).count('1')
        
        while uncovered:
            # Find best
            best_idx = 0
            best_count = uncovered_count[0]
            for i in range(1, num_words):
                if uncovered_count[i] > best_count:
                    best_count = uncovered_count[i]
                    best_idx = i
            
            if best_count == 0:
                break
            
            chosen.append(best_idx)
            
            # Words newly covered by best_idx:
            newly_covered = balls[best_idx] & uncovered
            uncovered &= ~newly_covered
            
            # Update uncovered_count for all candidates:
            # For each candidate j, uncovered_count[j] -= popcount(balls[j] & newly_covered)
            # This is O(num_words) per pick, and we pick up to ~243 times, so ~530K bit operations.
            # But popcount of 2187-bit ints is fast in Python.
            for j in range(num_words):
                if uncovered_count[j] > 0:
                    overlap = balls[j] & newly_covered
                    if overlap:
                        uncovered_count[j] -= bin(overlap).count('1')
        
        return chosen

    def is_cover(chosen_indices):
        """Check if the chosen set covers all words."""
        covered = 0
        for i in chosen_indices:
            covered |= balls[i]
        return covered == (1 << num_words) - 1

    def try_remove_redundant(chosen_indices):
        """Try to remove words that are redundant. Return a reduced list."""
        chosen = list(chosen_indices)
        improved = True
        while improved:
            improved = False
            for i in range(len(chosen)):
                # Try removing chosen[i]
                remaining = chosen[:i] + chosen[i+1:]
                # Check if remaining still covers
                covered = 0
                for j in remaining:
                    covered |= balls[j]
                if covered == (1 << num_words) - 1:
                    chosen = remaining
                    improved = True
                    break
        return chosen

    def local_search(chosen_indices, time_limit):
        """Local search: try removing pairs, then re-greedy, etc."""
        chosen = list(chosen_indices)
        best = len(chosen)
        
        while time.time() < time_limit:
            # Try single removals
            improved = True
            while improved:
                improved = False
                for i in range(len(chosen)):
                    remaining = chosen[:i] + chosen[i+1:]
                    covered = 0
                    for j in remaining:
                        covered |= balls[j]
                    if covered == (1 << num_words) - 1:
                        chosen = remaining
                        best = min(best, len(chosen))
                        improved = True
                        break
            
            if time.time() >= time_limit:
                break
            
            # Try perturb: remove k random words, then re-greedy cover the gaps
            k = random.randint(2, min(5, len(chosen)))
            if len(chosen) <= k:
                break
            removed = random.sample(chosen, k)
            remaining = [c for c in chosen if c not in set(removed)]
            
            # Find uncovered words
            covered = 0
            for c in remaining:
                covered |= balls[c]
            uncovered = ((1 << num_words) - 1) & ~covered
            
            if uncovered == 0:
                # Still covered, great
                chosen = remaining
                best = min(best, len(chosen))
                continue
            
            # Greedily cover the uncovered words
            # We need to pick words whose balls intersect uncovered the most
            # But we can only add words from all_words
            # Let's do a greedy fill:
            fill = list(remaining)
            un = uncovered
            while un:
                # Find word with max overlap with un
                best_idx = -1
                best_overlap = 0
                for i in range(num_words):
                    if i in set(fill):
                        continue
                    overlap = balls[i] & un
                    oc = bin(overlap).count('1')
                    if oc > best_overlap:
                        best_overlap = oc
                        best_idx = i
                if best_idx == -1 or best_overlap == 0:
                    break
                fill.append(best_idx)
                un &= ~balls[best_idx]
            
            if len(fill) < len(chosen):
                chosen = fill
                best = min(best, len(chosen))
        
        return chosen

    # Initial solution: greedy cover
    initial = greedy_cover_v2()
    # Verify it's a valid cover
    if not is_cover(initial):
        # Fallback to tetracode construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        result = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=3)]
        return result
    
    # Try to reduce
    reduced = try_remove_redundant(initial)
    if len(reduced) < len(initial):
        initial = reduced
    
    # Local search with time budget
    time_limit = deadline - 3.0  # Leave 3 seconds for return
    best_solution = local_search(initial, time_limit)
    
    # Verify final solution
    if not is_cover(best_solution):
        # Fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        result = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=3)]
        return result
    
    # Convert indices back to words
    return [list(all_words[i]) for i in best_solution]
# EVOLVE-BLOCK-END
