import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    space_size = q ** n
    powers = [q ** i for i in range(n)]

    # Precompute neighbor indices for each word
    # neighbor_indices[i] is a list of indices covered by codeword i
    neighbor_indices = []
    for x in range(space_size):
        neighbors = {x}
        for i in range(n):
            digit = (x // powers[i]) % q
            for delta in range(1, radius + 1):
                if digit - delta >= 0:
                    neighbors.add(x - delta * powers[i])
                if digit + delta < q:
                    neighbors.add(x + delta * powers[i])
        neighbor_indices.append(list(neighbors))
    
    # Create a sparse-like representation or just use lists for counting
    # For q=3, n=6, radius=1: space_size=729.
    # Each codeword covers 1 + n*(q-1) = 1 + 6*2 = 13 words.
    # We can use a matrix M (space_size x space_size) where M[i][j]=1 if i covers j.
    # But 729x729 is small enough for dense numpy operations if optimized, 
    # or we can just loop over the 13 neighbors.
    
    # Let's stick to lists for memory efficiency and simple iteration, 
    # but use numpy for the count vector updates if we batch them.
    # Actually, simple python loops over 13 items * 80 codewords * 729 checks is fast enough.
    # 80 * 729 * 13 ~ 750k ops. Very fast.

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # 1. Initial Construction: Direct sum of tetracode and all words of length n-4
    # Tetracode: 9 words of length 4 (for q=3)
    # General tetracode for q=3: [a, b, a+b, a+2b] mod 3
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    
    # Generate tails
    tails = list(itertools.product(range(q), repeat=n - 4))
    
    # Construct initial codewords as indices
    initial_indices = []
    for h in tetracode:
        for tail in tails:
            idx = 0
            for val in h + list(tail):
                idx = idx * q + val
            initial_indices.append(idx)

    def get_coverage_counts(code_set):
        """Compute coverage count for each word in the space."""
        counts = np.zeros(space_size, dtype=np.int32)
        for idx in code_set:
            for nb in neighbor_indices[idx]:
                counts[nb] += 1
        return counts

    def is_valid_covering(code_set):
        """Check if code_set covers all words."""
        if not code_set:
            return space_size == 0
        counts = get_coverage_counts(code_set)
        return np.all(counts > 0)

    def greedy_reduce(code_indices):
        """
        Greedily remove redundant codewords.
        A codeword is redundant if removing it leaves all words covered (count > 0 for all).
        Equivalent to: all words in its sphere have count > 1 in the current set.
        """
        code_set = set(code_indices)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            counts = get_coverage_counts(code_set)
            
            # Iterate through a copy of the set
            for idx in list(code_set):
                # Check if idx is critical
                # If any neighbor has count == 1, idx is critical (cannot remove)
                # If all neighbors have count > 1, idx is redundant
                is_critical = False
                for nb in neighbor_indices[idx]:
                    if counts[nb] == 1:
                        is_critical = True
                        break
                
                if not is_critical:
                    code_set.remove(idx)
                    changed = True
                    # Optional: break early and recompute counts? 
                    # Recomputing counts every removal is O(N*Space*Radius).
                    # N~80, Space=729, Radius~13. 80*729*13 ~ 750k ops.
                    # If we do this 80 times, ~60M ops. Might be tight in Python.
                    # Better to check all candidates against current counts, remove all safe ones at once?
                    # No, removing one might make another critical.
                    # So we must recompute or update incrementally.
                    # Incremental update: decrement counts for neighbors of removed idx.
                    # Let's try incremental update for speed.
                    for nb in neighbor_indices[idx]:
                        counts[nb] -= 1
                    # Note: We are modifying counts while iterating.
                    # We need to be careful. If we remove idx, counts change.
                    # The `is_critical` check for *other* indices should use updated counts.
                    # This loop structure (check all, remove all safe) is valid if we update counts as we go?
                    # No, because removing A might make B critical, but we already checked B.
                    # So we should only remove one at a time, or iterate until no more removals.
                    # Given the small size, let's stick to the safe but slightly slower recompute or careful incremental.
                    # Let's restart the while loop on first removal to be safe and simple.
                    break 

        return list(code_set)

    def local_search_swap(code_indices):
        """
        Try to swap codewords to enable further deletions.
        """
        code_set = set(code_indices)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to remove
            reduced = greedy_reduce(list(code_set))
            if len(reduced) < len(code_set):
                code_set = set(reduced)
                improved = True
                continue
            
            # Try to swap
            if not code_set:
                break
            
            # Pick a random codeword to replace
            idx_to_swap = random.choice(list(code_set))
            
            # Find a replacement that allows removal of idx_to_swap?
            # Or just a replacement that maintains coverage and allows future removals?
            # Heuristic: Replace idx_to_swap with a word that covers many currently
            # "weakly covered" words (count=1).
            
            counts = get_coverage_counts(code_set)
            weak_words = np.where(counts == 1)[0]
            
            best_candidate = None
            best_gain = 0
            
            # Sample candidates
            # We want a candidate c such that (code_set - {idx_to_swap}) U {c} is valid.
            # This is expensive to check directly.
            # Approximation: c should cover many weak_words that are NOT covered by others?
            # Actually, if we remove idx_to_swap, words in its sphere with count=1 become uncovered.
            # Let critical_words = [w in sphere(idx_to_swap) if counts[w]==1]
            # We need c to cover all critical_words.
            
            critical_words = []
            for nb in neighbor_indices[idx_to_swap]:
                if counts[nb] == 1:
                    critical_words.append(nb)
            
            if not critical_words:
                # idx_to_swap is redundant, but greedy_reduce should have caught it?
                # Maybe greedy_reduce missed it due to order?
                # If so, we can just remove it.
                code_set.remove(idx_to_swap)
                improved = True
                continue
            
            # Find c that covers all critical_words
            # And ideally covers other weak words too.
            
            # Precompute which words cover which critical words?
            # For each candidate c, check if it covers all critical_words.
            
            # Optimization: Only consider candidates that cover at least one critical word.
            # And must cover ALL.
            
            # Let's just sample random candidates and check validity.
            found_swap = False
            for _ in range(100):
                c = random.randint(0, space_size - 1)
                if c in code_set:
                    continue
                
                # Check if c covers all critical_words
                c_neighbors = set(neighbor_indices[c])
                if all(w in c_neighbors for w in critical_words):
                    # Potential valid swap. Verify full coverage.
                    new_set = (code_set - {idx_to_swap}) | {c}
                    if is_valid_covering(new_set):
                        # Try to reduce
                        reduced = greedy_reduce(list(new_set))
                        if len(reduced) < len(code_set):
                            code_set = set(reduced)
                            improved = True
                            found_swap = True
                            break
                        else:
                            # Accept swap even if size doesn't reduce immediately?
                            # This helps escape local minima.
                            # But we must ensure we don't loop forever.
                            # Let's accept it with small probability.
                            if random.random() < 0.05:
                                code_set = set(new_set)
                                improved = True
                                found_swap = True
                                break
            if found_swap:
                continue
            # If no swap found, try another random idx_to_swap?
            # To avoid getting stuck, we can just break or try a few more times.
            # For simplicity, we break the inner loop and try again in the outer loop
            # with a new random seed/shuffle if needed, but here we just break to avoid wasting time
            # if no swap is found for this specific idx_to_swap.
            # Actually, we should try a different idx_to_swap.
            # But the loop structure above picks one random idx_to_swap per iteration.
            # If it fails, `improved` remains False, and the while loop ends.
            # This is too aggressive. Let's try multiple swaps per pass.

        return list(code_set)

    best_code = None
    best_size = float('inf')

    # Multiple restarts
    for restart in range(50):
        if time.time() > deadline:
            break
            
        random.seed(42 + restart)
        
        if restart % 2 == 0:
            # Tetracode start
            code = list(initial_indices)
            random.shuffle(code)
        else:
            # Pure greedy start
            uncovered = set(range(space_size))
            code = []
            while uncovered:
                if time.time() > deadline:
                    break
                
                max_cov = -1
                best_cands = []
                for w in range(space_size):
                    # Count how many uncovered words w covers
                    # This is O(Space * Radius) per w -> O(Space^2 * Radius) total.
                    # 729^2 * 13 ~ 6.9M ops. Too slow for 50 restarts.
                    # Optimization: Pick a random uncovered word u.
                    # Best candidate is one that covers u and many others.
                    # Simpler greedy: Pick w that covers the most uncovered words.
                    # We can approximate by picking w that covers a random uncovered word.
                    # Let's stick to the bitmask approach if we had it, but we used lists.
                    # For speed, let's just pick a random uncovered word, and pick a random neighbor?
                    # No, that's not greedy.
                    
                    # Efficient Greedy:
                    # Maintain a count of how many times each candidate is "wanted" by uncovered words?
                    # Too complex.
                    
                    # Let's use a simpler heuristic for the greedy start:
                    # Pick a random uncovered word u.
                    # Pick a random neighbor w of u.
                    # Add w.
                    # This is "Randomized Greedy" or "Set Cover with random choices".
                    # It's fast and often good enough for a starting point.
                    
                    u = random.choice(list(uncovered))
                    w = random.choice(neighbor_indices[u])
                    code.append(w)
                    # Remove all covered words
                    for nb in neighbor_indices[w]:
                        if nb in uncovered:
                            uncovered.remove(nb)

        # Optimize
        code = greedy_reduce(code)
        code = local_search_swap(code)
        code = greedy_reduce(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code.copy()
            
        if best_size <= 73:
            # We found a very good code. Keep trying for a bit longer?
            # The deadline check handles this.
            pass

    if best_code is None:
        # Fallback
        best_code = initial_indices

    # Final verification and cleanup
    # Ensure distinctness
    unique_code = list(set(best_code))
    
    # Convert indices back to lists
    res = []
    for idx in unique_code:
        res.append(idx_to_word(idx))
        
    return res
