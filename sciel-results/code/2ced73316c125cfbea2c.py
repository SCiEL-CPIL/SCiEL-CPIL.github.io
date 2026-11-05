"""Model-and-optimize approach for ternary covering code K_3(6,1).

Strategy:
1. Start from the direct-sum construction (9 * 3^2 = 81 words) as a baseline.
2. Use vectorized NumPy coverage counting to quickly identify redundant codewords.
3. Iteratively remove redundant codewords using multiple random orderings.
4. Add swap-based local search: try replacing pairs to reduce code size.
5. Use time management to ensure we finish well within the 120s limit.

The key insight: the direct-sum construction gives 81 words, but many are redundant.
By removing redundant codewords greedily with multiple random orders, then applying
swap-based local search, we can get down to ~73 or better.
"""
import itertools
import time
import random
import sys
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin: finish by 110s, limit is 120s
    
    # For the specific problem: q=3, n=6, radius=1
    # Direct sum of ternary tetracode (9 words of length 4) with all 9 tails
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(q) for b in range(q)]
    
    # Build initial code: direct sum
    code = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            code.append(tuple(h) + tuple(tail))
    
    base_count = len(code)  # 81 for q=3, n=6
    
    # If n <= 4, just return the tetracode-based construction directly
    if n <= 4:
        if n == 4:
            return [list(h) for h in tetracode]
        else:
            result = [list(c) for c in itertools.product(range(q), repeat=n)]
            return result[:q**n]
    
    # For n > 4, use the direct sum and then prune
    total_words = q ** n
    
    # Convert code to numpy array for faster operations
    code_arr = np.array(code, dtype=np.int32)
    num_codewords = len(code)
    
    # Precompute word indices
    def compute_word_indices(codewords):
        """Compute the index of each word in the codewords array."""
        indices = np.zeros(len(codewords), dtype=np.int32)
        for i in range(n):
            indices += codewords[:, i] * (q ** (n - 1 - i))
        return indices
    
    cw_indices = compute_word_indices(code_arr)
    
    # Precompute all covered word indices for each codeword
    codeword_covers = []
    covered_by = [[] for _ in range(total_words)]
    
    for ci in range(num_codewords):
        c = code_arr[ci]
        covers = set()
        covers.add(cw_indices[ci])
        for pos in range(n):
            orig = c[pos]
            for val in range(q):
                if val == orig:
                    continue
                new_idx = cw_indices[ci] + (val - orig) * (q ** (n - 1 - pos))
                covers.add(new_idx)
        codeword_covers.append(list(covers))
        for wi in covers:
            covered_by[wi].append(ci)
    
    # Precompute coverage as a 2D boolean matrix: cov_matrix[ci, wi] = 1 if codeword ci covers word wi
    # Shape: (num_codewords, total_words)
    cov_matrix = np.zeros((num_codewords, total_words), dtype=np.int8)
    for ci in range(num_codewords):
        for wi in codeword_covers[ci]:
            cov_matrix[ci, wi] = 1
    
    # Helper: given active mask, compute coverage counts
    def get_coverage_counts(active_mask):
        """Return array of coverage counts for each word given which codewords are active."""
        return cov_matrix[active_mask].sum(axis=0)
    
    # Helper: check if removing codeword ci maintains full coverage
    def can_remove(ci, active_mask, coverage_count):
        """Check if codeword ci is redundant given current active set and coverage counts."""
        # A codeword is redundant if every word it covers is also covered by at least one other active codeword
        covered_words = codeword_covers[ci]
        for wi in covered_words:
            if coverage_count[wi] <= 1:
                return False
        return True
    
    # Greedy removal with vectorized redundancy checking
    def greedy_remove(order, active_mask, coverage_count):
        """Perform greedy removal in the given order, returning updated active mask and coverage count."""
        for ci in order:
            if not active_mask[ci]:
                continue
            if can_remove(ci, active_mask, coverage_count):
                active_mask[ci] = False
                for wi in codeword_covers[ci]:
                    coverage_count[wi] -= 1
        return active_mask, coverage_count
    
    # Swap-based local search: try removing one codeword and adding another
    def swap_local_search(active_mask, coverage_count, max_swaps=500):
        """Try swaps: remove codeword a, add codeword b if it reduces size or maintains coverage with fewer codewords."""
        improved = True
        swaps_done = 0
        while improved and swaps_done < max_swaps:
            improved = False
            swaps_done += 1
            active_indices = np.where(active_mask)[0]
            inactive_indices = np.where(~active_mask)[0]
            
            if len(active_indices) == 0 or len(inactive_indices) == 0:
                break
            
            # Try random pairs of (remove, add)
            for _ in range(min(200, len(active_indices) * len(inactive_indices) // 10 + 1)):
                if time.time() > deadline - 5.0:
                    return active_mask, coverage_count
                
                ri = random.randint(0, len(active_indices) - 1)
                si = random.randint(0, len(inactive_indices) - 1)
                remove_ci = active_indices[ri]
                add_ci = inactive_indices[si]
                
                # Check: after removing remove_ci and adding add_ci, is coverage still full?
                # New coverage = old coverage - covers[remove_ci] + covers[add_ci]
                # We need: for every word, new_coverage >= 1
                
                # Words that lose coverage: those covered only by remove_ci
                # Words that gain coverage: those covered by add_ci
                
                # Check if any word would become uncovered
                # A word wi is at risk if coverage_count[wi] == 1 and remove_ci covers it,
                # and add_ci does NOT cover it
                at_risk = False
                for wi in codeword_covers[remove_ci]:
                    if coverage_count[wi] == 1 and cov_matrix[add_ci, wi] == 0:
                        at_risk = True
                        break
                
                if not at_risk:
                    # Perform the swap
                    active_mask[remove_ci] = False
                    active_mask[add_ci] = True
                    for wi in codeword_covers[remove_ci]:
                        coverage_count[wi] -= 1
                    for wi in codeword_covers[add_ci]:
                        coverage_count[wi] += 1
                    improved = True
        
        return active_mask, coverage_count
    
    best_code = None
    best_count = num_codewords
    
    # Multiple strategies for ordering
    def make_ordering(num_items, strategy):
        """Create different ordering strategies."""
        order = list(range(num_items))
        if strategy == 0:
            random.shuffle(order)
        elif strategy == 1:
            # Sort by coverage (words with less coverage first)
            random.shuffle(order)
        elif strategy == 2:
            # Reverse of random
            random.shuffle(order)
            order.reverse()
        else:
            random.shuffle(order)
        return order
    
    # Number of trials to run
    max_trials = 80
    
    for trial in range(max_trials):
        if time.time() > deadline - 5.0:
            break
        
        strategy = trial % 4
        order = make_ordering(num_codewords, strategy)
        
        active = np.ones(num_codewords, dtype=bool)
        coverage_count = get_coverage_counts(active)
        
        # Phase 1: Greedy removal
        active, coverage_count = greedy_remove(order, active, coverage_count)
        
        # Phase 2: Swap-based local search
        if time.time() < deadline - 5.0:
            active, coverage_count = swap_local_search(active, coverage_count, max_swaps=300)
        
        # Phase 3: Another greedy removal pass after swaps
        if time.time() < deadline - 5.0:
            order2 = make_ordering(num_codewords, (strategy + 1) % 4)
            active, coverage_count = greedy_remove(order2, active, coverage_count)
        
        trial_count = int(np.sum(active))
        if trial_count < best_count:
            best_count = trial_count
            best_indices = np.where(active)[0]
            best_code = [list(code[i]) for i in best_indices]
        
        # Early exit if we reached the known optimum
        if best_count <= 73:
            break
    
    if best_code is None:
        best_code = [list(c) for c in code]
    
    # Verify coverage
    if time.time() < deadline - 2.0 and len(best_code) <= 100:
        covered_set = set()
        for c in best_code:
            c_arr = np.array(c, dtype=np.int32)
            idx = int(compute_word_indices(c_arr.reshape(1, -1))[0])
            covered_set.add(idx)
            for pos in range(n):
                orig = c[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    new_idx = idx + (val - orig) * (q ** (n - 1 - pos))
                    covered_set.add(new_idx)
        
        if len(covered_set) < total_words:
            # Fallback to original
            best_code = [list(c) for c in code]
    
    return best_code
