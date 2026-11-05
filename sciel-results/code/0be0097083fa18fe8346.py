"""Combined approach: direct-sum construction with incremental greedy pruning."""
import itertools
import time
import random
import sys


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # Direct sum of ternary tetracode with all tails
    if n >= 4:
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    else:
        # Fallback for small n
        codewords = [list(w) for w in itertools.product(range(q), repeat=n)]
    
    total_words = q ** n

    def word_to_idx(word):
        idx = 0
        for v in word:
            idx = idx * q + v
        return idx

    def get_covered_indices(cw):
        """Return a list of indices of words covered by cw (including itself)."""
        indices = [word_to_idx(cw)]
        if radius >= 1:
            for pos in range(n):
                orig = cw[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    lst = list(cw)
                    lst[pos] = new_val
                    indices.append(word_to_idx(lst))
        return indices

    # Precompute covered indices for each codeword
    codeword_covers = [get_covered_indices(cw) for cw in codewords]

    # Build initial coverage count
    def build_coverage_count(cw_indices):
        """Build coverage count from a list of codeword indices into the original codewords list."""
        counts = [0] * total_words
        for ci in cw_indices:
            for idx in codeword_covers[ci]:
                counts[idx] += 1
        return counts

    def try_remove_redundant(cw_list, cw_covers, counts, order, time_limit):
        """
        Try to remove redundant codewords from cw_list in the given order.
        Returns (new_cw_list, new_cw_covers, new_counts, num_removed)
        """
        active = [True] * len(cw_list)
        removed = 0
        
        for ci in order:
            if time.time() > time_limit:
                break
            if not active[ci]:
                continue
            
            # Check if codeword ci is redundant
            is_redundant = True
            for idx in cw_covers[ci]:
                if counts[idx] <= 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                active[ci] = False
                removed += 1
                # Update counts incrementally
                for idx in cw_covers[ci]:
                    counts[idx] -= 1
        
        if removed == 0:
            return cw_list, cw_covers, counts, 0
        
        # Build new lists
        new_cw_list = [cw_list[i] for i in range(len(cw_list)) if active[i]]
        new_cw_covers = [cw_covers[i] for i in range(len(cw_list)) if active[i]]
        
        return new_cw_list, new_cw_covers, counts, removed

    # Main search loop with multiple strategies
    best_cw = codewords
    best_covers = codeword_covers
    best_counts = build_coverage_count(range(len(codewords)))
    best_len = len(best_cw)

    strategies = []
    
    # Strategy 1: Reverse order
    strategies.append(('reverse', list(range(len(codewords) - 1, -1, -1))))
    
    # Strategy 2: Forward order
    strategies.append(('forward', list(range(len(codewords)))))
    
    # Strategy 3: Random order
    rand_order = list(range(len(codewords)))
    random.shuffle(rand_order)
    strategies.append(('random', rand_order))
    
    # Strategy 4: Priority order - remove codewords covering most critical words first
    # A codeword is "critical" if it covers words with low coverage
    # We'll compute a score for each codeword based on how many words it covers with count==1
    def compute_priority(cw_covers, counts):
        scores = []
        for ci in range(len(cw_covers)):
            critical_count = 0
            for idx in cw_covers[ci]:
                if counts[idx] == 1:
                    critical_count += 1
            # We want to remove codewords that cover FEW critical words first
            # (they are more likely to be redundant)
            scores.append((critical_count, ci))
        scores.sort()
        return [ci for _, ci in scores]

    priority_order = compute_priority(codeword_covers, best_counts)
    strategies.append(('priority', priority_order))

    # Try each strategy multiple times with restarts
    for trial in range(50):
        if time.time() > deadline - 10.0:
            break
        
        # Choose a strategy
        strat_name, order = strategies[trial % len(strategies)]
        
        # Make working copies
        cw_list = list(best_cw)
        cw_covers = list(best_covers)
        counts = list(best_counts)
        
        # For random and priority, reshuffle/recompute
        if strat_name == 'random':
            order = list(range(len(cw_list)))
            random.shuffle(order)
        elif strat_name == 'priority':
            order = compute_priority(cw_covers, counts)
        
        # Try to remove redundant codewords
        new_cw, new_covers, new_counts, removed = try_remove_redundant(
            cw_list, cw_covers, counts, order, deadline - 5.0
        )
        
        if removed > 0 and len(new_cw) < best_len:
            best_cw = new_cw
            best_covers = new_covers
            best_counts = new_counts
            best_len = len(best_cw)
            
            # Update strategies for next iteration
            if len(best_cw) > 0:
                strategies[0] = ('reverse', list(range(len(best_cw) - 1, -1, -1)))
                strategies[1] = ('forward', list(range(len(best_cw))))
                rand_order = list(range(len(best_cw)))
                random.shuffle(rand_order)
                strategies[2] = ('random', rand_order)
                priority_order = compute_priority(best_covers, best_counts)
                strategies[3] = ('priority', priority_order)
    
    # Final polish: try a few more rounds with different random seeds
    for trial in range(20):
        if time.time() > deadline - 2.0:
            break
        
        cw_list = list(best_cw)
        cw_covers = list(best_covers)
        counts = list(best_counts)
        
        order = list(range(len(cw_list)))
        random.shuffle(order)
        
        new_cw, new_covers, new_counts, removed = try_remove_redundant(
            cw_list, cw_covers, counts, order, deadline - 1.0
        )
        
        if removed > 0 and len(new_cw) < best_len:
            best_cw = new_cw
            best_covers = new_covers
            best_counts = new_counts
            best_len = len(best_cw)

    return [list(cw) for cw in best_cw]
