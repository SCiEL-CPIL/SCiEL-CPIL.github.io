"""Iterated Local Search for ternary covering code K_3(7,1)."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import defaultdict
from heapq import heappush, heappop

def construct(q, n, radius):
    # Only the known task parameters are specialized; fall back to direct sum otherwise.
    if (q, n, radius) != (3, 7, 1):
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    start_time = time.time()
    DEADLINE = start_time + 130.0  # safety margin before 150s kill

    # Precompute the universe as integers 0..2186 (base-3)
    # word[i] = integer representation of a length-7 base-3 word
    # We'll represent codewords as tuples of 7 ints for hashing, but use int for universe indexing.

    # Helper: convert tuple of 7 ints to int index (base 3)
    def tup_to_idx(tup):
        idx = 0
        for i in range(n):
            idx = idx * 3 + tup[i]
        return idx

    # Precompute all words as list of tuples
    all_words = []
    for idx in range(3**n):
        w = []
        x = idx
        for i in range(n):
            w.append(x % 3)
            x //= 3
        all_words.append(tuple(reversed(w)))
    # all_words[idx] is the tuple for index idx

    # For each codeword, we need to know which universe words it covers (within Hamming distance 1)
    # A codeword covers: itself + all words at Hamming distance 1
    # Number of words covered by one codeword: 1 + 7*2 = 15

    # Precompute for each position i and each change value, the set of indices at distance 1
    # Actually, let's precompute the "ball" of radius 1 around each possible word position
    # But 2187 * 15 = 32805, which is small. Let's precompute balls.

    # ball[idx] = list of indices covered by codeword at index idx
    balls = []
    for idx in range(3**n):
        w = all_words[idx]
        covered = [idx]
        for pos in range(n):
            for delta in [-1, 1]:
                new_val = w[pos] + delta
                if 0 <= new_val < 3:
                    new_w = list(w)
                    new_w[pos] = new_val
                    new_idx = tup_to_idx(tuple(new_w))
                    covered.append(new_idx)
        balls.append(covered)

    # Greedy set cover to get initial solution
    def greedy_cover():
        uncovered = set(range(3**n))
        code = []
        # For efficiency, track how many uncovered words each candidate covers
        # But we need to be smart. Let's use a priority queue.
        # Actually, for 2187 words and ~200 codewords, greedy is feasible.
        
        # Precompute for each index, the ball
        # Greedy: repeatedly pick the codeword that covers the most uncovered words
        # To speed up, we can use a lazy heap approach.
        
        uncovered_set = set(range(3**n))
        
        # For each candidate, count how many uncovered words it covers
        # We'll maintain a count array and a heap
        # count[i] = number of uncovered words in ball[i]
        counts = [0] * (3**n)
        for i in range(3**n):
            for j in balls[i]:
                if j in uncovered_set:
                    counts[i] += 1
        
        # Heap of (-count, index)
        heap = [(-counts[i], i) for i in range(3**n) if counts[i] > 0]
        import heapq
        heapq.heapify(heap)
        
        # Lazy deletion: when we pop, check if it's still valid
        # We need to track which indices are still in the heap and not already chosen
        chosen_set = set()
        
        while uncovered_set:
            # Pop until we find a valid entry
            while heap:
                neg_cnt, idx = heapq.heappop(heap)
                if idx in chosen_set:
                    continue
                # Check if count is still accurate (lazy)
                actual_cnt = 0
                for j in balls[idx]:
                    if j in uncovered_set:
                        actual_cnt += 1
                if actual_cnt == 0:
                    continue
                if -neg_cnt != actual_cnt:
                    # Re-push with correct count
                    heapq.heappush(heap, (-actual_cnt, idx))
                    continue
                # This is a good candidate
                chosen_set.add(idx)
                code.append(all_words[idx])
                # Remove covered words from uncovered
                for j in balls[idx]:
                    if j in uncovered_set:
                        uncovered_set.remove(j)
                # Update counts for all other candidates? Too expensive.
                # Instead, just re-evaluate lazily.
                break
            if not heap:
                break
        
        return code

    # Initial solution
    code = greedy_cover()
    best_code = list(code)
    best_size = len(code)
    
    # Local search: try to remove a codeword if its ball is fully covered by others
    def try_remove(code_list):
        """Try to remove each codeword if possible. Returns (new_code, removed_any)"""
        n_words = 3**n
        # For each codeword, check if all words in its ball are covered by other codewords
        # We need an efficient way. Let's maintain a coverage count.
        
        # coverage_count[j] = number of codewords in code_list that cover word j
        coverage_count = [0] * n_words
        for cw in code_list:
            idx = tup_to_idx(cw)
            for j in balls[idx]:
                coverage_count[j] += 1
        
        # Try to remove each codeword
        for i, cw in enumerate(code_list):
            idx = tup_to_idx(cw)
            # Check if all words in ball[idx] have coverage_count > 1
            can_remove = True
            for j in balls[idx]:
                if coverage_count[j] <= 1:
                    can_remove = False
                    break
            if can_remove:
                # Remove this codeword
                new_code = code_list[:i] + code_list[i+1:]
                return new_code, True
        return code_list, False

    # Swap: try replacing one codeword with another to allow removal of two
    def try_swap(code_list):
        """Try to find a swap that reduces size or enables removal."""
        n_words = 3**n
        coverage_count = [0] * n_words
        for cw in code_list:
            idx = tup_to_idx(cw)
            for j in balls[idx]:
                coverage_count[j] += 1
        
        # For each codeword i, check if removing it leaves some uncovered words
        # For each such word, see if adding a new codeword covers all remaining uncovered
        # This is expensive. Let's do a simpler version: try random swaps.
        
        return code_list, False

    # Iterated local search with perturbation
    def perturbation(code_list, num_changes=2):
        """Randomly modify the code."""
        new_code = list(code_list)
        for _ in range(num_changes):
            if len(new_code) > 0:
                # Remove a random codeword
                i = random.randint(0, len(new_code) - 1)
                new_code.pop(i)
            # Add a random codeword
            idx = random.randint(0, 3**n - 1)
            cw = all_words[idx]
            if cw not in new_code:
                new_code.append(cw)
        return new_code

    # Main ILS loop
    current_code = list(code)
    current_size = len(current_code)
    
    while time.time() < DEADLINE:
        # Local search: repeatedly try to remove
        improved = True
        while improved:
            current_code, improved = try_remove(current_code)
        
        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code = list(current_code)
        
        # Perturbation
        if time.time() > DEADLINE - 5:
            break
        current_code = perturbation(current_code, random.choice([1, 2, 3]))
        
        # Quick check: if perturbation broke coverage, fix it greedily
        # Check coverage
        n_words = 3**n
        covered = [False] * n_words
        for cw in current_code:
            idx = tup_to_idx(cw)
            for j in balls[idx]:
                covered[j] = True
        
        if not all(covered):
            # Fix: add codewords to cover uncovered words
            uncovered = [j for j in range(n_words) if not covered[j]]
            for j in uncovered:
                # Find a codeword that covers j and is not already in code
                # Just add the word itself
                cw = all_words[j]
                if cw not in current_code:
                    current_code.append(cw)
                    idx = tup_to_idx(cw)
                    for k in balls[idx]:
                        covered[k] = True
    
    return best_code
# EVOLVE-BLOCK-END
