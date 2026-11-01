import time
import sys
import math

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    if n == 1:
        return "1"
    
    total_perms = math.factorial(n)
    
    def suffix_priority_superperm(n, deadline):
        """
        Suffix-priority constructive algorithm:
        - Maintain sliding window of last n-1 symbols
        - Track all seen n-permutations
        - Greedily append symbol that creates lexicographically smallest new n-permutation
        - Prioritize transitions that maximize reuse of existing suffix structures
        """
        # Start with first permutation: 1, 2, 3, ..., n
        word = list(range(1, n + 1))
        seen_perms = {tuple(word)}
        
        # Build a priority queue of "next" permutations to try
        # For each position in the word, we can extend by adding a new symbol
        # Key insight: after placing a permutation, the suffix of length n-1 
        # can be the prefix of the next permutation if we append one more symbol
        
        # Strategy: maintain a set of "active suffixes" - suffixes of current word
        # that could be extended to form new permutations
        
        # More efficient approach: use the recursive structure but with optimized ordering
        # Build blocks for each permutation of 1..n-1, then order them optimally
        
        # Get all permutations of 1..n-1
        from itertools import permutations
        
        base_perms = list(permutations(range(1, n)))
        k = len(base_perms)
        
        # For each base permutation p, create block: p + [n] + p
        # But we can be smarter: the block is just p + [n] + p, length 2n-1
        # However, consecutive blocks can overlap
        
        # Precompute all blocks
        blocks = []
        for p in base_perms:
            block = list(p) + [n] + list(p)
            blocks.append(block)
        
        # Compute overlap matrix: overlap[i][j] = max o such that 
        # blocks[i][-o:] == blocks[j][:o]
        block_len = 2 * n - 1
        overlaps = [[0] * k for _ in range(k)]
        
        for i in range(k):
            bi = blocks[i]
            for j in range(k):
                bj = blocks[j]
                max_ov = min(block_len, block_len)
                ov = 0
                # Check from largest overlap to smallest
                # Since blocks have specific structure, we can be smarter
                # blocks[i] = p_i + [n] + p_i
                # The suffix of blocks[i] is p_i (last n-1 elements)
                # The prefix of blocks[j] is p_j (first n-1 elements)
                # So overlap can be at most n-1 (matching p_i suffix with p_j prefix)
                # Plus possibly more if n matches, but n only appears in middle
                
                # Actually, let's just do direct comparison
                for o in range(max_ov, 0, -1):
                    if bi[block_len - o:] == bj[:o]:
                        ov = o
                        break
                overlaps[i][j] = ov
        
        # Now we need to find an ordering of blocks that minimizes total length
        # Total length = len(blocks[order[0]]) + sum(len(blocks[order[t]]) - overlaps[order[t-1]][order[t]])
        # = k * block_len - sum(overlaps[order[t-1]][order[t]] for t in 1..k-1)
        # To minimize length, maximize sum of overlaps
        
        # This is a TSP-like problem. Use greedy + local search.
        
        # Greedy initialization: start with block 0, repeatedly add the block with max overlap
        order = [0]
        remaining = set(range(1, k))
        
        while remaining and time.time() < deadline:
            last = order[-1]
            best_next = None
            best_ov = -1
            for j in remaining:
                ov = overlaps[last][j]
                if ov > best_ov:
                    best_ov = ov
                    best_next = j
                elif ov == best_ov and best_next is not None:
                    # Tie-breaking: prefer lexicographically smaller block
                    if blocks[j] < blocks[best_next]:
                        best_next = j
            if best_next is None:
                break
            order.append(best_next)
            remaining.remove(best_next)
        
        # If not all blocks added, add remaining in arbitrary order
        if remaining:
            order.extend(sorted(remaining))
        
        # Local search: 2-opt to improve overlap sum
        def calc_overlap_sum(order):
            total = 0
            for t in range(1, len(order)):
                total += overlaps[order[t-1]][order[t]]
            return total
        
        current_ov_sum = calc_overlap_sum(order)
        
        # 2-opt local search
        improved = True
        max_iterations = 5000
        iteration = 0
        
        while improved and iteration < max_iterations and time.time() < deadline - 10:
            improved = False
            iteration += 1
            for i in range(k - 1):
                if time.time() > deadline - 10:
                    break
                for j in range(i + 1, k):
                    # Reverse segment order[i+1..j]
                    new_order = order[:i+1] + order[i+1:j+1][::-1] + order[j+1:]
                    new_ov_sum = calc_overlap_sum(new_order)
                    if new_ov_sum > current_ov_sum:
                        order = new_order
                        current_ov_sum = new_ov_sum
                        improved = True
        
        # Construct final word
        if not order:
            return None
            
        best_word = list(blocks[order[0]])
        for t in range(1, len(order)):
            i = order[t-1]
            j = order[t]
            ov = overlaps[i][j]
            best_word += blocks[j][ov:]
        
        return best_word
    
    def is_valid_superperm(word, n):
        """Check if word contains all permutations of 1..n"""
        if len(word) < n:
            return False
        perms_seen = set()
        for i in range(len(word) - n + 1):
            window = tuple(word[i:i + n])
            if len(set(window)) == n:
                perms_seen.add(window)
        return len(perms_seen) == math.factorial(n)
    
    # Try suffix-priority algorithm
    result = suffix_priority_superperm(n, deadline)
    
    if result is not None and is_valid_superperm(result, n):
        return "".join(map(str, result))
    
    # Fallback: standard recursive construction
    word = [1]
    for m in range(2, n + 1):
        if time.time() > deadline:
            break
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    return "".join(map(str, word))
