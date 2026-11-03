"""Greedy set-cover for K_q(n, radius) with a time guard."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import math
import heapq

def construct(q, n, radius):
    start = time.time()
    deadline = start + 110.0

    total = q ** n
    # Precompute powers for index conversion
    pows = [q ** (n - 1 - i) for i in range(n)]
    
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute ball lists for all indices
    ball_lists = [[] for _ in range(total)]
    if radius == 1:
        for idx in range(total):
            w = idx_to_word(idx)
            ball = [idx]
            for i in range(n):
                pw = pows[i]
                w_i = w[i]
                for v in range(q):
                    if v != w_i:
                        new_idx = idx + (v - w_i) * pw
                        ball.append(new_idx)
            ball_lists[idx] = ball
    elif radius == 0:
        for idx in range(total):
            ball_lists[idx] = [idx]
    else:
        from itertools import combinations
        for idx in range(total):
            w = idx_to_word(idx)
            ball = {idx}
            for r in range(1, radius + 1):
                for positions in combinations(range(n), r):
                    for vals in itertools.product(range(q), repeat=r):
                        new_w = w[:]
                        for pos, val in zip(positions, vals):
                            if val == new_w[pos]:
                                break
                        else:
                            new_idx = word_to_idx(new_w)
                            ball.add(new_idx)
            ball_lists[idx] = list(ball)

    # --- Tetracode-based structured approach ---
    if q == 3 and n >= 4 and radius == 1:
        # Tetracode words
        tetracode = []
        for a in range(3):
            for b in range(3):
                tetracode.append([a, b, (a + b) % 3, (a + 2 * b) % 3])
        
        num_tc = len(tetracode)  # 9
        tail_len = n - 4
        num_tails = q ** tail_len  # 9 for n=6
        
        # Precompute: for each tetracode word and tail, what's the index
        tc_tail_indices = []
        for tc in tetracode:
            tail_indices = []
            for tail in itertools.product(range(q), repeat=tail_len):
                word = tc + list(tail)
                tail_indices.append(word_to_idx(word))
            tc_tail_indices.append(tail_indices)
        
        # For each tetracode word, compute which words it covers for each tail subset
        # Strategy: for each tetracode word, we need to choose a subset of tails
        # such that the union of balls covers all words in that "region"
        
        # First, identify which words are "close" to each tetracode word
        # A word is covered by tetracode word tc with tail t if dist(word, tc+t) <= 1
        
        # For each tetracode word, determine which words MUST be covered by it
        # (i.e., words that are far from all other tetracode words)
        
        # Actually, let's use a different approach:
        # Start with all tails for each tetracode word (81 total)
        # Then iteratively remove redundant tails
        
        # Initialize with all 81 codewords
        all_indices = []
        for tc_idx in range(num_tc):
            for tail_idx in range(num_tails):
                all_indices.append(tc_tail_indices[tc_idx][tail_idx])
        
        code = list(all_indices)
        
        # Compute coverage
        coverage_count = [0] * total
        for c in code:
            for idx in ball_lists[c]:
                coverage_count[idx] += 1
        
        # Iteratively remove redundant codewords
        # Use a priority queue: codewords that cover the fewest unique words first
        import heapq
        
        # For each codeword, compute how many words it uniquely covers
        def get_unique_coverage(c, coverage_count, code_set):
            unique = 0
            for idx in ball_lists[c]:
                if coverage_count[idx] == 1:
                    unique += 1
            return unique
        
        # Remove phase with multiple strategies
        best_code = list(code)
        best_count = len(code)
        
        max_restarts = 15
        for restart in range(max_restarts):
            if time.time() > deadline - 15:
                break
            
            current_code = list(best_code)
            current_set = set(current_code)
            
            # Recompute coverage
            coverage_count = [0] * total
            for c in current_code:
                for idx in ball_lists[c]:
                    coverage_count[idx] += 1
            
            # Check if covered
            is_covered = all(coverage_count[j] > 0 for j in range(total))
            
            if not is_covered:
                continue
            
            # Removal phase: try multiple removal orders
            for strategy in range(3):
                if time.time() > deadline - 10:
                    break
                
                remove_order = list(current_code)
                
                if strategy == 0:
                    # Remove in order of least unique coverage first
                    def unique_cov(c):
                        cnt = 0
                        for idx in ball_lists[c]:
                            if coverage_count[idx] == 1:
                                cnt += 1
                        return cnt
                    remove_order.sort(key=unique_cov)
                elif strategy == 1:
                    # Random order
                    random.shuffle(remove_order)
                else:
                    # Remove in reverse index order
                    remove_order.sort(reverse=True)
                
                new_code = []
                for c in remove_order:
                    # Check if c is redundant
                    is_redundant = True
                    for idx in ball_lists[c]:
                        if coverage_count[idx] <= 1:
                            is_redundant = False
                            break
                    
                    if is_redundant:
                        for idx in ball_lists[c]:
                            coverage_count[idx] -= 1
                    else:
                        new_code.append(c)
                
                current_code = new_code
                
                if len(current_code) < best_count:
                    best_count = len(current_code)
                    best_code = list(current_code)
            
            # If we've reached the theoretical lower bound, stop
            if best_count <= 71:
                break
        
        # Final greedy fill if needed
        coverage_count = [0] * total
        for c in best_code:
            for idx in ball_lists[c]:
                coverage_count[idx] += 1
        
        uncovered = [j for j in range(total) if coverage_count[j] == 0]
        
        while uncovered and time.time() < deadline - 5:
            best_idx = -1
            best_gain = 0
            for idx in range(total):
                gain = sum(1 for b in ball_lists[idx] if coverage_count[b] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
            
            if best_gain == 0:
                break
                
            best_code.append(best_idx)
            for b in ball_lists[best_idx]:
                coverage_count[b] += 1
            
            uncovered = [j for j in range(total) if coverage_count[j] == 0]
        
        result = [idx_to_word(idx) for idx in best_code]
        return result
    else:
        # Fallback: generic greedy
        code = []
        coverage_count = [0] * total
        
        while time.time() < deadline - 10:
            # Find uncovered words
            uncovered = [j for j in range(total) if coverage_count[j] == 0]
            if not uncovered:
                break
            
            # Greedy: pick codeword covering most uncovered
            best_idx = -1
            best_gain = 0
            for idx in range(total):
                gain = sum(1 for b in ball_lists[idx] if coverage_count[b] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
            
            if best_gain == 0:
                break
            
            code.append(best_idx)
            for b in ball_lists[best_idx]:
                coverage_count[b] += 1
        
        result = [idx_to_word(idx) for idx in code]
        return result
# EVOLVE-BLOCK-END
