import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    # This task is specifically q=3, n=6, radius=1 (K_3(6,1))
    # Best known: 73 codewords. Lower bound: 71.
    
    if q == 3 and n == 6 and radius == 1:
        total = q ** n  # 729
        
        # Precompute masks: mask[i] is a bitset of all words within distance 1 of word i
        # Use Python integers as bitsets (729 bits = ~100 bytes, very fast bit ops)
        
        def word_to_idx(w):
            idx = 0
            for i in range(n):
                idx = idx * q + w[i]
            return idx

        def idx_to_word(idx):
            w = [0] * n
            for i in range(n - 1, -1, -1):
                w[i] = idx % q
                idx //= q
            return w

        # Precompute all masks
        masks = [0] * total
        for i in range(total):
            w_i = idx_to_word(i)
            mask = 1 << i  # self
            for pos in range(n):
                for val in range(q):
                    if val == w_i[pos]:
                        continue
                    w_new = list(w_i)
                    w_new[pos] = val
                    idx_new = word_to_idx(tuple(w_new))
                    mask |= (1 << idx_new)
            masks[i] = mask

        all_bits = (1 << total) - 1

        def greedy_once(tie_break='max'):
            """Run greedy set cover. tie_break: 'max', 'min', or 'random'"""
            uncovered = all_bits
            code = []
            while uncovered:
                if time.time() > deadline:
                    break
                best_idx = -1
                best_count = -1
                candidates = []
                for i in range(total):
                    common = uncovered & masks[i]
                    if common:
                        cnt = common.bit_count()
                        if cnt > best_count:
                            best_count = cnt
                            candidates = [i]
                        elif cnt == best_count:
                            candidates.append(i)
                if not candidates:
                    break
                if tie_break == 'max':
                    best_idx = candidates[0]
                elif tie_break == 'min':
                    best_idx = candidates[-1]
                else:
                    best_idx = random.choice(candidates)
                code.append(best_idx)
                uncovered &= ~masks[best_idx]
            return code

        def try_remove_redundant(code):
            """Try to remove redundant codewords from the code."""
            code_set = set(code)
            # For each codeword, check if removing it still leaves full coverage
            # This is O(K * 2^729) which is too slow for bitsets, so use a different approach
            # Actually, we can check: for each codeword c, see if all words in mask[c] 
            # are covered by other codewords.
            # Build coverage count for each word
            coverage_count = [0] * total
            for c in code:
                m = masks[c]
                # Count bits in m
                # We need per-word coverage, so iterate
                # This is expensive: 729 * K operations per check
                pass
            
            # Simpler: try removing each codeword and check if uncovered becomes nonzero
            # Only do this for small codes
            if len(code) > 80:
                return code
            
            for i in range(len(code)):
                candidate_code = code[:i] + code[i+1:]
                # Check coverage
                covered = 0
                for c in candidate_code:
                    covered |= masks[c]
                if covered == all_bits:
                    return candidate_code
            return code

        def local_search_improve(code):
            """Try to improve the code via swap operations."""
            code_set = set(code)
            improved = True
            while improved:
                improved = False
                if time.time() > deadline - 5:
                    break
                # Try swapping each codeword with a non-codeword
                for i, c in enumerate(code):
                    if time.time() > deadline - 5:
                        break
                    # Try replacing c with some other word w
                    # After replacement, check if coverage is still full
                    # Then try to remove another codeword
                    # This is expensive, so limit searches
                    # Instead, try: remove c, then greedily add back minimum needed
                    pass
                
                # Simpler local search: for each pair (remove c1, add c2), check if valid
                # And then see if we can remove a third
                # Too expensive. Instead, use a different approach:
                # Try to find a codeword that's redundant when combined with others
                pass
            
            return code

        # Strategy:
        # 1. Run greedy multiple times with different tie-breaking
        # 2. For each result, try to remove redundant codewords
        # 3. Keep the best (smallest) code
        
        best_code = None
        best_size = float('inf')
        
        # Run multiple greedy variants
        num_restarts = 30
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            if restart == 0:
                code = greedy_once('max')
            elif restart == 1:
                code = greedy_once('min')
            else:
                code = greedy_once('random')
            
            if not code:
                continue
            
            # Try to remove redundant codewords
            code = try_remove_redundant(code)
            
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]
                if best_size <= 73:
                    # Can't do much better than known best
                    # Still try a bit more
                    if best_size == 71:
                        break
        
        # If we have time, do more randomized restarts
        while time.time() < deadline - 2:
            code = greedy_once('random')
            if not code:
                break
            code = try_remove_redundant(code)
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]
                if best_size == 71:
                    break
        
        if best_code is None:
            # Fallback: should not happen
            best_code = greedy_once('max')
        
        # Convert indices to words
        result = [idx_to_word(i) for i in best_code]
        return result
    
    else:
        # Generic fallback for other (q, n, radius)
        total = q ** n
        all_bits = (1 << total) - 1
        
        def word_to_idx(w):
            idx = 0
            for i in range(n):
                idx = idx * q + w[i]
            return idx

        def idx_to_word(idx):
            w = [0] * n
            for i in range(n - 1, -1, -1):
                w[i] = idx % q
                idx //= q
            return w

        # Precompute masks for radius
        masks = [0] * total
        for i in range(total):
            w_i = idx_to_word(i)
            mask = 0
            # Generate all words within Hamming distance <= radius
            for positions in range(radius + 1):
                for pos_comb in itertools.combinations(range(n), positions):
                    for vals in itertools.product(range(q), repeat=positions):
                        w_new = list(w_i)
                        for p, v in zip(pos_comb, vals):
                            w_new[p] = v
                        idx_new = word_to_idx(tuple(w_new))
                        mask |= (1 << idx_new)
            masks[i] = mask

        uncovered = all_bits
        code = []
        while uncovered:
            if time.time() > deadline:
                break
            best_idx = -1
            best_count = -1
            for i in range(total):
                common = uncovered & masks[i]
                if common:
                    cnt = common.bit_count()
                    if cnt > best_count:
                        best_count = cnt
                        best_idx = i
            if best_idx == -1:
                break
            code.append(best_idx)
            uncovered &= ~masks[best_idx]
        
        return [idx_to_word(i) for i in code]
