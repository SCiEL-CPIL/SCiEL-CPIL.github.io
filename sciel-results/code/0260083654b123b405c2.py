import numpy as np
import time
import itertools
import random
from collections import deque

def construct(q, n, radius):
    if q == 3 and n == 8 and radius == 1:
        deadline = time.time() + 115.0
        N = q ** n  # 6561
        powers = [q ** i for i in range(n)]
        
        # Precompute neighbors
        neighbors = np.empty((N, 1 + 2 * n), dtype=np.int32)
        for w in range(N):
            neighbors[w, 0] = w
            idx = 1
            for i in range(n):
                d = (w // powers[i]) % q
                diff1 = ((d + 1) % q) - d
                w1 = w + diff1 * powers[i]
                neighbors[w, idx] = w1
                idx += 1
                diff2 = ((d + 2) % q) - d
                w2 = w + diff2 * powers[i]
                neighbors[w, idx] = w2
                idx += 1
        
        # Precompute neighbor masks for fast intersection checks
        # neighbor_mask[w, d] is True if d is a neighbor of w (dist <= 1)
        neighbor_mask = np.zeros((N, N), dtype=bool)
        for w in range(N):
            neighbor_mask[w, neighbors[w, :]] = True
        
        # Precompute distance-2 neighbors for swap candidates
        # A word d is a valid replacement for c if dist(c,d) <= 2
        # and d covers all essential words of c
        # We'll compute this on-the-fly but cache the neighbor sets
        
        def greedy_init(rng, mode=0, is_in_code=None):
            """
            mode 0: deterministic greedy (max gain, first candidate)
            mode 1: randomized top-k greedy
            mode 2: random start, then greedy
            """
            if is_in_code is None:
                is_in_code = np.zeros(N, dtype=bool)
            uncovered = np.ones(N, dtype=np.bool_)
            code = set()
            
            if mode == 2:
                # Start with a random word
                w = rng.integers(0, N)
                code.add(w)
                is_in_code[w] = True
                uncovered[neighbors[w, :]] = False
            
            while np.any(uncovered) and time.time() < deadline:
                gains = uncovered[neighbors].sum(axis=1)
                max_gain = np.max(gains)
                if max_gain == 0:
                    break
                candidates = np.where(gains == max_gain)[0]
                
                if mode == 1 and len(candidates) > 1:
                    top_k = min(10, len(candidates))
                    w = rng.choice(candidates[:top_k])
                elif mode == 0:
                    w = candidates[0]
                else:
                    w = rng.choice(candidates)
                
                code.add(w)
                is_in_code[w] = True
                uncovered[neighbors[w, :]] = False
            
            return code, is_in_code
        
        def reduce_code(code_set, coverage, neighbors, is_in_code, deadline):
            if not code_set:
                return code_set, coverage, is_in_code
            
            changed = True
            while changed and time.time() < deadline:
                changed = False
                for c in list(code_set):
                    if np.min(coverage[neighbors[c, :]]) > 1:
                        code_set.remove(c)
                        is_in_code[c] = False
                        coverage[neighbors[c, :]] -= 1
                        changed = True
            
            return code_set, coverage, is_in_code
        
        def compute_coverage(code_set, neighbors):
            coverage = np.zeros(N, dtype=np.int32)
            for c in code_set:
                coverage[neighbors[c, :]] += 1
            return coverage
        
        def get_replacement(essential_words, is_in_code, neighbor_mask, rng):
            """Find a word d not in code_set that covers all essential_words."""
            if len(essential_words) == 0:
                return None
            
            valid_mask = np.ones(N, dtype=bool)
            for ew in essential_words:
                valid_mask &= neighbor_mask[ew, :]
            
            valid_mask &= ~is_in_code
            
            candidates = np.where(valid_mask)[0]
            if len(candidates) == 0:
                return None
            
            return rng.choice(candidates)
        
        def try_single_swap(code_set, coverage, neighbors, is_in_code, neighbor_mask, deadline, rng):
            """Try to remove one codeword and add another (or just remove if redundant)."""
            if not code_set:
                return code_set, coverage, is_in_code, False
            
            code_list = list(code_set)
            rng.shuffle(code_list)
            
            for c in code_list:
                if time.time() > deadline - 5:
                    break
                c_neighbors = neighbors[c, :]
                essential_words = c_neighbors[coverage[c_neighbors] == 1]
                
                if len(essential_words) == 0:
                    # c is redundant
                    code_set.remove(c)
                    is_in_code[c] = False
                    coverage[neighbors[c, :]] -= 1
                    return code_set, coverage, is_in_code, True
                
                if len(essential_words) > 3:
                    continue  # Too hard to cover with one word efficiently
                
                # Find a replacement
                d = get_replacement(essential_words, is_in_code, neighbor_mask, rng)
                
                if d is not None:
                    code_set.remove(c)
                    code_set.add(d)
                    is_in_code[c] = False
                    is_in_code[d] = True
                    coverage[neighbors[c, :]] -= 1
                    coverage[neighbors[d, :]] += 1
                    
                    # Try to reduce further
                    code_set, coverage, is_in_code = reduce_code(code_set, coverage, neighbors, is_in_code, deadline)
                    return code_set, coverage, is_in_code, True
            
            return code_set, coverage, is_in_code, False
        
        def try_double_swap(code_set, coverage, neighbors, is_in_code, neighbor_mask, deadline, rng):
            """Try to remove two codewords and add one or two new ones."""
            if len(code_set) < 3:
                return code_set, coverage, is_in_code, False
            
            code_list = list(code_set)
            rng.shuffle(code_list)
            
            max_pairs = min(30, len(code_list) - 1)
            
            for i in range(max_pairs):
                if time.time() > deadline - 5:
                    break
                c1 = code_list[i]
                c2 = code_list[(i + 1) % len(code_list)]
                if c1 == c2:
                    continue
                
                # Compute combined essential words
                combined_cov = coverage.copy()
                combined_cov[neighbors[c1, :]] -= 1
                combined_cov[neighbors[c2, :]] -= 1
                
                # Words that become uncovered
                uncovered_words = np.where(combined_cov == 0)[0]
                
                if len(uncovered_words) > 4:
                    continue  # Too many to cover with 1-2 new words
                
                if len(uncovered_words) == 0:
                    # Both are redundant
                    code_set.remove(c1)
                    code_set.remove(c2)
                    is_in_code[c1] = False
                    is_in_code[c2] = False
                    coverage[neighbors[c1, :]] -= 1
                    coverage[neighbors[c2, :]] -= 1
                    return code_set, coverage, is_in_code, True
                
                # Try to find 1-2 words that cover all uncovered_words
                uw = uncovered_words.tolist()
                
                # Try single word
                if len(uw) <= 3:
                    d = get_replacement(uw, is_in_code, neighbor_mask, rng)
                    if d is not None:
                        code_set.remove(c1)
                        code_set.remove(c2)
                        code_set.add(d)
                        is_in_code[c1] = False
                        is_in_code[c2] = False
                        is_in_code[d] = True
                        coverage[neighbors[c1, :]] -= 1
                        coverage[neighbors[c2, :]] -= 1
                        coverage[neighbors[d, :]] += 1
                        
                        code_set, coverage, is_in_code = reduce_code(code_set, coverage, neighbors, is_in_code, deadline)
                        return code_set, coverage, is_in_code, True
                
                # Try two words
                if len(uw) <= 5:
                    # Split uncovered words into two groups
                    for split in range(1, len(uw)):
                        group1 = uw[:split]
                        group2 = uw[split:]
                        
                        # Find word for group1
                        d1 = get_replacement(group1, is_in_code, neighbor_mask, rng)
                        if d1 is None:
                            continue
                        
                        # Temporarily mark d1 as in_code to exclude it from d2 search?
                        # No, d2 just needs to cover group2 and not be in code.
                        # If d2 == d1, it doesn't help cover group2 if group2 is disjoint from group1's coverage?
                        # Actually, if d1 covers group1, does it cover group2?
                        # We need d2 to cover group2.
                        # If d1 also covers group2, then d2 is not needed?
                        # But we are adding 2 words.
                        
                        # Simple check: find d2 covering group2, not in code, and d2 != d1
                        d2 = get_replacement(group2, is_in_code, neighbor_mask, rng)
                        if d2 is None or d2 == d1:
                            continue
                        
                        # Verify the swap actually works
                        new_cov = coverage.copy()
                        new_cov[neighbors[c1, :]] -= 1
                        new_cov[neighbors[c2, :]] -= 1
                        new_cov[neighbors[d1, :]] += 1
                        new_cov[neighbors[d2, :]] += 1
                        
                        if np.min(new_cov) >= 1:
                            code_set.remove(c1)
                            code_set.remove(c2)
                            code_set.add(d1)
                            code_set.add(d2)
                            is_in_code[c1] = False
                            is_in_code[c2] = False
                            is_in_code[d1] = True
                            is_in_code[d2] = True
                            coverage = new_cov
                            
                            code_set, coverage, is_in_code = reduce_code(code_set, coverage, neighbors, is_in_code, deadline)
                            return code_set, coverage, is_in_code, True
            
            return code_set, coverage, is_in_code, False
        
        def add_reduce(code_set, coverage, neighbors, is_in_code, deadline, rng):
            """Add a random word, then reduce to escape local minima."""
            if not code_set:
                return code_set, coverage, is_in_code, False
            
            w = rng.integers(0, N)
            if is_in_code[w]:
                return code_set, coverage, is_in_code, False
            
            new_set = set(code_set)
            new_is_in_code = is_in_code.copy()
            new_set.add(w)
            new_is_in_code[w] = True
            new_coverage = coverage.copy()
            new_coverage[neighbors[w, :]] += 1
            
            new_set, new_coverage, new_is_in_code = reduce_code(new_set, new_coverage, neighbors, new_is_in_code, deadline)
            
            if len(new_set) < len(code_set):
                return new_set, new_coverage, new_is_in_code, True
            
            return code_set, coverage, is_in_code, False
        
        best_code = None
        best_size = float('inf')
        
        # Run multiple restarts with different strategies
        modes = [0, 1, 2, 1, 0, 2, 1, 0, 2, 1, 0, 1, 2, 0, 1, 2, 0, 1, 2, 1]
        
        for i in range(20):
            if time.time() > deadline - 5:
                break
            
            rng = np.random.default_rng(42 + i * 13)
            mode = modes[i % len(modes)]
            
            code_set, is_in_code = greedy_init(rng, mode)
            coverage = compute_coverage(code_set, neighbors)
            code_set, coverage, is_in_code = reduce_code(code_set, coverage, neighbors, is_in_code, deadline)
            
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for _ in range(150):
                    if time.time() > deadline - 5:
                        break
                    
                    r = rng.random()
                    if r < 0.5:
                        new_set, new_coverage, new_is_in_code, was_improved = try_single_swap(
                            code_set, coverage, neighbors, is_in_code, neighbor_mask, deadline, rng
                        )
                    elif r < 0.8:
                        new_set, new_coverage, new_is_in_code, was_improved = try_double_swap(
                            code_set, coverage, neighbors, is_in_code, neighbor_mask, deadline, rng
                        )
                    else:
                        new_set, new_coverage, new_is_in_code, was_improved = add_reduce(
                            code_set, coverage, neighbors, is_in_code, deadline, rng
                        )
                    
                    if was_improved:
                        code_set = new_set
                        coverage = new_coverage
                        is_in_code = new_is_in_code
                        improved = True
            
            if len(code_set) < best_size:
                best_size = len(code_set)
                best_code = set(code_set)
                
            if best_size <= 486:
                break
        
        # Polish pass: thorough 1-swap and 2-swap on the best code
        if best_code is not None and time.time() < deadline - 3:
            code_set = set(best_code)
            is_in_code = np.zeros(N, dtype=bool)
            for c in code_set:
                is_in_code[c] = True
            coverage = compute_coverage(code_set, neighbors)
            rng = np.random.default_rng(999)
            
            # First, do a full systematic reduction pass
            code_set, coverage, is_in_code = reduce_code(code_set, coverage, neighbors, is_in_code, deadline)
            
            # Then, multiple rounds of local search
            for round_num in range(5):
                if time.time() > deadline - 3:
                    break
                
                improved_in_round = True
                while improved_in_round and time.time() < deadline - 3:
                    improved_in_round = False
                    
                    # Try 1-swaps on all codewords
                    for c in list(code_set):
                        if time.time() > deadline - 3:
                            break
                        c_neighbors = neighbors[c, :]
                        essential_words = c_neighbors[coverage[c_neighbors] == 1]
                        
                        if len(essential_words) == 0:
                            code_set.remove(c)
                            is_in_code[c] = False
                            coverage[neighbors[c, :]] -= 1
                            improved_in_round = True
                            break
                        
                        if len(essential_words) <= 3:
                            d = get_replacement(essential_words, is_in_code, neighbor_mask, rng)
                            if d is not None:
                                code_set.remove(c)
                                code_set.add(d)
                                is_in_code[c] = False
                                is_in_code[d] = True
                                coverage[neighbors[c, :]] -= 1
                                coverage[neighbors[d, :]] += 1
                                improved_in_round = True
                                break
                    
                    if improved_in_round:
                        continue
                    
                    # Try 2-swaps
                    code_list = list(code_set)
                    rng.shuffle(code_list)
                    for i in range(min(20, len(code_list) - 1)):
                        if time.time() > deadline - 3:
                            break
                        c1 = code_list[i]
                        c2 = code_list[(i + 1) % len(code_list)]
                        if c1 == c2:
                            continue
                        
                        combined_cov = coverage.copy()
                        combined_cov[neighbors[c1, :]] -= 1
                        combined_cov[neighbors[c2, :]] -= 1
                        uncovered_words = np.where(combined_cov == 0)[0]
                        
                        if len(uncovered_words) == 0:
                            code_set.remove(c1)
                            code_set.remove(c2)
                            is_in_code[c1] = False
                            is_in_code[c2] = False
                            coverage[neighbors[c1, :]] -= 1
                            coverage[neighbors[c2, :]] -= 1
                            improved_in_round = True
                            break
                        
                        if len(uncovered_words) <= 4:
                            uw = uncovered_words.tolist()
                            # Try single replacement
                            if len(uw) <= 3:
                                d = get_replacement(uw, is_in_code, neighbor_mask, rng)
                                if d is not None:
                                    new_cov = coverage.copy()
                                    new_cov[neighbors[c1, :]] -= 1
                                    new_cov[neighbors[c2, :]] -= 1
                                    new_cov[neighbors[d, :]] += 1
                                    if np.min(new_cov) >= 1:
                                        code_set.remove(c1)
                                        code_set.remove(c2)
                                        code_set.add(d)
                                        is_in_code[c1] = False
                                        is_in_code[c2] = False
                                        is_in_code[d] = True
                                        coverage = new_cov
                                        improved_in_round = True
                                        break
                            
                            # Try two replacements
                            if len(uw) <= 5:
                                for split in range(1, len(uw)):
                                    group1 = uw[:split]
                                    group2 = uw[split:]
                                    
                                    d1 = get_replacement(group1, is_in_code, neighbor_mask, rng)
                                    if d1 is None:
                                        continue
                                    
                                    d2 = get_replacement(group2, is_in_code, neighbor_mask, rng)
                                    if d2 is None or d2 == d1:
                                        continue
                                    
                                    new_cov = coverage.copy()
                                    new_cov[neighbors[c1, :]] -= 1
                                    new_cov[neighbors[c2, :]] -= 1
                                    new_cov[neighbors[d1, :]] += 1
                                    new_cov[neighbors[d2, :]] += 1
                                    
                                    if np.min(new_cov) >= 1:
                                        code_set.remove(c1)
                                        code_set.remove(c2)
                                        code_set.add(d1)
                                        code_set.add(d2)
                                        is_in_code[c1] = False
                                        is_in_code[c2] = False
                                        is_in_code[d1] = True
                                        is_in_code[d2] = True
                                        coverage = new_cov
                                        improved_in_round = True
                                        break
                                    if improved_in_round:
                                        break
                    
                    # Final reduction
                    code_set, coverage, is_in_code = reduce_code(code_set, coverage, neighbors, is_in_code, deadline)
            
            if len(code_set) < best_size:
                best_size = len(code_set)
                best_code = code_set
        
        if best_code is None:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            best_code = set()
            for h in tetracode:
                for tail in itertools.product(range(3), repeat=n - 4):
                    word = h + list(tail)
                    val = 0
                    for i, d in enumerate(word):
                        val += d * (q ** i)
                    best_code.add(val)
        
        final_code = []
        for w in best_code:
            word = []
            temp = w
            for _ in range(n):
                word.append(temp % q)
                temp //= q
            final_code.append(word)
        
        return final_code
    
    else:
        N = q ** n
        deadline = time.time() + 110.0
        neighbors = np.empty((N, 1 + 2 * n), dtype=np.int32)
        powers = [q ** i for i in range(n)]
        for w in range(N):
            neighbors[w, 0] = w
            idx = 1
            for i in range(n):
                d = (w // powers[i]) % q
                diff1 = ((d + 1) % q) - d
                w1 = w + diff1 * powers[i]
                neighbors[w, idx] = w1
                idx += 1
                diff2 = ((d + 2) % q) - d
                w2 = w + diff2 * powers[i]
                neighbors[w, idx] = w2
                idx += 1
        
        rng = np.random.default_rng()
        uncovered = np.ones(N, dtype=np.bool_)
        code = set()
        while np.any(uncovered) and time.time() < deadline:
            gains = uncovered[neighbors].sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            w = rng.choice(candidates)
            code.add(w)
            uncovered[neighbors[w, :]] = False
        
        final_code = []
        for w in code:
            word = []
            temp = w
            for _ in range(n):
                word.append(temp % q)
                temp //= q
            final_code.append(word)
        return final_code
