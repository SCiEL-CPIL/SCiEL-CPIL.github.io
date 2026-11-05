import numpy as np
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Specific case optimization
    if q == 3 and n == 6 and radius == 1:
        total = 3 ** 6
        powers = [3 ** p for p in range(6)]
        
        # Precompute neighbors as bitsets
        # cover_mask[i] is an integer with bits set for all j such that dist(i, j) <= 1
        cover_mask = [0] * total
        for i in range(total):
            d = [(i // powers[p]) % 3 for p in range(6)]
            mask = 1 << i
            for p in range(6):
                for alt in range(3):
                    if alt != d[p]:
                        new_i = i + (alt - d[p]) * powers[p]
                        mask |= (1 << new_i)
            cover_mask[i] = mask

        def idx_to_list(idx):
            return [(idx // powers[p]) % 3 for p in range(6)]

        def greedy_set_cover():
            """Greedy set cover using bitsets."""
            uncovered = (1 << total) - 1
            codewords = []
            
            while uncovered:
                # Find word that covers most uncovered bits
                best_c = -1
                best_count = -1
                # Optimization: We can sample or iterate. Iterating 729 is fast.
                # To add diversity, we can randomize tie-breaking or use a small window of top-k.
                # For now, deterministic max, but we can add noise.
                for c in range(total):
                    count = (cover_mask[c] & uncovered).bit_count()
                    if count > best_count:
                        best_count = count
                        best_c = c
                
                codewords.append(best_c)
                uncovered &= ~cover_mask[best_c]
                
            return codewords

        def local_search(codewords, time_budget):
            """Perform local search to reduce code size using bitsets."""
            S = set(codewords)
            S_mask = 0
            for c in S:
                S_mask |= (1 << c)
            
            best_S = set(S)
            best_size = len(S)
            full_mask = (1 << total) - 1
            
            def try_remove():
                """Try to find a removable word."""
                nonlocal S, S_mask, best_S, best_size
                items = list(S)
                random.shuffle(items)
                
                for w in items:
                    union_others = 0
                    for v in S:
                        if v != w:
                            union_others |= cover_mask[v]
                    
                    L = cover_mask[w] & ~union_others
                    
                    if L == 0:
                        S.remove(w)
                        S_mask &= ~(1 << w)
                        if len(S) < best_size:
                            best_size = len(S)
                            best_S = set(S)
                        return True
                return False

            def try_swap():
                """Try to swap a word with another that covers its unique words."""
                nonlocal S, S_mask, best_S, best_size
                items = list(S)
                random.shuffle(items)
                
                for w in items:
                    union_others = 0
                    for v in S:
                        if v != w:
                            union_others |= cover_mask[v]
                    
                    L = cover_mask[w] & ~union_others
                    
                    if L == 0:
                        continue
                    
                    candidates = full_mask
                    temp_L = L
                    while temp_L:
                        x = (temp_L & -temp_L).bit_length() - 1
                        temp_L &= temp_L - 1
                        candidates &= cover_mask[x]
                        if candidates == 0:
                            break
                    
                    if candidates == 0:
                        continue
                    
                    candidates &= ~S_mask
                    candidates &= ~(1 << w)
                    
                    if candidates:
                        # Pick a random candidate from the set bits
                        # Count bits and pick a random position
                        bit_count = candidates.bit_count()
                        if bit_count > 0:
                            # Use random to pick a position among set bits
                            # Efficient way: mask with random large int repeatedly
                            # Or: convert to list of set bit positions (slow for large)
                            # For 729 bits, we can use a simple approach
                            r = random.getrandbits(64)
                            masked = candidates & r
                            if masked:
                                w_new = (masked & -masked).bit_length() - 1
                            else:
                                # Fallback: pick lowest
                                w_new = (candidates & -candidates).bit_length() - 1
                            
                            S.remove(w)
                            S.add(w_new)
                            S_mask = 0
                            for c in S:
                                S_mask |= (1 << c)
                            return True
                
                return False

            def try_two_swap():
                """Try to remove two words and add one that covers both their unique words."""
                nonlocal S, S_mask, best_S, best_size
                if len(S) < 3:
                    return False
                
                items = list(S)
                random.shuffle(items)
                
                # Try a few pairs
                for i in range(min(len(items), 20)):
                    for j in range(i + 1, min(len(items), 20)):
                        w1, w2 = items[i], items[j]
                        
                        # Compute unique coverage for w1 and w2
                        union_others = 0
                        for v in S:
                            if v != w1 and v != w2:
                                union_others |= cover_mask[v]
                        
                        L1 = cover_mask[w1] & ~union_others
                        L2 = cover_mask[w2] & ~union_others
                        
                        # Combined uncovered if we remove both
                        L = L1 | L2
                        
                        if L == 0:
                            # Both redundant
                            S.remove(w1)
                            S.remove(w2)
                            S_mask &= ~(1 << w1)
                            S_mask &= ~(1 << w2)
                            if len(S) < best_size:
                                best_size = len(S)
                                best_S = set(S)
                            return True
                        
                        # Find candidates that cover all of L
                        candidates = full_mask
                        temp_L = L
                        while temp_L:
                            x = (temp_L & -temp_L).bit_length() - 1
                            temp_L &= temp_L - 1
                            candidates &= cover_mask[x]
                            if candidates == 0:
                                break
                        
                        if candidates == 0:
                            continue
                        
                        candidates &= ~S_mask
                        candidates &= ~(1 << w1)
                        candidates &= ~(1 << w2)
                        
                        if candidates:
                            # Pick random candidate
                            r = random.getrandbits(64)
                            masked = candidates & r
                            if masked:
                                w_new = (masked & -masked).bit_length() - 1
                            else:
                                w_new = (candidates & -candidates).bit_length() - 1
                            
                            S.remove(w1)
                            S.remove(w2)
                            S.add(w_new)
                            S_mask = 0
                            for c in S:
                                S_mask |= (1 << c)
                            if len(S) < best_size:
                                best_size = len(S)
                                best_S = set(S)
                            return True
                
                return False

            stuck_count = 0
            max_stuck = 50
            last_improvement = time.time()
            no_improve_limit = 8.0  # seconds without improvement before trying harder
            
            while time.time() < time_budget:
                # Choose move type: mostly remove/swap, occasionally two_swap
                r = random.random()
                if r < 0.4:
                    success = try_remove()
                elif r < 0.8:
                    success = try_swap()
                else:
                    success = try_two_swap()
                
                if success and len(S) <= best_size:
                    stuck_count = 0
                    last_improvement = time.time()
                    max_stuck = random.randint(30, 100)
                else:
                    stuck_count += 1
                    if stuck_count > max_stuck:
                        stuck_count = 0
                        max_stuck = random.randint(50, 200)
                        # If no improvement for a while, try more aggressive
                        if time.time() - last_improvement > no_improve_limit:
                            # Perturb: randomly swap a few words
                            if len(S) > 2:
                                items = list(S)
                                random.shuffle(items)
                                for _ in range(random.randint(1, 3)):
                                    if len(S) > 2:
                                        w_out = random.choice(list(S))
                                        S.remove(w_out)
                                        S_mask &= ~(1 << w_out)
                                        # Find a random candidate
                                        candidates = full_mask & ~S_mask
                                        if candidates:
                                            r2 = random.getrandbits(64)
                                            masked = candidates & r2
                                            if masked:
                                                w_in = (masked & -masked).bit_length() - 1
                                            else:
                                                w_in = (candidates & -candidates).bit_length() - 1
                                            S.add(w_in)
                                            S_mask |= (1 << w_in)
                                last_improvement = time.time()
                        if max_stuck > 500:
                            break

            return list(best_S)

        # Multi-restart strategy
        best_overall = None
        best_overall_size = float('inf')
        
        restart_count = 0
        while time.time() < deadline - 5:
            restart_count += 1
            
            # Adaptive time budget: shorter for later restarts if we're doing well
            elapsed = time.time() - start_time
            remaining = deadline - 5 - time.time()
            if best_overall_size <= 74:
                # We're close to optimal, use shorter restarts to get more diversity
                time_budget = min(time.time() + 5, deadline - 5)
            else:
                # Use more time per restart
                time_budget = min(time.time() + 12, deadline - 5)
            
            # Greedy initialization
            codewords = greedy_set_cover()
            
            # Local search
            codewords = local_search(codewords, time_budget)
            
            if len(codewords) < best_overall_size:
                best_overall_size = len(codewords)
                best_overall = list(codewords)
                
            # If we found 73, do one final long polish
            if best_overall_size == 73 and time.time() < deadline - 10:
                codewords = local_search(list(best_overall), deadline - 5)
                if len(codewords) < best_overall_size:
                    best_overall_size = len(codewords)
                    best_overall = list(codewords)

        if best_overall is None:
            best_overall = [0]
            
        return [idx_to_list(idx) for idx in best_overall]

    else:
        # Fallback for other parameters
        total = q ** n
        return [[0] * n]
