import time
import random
import sys
import heapq

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 112  # 8s margin from 120s limit
    
    total_words = q ** n
    
    def word_to_idx(word):
        idx = 0
        for d in word:
            idx = idx * q + d
        return idx

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n-1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Precompute balls and masks
    balls = []
    ball_masks = []
    
    for idx in range(total_words):
        w = idx_to_word(idx)
        neighbors = [idx]
        for pos in range(n):
            original = w[pos]
            for new_val in range(q):
                if new_val == original:
                    continue
                w[pos] = new_val
                n_idx = word_to_idx(w)
                neighbors.append(n_idx)
            w[pos] = original
        
        balls.append(neighbors)
        mask = 0
        for nb in neighbors:
            mask |= (1 << nb)
        ball_masks.append(mask)
        
    all_mask = (1 << total_words) - 1
    
    # Precompute for each word, which ball indices cover it (reverse mapping)
    # coverers[word] = list of ball indices whose ball contains word
    coverers = [[] for _ in range(total_words)]
    for c in range(total_words):
        for w in balls[c]:
            coverers[w].append(c)
    
    best_solution = None
    best_size = float('inf')
    
    def greedy_cover_lazy(seed):
        """Lazy greedy: use a heap to avoid recomputing gains for all candidates."""
        random.seed(seed)
        covered_mask = 0
        code = []
        
        # Heap entries: (-gain, tiebreak, candidate)
        # We'll recompute gains lazily
        heap = []
        for c in range(total_words):
            gain = ball_masks[c].bit_count()
            heapq.heappush(heap, (-gain, random.random(), c))
        
        while covered_mask != all_mask and time.time() < deadline:
            # Pop until we find a valid entry
            while heap:
                neg_gain, _, c = heapq.heappop(heap)
                actual_gain = (ball_masks[c] & ~covered_mask).bit_count()
                if -neg_gain == actual_gain:
                    break
            else:
                break
            
            # This is the best candidate
            code.append(c)
            covered_mask |= ball_masks[c]
            
            # Update heap: for all candidates, their gain may have decreased
            # Instead of updating all, we'll just push new entries with current gains
            # when we need them. For now, we rely on lazy deletion.
            # To keep heap manageable, periodically rebuild
            if len(code) % 20 == 0:
                heap = []
                for c in range(total_words):
                    gain = (ball_masks[c] & ~covered_mask).bit_count()
                    if gain > 0:
                        heapq.heappush(heap, (-gain, random.random(), c))
        
        return code

    def reduce_cover(code):
        """Remove redundant codewords."""
        if not code:
            return code
            
        cover_count = [0] * total_words
        for c in code:
            for w in balls[c]:
                cover_count[w] += 1
        
        changed = True
        while changed:
            changed = False
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                is_redundant = True
                for w in balls[c]:
                    if cover_count[w] <= 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for w in balls[c]:
                        cover_count[w] -= 1
                    code.pop(i)
                    changed = True
        
        return code

    def local_search(code):
        """
        Local search with smarter swap strategy.
        Key idea: For each codeword c, compute its "loss set" (words only covered by c).
        Then find a replacement c' that covers the loss set.
        Use coverers to quickly find candidates.
        """
        if not code:
            return code
            
        cover_count = [0] * total_words
        for c in code:
            for w in balls[c]:
                cover_count[w] += 1
                
        in_code = [False] * total_words
        for c in code:
            in_code[c] = True
            
        improved = True
        iterations = 0
        max_iterations = 500
        
        while improved and time.time() < deadline and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # Phase 1: Remove redundant codewords
            for i in range(len(code)):
                c = code[i]
                redundant = True
                for w in balls[c]:
                    if cover_count[w] == 1:
                        redundant = False
                        break
                if redundant:
                    for w in balls[c]:
                        cover_count[w] -= 1
                    in_code[c] = False
                    code.pop(i)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Phase 2: Try swaps
            # For each codeword, compute loss set
            for i in range(len(code)):
                c = code[i]
                loss_words = []
                for w in balls[c]:
                    if cover_count[w] == 1:
                        loss_words.append(w)
                
                if not loss_words:
                    continue  # Shouldn't happen after phase 1
                
                # Find candidates c' that cover all loss_words
                # Use coverers: for each loss word, get its coverers
                # Intersection of coverers[loss_words] gives candidates
                # But that's expensive. Instead, pick the loss word with fewest coverers
                # and iterate through its coverers.
                
                # Find loss word with minimum coverers count
                min_coverer_word = min(loss_words, key=lambda w: len(coverers[w]))
                
                found_swap = False
                for c_prime in coverers[min_coverer_word]:
                    if in_code[c_prime]:
                        continue
                    # Check if c_prime covers all loss_words
                    covers_all = True
                    for w in loss_words:
                        if (ball_masks[c_prime] >> w) & 1 == 0:
                            covers_all = False
                            break
                    if covers_all:
                        # Perform swap
                        for w in balls[c]:
                            cover_count[w] -= 1
                        in_code[c] = False
                        code[i] = c_prime
                        
                        for w in balls[c_prime]:
                            cover_count[w] += 1
                        in_code[c_prime] = True
                        
                        # Check if this enabled a reduction
                        reduced = False
                        for j in range(len(code)):
                            cj = code[j]
                            redundant = True
                            for w in balls[cj]:
                                if cover_count[w] == 1:
                                    redundant = False
                                    break
                            if redundant:
                                for w in balls[cj]:
                                    cover_count[w] -= 1
                                in_code[cj] = False
                                code.pop(j)
                                reduced = True
                                break
                            
                        if reduced:
                            improved = True
                            break
                        else:
                            # Revert swap
                            for w in balls[c_prime]:
                                cover_count[w] -= 1
                            in_code[c_prime] = False
                            code[i] = c
                            for w in balls[c]:
                                cover_count[w] += 1
                            in_code[c] = True
                        
                        found_swap = True
                        break
                
                if found_swap and improved:
                    break
                elif found_swap and not improved:
                    # Tried a swap but no reduction; try next codeword
                    continue
            
        return code

    # Main loop
    initial_code = greedy_cover_lazy(0)
    if initial_code:
        reduced_code = reduce_cover(initial_code)
        local_searched_code = local_search(reduced_code)
        best_solution = local_searched_code
        best_size = len(local_searched_code)
    
    seed = 1
    while time.time() < deadline:
        code = greedy_cover_lazy(seed)
        if code is None:
            break
        reduced_code = reduce_cover(code)
        local_searched_code = local_search(reduced_code)
        if len(local_searched_code) < best_size:
            best_solution = local_searched_code
            best_size = len(local_searched_code)
        seed += 1
        
    if best_solution is None:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    result = []
    for idx in best_solution:
        result.append(idx_to_word(idx))
        
    return result
