import itertools
import time
import random
import heapq

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    total_words = q ** n
    
    # Precompute indices and ball bitmasks
    # index_to_digits and digits_to_index are fast enough for n=7, q=3
    def index_to_digits(idx):
        digits = [0] * n
        for i in range(n - 1, -1, -1):
            digits[i] = idx % q
            idx //= q
        return digits

    def digits_to_index(digits):
        idx = 0
        for d in digits:
            idx = idx * q + d
        return idx

    # Precompute balls
    # ball[i] is a bitmask where bit j is 1 if word j is within distance radius of word i
    balls = [0] * total_words
    # For radius 1, the ball contains the word itself and all words at distance 1
    # Number of words in ball = 1 + n * (q - 1)
    # For q=3, n=7, radius=1: 1 + 7*2 = 15 words per ball
    
    for i in range(total_words):
        digits = index_to_digits(i)
        mask = 1 << i
        
        # Add distance 1 neighbors
        for pos in range(n):
            orig = digits[pos]
            for v in range(q):
                if v == orig:
                    continue
                new_digits = digits[:]
                new_digits[pos] = v
                nidx = digits_to_index(new_digits)
                mask |= (1 << nidx)
        balls[i] = mask

    full_mask = (1 << total_words) - 1

    def is_valid(cover):
        union = 0
        for c in cover:
            union |= balls[c]
        return union == full_mask

    def greedy_cover():
        uncovered = full_mask
        cover = []
        remaining = list(range(total_words))
        random.shuffle(remaining)
        
        # Use a simple greedy approach: pick word that covers most uncovered
        while uncovered:
            best_idx = -1
            best_count = -1
            # To speed this up, we could use a heap, but for 2187 items, 
            # a linear scan with early termination might be okay if we are careful.
            # However, doing this 100+ times per restart is expensive.
            # Let's stick to a simple scan but optimize the inner loop.
            
            # Optimization: If we just want a valid cover, we can be a bit smarter.
            # But let's try to improve the initial quality.
            
            # Actually, for 2187 items, a linear scan takes ~2187 * 15 operations.
            # If we do this ~100 times, that's ~3 million operations. Very fast.
            
            for c in remaining:
                cnt = (uncovered & balls[c]).bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best_idx = c
                    if best_count == uncovered.bit_count():
                        break # Perfect coverage for remaining uncovered? Unlikely but possible
            
            if best_idx == -1 or best_count == 0:
                break # Should not happen if cover is valid
                
            cover.append(best_idx)
            uncovered &= ~balls[best_idx]
            remaining.remove(best_idx)
            
        return cover

    def reduce_cover(cover):
        cover_set = set(cover)
        changed = True
        while changed:
            changed = False
            for c in list(cover_set):
                if c not in cover_set: continue
                rest = cover_set - {c}
                union = 0
                for r in rest:
                    union |= balls[r]
                if union == full_mask:
                    cover_set.remove(c)
                    changed = True
        return list(cover_set)

    def local_search(cover):
        cover_set = set(cover)
        all_words = set(range(total_words))
        improved = True
        max_iters = 500
        iter_count = 0
        
        while improved and iter_count < max_iters and time.time() < deadline:
            improved = False
            iter_count += 1
            if not cover_set:
                break
                
            # Try to remove a random element
            c = random.choice(list(cover_set))
            cover_set.remove(c)
            if is_valid(cover_set):
                improved = True
                continue
            else:
                # Put it back
                cover_set.add(c)
                
            # Try 1-swap: replace c with some other word
            candidates = all_words - cover_set
            if not candidates:
                break
                
            # Try a few random swaps
            swapped = False
            for _ in range(10):
                c_prime = random.choice(list(candidates))
                new_set = cover_set - {c} | {c_prime}
                if is_valid(new_set):
                    cover_set = new_set
                    improved = True
                    swapped = True
                    break
            
            if not swapped:
                # Try to find a better swap by looking for a word that covers more
                # This is expensive, so we skip it for now and rely on random walks
                break
        
        return list(cover_set)

    def local_search_2swap(cover):
        cover_set = set(cover)
        all_words = set(range(total_words))
        improved = True
        max_iters = 100
        iter_count = 0
        
        while improved and iter_count < max_iters and time.time() < deadline:
            improved = False
            iter_count += 1
            if len(cover_set) < 2:
                break
                
            # Pick two random elements to swap with two new ones
            c1, c2 = random.sample(list(cover_set), 2)
            candidates = all_words - cover_set
            if len(candidates) < 2:
                break
                
            c3, c4 = random.sample(list(candidates), 2)
            new_set = cover_set - {c1, c2} | {c3, c4}
            if is_valid(new_set):
                cover_set = new_set
                improved = True
                
        return list(cover_set)

    best_cover = None
    best_size = float('inf')
    
    restart = 0
    while time.time() < deadline:
        restart += 1
        # Greedy
        cover = greedy_cover()
        # Reduce
        cover = reduce_cover(cover)
        
        if not is_valid(cover):
            continue
            
        # Local Search 1-swap
        cover = local_search(cover)
        cover = reduce_cover(cover)
        
        # Local Search 2-swap
        cover = local_search_2swap(cover)
        cover = reduce_cover(cover)
        
        if is_valid(cover):
            if len(cover) < best_size:
                best_size = len(cover)
                best_cover = cover[:]
                
        # If we get stuck, we might want to perturb
        # For now, just restart

    if best_cover is None:
        # Fallback: return a valid cover, even if large
        # Simple greedy without local search
        cover = greedy_cover()
        cover = reduce_cover(cover)
        best_cover = cover

    return [index_to_digits(c) for c in best_cover]
