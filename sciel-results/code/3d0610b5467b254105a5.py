import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Precompute all words
    all_words = list(itertools.product(range(q), repeat=n))
    num_words = len(all_words)
    
    # Map word to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute balls
    # ball[idx] is a set of indices of words within distance <= radius
    balls = []
    for w in all_words:
        ball = set()
        for other in all_words:
            dist = sum(1 for a, b in zip(w, other) if a != b)
            if dist <= radius:
                ball.add(word_to_idx[other])
        balls.append(ball)
        
    # Initialize with the seed's construction (Direct sum of tetracode)
    # This provides a valid starting point of size 81 for n=6, q=3, r=1
    if q == 3 and n == 6 and radius == 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        initial_codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    else:
        # Fallback: Greedy initialization for other parameters
        initial_codewords = []
        covered = [False] * num_words
        while any(not c for c in covered):
            best_cw = None
            best_count = -1
            # Heuristic: pick random candidates to avoid bias, or check all
            # For small N, check all
            for i in range(num_words):
                if i in [word_to_idx[tuple(c)] for c in initial_codewords]:
                    continue
                count = 0
                for j in balls[i]:
                    if not covered[j]:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_cw = i
                if time.time() > deadline:
                    break
            if best_cw is None:
                break
            initial_codewords.append(all_words[best_cw])
            for j in balls[best_cw]:
                covered[j] = True
            if time.time() > deadline:
                break

    # Convert to set of indices for fast lookup
    current_code = set(word_to_idx[tuple(c)] for c in initial_codewords)
    
    # Calculate coverage counts
    # cover_count[j] = number of codewords covering word j
    cover_count = [0] * num_words
    for c_idx in current_code:
        for j in balls[c_idx]:
            cover_count[j] += 1
            
    # Helper to check if code is valid (all covered)
    def is_valid():
        return all(c > 0 for c in cover_count)
        
    # Reduction Loop
    def try_reduce():
        nonlocal current_code, cover_count
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Find a redundant codeword
            candidates = []
            for c_idx in current_code:
                # Check if removing c_idx leaves all its neighbors covered by others
                redundant = True
                for j in balls[c_idx]:
                    if cover_count[j] == 1:
                        redundant = False
                        break
                if redundant:
                    candidates.append(c_idx)
                    
            if candidates:
                # Pick one (e.g., random or first)
                c_idx = random.choice(candidates)
                current_code.remove(c_idx)
                for j in balls[c_idx]:
                    cover_count[j] -= 1
                improved = True
                
    # Swap Loop
    def try_swap():
        nonlocal current_code, cover_count
        # Try to find a swap that allows a subsequent reduction
        # Or simply a swap that maintains coverage but changes the set
        # A more powerful move: 1-swap that enables a reduction.
        # We simulate: remove c, add c'. If valid, keep.
        # To allow reduction, we might want to swap c with c' such that
        # some other codeword d becomes redundant.
        
        # Simple approach: Try all swaps that maintain validity.
        # If we find one, do it and run reduction again.
        # This can be expensive, so we limit iterations.
        
        c_list = list(current_code)
        random.shuffle(c_list)
        
        for c_idx in c_list:
            if time.time() > deadline:
                break
            
            # Temporarily remove c_idx
            for j in balls[c_idx]:
                cover_count[j] -= 1
                
            # Find a c' that covers the now-uncovered words (if any)
            # Actually, if removing c_idx makes some words uncovered, c' MUST cover them.
            # Let U be the set of words with cover_count 0 after removing c_idx.
            uncovered = set()
            for j in balls[c_idx]:
                if cover_count[j] == 0:
                    uncovered.add(j)
                    
            if not uncovered:
                # c_idx was redundant, should have been caught by try_reduce
                # Restore
                for j in balls[c_idx]:
                    cover_count[j] += 1
                continue
            
            # Find c' not in current_code that covers all words in uncovered
            # And ideally, c' doesn't break other things (it only adds coverage)
            # We also want c' to be "good", e.g., covers many words that are currently covered only once?
            # No, we just need validity.
            
            best_c_prime = None
            # Heuristic: pick c' that maximizes the number of words it covers that are currently covered exactly once?
            # Or just pick the first one that works.
            # To allow reduction, we might prefer c' that covers words heavily covered by others?
            # Actually, if we just maintain validity, we might get stuck.
            # Let's just find ANY valid c'.
            
            for c_prime in range(num_words):
                if c_prime in current_code:
                    continue
                if time.time() > deadline:
                    break
                    
                # Check if c_prime covers all words in uncovered
                valid_swap = True
                for u in uncovered:
                    if u not in balls[c_prime]:
                        valid_swap = False
                        break
                        
                if valid_swap:
                    best_c_prime = c_prime
                    break
                    
            if best_c_prime is not None:
                # Perform swap
                current_code.add(best_c_prime)
                for j in balls[best_c_prime]:
                    cover_count[j] += 1
                # We removed c_idx earlier, so we are good.
                # We should return to allow try_reduce to run
                return True
            else:
                # Restore c_idx
                for j in balls[c_idx]:
                    cover_count[j] += 1
                    
        return False

    # Main loop
    # First, reduce
    try_reduce()
    
    # Then, swap and reduce iteratively
    max_iters = 1000
    iter_count = 0
    while time.time() < deadline and iter_count < max_iters:
        iter_count += 1
        if try_swap():
            try_reduce()
        else:
            # If no simple swap works, try a more aggressive search?
            # For now, break if no swap found in one pass.
            # To avoid infinite loops on same state, we could track states, 
            # but given time limit, we just break.
            break

    # Convert back to list of lists
    result = [all_words[i] for i in current_code]
    return result
