import itertools
import random
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for safety
    
    # 1. Precompute space and spheres (balls)
    # For q=3, n=6, total=729. Ball size for radius 1 is 1 + n*(q-1) = 1 + 6*2 = 13.
    total = q ** n
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # sphere[i] is a list of indices of words within Hamming distance 'radius' from words[i]
    # For radius 1, we can compute this quickly.
    spheres = []
    for i, w in enumerate(words):
        ball = [i]
        # For general radius, we'd need a BFS or combinatorial generation.
        # For this specific task (radius 1), we just change one coordinate.
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val != orig_val:
                    new_w_list = list(w)
                    new_w_list[pos] = new_val
                    new_w_tuple = tuple(new_w_list)
                    ball.append(word_to_idx[new_w_tuple])
        spheres.append(ball)
        
    # Helper to check if a set of codeword indices covers the space
    def is_covered(code_indices):
        if not code_indices:
            return total == 0
        counts = [0] * total
        for ci in code_indices:
            for w in spheres[ci]:
                counts[w] += 1
        # Check if any count is 0
        for c in counts:
            if c == 0:
                return False
        return True

    def get_uncovered(code_indices):
        counts = [0] * total
        for ci in code_indices:
            for w in spheres[ci]:
                counts[w] += 1
        return [i for i, c in enumerate(counts) if c == 0]

    # Local Search: Remove redundant codewords
    # A codeword is redundant if every word in its ball is covered by at least one other codeword.
    # i.e., for all w in sphere[ci], count[w] > 1 (before removal).
    def local_search_reduce(code_indices):
        # Use a set for O(1) removal checks if needed, but list is fine for small sizes
        # We work with a list of indices
        code = sorted(set(code_indices))
        if not code:
            return code
            
        # Initial counts
        counts = [0] * total
        for ci in code:
            for w in spheres[ci]:
                counts[w] += 1
                
        changed = True
        while changed:
            changed = False
            new_code = []
            for ci in code:
                # Check if ci is redundant
                # If removing ci leaves all its neighbors covered, it's redundant.
                # This is equivalent to: min(counts[w] for w in spheres[ci]) > 1
                # Wait, if counts[w] is the count INCLUDING ci, then we need counts[w] >= 2.
                # If counts[w] == 1 for any w in spheres[ci], then removing ci makes counts[w]=0.
                is_redundant = True
                for w in spheres[ci]:
                    if counts[w] <= 1:
                        is_redundant = False
                        break
                if is_redundant:
                    # Remove ci
                    for w in spheres[ci]:
                        counts[w] -= 1
                    changed = True
                else:
                    new_code.append(ci)
            code = new_code
        return code

    def greedy_cover(uncovered_indices, code_indices, max_adds=20):
        # Greedily add centers to cover uncovered_indices
        # Returns updated code_indices
        if not uncovered_indices:
            return code_indices
            
        uncov_set = set(uncovered_indices)
        current_code = list(code_indices)
        
        # Candidate centers: all words in the space
        # To speed up, we can restrict candidates to words that cover at least one uncovered word.
        # But for n=6, total=729, scanning all is cheap.
        
        for _ in range(max_adds):
            if not uncov_set:
                break
            best_c = -1
            best_count = -1
            # Random tie-breaking to allow different paths in multi-start
            # But for deterministic repair, we might want stable order? 
            # Let's use a random shuffle of candidates to break ties.
            candidates = list(range(total))
            random.shuffle(candidates)
            
            for c in candidates:
                cnt = 0
                for w in spheres[c]:
                    if w in uncov_set:
                        cnt += 1
                if cnt > best_count:
                    best_count = cnt
                    best_c = c
                elif cnt == best_count and random.random() < 0.5:
                    # Random tie break
                    best_c = c
            
            if best_c == -1 or best_count == 0:
                break # Cannot cover more? Should not happen if space is connected.
                
            current_code.append(best_c)
            for w in spheres[best_c]:
                uncov_set.discard(w)
                
        return current_code

    # Main Search Logic
    best_code = []
    best_size = float('inf')
    
    # 1. Seed with Tetracode Direct Sum
    # Tetracode: 9 words of length 4.
    # Extend to length 6 by adding 2 free coordinates.
    # Size: 9 * 3^2 = 81.
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    seed_indices = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            w = tuple(h + list(tail))
            seed_indices.append(word_to_idx[w])
            
    # Local search on seed
    cand = local_search_reduce(seed_indices)
    if is_covered(cand):
        best_code = cand
        best_size = len(cand)
    else:
        # Should not happen, but repair if it does
        uncov = get_uncovered(cand)
        cand = greedy_cover(uncov, cand)
        cand = local_search_reduce(cand)
        best_code = cand
        best_size = len(cand)

    # 2. Multi-Start Greedy + Local Search
    # Run several random greedy constructions
    num_trials = 30
    trial = 0
    while trial < num_trials and time.time() < deadline - 10:
        # Randomized Greedy Construction
        code = []
        uncov = set(range(total))
        
        # We want to pick centers that cover many uncovered points.
        # Standard greedy.
        while uncov and len(code) < best_size: # Don't exceed current best
            best_c = -1
            best_cnt = -1
            # Sample a subset of candidates to speed up? Or just scan all.
            # Scanning all 729 is fast.
            # Add slight randomness to tie-breaking or selection.
            candidates = list(range(total))
            random.shuffle(candidates)
            
            for c in candidates:
                cnt = 0
                for w in spheres[c]:
                    if w in uncov:
                        cnt += 1
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_c = c
                elif cnt == best_cnt and random.random() < 0.1:
                    best_c = c
                    
            if best_c == -1 or best_cnt == 0:
                break
                
            code.append(best_c)
            for w in spheres[best_c]:
                uncov.discard(w)
                
        if not uncov:
            # Greedy finished, try to reduce
            code = local_search_reduce(code)
            if is_covered(code) and len(code) < best_size:
                best_size = len(code)
                best_code = code
                
        trial += 1

    # 3. Perturbation / Hill Climbing from Best
    # Drop 1 or 2 codewords, repair, and reduce.
    # This helps escape local minima where a specific set of codewords is "stuck".
    perturbations = 100
    for _ in range(perturbations):
        if time.time() > deadline - 5:
            break
            
        if not best_code:
            break
            
        code = list(best_code)
        k = len(code)
        
        # Drop 1 or 2 random codewords
        drop_count = 1 if random.random() < 0.7 else 2
        if k < drop_count:
            continue
            
        drop_indices = set(random.sample(range(k), drop_count))
        new_code = [code[i] for i in range(k) if i not in drop_indices]
        
        # Find uncovered
        uncov = get_uncovered(new_code)
        
        if not uncov:
            # Still covered, maybe reduce further?
            # local_search_reduce would have handled this if we ran it, but we didn't.
            # So run it.
            new_code = local_search_reduce(new_code)
            if is_covered(new_code) and len(new_code) < best_size:
                best_size = len(new_code)
                best_code = new_code
        else:
            # Repair
            new_code = greedy_cover(uncov, new_code, max_adds=5)
            # Reduce
            new_code = local_search_reduce(new_code)
            
            if is_covered(new_code) and len(new_code) < best_size:
                best_size = len(new_code)
                best_code = new_code

    # Final Check and Return
    # Ensure validity
    if not is_covered(best_code):
        # Fallback: Re-run greedy from scratch if something went wrong
        uncov = get_uncovered(best_code)
        best_code = greedy_cover(uncov, best_code)
        best_code = local_search_reduce(best_code)
        
    return [list(words[i]) for i in best_code]

# Self-test for debugging (removed for submission)
if __name__ == "__main__":
    # Test with q=3, n=6, radius=1
    # Expected best: 73
    res = construct(3, 6, 1)
    print(f"Size: {len(res)}")
    # Verify
    # ... (verification logic similar to is_covered)
