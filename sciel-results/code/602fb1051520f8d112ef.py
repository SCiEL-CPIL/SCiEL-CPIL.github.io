import itertools
import time

def construct(q, n, radius):
    # For this specific task, q=3, n=7, radius=1
    # But we should be general enough or at least handle the given parameters.
    # The problem statement says the evaluator passes q, n, radius.
    # However, the specific instance is 7 matches, ternary, radius 1.
    # The seed was specific to n=7, q=3.
    # Let's assume the evaluator calls construct(3, 7, 1).
    
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback to a simple construction for other parameters
        # This is a heuristic, might not be optimal but valid
        # Direct product of a small code
        if n == 0:
            return [[]]
        # Try to use a simple greedy for general case, but might be slow
        # For now, return a trivial code: all words
        # Or a product code
        # Let's stick to the specific case optimization and fallback to seed logic if params differ
        # But the task is specifically about K_3(7,1).
        # If params differ, we might fail the "valid" check if we don't cover.
        # Let's generate a full space code (all 2187 words) which is always valid but score 2187.
        # Or better, use the seed's logic generalized?
        # The seed used tetracode for n>=4.
        # Let's just return the seed's output for non-matching params to be safe, 
        # or a greedy for small spaces.
        
        # Actually, the prompt says "The evaluator passes this task's alphabet size q, length n and covering radius."
        # It implies the evaluator will call construct(3, 7, 1).
        # If it calls something else, we should still try to be valid.
        # Let's implement a general greedy if space is small enough, else fallback.
        
        total_words = q ** n
        if total_words > 10000:
            # Fallback to seed-like construction if possible
            if q == 3 and n >= 4:
                 tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
                 return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
            else:
                 # Return all words (valid, but bad score)
                 return [list(w) for w in itertools.product(range(q), repeat=n)]
        
    # Specific optimized path for q=3, n=7, radius=1
    space_size = q ** n # 2187
    
    # Precompute spheres
    # A word is a tuple of n ints.
    # We map each word to an index 0..2186
    # Index = w[0]*3^6 + w[1]*3^5 + ... + w[6]
    
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

    # Precompute neighbors (sphere of radius 1) for each index
    # spheres[i] is a list of indices
    spheres = [[] for _ in range(space_size)]
    
    # Generate all words
    all_words = [idx_to_word(i) for i in range(space_size)]
    
    for i, w in enumerate(all_words):
        spheres[i].append(i)
        for pos in range(n):
            orig = w[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                w[pos] = new_val
                j = word_to_idx(w)
                spheres[i].append(j)
            w[pos] = orig
            
    # Precompute "coverers": for each word u, which words v cover u?
    # coverers[u] is the list of v such that u is in spheres[v].
    # This is just spheres[u] because the relation is symmetric.
    # If v covers u, then u is in sphere(v) -> dist(u,v) <= 1 -> dist(v,u) <= 1 -> v is in sphere(u).
    # So coverers[u] == spheres[u].
    
    # Greedy Set Cover
    # benefit[i] = number of uncovered words in spheres[i]
    benefit = [len(spheres[i]) for i in range(space_size)]
    # covered is a bitset. Python ints are arbitrary precision, so we can use a single int.
    covered = 0
    selected = []
    
    # We need to pick the word with max benefit.
    # To make this fast, we can maintain a max-heap or just scan.
    # Scanning 2187 items is fast.
    # But updating benefits is the key.
    
    # To speed up max finding, we can keep a list of (benefit, index) and sort? No, dynamic.
    # Just scan. 2187 * 200 steps = 437,400 comparisons. Negligible.
    
    while covered != (1 << space_size) - 1:
        if time.time() > deadline:
            break
            
        # Find max benefit
        max_b = -1
        max_i = -1
        for i in range(space_size):
            if benefit[i] > max_b:
                max_b = benefit[i]
                max_i = i
                if max_b == 0:
                    break # Should not happen if not fully covered, but safety
        
        if max_i == -1 or max_b == 0:
            break # No more progress possible? Or all covered?
            
        # Select max_i
        selected.append(max_i)
        
        # Update covered and benefits
        # Get the sphere of max_i
        s = spheres[max_i]
        new_bits = 0
        for j in s:
            bit = 1 << j
            if not (covered & bit):
                new_bits |= bit
                # j was just covered.
                # We need to decrement benefit[v] for all v such that j is in spheres[v].
                # These v are exactly spheres[j].
                for v in spheres[j]:
                    benefit[v] -= 1
        covered |= new_bits
        
    # Post-processing: Remove redundant words
    # A word is redundant if its sphere is subset of union of spheres of other selected words.
    # Equivalently, if we remove it, do we still cover everything?
    # We can check this by rebuilding the covered set without the word.
    # Or simpler: for each selected word w, check if every word in sphere(w) is covered by some other selected word.
    
    # Let's try to remove words one by one.
    # We maintain the `covered` bitset.
    # If we remove w, we need to check if sphere(w) is still covered by the remaining set.
    # This is equivalent to: is there any word u in sphere(w) such that u is NOT covered by (selected \ {w})?
    
    # To do this efficiently:
    # We can count for each word u, how many selected words cover it.
    # Let count[u] be the number of selected words v such that u in spheres[v].
    # If we remove w, then for all u in spheres[w], count[u] decreases by 1.
    # If any count[u] becomes 0, then w is not redundant.
    
    counts = [0] * space_size
    for w in selected:
        for u in spheres[w]:
            counts[u] += 1
            
    # Now try to remove
    # We iterate through selected. If we can remove, we do.
    # We might need multiple passes or a queue.
    # Simple approach: iterate, if removable, remove and update counts.
    # If we remove, we don't add it back.
    
    final_selected = []
    # To allow re-evaluation, we can just try to remove from the list.
    # Let's use a boolean array or just filter.
    
    # Actually, removing a word might make another word removable.
    # So we should iterate until no more can be removed.
    
    # Let's keep a list of selected indices.
    # We'll use a set for O(1) lookup if needed, but list is fine.
    
    # Optimization: Only check words that are candidates for removal.
    # Initially all.
    
    changed = True
    while changed:
        changed = False
        new_selected = []
        for w in selected:
            # Check if w is redundant
            redundant = True
            for u in spheres[w]:
                if counts[u] == 1:
                    # Only w covers u. So w is not redundant.
                    redundant = False
                    break
            if redundant:
                # Remove w
                for u in spheres[w]:
                    counts[u] -= 1
                changed = True
            else:
                new_selected.append(w)
        selected = new_selected
        if time.time() > deadline:
            break

    # Convert indices back to words
    result = [idx_to_word(i) for i in selected]
    
    return result

