import itertools
import time

def construct(q, n, radius):
    # Parameters
    Q = q
    N = n
    R = radius
    
    # Total words in space
    total_words = Q ** N
    
    # Precompute neighbors (balls) for all words in the space
    # This allows fast coverage updates
    # ball_indices[i] is a list of indices of words within Hamming distance R of word i
    # Since R=1 and Q=3, each word has 1 + N*(Q-1) = 1 + 8*2 = 17 neighbors? 
    # Wait, Hamming distance 1 means changing exactly 1 position or 0 positions.
    # So 1 (self) + N * (Q-1) = 1 + 8 * 2 = 17 neighbors.
    # The seed description said "differs in at most one position". Yes.
    
    # Let's verify the size.
    # For Q=3, N=8: 1 + 8*2 = 17.
    
    ball_indices = []
    for i in range(total_words):
        # Convert i to base Q string
        digits = [0] * N
        temp = i
        for j in range(N):
            digits[j] = temp % Q
            temp //= Q
        # digits are reversed in standard base conversion if we do this, but for symmetry it doesn't matter as long as consistent
        # Actually, let's just use itertools.product for clarity and correctness, 
        # but precomputing 6561 balls of size 17 is fast enough.
        
        neighbors = [i]
        for pos in range(N):
            orig_val = digits[pos]
            for new_val in range(Q):
                if new_val == orig_val:
                    continue
                # Calculate new index
                # This is slow if done naively. 
                # Better to precompute the "step" values.
                pass
        
    # Optimized neighbor generation
    # Let's generate the list of offsets for distance 1.
    # An offset is a change in one position by delta.
    # We can precompute the list of (position, delta) pairs.
    # But index calculation depends on the position.
    # Let's just compute neighbors for each index on the fly once.
    
    neighbors = []
    for i in range(total_words):
        digits = [0] * N
        temp = i
        for j in range(N):
            digits[j] = temp % Q
            temp //= Q
            
        nbrs = [i]
        for pos in range(N):
            orig = digits[pos]
            # Base value of the index at this position is orig * (Q ** pos)
            # If we change digit at pos to v, new index = i - orig * (Q ** pos) + v * (Q ** pos)
            base_val = Q ** pos
            for v in range(Q):
                if v == orig:
                    continue
                new_idx = i - orig * base_val + v * base_val
                nbrs.append(new_idx)
        neighbors.append(nbrs)

    # Initialize with direct sum code
    # Tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    def list_to_idx(lst):
        idx = 0
        for j, val in enumerate(lst):
            idx += val * (Q ** j)
        return idx

    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(Q), repeat=N - 4):
            word = h + list(tail)
            initial_codewords.append(list_to_idx(word))
            
    codewords = set(initial_codewords)
    
    # Coverage counts
    coverage = [0] * total_words
    for c in codewords:
        for w in neighbors[c]:
            coverage[w] += 1
            
    # Critical words are those with coverage 1
    # We want to minimize len(codewords)
    
    # Heuristic:
    # 1. Try to remove codewords that don't cover any critical words.
    # 2. Try to replace codewords that cover critical words with others that cover them better.
    
    # To speed up, we can iterate.
    # Track time
    start_time = time.time()
    timeout = 140 # Safety margin for 150s limit
    
    # Helper to check if a set of words is covered by a candidate codeword
    # We just need to check if all critical words assigned to c are in neighbors[c']
    
    # We'll maintain a list of critical words for each codeword? 
    # That's expensive to update. 
    # Instead, just scan.
    
    # Let's try a simple loop:
    # While time permits:
    #   improved = False
    #   for c in list(codewords):
    #       # Find critical words covered by c
    #       crits = [w for w in neighbors[c] if coverage[w] == 1]
    #       if not crits:
    #           # Can remove c
    #           for w in neighbors[c]:
    #               coverage[w] -= 1
    #           codewords.remove(c)
    #           improved = True
    #           break # Restart loop to update crits
    #       else:
    #           # Try to find replacement c' such that crits are subset of neighbors[c']
    #           # And c' is not in codewords
    #           # To be efficient, pick the first critical word w0 in crits.
    #           # Any replacement c' must be in neighbors[w0].
    #           candidates = [x for x in neighbors[crits[0]] if x not in codewords]
    #           for c_prime in candidates:
    #               # Check if all crits are in neighbors[c_prime]
    #               if all(w in neighbors[c_prime] for w in crits):
    #                   # Check if c_prime covers at least as many non-critical words? 
    #                   # Or just accept it? 
    #                   # Accepting it might not reduce count immediately, but might enable future removals.
    #                   # To ensure progress, maybe require that c_prime covers some word that c does not?
    #                   # Or simply replace and hope for the best?
    #                   # Let's replace.
    #                   for w in neighbors[c]:
    #                       coverage[w] -= 1
    #                   codewords.remove(c)
    #                   for w in neighbors[c_prime]:
    #                       coverage[w] += 1
    #                   codewords.add(c_prime)
    #                   improved = True
    #                   break
    #           if improved:
    #               break
    #   if not improved:
    #       break
    #   if time.time() - start_time > timeout:
    #       break
        
    # This might be slow if we scan all codewords and all candidates.
    # 729 codewords. 17 neighbors.
    # Candidates for a critical word: up to 17.
    # Checking all crits (up to 17) against 17 neighbors.
    # 17*17 = 289 ops per candidate.
    # 17 candidates.
    # ~5000 ops per codeword check.
    # 729 codewords -> ~3.6M ops per pass.
    # Python can do ~10M-50M simple ops/sec. So one pass is < 1s.
    # We can do many passes.
    
    # Let's implement this loop carefully.
    
    # To make "w in neighbors[c_prime]" fast, we can use a set for neighbors?
    # Or just list membership check is O(N). 17 is small.
    
    # Optimization: 
    # If we replace c with c', we might want to prioritize replacements that cover MORE words 
    # or cover words that are "hard" to cover.
    # For now, simple replacement.
    
    # One issue: if we replace c with c', we might create new critical words.
    # The loop handles this by restarting.
    
    # Let's add a check: only replace if it doesn't increase the number of critical words?
    # Or if it decreases the "total coverage weight"?
    # Simplest: just replace if it covers the criticals.
    
    # Let's run this.
    
    while time.time() - start_time < timeout:
        improved = False
        # Sort codewords to have deterministic behavior? Or randomize?
        # Randomization helps escape local minima.
        import random
        current_list = list(codewords)
        random.shuffle(current_list)
        
        for c in current_list:
            if time.time() - start_time > timeout:
                break
            
            # Find critical words covered by c
            crits = [w for w in neighbors[c] if coverage[w] == 1]
            
            if not crits:
                # Remove c
                for w in neighbors[c]:
                    coverage[w] -= 1
                codewords.remove(c)
                improved = True
                break # Restart to update state
            
            # Try to replace
            # Candidates must cover all crits.
            # Start with first crit
            c0 = crits[0]
            candidates = [x for x in neighbors[c0] if x not in codewords]
            
            # Filter candidates that cover all crits
            valid_candidates = []
            for c_prime in candidates:
                # Check if all crits are in neighbors[c_prime]
                # Using set intersection might be faster if we had sets, but lists are small.
                # Let's just check.
                is_valid = True
                for w in crits:
                    if w not in neighbors[c_prime]:
                        is_valid = False
                        break
                if is_valid:
                    valid_candidates.append(c_prime)
            
            if valid_candidates:
                # Pick one. 
                # Heuristic: pick the one that covers the most words with coverage 2? 
                # Or just random.
                c_prime = random.choice(valid_candidates)
                
                # Apply replacement
                for w in neighbors[c]:
                    coverage[w] -= 1
                codewords.remove(c)
                for w in neighbors[c_prime]:
                    coverage[w] += 1
                codewords.add(c_prime)
                improved = True
                break # Restart
                
        if not improved:
            break
            
    # Convert back to list of lists
    result = []
    for c in codewords:
        digits = [0] * N
        temp = c
        for j in range(N):
            digits[j] = temp % Q
            temp //= Q
        result.append(digits)
        
    return result
