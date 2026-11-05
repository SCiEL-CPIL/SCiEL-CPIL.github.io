import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    # Parameters
    q_val = q
    n_val = n
    r_val = radius
    
    if q_val != 3 or n_val != 7 or r_val != 1:
        # Fallback for other parameters (though task is specific)
        # Just return a trivial cover or raise
        # For this task, we assume 3,7,1
        pass

    # Precomputation for q=3, n=7, r=1
    total_words = q_val ** n_val
    
    # Generate all words
    all_words = [list(w) for w in itertools.product(range(q_val), repeat=n_val)]
    
    # Map word tuple to index
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    # Precompute balls (neighbors within distance 1)
    # ball[i] is a list of indices of words covered by codeword i
    # covered_by[i] is a list of indices of codewords that cover word i
    # Note: ball[i] and covered_by[i] are the same set in an undirected graph, 
    # but conceptually:
    # If we choose codeword c, it covers words in ball[c].
    # If a word w is uncovered, we can choose any codeword c in covered_by[w].
    
    balls = [[] for _ in range(total_words)]
    covered_by = [[] for _ in range(total_words)]
    
    # To speed up, we can generate neighbors on the fly or precompute.
    # Precomputing 2187 * 22 is small.
    
    for i, w in enumerate(all_words):
        # w itself
        balls[i].append(i)
        covered_by[i].append(i)
        
        for pos in range(n_val):
            for delta in [1, -1]:
                new_w = w.copy()
                new_w[pos] = (new_w[pos] + delta) % q_val
                j = word_to_idx[tuple(new_w)]
                balls[i].append(j)
                covered_by[j].append(i) # i covers j? No. 
                # Wait. 
                # If codeword is i, it covers j if d(i,j)<=1.
                # So if i is at pos with val v, and j is at pos with val v+1, 
                # then i covers j. 
                # So j is in ball[i].
                # And i is in covered_by[j].
                
                # Let's re-verify.
                # Loop i (potential codeword).
                # Find neighbors j.
                # j is covered by i.
                # So append j to balls[i].
                # And append i to covered_by[j].
                
                # The code above:
                # i is current word.
                # new_w is neighbor. j is index of new_w.
                # balls[i].append(j) -> Correct. i covers j.
                # covered_by[j].append(i) -> Correct. i covers j.
                
                # However, the loop structure above adds to balls[i] and covered_by[j].
                # It does NOT add to covered_by[i] or balls[j] in this iteration.
                # But since the graph is symmetric, when we reach j in the outer loop, 
                # we will find i as a neighbor and add i to balls[j] and j to covered_by[i].
                # So this is correct and complete.

    # Initial Solution: Direct Sum of Tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n_val - 4):
            cw = h + list(tail)
            idx = word_to_idx[tuple(cw)]
            initial_codewords.append(idx)
            
    # Ensure unique
    initial_codewords = list(set(initial_codewords))
    
    # State
    C = set(initial_codewords)
    cover_count = np.zeros(total_words, dtype=np.int8)
    
    # Initialize cover_count
    for c in C:
        for w in balls[c]:
            cover_count[w] += 1
            
    best_size = len(C)
    best_C = set(C)
    
    deadline = time.time() + 110 # 10s margin
    
    def try_deletion(C, cover_count):
        """
        Greedily remove redundant codewords.
        Returns number of removals.
        """
        removed = 0
        # We need to check if removing c is safe.
        # c is safe if for all w in balls[c], cover_count[w] > 1.
        
        # To speed up, we can just iterate through C.
        # If C is large, this is O(|C| * 22).
        
        # Optimization: Only check candidates that might be redundant?
        # No, just scan.
        
        # We need to be careful with iteration while modifying set.
        # Make a list.
        candidates = list(C)
        
        for c in candidates:
            if time.time() > deadline:
                break
                
            # Check if c is redundant
            redundant = True
            for w in balls[c]:
                if cover_count[w] <= 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove c
                C.remove(c)
                for w in balls[c]:
                    cover_count[w] -= 1
                removed += 1
                # After removal, previous candidates might still be valid or invalid.
                # We continue scanning the original list.
                # If a previous candidate was kept, it might now be removable.
                # But we already passed it. 
                # To be thorough, we could loop until no removals happen in a full pass.
                
        return removed
        
    def repair(C, cover_count):
        """
        Add codewords to cover all holes.
        """
        while True:
            if time.time() > deadline:
                break
                
            # Find a hole (word with cover_count 0)
            # Using np.argmin is fast
            min_val = np.min(cover_count)
            if min_val > 0:
                break
                
            h_idx = np.argmin(cover_count)
            
            # Candidates are covered_by[h_idx]
            # These are codewords that cover h_idx.
            # Since cover_count[h_idx] == 0, none of these candidates are in C.
            
            best_c = -1
            best_score = -1
            
            for c in covered_by[h_idx]:
                if c in C:
                    continue # Should not happen if logic is correct, but safe
                    
                # Score: how many holes does c cover?
                # Or how many words with low count?
                # Let's count words with count 0.
                score = 0
                for w in balls[c]:
                    if cover_count[w] == 0:
                        score += 1
                        
                # Tie breaker: total coverage potential? Or just random.
                # Let's add a small random noise to break ties and encourage diversity
                score += random.random() * 0.01
                
                if score > best_score:
                    best_score = score
                    best_c = c
                    
            if best_c == -1:
                # Should not happen in a covering code problem with radius 1
                # unless C is empty and we picked a bad hole? 
                # No, every word has 22 neighbors.
                break
                
            # Add best_c
            C.add(best_c)
            for w in balls[best_c]:
                cover_count[w] += 1
                
    # Main Loop
    # 1. Initial Deletion
    try_deletion(C, cover_count)
    if len(C) < best_size:
        best_size = len(C)
        best_C = set(C)
        
    # 2. Kick-Repair Loop
    kick_size = 2
    # Maybe increase kick size if stuck?
    # Start with small kicks.
    
    while time.time() < deadline:
        # Save state
        C_copy = set(C)
        cc_copy = cover_count.copy()
        
        # Kick: Remove k random codewords
        if len(C) > kick_size:
            to_remove = random.sample(list(C), kick_size)
            for c in to_remove:
                C.remove(c)
                for w in balls[c]:
                    cover_count[w] -= 1
                    
            # Repair
            repair(C, cover_count)
            
            # Deletion
            try_deletion(C, cover_count)
            
            if len(C) < best_size:
                best_size = len(C)
                best_C = set(C)
            else:
                # Revert if not improved? 
                # Or keep exploring? 
                # Usually, if it doesn't improve immediately, it's stuck in a local min 
                # similar to the previous one. Reverting is safer to maintain a good baseline.
                # But sometimes a "bad" move enables a better one later? 
                # In this deterministic repair, it's likely we just got back to a similar state.
                # Let's revert if not improved to save time exploring equivalent states.
                C = C_copy
                cover_count = cc_copy
                
        # Occasionally try larger kicks?
        if random.random() < 0.1:
            kick_size = 3
        elif random.random() < 0.1:
            kick_size = 1
        else:
            kick_size = 2 # default

    # Convert best_C back to list of lists
    result = [all_words[i] for i in best_C]
    
    return result

