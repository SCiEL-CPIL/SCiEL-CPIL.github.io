import time
import random
import itertools
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Total words
    total_words = q ** n
    
    # Helper to generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Helper to check distance
    def dist(w1, w2):
        return sum(1 for a, b in zip(w1, w2) if a != b)
    
    # Precompute neighbors for each word? 
    # Number of words within distance 1: 1 + n*(q-1)
    # For q=3, n=8: 1 + 8*2 = 17 neighbors.
    # Total 6561 words. 6561 * 17 ~ 111k pairs. Manageable.
    
    neighbors = []
    for w in all_words:
        nbrs = [w]
        for i in range(n):
            orig = w[i]
            for val in range(q):
                if val != orig:
                    new_w = list(w)
                    new_w[i] = val
                    nbrs.append(tuple(new_w))
        neighbors.append([word_to_idx[t] for t in nbrs])
        
    # Greedy Construction
    covered = [False] * total_words
    code = []
    uncovered_count = total_words
    
    # Precompute coverage count for each candidate?
    # Initially, every word covers itself and its neighbors.
    # We want to pick the word that covers the most *uncovered* words.
    
    # To speed up, we can maintain a list of uncovered words.
    # But checking coverage of a candidate requires iterating its neighbors.
    
    # Simple Greedy:
    # 1. List of all words.
    # 2. While uncovered > 0:
    #    a. Find word w that maximizes count of uncovered neighbors.
    #    b. Add w to code.
    #    c. Mark all neighbors of w as covered.
    #    d. Update uncovered count.
    
    # This is O(N^2 * K) where K is neighbors. 
    # N=6561. N^2 is 43M. K=17. 700M ops. Might be too slow in Python.
    
    # Optimized Greedy:
    # Use a heap or just scan. Scanning 6561 words, each checking 17 neighbors.
    # 6561 * 17 ~ 111k checks per iteration.
    # How many iterations? Code size ~ 486.
    # 486 * 111k ~ 54M ops. Should be fine in < 5s.
    
    uncovered_indices = set(range(total_words))
    
    while uncovered_indices and time.time() < deadline:
        best_word = -1
        best_count = -1
        # Randomize start to avoid symmetry? Or just scan all.
        for i in range(total_words):
            if i not in uncovered_indices:
                # If the center is covered, it might still cover other uncovered neighbors?
                # No, if i is covered, it doesn't mean its neighbors are covered.
                # We want to cover *uncovered* words.
                # A word w covers w and its neighbors.
                # We should evaluate all words, not just uncovered ones.
                # But if w is already in code, skip? No, we haven't added it yet.
                pass
        
        # Actually, we should evaluate all words as candidates.
        # But we can skip words that are already in the code? No, we are building the code.
        
        # Optimization: Only consider words that are currently uncovered?
        # If a word is covered, adding it might cover its neighbors which are uncovered.
        # So we must consider all words.
        
        for i in range(total_words):
            count = 0
            for nbr_idx in neighbors[i]:
                if nbr_idx in uncovered_indices:
                    count += 1
            if count > best_count:
                best_count = count
                best_word = i
                
        if best_word == -1 or best_count == 0:
            break
            
        code.append(all_words[best_word])
        for nbr_idx in neighbors[best_word]:
            if nbr_idx in uncovered_indices:
                uncovered_indices.remove(nbr_idx)
                
    # Local Search
    # Try to remove redundant codewords
    # Try to swap codewords to allow removal of others
    
    # Convert code to indices for faster lookup
    code_indices = [word_to_idx[w] for w in code]
    
    # Function to check if a code index is redundant given the set of code indices
    # A codeword c is redundant if all words it covers are covered by other codewords in the set.
    
    def is_redundant(c_idx, code_set):
        # Check all neighbors of c_idx
        # If any neighbor is NOT covered by any other codeword in code_set, then c_idx is NOT redundant.
        # If all neighbors are covered by others, it is redundant.
        
        # Optimization: Count how many codewords cover each neighbor?
        # Too slow to recompute for every check.
        
        # Simple check:
        for nbr_idx in neighbors[c_idx]:
            # Check if nbr_idx is covered by any OTHER codeword
            covered_by_other = False
            for other_idx in code_set:
                if other_idx == c_idx:
                    continue
                # Check if nbr_idx is in neighbors[other_idx]
                # We can precompute a set of neighbors for each index?
                # neighbors is a list of lists.
                # Checking membership in a list is O(K).
                # So this is O(K * |code| * K) = O(K^2 * |code|).
                # K=17, |code|~500. 17*17*500 ~ 144k ops per redundancy check.
                # If we do this for 500 codewords, 72M ops.
                # Might be slow in Python if done naively.
                
                # Better: Precompute a boolean matrix or set of covered neighbors for each code word?
                # Or maintain a count array: cover_count[word_idx] = number of codewords covering it.
                
            if not covered_by_other:
                return False
        return True

    # Let's use the cover_count approach.
    # cover_count[i] = number of codewords in current code that cover word i.
    
    cover_count = [0] * total_words
    
    def update_cover_count(idx, delta):
        for nbr_idx in neighbors[idx]:
            cover_count[nbr_idx] += delta
            
    # Initialize cover_count for the greedy code
    for c_idx in code_indices:
        update_cover_count(c_idx, 1)
        
    # Local Search Loop
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try to remove redundant codewords
        # Iterate through code_indices
        # If cover_count[nbr] > 1 for all nbr in neighbors[c_idx], then c_idx is redundant?
        # No, it means every word covered by c_idx is covered by at least one OTHER codeword.
        # So we can remove c_idx.
        
        new_code_indices = []
        for c_idx in code_indices:
            # Check if redundant
            redundant = True
            for nbr_idx in neighbors[c_idx]:
                if cover_count[nbr_idx] <= 1:
                    # This neighbor is only covered by c_idx (or no one, but it's covered)
                    # So c_idx is essential for this neighbor.
                    redundant = False
                    break
                    
            if redundant:
                # Remove it
                update_cover_count(c_idx, -1)
                improved = True
            else:
                new_code_indices.append(c_idx)
                
        code_indices = new_code_indices
        
        # 2. Try swaps?
        # If we can't remove, try to swap a codeword to a better position.
        # A swap is useful if it allows a removal later.
        # Heuristic: Pick a codeword that is "weak" (covers few unique words) and move it to a position that covers many unique words?
        # Or just random restarts?
        
        # Given time, maybe just stick to removal for now.
        # If we hit a local minimum with removal, we might need swaps.
        
        # Let's add a simple swap heuristic:
        # Pick a random codeword c.
        # Pick a random word w not in code.
        # If replacing c with w allows the removal of another codeword, do it.
        # This is expensive.
        
        # Alternative: If no removals, break?
        # The knowledge says "typically achieving the known optimal 486".
        # Greedy often gets ~500-540. Removal gets it down to 486.
        
    # Convert back to list of lists
    return [list(all_words[i]) for i in code_indices]

# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    total_words = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors
    neighbors = []
    for w in all_words:
        nbrs = [w]
        for i in range(n):
            orig = w[i]
            for val in range(q):
                if val != orig:
                    new_w = list(w)
                    new_w[i] = val
                    nbrs.append(tuple(new_w))
        neighbors.append([word_to_idx[t] for t in nbrs])
        
    # Greedy Construction
    covered = [False] * total_words
    code_indices = []
    uncovered_indices = set(range(total_words))
    
    # To speed up greedy, we can use a heuristic.
    # Simple scan is okay for 6561 words.
    
    while uncovered_indices and time.time() < deadline:
        best_word = -1
        best_count = -1
        for i in range(total_words):
            count = 0
            for nbr_idx in neighbors[i]:
                if nbr_idx in uncovered_indices:
                    count += 1
            if count > best_count:
                best_count = count
                best_word = i
                
        if best_word == -1 or best_count == 0:
            break
            
        code_indices.append(best_word)
        for nbr_idx in neighbors[best_word]:
            if nbr_idx in uncovered_indices:
                uncovered_indices.remove(nbr_idx)
                
    # Local Search: Removal
    cover_count = [0] * total_words
    
    def update_cover_count(idx, delta):
        for nbr_idx in neighbors[idx]:
            cover_count[nbr_idx] += delta
            
    for c_idx in code_indices:
        update_cover_count(c_idx, 1)
        
    improved = True
    while improved and time.time() < deadline:
        improved = False
        new_code_indices = []
        for c_idx in code_indices:
            redundant = True
            for nbr_idx in neighbors[c_idx]:
                if cover_count[nbr_idx] <= 1:
                    redundant = False
                    break
                    
            if redundant:
                update_cover_count(c_idx, -1)
                improved = True
            else:
                new_code_indices.append(c_idx)
                
        code_indices = new_code_indices
        
    return [list(all_words[i]) for i in code_indices]
# EVOLVE-BLOCK-END
