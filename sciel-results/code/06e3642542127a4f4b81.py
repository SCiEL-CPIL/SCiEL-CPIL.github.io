"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time


def construct(q, n, radius):
    # Direct construction using a greedy set-cover heuristic on the residual uncovered space.
    # For the specific task (q=3, n=8, radius=1), we aim to beat the 486 direct-sum bound.
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback: simple greedy for general cases, or direct sum if applicable
        # For non-matching parameters, we just return a valid (possibly suboptimal) code.
        # Since the task is specific, we focus optimization on 3,8,1.
        # A safe fallback is the direct sum of tetracode for n>=4.
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Brute force for small n
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Specific optimization for q=3, n=8, radius=1
    # We use a greedy algorithm that picks the codeword covering the most *currently uncovered* words.
    # To make this efficient, we maintain a count of how many codewords cover each word.
    # We pick the word with the highest "coverage potential" among uncovered words.
    # Actually, a standard greedy set cover works on the universe of words.
    # Universe size is 3^8 = 6561.
    # Each codeword covers itself and its 7 neighbors (distance 1). Total 8 words per codeword.
    # We need to cover all 6561 words.
    
    # To speed up, we can use a randomized greedy approach with multiple restarts if time permits.
    # However, 120s is generous. A single good greedy run might suffice.
    # Let's implement a greedy set cover.
    
    # Precompute neighbors for each word index?
    # Indexing: word w (list of 8 ints) -> index = w[0]*3^7 + ... + w[7]
    # Or simpler: use tuples as keys in a set.
    
    # Universe: all tuples of length 8 over {0,1,2}
    # This is 6561 items.
    # For each potential codeword c, its coverage is {c} U {c with one coordinate changed}.
    
    # Greedy Set Cover:
    # 1. Start with empty set of codewords.
    # 2. Maintain a set of uncovered words.
    # 3. While uncovered is not empty:
    #    a. Find the codeword c (from all 6561 possible) that covers the maximum number of words in 'uncovered'.
    #    b. Add c to the code.
    #    c. Remove covered words from 'uncovered'.
    
    # Optimization:
    # Iterating over all 6561 candidates at each step is O(6561 * 8 * |uncovered|) which might be slow if done naively.
    # |uncovered| starts at 6561 and shrinks.
    # Steps ~ 400-500.
    # 500 * 6561 * 8 * (avg uncovered size ~ 3000) -> 500 * 6561 * 24000 ~ 79 billion ops. Too slow in Python.
    
    # We need a smarter greedy or a heuristic.
    # Alternative: Local Search / Hill Climbing.
    # Start with a known good code (e.g., the 486 direct sum code).
    # Try to remove codewords and see if coverage is maintained.
    # Or, try to swap codewords to reduce count? No, count is fixed if we just remove.
    # We want to *minimize* the number of codewords.
    # Start with a larger set, then remove redundant codewords.
    
    # Let's try:
    # 1. Generate the 486 codewords from the direct sum.
    # 2. Compute coverage counts for all 6561 words.
    # 3. While there is a codeword whose removal does not leave any word with count 0:
    #    - Remove it.
    #    - Update counts.
    # This is a "greedy removal" heuristic. It might get stuck, but it's fast.
    
    # Even better: Randomized removal with restarts?
    # Or, start with a denser random code and remove?
    
    # Let's try the removal heuristic on the 486 code.
    
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    base_code = [tuple(h + list(tail)) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Map tuple to index for faster counting?
    # Or just use a dictionary of counts.
    
    def get_neighbors(word):
        # Returns list of neighbors including self
        neigh = [word]
        for i in range(n):
            for d in range(q):
                if d == word[i]:
                    continue
                new_word = list(word)
                new_word[i] = d
                neigh.append(tuple(new_word))
        return neigh
    
    # Precompute neighbors for all 6561 words?
    # 6561 * 8 = 52488 tuples. Memory is fine.
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    neighbors = []
    for w in all_words:
        neigh = [w]
        for i in range(n):
            for d in range(q):
                if d == w[i]:
                    continue
                new_word = list(w)
                new_word[i] = d
                neigh.append(tuple(new_word))
        neighbors.append(neigh)
    
    # Initialize counts
    counts = [0] * len(all_words)
    code_set = set(base_code)
    
    for c in code_set:
        idx = word_to_idx[c]
        for nb in neighbors[idx]:
            counts[word_to_idx[nb]] += 1
    
    # Greedy Removal
    # We want to remove a codeword c if for all nb in neighbors[c], counts[word_to_idx[nb]] > 1.
    # If we remove c, counts decrease by 1 for all its neighbors.
    
    # To be efficient, we can maintain a list of "removable" codewords.
    # A codeword is removable if min(counts[nb] for nb in neighbors[c]) > 1.
    
    # Let's iterate.
    # We'll use a list of current codewords.
    current_code = list(code_set)
    # To allow removal, we need to check removability dynamically.
    
    # Optimization: Instead of checking all, we can randomly pick a codeword and check if it's removable.
    # If yes, remove it. If no, try another.
    # This is a stochastic local search.
    
    # Let's do a deterministic pass first:
    # Repeat until no more removals:
    #   For each codeword in current_code:
    #       Check if removable.
    #       If yes, remove it and update counts.
    
    # To avoid O(N^2) if N is large, we can just do a few passes.
    # 486 is small.
    
    changed = True
    while changed:
        changed = False
        # Shuffle to avoid bias? Or just iterate.
        # Let's iterate in reverse to handle list removal safely.
        for i in range(len(current_code) - 1, -1, -1):
            c = current_code[i]
            idx = word_to_idx[c]
            # Check if all neighbors have count > 1
            removable = True
            for nb in neighbors[idx]:
                if counts[word_to_idx[nb]] <= 1:
                    removable = False
                    break
            if removable:
                # Remove c
                current_code.pop(i)
                # Update counts
                for nb in neighbors[idx]:
                    counts[word_to_idx[nb]] -= 1
                changed = True
    
    # Now, try to improve further with randomized restarts or swaps?
    # The greedy removal might get stuck in a local minimum.
    # We can try random restarts of the removal process from the 486 code.
    # Or, start from a slightly larger random code and remove.
    
    # Given time constraints, let's see if this deterministic removal yields a result < 486.
    # If not, we might need a more aggressive search.
    
    # Let's add a simple randomized component:
    # Try a few times with random order of checking/removal.
    
    best_code = current_code
    best_len = len(best_code)
    
    # Try 10 random restarts of the removal process
    for _ in range(10):
        # Reset counts and code
        counts = [0] * len(all_words)
        code_set = set(base_code)
        for c in code_set:
            idx = word_to_idx[c]
            for nb in neighbors[idx]:
                counts[word_to_idx[nb]] += 1
        
        current_code = list(code_set)
        random.shuffle(current_code)
        
        changed = True
        while changed:
            changed = False
            # Iterate in random order? Or just reverse of shuffled list?
            # We need to be careful with indices if we remove.
            # Let's use a set for current_code and a list for iteration?
            # Simpler: iterate over a copy of the list, and check if it's still in the set.
            
            # Actually, the deterministic loop above is fine. Let's just shuffle the initial list.
            for i in range(len(current_code) - 1, -1, -1):
                c = current_code[i]
                idx = word_to_idx[c]
                removable = True
                for nb in neighbors[idx]:
                    if counts[word_to_idx[nb]] <= 1:
                        removable = False
                        break
                if removable:
                    current_code.pop(i)
                    for nb in neighbors[idx]:
                        counts[word_to_idx[nb]] -= 1
                    changed = True
        
        if len(current_code) < best_len:
            best_len = len(current_code)
            best_code = current_code
    
    return [list(c) for c in best_code]

# EVOLVE-BLOCK-END
