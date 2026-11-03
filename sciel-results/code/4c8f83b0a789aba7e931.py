import itertools
import random
import time

# Precomputation for specific task parameters if they are fixed, 
# but the interface allows general q, n, radius.
# However, the task specifically mentions K_3(6,1).
# The evaluator passes q, n, radius. 
# We should handle the general case but optimize for the specific known instance if possible, 
# or just ensure the general algorithm is fast enough.
# Given the 120s limit, a general greedy is safe for small n.
# For n=6, q=3, space is 729.

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Generate all words in the space
    # Represent words as tuples of integers for hashing and comparison
    all_words = list(itertools.product(range(q), repeat=n))
    num_words = len(all_words)
    
    # Map word to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors for each word
    # neighbors[i] is a list of indices of words within Hamming distance <= radius
    neighbors = []
    for i, w in enumerate(all_words):
        nbrs = set()
        # Generate all words within distance radius
        # For radius 1, it's just the word itself and words with 1 position changed
        # For general radius, we can use a recursive generation or simply check all words (too slow for large n)
        # Here n is small (6), so we can just check all words if n is small, 
        # but for efficiency, let's generate neighbors properly.
        
        # Efficient neighbor generation for small radius
        def gen_neighbors(word, rad):
            # yields words within distance rad
            # This is a standard problem.
            # For rad=1:
            #   self
            #   for each pos, for each val != word[pos], change
            # For rad>1, it gets more complex.
            # Given the specific task is radius 1, let's optimize for that, 
            # but keep a fallback.
            
            if rad == 1:
                yield word
                for pos in range(len(word)):
                    orig_val = word[pos]
                    for new_val in range(q):
                        if new_val != orig_val:
                            new_word = list(word)
                            new_word[pos] = new_val
                            yield tuple(new_word)
            else:
                # Fallback for general radius: check all words (only feasible for small n)
                # This is O(q^n * n) per word, which is O(q^n * q^n * n) total.
                # For n=6, q=3, 729*729*6 ~ 3M ops. Acceptable.
                for other in all_words:
                    dist = sum(1 for a, b in zip(word, other) if a != b)
                    if dist <= rad:
                        yield other

        for nbr in gen_neighbors(w, radius):
            nbrs.add(word_to_idx[nbr])
        neighbors.append(list(nbrs))
        
    best_code = None
    best_size = float('inf')
    
    # Greedy with randomization
    # We run multiple iterations
    max_iter = 5000 # Just a cap, time is the real constraint
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
            
        random.seed(i)
        uncovered = set(range(num_words))
        selected = []
        
        # To speed up greedy, we can maintain a priority queue or just scan.
        # Scanning is O(N^2) per iteration. N=729. 729^2 ~ 500k.
        # If we select ~70 codewords, 70 * 500k = 35M ops.
        # Python might be slow for 35M ops in a tight loop if not optimized.
        # Let's try to optimize the inner loop.
        
        # Instead of scanning all candidates every time, we can use a "lazy" priority queue approach
        # or just rely on the fact that N is small.
        
        # Let's try a simple greedy first.
        while uncovered:
            best_cand = -1
            max_gain = 0
            # Randomize tie-breaking to explore different solutions
            # We can shuffle the candidate list once per outer loop iteration?
            # No, we need to pick the best at each step.
            # To randomize ties, we can add a small random noise to the gain.
            
            for c in range(num_words):
                if c in [s for s in selected]: # This check is slow. Use a set.
                    continue
                # Calculate gain: number of neighbors of c that are in uncovered
                gain = 0
                for nbr in neighbors[c]:
                    if nbr in uncovered:
                        gain += 1
                # Add small random noise to break ties
                noise = random.random() * 0.01
                if gain + noise > max_gain:
                    max_gain = gain + noise
                    best_cand = c
            
            if best_cand == -1:
                break # Should not happen if space is connected
                
            selected.append(best_cand)
            # Remove covered points
            for nbr in neighbors[best_cand]:
                if nbr in uncovered:
                    uncovered.discard(nbr)
                    
        # Pruning step
        # Check if any selected codeword is redundant
        # A codeword c is redundant if all its neighbors are covered by other selected codewords.
        # i.e., N(c) subset union(N(c') for c' in selected \ {c})
        # We can do this iteratively.
        
        changed = True
        while changed:
            changed = False
            new_selected = []
            for c in selected:
                # Check if c is redundant
                is_redundant = True
                # Get all neighbors of c
                # If any neighbor is NOT covered by others, c is not redundant
                for nbr in neighbors[c]:
                    covered_by_others = False
                    for other in selected:
                        if other == c:
                            continue
                        if nbr in neighbors[other]:
                            covered_by_others = True
                            break
                    if not covered_by_others:
                        is_redundant = False
                        break
                if is_redundant:
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
            
        if len(selected) < best_size:
            best_size = len(selected)
            best_code = [list(all_words[c]) for c in selected]
            
    if best_code is None:
        # Fallback to the seed solution if something went wrong
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return best_code
