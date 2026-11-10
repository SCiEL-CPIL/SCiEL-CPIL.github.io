import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    deadline = time.time() + 110.0
    rng = random.Random(42)
    
    total_words = q ** n
    if total_words > 100000:
        # Fallback for larger spaces (not expected for this specific task q=3, n=7)
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        base = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return base

    # Precompute all words
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute cover sets: for each word index, which word indices does it cover?
    # Cover(w) = {w} U {w' : d(w, w') = 1}
    cover_sets = []
    for w in all_words:
        covered = {w}
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                covered.add(tuple(new_w))
        cover_sets.append([word_to_idx[cw] for cw in covered])
        
    # Convert cover_sets to a list of lists of ints for fast access
    # cover_sets[i] is a list of indices
    
    # State variables
    # selected: set of indices of selected codewords
    # coverage: array of length total_words, coverage[i] = number of selected codewords covering i
    
    def init_greedy():
        """Greedy set cover initialization."""
        uncovered = np.ones(total_words, dtype=np.int32)
        selected = set()
        coverage = np.zeros(total_words, dtype=np.int32)
        
        while uncovered.sum() > 0:
            best_idx = -1
            best_gain = 0
            # Find word that covers most uncovered words
            # Optimization: only check words not yet selected
            # For small spaces, scanning all is fine. 2187 words.
            for i in range(total_words):
                if i in selected:
                    continue
                gain = 0
                for w in cover_sets[i]:
                    if uncovered[w]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1 or best_gain == 0:
                break
            selected.add(best_idx)
            for w in cover_sets[best_idx]:
                coverage[w] += 1
                if coverage[w] == 1:
                    uncovered[w] = 0
        return selected, coverage

    def try_remove_redundant(selected, coverage):
        """Remove any redundant codewords. Returns True if any were removed."""
        removed = False
        # Iterate over a copy of the list to allow deletion
        for c in list(selected):
            redundant = True
            for w in cover_sets[c]:
                if coverage[w] <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove c
                selected.remove(c)
                for w in cover_sets[c]:
                    coverage[w] -= 1
                removed = True
        return removed

    def repair(selected, coverage):
        """Greedy repair of uncovered words."""
        uncovered = (coverage == 0)
        while uncovered.any():
            best_idx = -1
            best_gain = 0
            # Find word that covers most uncovered words
            # We can restrict search to words that are NOT in selected to avoid duplicates, 
            # though adding a duplicate doesn't help.
            for i in range(total_words):
                if i in selected:
                    continue
                gain = 0
                for w in cover_sets[i]:
                    if uncovered[w]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1 or best_gain == 0:
                break
            selected.add(best_idx)
            for w in cover_sets[best_idx]:
                coverage[w] += 1
                if coverage[w] == 1:
                    uncovered[w] = 0

    # Main Loop: Local Search with Kicks
    best_selected, best_coverage = init_greedy()
    best_size = len(best_selected)
    
    # Try to clean up the initial greedy solution
    while try_remove_redundant(best_selected, best_coverage):
        pass
    best_size = len(best_selected)
    
    current_selected, current_coverage = set(best_selected), best_coverage.copy()
    current_size = len(current_selected)
    
    # Local Search
    # Strategy: 
    # 1. Try to remove redundant words.
    # 2. If stuck, "kick": remove a random word (creating holes), then repair.
    #    Or: Remove a random word, repair, then try to remove redundancies.
    
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 500 # Reset kick strength or something? Just keep going.
    
    while time.time() < deadline - 5:
        # Phase 1: Remove redundancies
        if try_remove_redundant(current_selected, current_coverage):
            current_size = len(current_selected)
            if current_size < best_size:
                best_size = current_size
                best_selected = set(current_selected)
                best_coverage = current_coverage.copy()
            no_improve_count = 0
            continue
        
        # Phase 2: Kick
        # If no redundant words, we are in a local minimum.
        # Perform a kick: remove a random selected word, then repair.
        if len(current_selected) == 0:
            break
            
        # Choose a word to remove. 
        # Heuristic: Remove a word that is "least useful" or just random.
        # Random is simple and effective with many iterations.
        kick_word = rng.choice(list(current_selected))
        
        # Remove it
        current_selected.remove(kick_word)
        for w in cover_sets[kick_word]:
            current_coverage[w] -= 1
            
        # Repair
        repair(current_selected, current_coverage)
        current_size = len(current_selected)
        
        # Phase 3: Clean up after repair
        # The repair might have added words that make others redundant.
        # Also, the repair might have added the same word back or similar.
        # Run redundancy removal again.
        if try_remove_redundant(current_selected, current_coverage):
             # It's possible we got smaller
             pass
        # Run it in a loop to ensure full cleanup
        while try_remove_redundant(current_selected, current_coverage):
            pass
        current_size = len(current_selected)
        
        if current_size < best_size:
            best_size = current_size
            best_selected = set(current_selected)
            best_coverage = current_coverage.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # Optional: If we get stuck for a long time, restart from best?
        # Or just keep kicking. Kicking from a local minimum usually explores nearby minima.
        # If we restart, we might lose the "memory" of the good structure.
        # However, if we are far from best, restarting might be good.
        # Let's stick to kicking for now as it's fast.
        
    # Final cleanup on best
    while try_remove_redundant(best_selected, best_coverage):
        pass
        
    result = [list(all_words[i]) for i in best_selected]
    return result
