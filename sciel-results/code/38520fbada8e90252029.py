import time
import random
import itertools
import sys
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code for q=3, n=8, radius=1.
    Returns a list of codewords (each a list of n integers).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 115s deadline to be safe
    
    # For this specific problem: q=3, n=8, radius=1
    # Space size: 3^8 = 6561
    # We want to find a minimal set of codewords such that every word is within
    # Hamming distance 1 of some codeword.
    
    # Strategy: 
    # 1. Use a greedy set-cover approach to build an initial code
    # 2. Then use local search (swap, remove) to reduce the size
    # 3. Use randomized restarts to escape local optima
    
    # Precompute all words and their neighbors (words within distance 1)
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # For each word, precompute the set of words it covers (itself + neighbors)
    # A word w covers: w itself, and all words differing in exactly 1 position
    covers = [set() for _ in range(len(all_words))]
    for idx, w in enumerate(all_words):
        covers[idx].add(idx)
        w_list = list(w)
        for i in range(n):
            orig = w_list[i]
            for new_val in range(q):
                if new_val != orig:
                    w_list[i] = new_val
                    covers[idx].add(word_to_idx[tuple(w_list)])
            w_list[i] = orig
    
    # Convert covers to frozensets for hashing
    covers_frozen = [frozenset(c) for c in covers]
    
    # Map from cover-set to list of words that have that cover
    # This helps in greedy: when we pick a word, we know which uncovered points it covers
    cover_to_words = {}
    for idx, c in enumerate(covers_frozen):
        if c not in cover_to_words:
            cover_to_words[c] = []
        cover_to_words[c].append(idx)
    
    # Greedy set cover: repeatedly pick the word that covers the most uncovered points
    def greedy_set_cover():
        uncovered = set(range(len(all_words)))
        chosen = []
        
        # For efficiency, maintain for each word how many of its covered points are uncovered
        # But that's expensive. Instead, just compute on the fly.
        
        while uncovered:
            if time.time() > deadline:
                break
            best_word = None
            best_count = -1
            for w_idx in range(len(all_words)):
                if w_idx in chosen:
                    continue
                count = len(covers_frozen[w_idx] & uncovered)
                if count > best_count:
                    best_count = count
                    best_word = w_idx
                    if best_count == len(uncovered):
                        break
            if best_word is None:
                break
            chosen.append(best_word)
            uncovered -= covers_frozen[best_word]
        
        return chosen
    
    # Better greedy with randomization
    def randomized_greedy():
        uncovered = set(range(len(all_words)))
        chosen = []
        
        while uncovered:
            if time.time() > deadline:
                break
            # Pick from top-k candidates randomly
            counts = []
            for w_idx in range(len(all_words)):
                if w_idx in chosen:
                    continue
                count = len(covers_frozen[w_idx] & uncovered)
                if count > 0:
                    counts.append((count, w_idx))
            
            if not counts:
                break
            
            counts.sort(key=lambda x: -x[0])
            # Pick from top 10 or so
            k = min(10, len(counts))
            best_count, best_word = counts[random.randint(0, k - 1)]
            chosen.append(best_word)
            uncovered -= covers_frozen[best_word]
        
        return chosen
    
    # Local search: try to remove words, or swap words
    def local_search(chosen):
        if time.time() > deadline:
            return chosen
        
        chosen_set = set(chosen)
        # Compute coverage counts
        coverage = [0] * len(all_words)
        for w_idx in chosen:
            for c in covers_frozen[w_idx]:
                coverage[c] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try to remove words
            for i in range(len(chosen)):
                if time.time() > deadline:
                    break
                w_idx = chosen[i]
                # Check if removing w_idx keeps all covered
                can_remove = True
                for c in covers_frozen[w_idx]:
                    if coverage[c] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    for c in covers_frozen[w_idx]:
                        coverage[c] -= 1
                    chosen.remove(w_idx)
                    chosen_set.discard(w_idx)
                    improved = True
                    break
            
            # Try to swap: remove one, add another
            if not improved and len(chosen) > 0:
                # Try random swaps
                for _ in range(100):
                    if time.time() > deadline:
                        break
                    if len(chosen) == 0:
                        break
                    i = random.randint(0, len(chosen) - 1)
                    w_remove = chosen[i]
                    # Find a word not in chosen that could replace it
                    candidates = []
                    for w_idx in range(len(all_words)):
                        if w_idx in chosen_set:
                            continue
                        # Check if adding w_idx and removing w_remove still covers everything
                        # This is expensive. Just try it.
                        new_coverage = coverage[:]
                        for c in covers_frozen[w_remove]:
                            new_coverage[c] -= 1
                        for c in covers_frozen[w_idx]:
                            new_coverage[c] += 1
                        if all(new_coverage[c] >= 1 for c in range(len(all_words))):
                            candidates.append(w_idx)
                    
                    if candidates:
                        w_add = random.choice(candidates)
                        # Perform swap
                        for c in covers_frozen[w_remove]:
                            coverage[c] -= 1
                        for c in covers_frozen[w_add]:
                            coverage[c] += 1
                        chosen[i] = w_add
                        chosen_set.discard(w_remove)
                        chosen_set.add(w_add)
                        improved = True
                        break
        
        return chosen
    
    # Better approach: use the known structure. The best known code has 486 words.
    # Let's try to get close to that.
    
    # Start with a good initial solution using greedy
    best_solution = None
    best_size = float('inf')
    
    # Multiple restarts with different random seeds
    for restart in range(50):
        if time.time() > deadline:
            break
        
        random.seed(restart)
        
        # Greedy with randomization
        solution = randomized_greedy()
        
        if not solution:
            continue
        
        # Local search
        solution = local_search(solution)
        
        if len(solution) < best_size:
            best_size = len(solution)
            best_solution = solution[:]
        
        # If we're getting close to the known best, do more intensive local search
        if best_size <= 500 and time.time() < deadline:
            # Try more aggressive local search
            for _ in range(10):
                if time.time() > deadline:
                    break
                solution = local_search(solution)
                if len(solution) < best_size:
                    best_size = len(solution)
                    best_solution = solution[:]
    
    if best_solution is None:
        # Fallback: return all words (trivially valid but large)
        best_solution = list(range(len(all_words)))
    
    # Convert indices back to word lists
    return [list(all_words[i]) for i in best_solution]


if __name__ == "__main__":
    # Test
    code = construct(3, 8, 1)
    print(f"Code size: {len(code)}")
    # Verify
    all_words = list(itertools.product(range(3), repeat=8))
    word_set = set(tuple(w) for w in code)
    
    def hamming_dist(a, b):
        return sum(1 for x, y in zip(a, b) if x != y)
    
    covered = 0
    for w in all_words:
        for c in word_set:
            if hamming_dist(w, c) <= 1:
                covered += 1
                break
    print(f"Covered: {covered}/{len(all_words)}")
