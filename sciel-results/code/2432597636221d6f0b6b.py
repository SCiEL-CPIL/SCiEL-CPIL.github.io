import time
import random
import numpy as np

def construct():
    N = 18
    MIN_DIST = 6
    TOTAL_BUDGET = 80.0  # Safety margin before 90s kill
    t_start = time.time()
    deadline = t_start + TOTAL_BUDGET

    # Precompute popcount for all 18-bit values
    popcount = np.zeros(1 << N, dtype=np.uint8)
    for i in range(1, 1 << N):
        popcount[i] = popcount[i >> 1] + (i & 1)

    # Strategy: Use a structured construction based on quadratic residues or
    # a carefully designed greedy with local search.
    
    # First, let's try to build a good code using a greedy approach with
    # a specific ordering that exploits structure.
    
    # For A_2(18,6), known constructions give codes of size 512 (from QR codes)
    # Let's try to find a good code using randomized greedy + local search.
    
    def greedy_construct(seed=None):
        """Greedy construction with random ordering"""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        # All valid candidates: words with weight >= 3
        # (to be compatible with 0 at distance >= 6)
        candidates = [c for c in range(1, 1 << N) if popcount[c] >= 3]
        rng.shuffle(candidates)
        
        words = [0]
        word_set = {0}
        
        for c in candidates:
            if time.time() > deadline - 1.0:
                break
            # Check compatibility with all existing words
            compatible = True
            for w in words:
                if popcount[c ^ w] < MIN_DIST:
                    compatible = False
                    break
            if compatible:
                words.append(c)
                word_set.add(c)
        
        return words

    def local_search(words, seed=None):
        """Local search: try to add more words by swapping or adding"""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        current_words = list(words)
        current_set = set(current_words)
        best_words = list(current_words)
        best_size = len(current_words)
        
        # Try multiple rounds of improvement
        for _ in range(50):
            if time.time() > deadline - 1.0:
                break
            
            improved = False
            
            # Try to add a random word
            for _ in range(1000):
                if time.time() > deadline - 1.0:
                    break
                c = rng.randint(1, (1 << N) - 1)
                if c in current_set:
                    continue
                if popcount[c] < 3:
                    continue
                
                compatible = True
                for w in current_words:
                    if popcount[c ^ w] < MIN_DIST:
                        compatible = False
                        break
                
                if compatible:
                    current_words.append(c)
                    current_set.add(c)
                    improved = True
            
            if len(current_words) > best_size:
                best_size = len(current_words)
                best_words = list(current_words)
            
            if not improved:
                # Try swap: remove one word, add another
                if len(current_words) > 1:
                    idx = rng.randint(0, len(current_words) - 1)
                    removed = current_words[idx]
                    current_words.pop(idx)
                    current_set.discard(removed)
                    
                    # Try to add a new word
                    for _ in range(200):
                        if time.time() > deadline - 1.0:
                            break
                        c = rng.randint(1, (1 << N) - 1)
                        if c in current_set or c == removed:
                            continue
                        if popcount[c] < 3:
                            continue
                        
                        compatible = True
                        for w in current_words:
                            if popcount[c ^ w] < MIN_DIST:
                                compatible = False
                                break
                        
                        if compatible:
                            current_words.append(c)
                            current_set.add(c)
                            if len(current_words) > best_size:
                                best_size = len(current_words)
                                best_words = list(current_words)
                            break
                    else:
                        # Couldn't add, put back
                        current_words.insert(idx, removed)
                        current_set.add(removed)
        
        return best_words

    # Get initial solution
    best_words = greedy_construct(seed=42)
    best_len = len(best_words)
    
    # Run local search
    if time.time() < deadline - 2.0:
        improved = local_search(best_words, seed=123)
        if len(improved) > best_len:
            best_words = improved
            best_len = len(best_words)
    
    # Try more greedy constructions with different seeds
    for seed in range(10, 50):
        if time.time() > deadline - 2.0:
            break
        cand = greedy_construct(seed=seed)
        if len(cand) > best_len:
            best_words = cand
            best_len = len(cand)
            # Try local search on this
            if time.time() < deadline - 2.0:
                improved = local_search(best_words, seed=seed + 100)
                if len(improved) > best_len:
                    best_words = improved
                    best_len = len(best_words)
    
    # Final systematic pass: try to add any remaining compatible words
    if time.time() < deadline - 1.0:
        current_set = set(best_words)
        for c in range(1, 1 << N):
            if time.time() > deadline - 0.5:
                break
            if c in current_set:
                continue
            if popcount[c] < 3:
                continue
            compatible = True
            for w in best_words:
                if popcount[c ^ w] < MIN_DIST:
                    compatible = False
                    break
            if compatible:
                best_words.append(c)
                current_set.add(c)
    
    # Ensure non-empty
    if not best_words:
        return [0]
    
    return best_words
