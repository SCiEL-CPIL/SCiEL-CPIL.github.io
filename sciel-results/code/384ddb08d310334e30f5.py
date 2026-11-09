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

    # Build the "conflict" graph: two words conflict if their Hamming distance < MIN_DIST
    # We'll use bit-parallel representation: for each word, a bitmask of conflicting words
    # But 2^18 = 262144 bits per mask is large. Instead, use a smarter approach.
    
    # Strategy: 
    # 1. Use a greedy construction to get a good initial independent set
    # 2. Use local search with bit-parallel conflict checking to improve it
    # 3. The key structural change: use a different ordering and conflict representation
    
    # Precompute for each word, the list of words at distance < 6 (its "forbidden" set)
    # This is expensive to store fully, so we'll compute conflicts on-the-fly using popcount
    
    def greedy_construct(seed=None):
        """Greedy construction with weighted scoring for better initial solution"""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        # Start with 0
        words = [0]
        word_set = {0}
        
        # Candidate pool: all words with weight >= 3
        # Sort by weight (heuristic: lower weight words are "easier" to place)
        candidates_by_weight = {}
        for c in range(1, 1 << N):
            w = popcount[c]
            if w >= 3:
                if w not in candidates_by_weight:
                    candidates_by_weight[w] = []
                candidates_by_weight[w].append(c)
        
        # Process weights in order
        for weight in sorted(candidates_by_weight.keys()):
            if time.time() > deadline - 1.0:
                break
            cand_list = candidates_by_weight[weight]
            rng.shuffle(cand_list)
            
            for c in cand_list:
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
        """Local search: try to add more words by random sampling and swaps"""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        current_words = list(words)
        current_set = set(current_words)
        best_words = list(current_words)
        best_size = len(current_words)
        
        # Try multiple rounds of improvement
        for _ in range(30):
            if time.time() > deadline - 1.0:
                break
            
            improved = False
            
            # Try to add random words
            for _ in range(500):
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
                # Try swap: remove one word, try to add two
                if len(current_words) > 1:
                    idx = rng.randint(0, len(current_words) - 1)
                    removed = current_words[idx]
                    current_words.pop(idx)
                    current_set.discard(removed)
                    
                    # Try to add a new word that doesn't conflict
                    added = False
                    for _ in range(100):
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
                            added = True
                            break
                    if not added:
                        # Put back
                        current_words.insert(idx, removed)
                        current_set.add(removed)
        
        return best_words

    # Get initial solution with multiple seeds
    best_words = greedy_construct(seed=42)
    best_len = len(best_words)
    
    # Run local search
    if time.time() < deadline - 2.0:
        improved = local_search(best_words, seed=123)
        if len(improved) > best_len:
            best_words = improved
            best_len = len(best_words)
    
    # Try more greedy constructions with different seeds
    for seed in range(10, 40):
        if time.time() > deadline - 2.0:
            break
        cand = greedy_construct(seed=seed)
        if len(cand) > best_len:
            best_words = cand
            best_len = len(cand)
            if time.time() < deadline - 2.0:
                improved = local_search(best_words, seed=seed + 100)
                if len(improved) > best_len:
                    best_words = improved
                    best_len = len(best_words)
    
    # Final systematic pass
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
    
    if not best_words:
        return [0]
    
    return best_words
