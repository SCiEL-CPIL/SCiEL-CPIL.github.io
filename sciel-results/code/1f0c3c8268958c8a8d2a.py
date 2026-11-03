import itertools
import random

def construct(q, n, radius):
    # For the specific case q=3, n=6, radius=1
    # We will use a Greedy Set Cover followed by Local Search (Pruning + Swaps)
    
    if q == 3 and n == 6 and radius == 1:
        total_words = 3 ** n
        all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
        
        # Map word to index for fast lookup
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        # Precompute coverage: for each word c, which words are within distance <= radius
        # Coverage is a list of sets of indices
        coverage = [set() for _ in range(total_words)]
        
        # Efficient Hamming distance for small n
        # Precompute distance between all pairs? 729^2 = 531k. Fast enough.
        # Or just compute on the fly during coverage building.
        
        for i, c in enumerate(all_words):
            cov = set()
            for j, w in enumerate(all_words):
                # Calculate Hamming distance
                dist = 0
                for k in range(n):
                    if c[k] != w[k]:
                        dist += 1
                        if dist > radius:
                            break
                if dist <= radius:
                    cov.add(j)
            coverage[i] = cov
            
        # Greedy Set Cover
        uncovered = set(range(total_words))
        code_indices = []
        
        while uncovered:
            best_idx = -1
            best_cover_count = -1
            for i in range(total_words):
                # Optimization: skip if i is already in code? 
                # Not strictly necessary, but picking a new one is better.
                # However, we can pick any i.
                new_cover = len(coverage[i] & uncovered)
                if new_cover > best_cover_count:
                    best_cover_count = new_cover
                    best_idx = i
                    if best_cover_count == len(uncovered):
                        break
            
            if best_idx == -1:
                break
                
            code_indices.append(best_idx)
            uncovered -= coverage[best_idx]
            
        # Local Search: Pruning + Swaps
        # We represent the code as a set for fast membership check
        code_set = set(code_indices)
        
        # Helper to check if a set of indices covers all words
        def is_valid_code(codes):
            covered = set()
            for c in codes:
                covered.update(coverage[c])
            return len(covered) == total_words
            
        # Helper to calculate cover counts for pruning
        def get_cover_counts(codes):
            counts = [0] * total_words
            for c in codes:
                for w in coverage[c]:
                    counts[w] += 1
            return counts
            
        improved = True
        max_iterations = 500 # Safety
        
        while improved and max_iterations > 0:
            improved = False
            max_iterations -= 1
            
            # 1. Try Pruning: Remove any redundant codeword
            counts = get_cover_counts(code_set)
            for c in list(code_set):
                # Check if removing c leaves any word with count 0
                can_remove = True
                for w in coverage[c]:
                    if counts[w] == 1:
                        can_remove = False
                        break
                if can_remove:
                    code_set.remove(c)
                    improved = True
                    break # Restart loop
            
            if improved:
                continue
                
            # 2. Try Swaps: Replace one codeword with another to enable pruning
            # Randomize order to escape local minima
            current_list = list(code_set)
            random.shuffle(current_list)
            
            swapped = False
            for c in current_list:
                # Try swapping c with some other word w not in code
                # We want to find a w such that (code_set - {c} + {w}) is valid
                # And ideally, it allows further pruning.
                # Heuristic: Pick w that covers many "critically covered" words?
                # Simpler: Random sampling of candidates.
                
                # Identify words that are ONLY covered by c? 
                # If a word is only covered by c, we MUST keep c or replace it with a w that also covers it.
                # Let's find "critical" words for c.
                critical_words = []
                for w in coverage[c]:
                    if counts[w] == 1:
                        critical_words.append(w)
                        
                if not critical_words:
                    # c is redundant? Should have been pruned. 
                    # But maybe counts changed? No, we just checked.
                    # If no critical words, c can be removed.
                    # But we didn't remove it in step 1?
                    # Step 1 checks if ANY word in coverage[c] has count 1.
                    # If none do, c is removable.
                    # So if we are here, c is NOT removable.
                    pass
                
                # Candidate w must cover all critical_words of c.
                # Let's find candidates.
                candidates = []
                # To speed up, we can precompute for each word x, the set of codewords that cover it?
                # Or just iterate through all possible w? 729 is small.
                
                for w_idx in range(total_words):
                    if w_idx in code_set:
                        continue
                    # Check if w_idx covers all critical_words
                    valid_swap = True
                    for cw in critical_words:
                        if cw not in coverage[w_idx]:
                            valid_swap = False
                            break
                    if valid_swap:
                        candidates.append(w_idx)
                        
                if candidates:
                    # Try a random candidate
                    w_new = random.choice(candidates)
                    new_code_set = code_set.copy()
                    new_code_set.remove(c)
                    new_code_set.add(w_new)
                    
                    if is_valid_code(new_code_set):
                        code_set = new_code_set
                        swapped = True
                        improved = True
                        break # Restart loop to allow pruning
                        
        return [list(all_words[i]) for i in code_set]
    
    else:
        # Fallback for other parameters: Simple Greedy
        all_words = list(itertools.product(range(q), repeat=n))
        total_words = len(all_words)
        coverage = [set() for _ in range(total_words)]
        
        def hamming_distance(a, b):
            return sum(1 for x, y in zip(a, b) if x != y)
            
        for i, c in enumerate(all_words):
            for j, w in enumerate(all_words):
                if hamming_distance(c, w) <= radius:
                    coverage[i].add(j)
        
        uncovered = set(range(total_words))
        code_indices = []
        
        while uncovered:
            best_idx = -1
            best_count = -1
            for i in range(total_words):
                count = len(coverage[i] & uncovered)
                if count > best_count:
                    best_count = count
                    best_idx = i
            
            if best_idx == -1: break
            code_indices.append(best_idx)
            uncovered -= coverage[best_idx]
            
        return [list(all_words[i]) for i in code_indices]
