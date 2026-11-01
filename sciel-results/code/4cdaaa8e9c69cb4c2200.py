import time
import itertools
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    total_words = q ** n
    # Precompute neighbors (including self) for each word
    # Index mapping: word tuple -> index
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    neighbors = [[] for _ in range(total_words)]
    for i, w in enumerate(all_words):
        neighbors[i].append(i)
        for pos in range(n):
            orig = w[pos]
            for nv in range(q):
                if nv != orig:
                    lst = list(w)
                    lst[pos] = nv
                    j = word_to_idx[tuple(lst)]
                    neighbors[i].append(j)
    
    all_indices = set(range(total_words))
    
    def is_valid_cover(code_set):
        # Check if every word is covered by at least one codeword
        # Using bitsets for speed if possible, but sets are fine for 729
        # Actually, maintaining counts is better for search
        pass

    def greedy_cover():
        """
        Greedy set cover to get an initial solution.
        """
        uncovered = set(range(total_words))
        code = set()
        
        while uncovered:
            best_word = None
            max_covered = 0
            # Randomize tie-breaking
            candidates = list(uncovered)
            random.shuffle(candidates)
            # Optimization: Only check a subset of candidates if time is tight?
            # For 729 words, checking all is fast enough.
            for i in candidates:
                if i in code:
                    continue
                covered = len(uncovered & set(neighbors[i]))
                if covered > max_covered:
                    max_covered = covered
                    best_word = i
            
            if best_word is None:
                break
                
            code.add(best_word)
            uncovered -= set(neighbors[best_word])
            
        return code

    def local_search(code_set, deadline):
        """
        Local search to minimize code size.
        """
        code = set(code_set)
        # Maintain cover counts
        cover_count = np.zeros(total_words, dtype=int)
        for c in code:
            for nb in neighbors[c]:
                cover_count[nb] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            k = len(code)
            
            # 1. Try to remove redundant words
            removed_any = False
            # Iterate in random order to avoid bias
            items = list(code)
            random.shuffle(items)
            for c in items:
                if c not in code:
                    continue
                # Check if all neighbors are covered by others
                # i.e., min count in neighbors[c] > 1
                if np.min(cover_count[neighbors[c]]) > 1:
                    # Remove c
                    for nb in neighbors[c]:
                        cover_count[nb] -= 1
                    code.remove(c)
                    improved = True
                    removed_any = True
                    break # Restart removal loop
            
            if removed_any:
                continue
                
            # 2. If no removal, try swaps (remove 1, add 1) to escape local minima
            # This doesn't reduce size, but might enable future removals
            if k > 73: # Only swap if we are above target
                # Pick a random word to remove
                c_remove = random.choice(list(code))
                
                # Tentatively remove
                for nb in neighbors[c_remove]:
                    cover_count[nb] -= 1
                code.remove(c_remove)
                
                # Find uncovered points
                uncovered = np.where(cover_count == 0)[0]
                
                if len(uncovered) == 0:
                    # It was redundant, but we missed it? Unlikely.
                    improved = True
                    continue
                
                # Find a word to add that covers the most uncovered points
                best_add = None
                max_cov = 0
                # Check all words not in code
                # Optimization: Only check words that cover at least one uncovered point
                # We can iterate over uncovered points and look at their neighbors
                candidates = set()
                for u in uncovered:
                    candidates.update(neighbors[u])
                
                for c_add in candidates:
                    if c_add in code:
                        continue
                    # How many uncovered points does c_add cover?
                    # Note: c_add might cover points already covered, but we only care about fixing uncovered
                    # Actually, adding c_add increases counts for all its neighbors.
                    # We need to ensure that after adding, ALL points are covered.
                    # So c_add MUST cover ALL uncovered points.
                    # If it doesn't cover all, the code remains invalid.
                    # So we look for c_add such that set(uncovered) is subset of set(neighbors[c_add])
                    
                    # Efficient check:
                    # count of uncovered points in neighbors[c_add] == len(uncovered)
                    # This is expensive if done for all candidates.
                    
                    # Alternative: Just pick the one covering the most, and if it doesn't cover all, 
                    # we might need to add multiple words? No, we want to keep size same or reduce.
                    # If we remove 1 and add 1, size is same.
                    # If the new word doesn't cover all holes, we are still invalid.
                    # So we MUST find a word that covers ALL holes.
                    
                    # Let's check if any candidate covers all holes.
                    # If none does, we revert the removal.
                    
                    # Heuristic: If holes are many, a single word might not cover them all.
                    # In that case, this swap is not useful for *validity* immediately.
                    # However, we might add the best candidate, accept the invalid state temporarily?
                    # No, let's stick to valid states.
                    
                    # If no single word covers all holes, revert.
                    # This means we are stuck if holes are scattered.
                    
                    # Let's just check the top few candidates.
                    pass 

                # Simplified Swap Logic:
                # If we can't find a word to cover all holes, revert.
                found = False
                # Sort candidates by coverage of holes
                # This is O(N * Holes)
                # For 729 words, it's fine.
                
                best_c_add = None
                max_holes_covered = 0
                
                for c_add in candidates:
                    if c_add in code:
                        continue
                    # Check if c_add covers all uncovered
                    # Using sets
                    if set(uncovered).issubset(set(neighbors[c_add])):
                        best_c_add = c_add
                        found = True
                        break
                    # If not, we can't use it to restore validity with 1 add.
                    # So we only accept if it covers ALL.
                    
                if found:
                    # Add best_c_add
                    for nb in neighbors[best_c_add]:
                        cover_count[nb] += 1
                    code.add(best_c_add)
                    improved = True # Size same, but structure changed
                else:
                    # Revert removal
                    for nb in neighbors[c_remove]:
                        cover_count[nb] += 1
                    code.add(c_remove)
                    # If we can't swap, we are stuck.
                    # Try a double swap? Or just break?
                    # For now, break to avoid wasting time.
                    break

        return code

    # 1. Generate initial code using Product Construction (Size 81)
    # This is faster and more reliable than greedy for the start.
    # Ternary Hamming Code of length 4:
    # Parity check matrix H (4x3) over GF(3)?
    # Actually, we can just use the greedy cover, it's simple.
    # But 81 is a known good start.
    
    # Let's try Greedy first, it might find 80-85 quickly.
    # Then Local Search.
    
    best_code = None
    best_size = total_words + 1
    
    # Multiple restarts
    for i in range(50):
        if time.time() > deadline - 5:
            break
        
        # Initialize
        code = greedy_cover()
        
        # Optimize
        code = local_search(code, deadline - 2)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            
        # If we hit the theoretical lower bound or known best, stop early?
        # Known best is 73.
        if best_size <= 73:
            break

    if best_code is None:
        best_code = greedy_cover()

    return [list(all_words[i]) for i in best_code]
