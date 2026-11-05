"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import sys

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet of size q with covering radius r.
    """
    if q == 3 and n == 6 and radius == 1:
        return _cover_ternary_6_1_optimized()
    else:
        return _generic_greedy_cover(q, n, radius)

def _cover_ternary_6_1_optimized():
    """
    Optimized solver for K_3(6,1) using Iterated Local Search (ILS) with Greedy initialization 
    and aggressive deletion-based local search.
    """
    alphabet = range(3)
    all_words = list(itertools.product(alphabet, repeat=6))
    n_words = len(all_words) # 729
    
    # Map word tuple to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute coverage lists (list of indices) for faster iteration than sets if we manage counts carefully
    # But for greedy, we need to count intersections. Sets are easier for intersection size.
    # Let's use sets for coverage, but optimize the greedy loop.
    cover_sets = []
    for w in all_words:
        covered = {word_to_idx[w]}
        for pos in range(6):
            for val in alphabet:
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                covered.add(word_to_idx[tuple(new_w)])
        cover_sets.append(covered)
        
    full_set = set(range(n_words))

    def greedy_cover(uncovered_set, random_seed=None, codewords_init=None):
        """
        Greedy set cover that can start from a set of uncovered words and existing codewords.
        """
        if random_seed is not None:
            random.seed(random_seed)
            
        uncovered = set(uncovered_set)
        codewords = list(codewords_init) if codewords_init else []
        
        while uncovered:
            best_word = -1
            best_count = -1
            candidates = []
            
            # Optimization: Iterate all words, find max coverage of uncovered
            # 729 words * 729 checks per iteration is ~500k ops. 
            # Number of iterations ~70. Total ~35M ops per greedy run.
            # This might be slow in Python if done naively.
            # We can speed up by only checking words that cover *any* uncovered word?
            # Or just accept the cost and rely on fewer restarts?
            
            for i in range(n_words):
                # Quick check: if the set is disjoint from uncovered, count is 0
                # len(cover_sets[i] & uncovered)
                count = len(cover_sets[i] & uncovered)
                if count > best_count:
                    best_count = count
                    best_word = i
                    candidates = [i]
                elif count == best_count and count > 0:
                    candidates.append(i)
            
            if not candidates:
                break # Should not happen
                
            if len(candidates) > 1:
                best_word = random.choice(candidates)
                
            codewords.append(best_word)
            uncovered -= cover_sets[best_word]
            
        return codewords

    def local_search(codewords):
        """
        Aggressive deletion local search.
        """
        # Convert to list of indices
        cw_set = set(codewords)
        
        # Calculate counts
        counts = [0] * n_words
        for cw in cw_set:
            for idx in cover_sets[cw]:
                counts[idx] += 1
                
        improved = True
        while improved:
            improved = False
            # Shuffle order to avoid bias
            cw_list = list(cw_set)
            random.shuffle(cw_list)
            
            for cw in cw_list:
                # Check if removing cw leaves any word with count 0
                can_remove = True
                for idx in cover_sets[cw]:
                    if counts[idx] == 1:
                        can_remove = False
                        break
                
                if can_remove:
                    # Remove it
                    for idx in cover_sets[cw]:
                        counts[idx] -= 1
                    cw_set.remove(cw)
                    improved = True
                    break # Restart the loop as the set changed
        
        return list(cw_set)

    def perturb(codewords, k):
        """
        Randomly remove k codewords to create a perturbed solution.
        """
        cw_list = list(codewords)
        random.shuffle(cw_list)
        # Remove k random ones
        removed = cw_list[:k]
        remaining = cw_list[k:]
        return remaining

    best_codewords = None
    best_size = float('inf')
    
    start_time = time.time()
    timeout = 115 # Leave 5s for safety
    
    # Initial solution
    random.seed(42)
    init_cw = greedy_cover(full_set, random_seed=42)
    init_cw = local_search(init_cw)
    best_size = len(init_cw)
    best_codewords = init_cw
    
    if best_size <= 73:
        return [list(all_words[i]) for i in best_codewords]

    # ILS Loop
    current_cw = best_codewords[:]
    
    while time.time() - start_time < timeout:
        # Perturb
        # Remove 5 to 10 random codewords
        k = random.randint(5, 10)
        perturbed = perturb(current_cw, k)
        
        # Re-greedy to cover the holes
        # We need to know what is uncovered by the perturbed set
        # Calculate uncovered
        covered = set()
        for cw in perturbed:
            covered |= cover_sets[cw]
        uncovered = full_set - covered
        
        # Run greedy to fill uncovered
        new_cw = greedy_cover(uncovered, codewords_init=perturbed)
        
        # Local Search
        new_cw = local_search(new_cw)
        
        if len(new_cw) < best_size:
            best_size = len(new_cw)
            best_codewords = new_cw
            current_cw = new_cw[:]
            
            if best_size <= 73:
                break
        
        # If no improvement, we can either keep current or accept a worse one (simulated annealing style)
        # For simplicity, we stick to "best so far" but perturb the "best so far" or "current"
        # Let's perturb the best so far to escape local minima more aggressively
        # Actually, standard ILS perturbs the current solution. If we only perturb best, we might get stuck.
        # Let's mix: 50% perturb best, 50% perturb current (which is best if improved, else previous)
        # For now, just perturb the best solution found so far to ensure exploration from the best peak.
        # If we don't improve, current_cw remains the previous best.
        # This is a "Hill Climbing with Random Restarts from Best" variant.
        
    # Convert indices back to words
    result = [list(all_words[i]) for i in best_codewords]
    return result

def _generic_greedy_cover(q, n, radius):
    """Generic greedy set cover for arbitrary q, n, radius."""
    alphabet = range(q)
    all_words = list(itertools.product(alphabet, repeat=n))
    n_words = len(all_words)
    
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute coverage sets
    covered_sets = []
    for w in all_words:
        covered = []
        for candidate in all_words:
            dist = sum(1 for a, b in zip(w, candidate) if a != b)
            if dist <= radius:
                covered.append(candidate)
        covered_sets.append(set(word_to_idx[c] for c in covered))
    
    # Greedy set cover
    uncovered = set(range(n_words))
    codewords = []
    
    while uncovered:
        best_word = -1
        best_count = -1
        for i in range(n_words):
            count = len(covered_sets[i] & uncovered)
            if count > best_count:
                best_count = count
                best_word = i
            if best_count == len(uncovered):
                break
        
        codewords.append(best_word)
        uncovered -= covered_sets[best_word]
    
    # Try to remove redundant codewords
    improved = True
    while improved:
        improved = False
        for i in range(len(codewords)):
            temp_codewords = codewords[:i] + codewords[i+1:]
            covered = set()
            for cw in temp_codewords:
                covered |= covered_sets[cw]
            if covered == set(range(n_words)):
                codewords = temp_codewords
                improved = True
                break
    
    result = [list(all_words[i]) for i in codewords]
    return result
# EVOLVE-BLOCK-END
