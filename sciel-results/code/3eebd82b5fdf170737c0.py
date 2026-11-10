"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random

def construct(q, n, radius):
    """
    Find a covering code of size as small as possible for q-ary length n radius r.
    For the specific case q=3, n=6, r=1, we aim for 73 codewords.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin

    # --- Helper functions ---

    def word_to_idx(word):
        """Convert a word (list of ints) to an integer index."""
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    def idx_to_word(idx):
        """Convert an integer index to a word (list of ints)."""
        word = [0] * n
        for i in range(n):
            word[n - 1 - i] = idx % q
            idx //= q
        return word

    def get_neighbors_idx(idx, radius=1):
        """
        Get the set of indices within Hamming distance `radius` of the given index.
        Optimized for small radius (1 or 2).
        """
        neighbors = {idx}
        word = idx_to_word(idx)
        
        if radius >= 1:
            # Distance 1
            for pos in range(n):
                orig_val = word[pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    word[pos] = new_val
                    new_idx = word_to_idx(word)
                    neighbors.add(new_idx)
                    if radius >= 2:
                        # Distance 2: change another position
                        for pos2 in range(n):
                            if pos2 == pos:
                                continue
                            orig_val2 = word[pos2]
                            for new_val2 in range(q):
                                if new_val2 == orig_val2:
                                    continue
                                word[pos2] = new_val2
                                new_idx2 = word_to_idx(word)
                                neighbors.add(new_idx2)
                            word[pos2] = orig_val2
                    word[pos] = orig_val
        
        return neighbors

    def check_coverage(codewords, all_indices, balls):
        """
        Check if the given codewords cover all indices.
        Returns True if covered, False otherwise.
        """
        covered = [0] * len(all_indices)
        for cw in codewords:
            for idx in balls[cw]:
                covered[idx] += 1
        for i in range(len(all_indices)):
            if covered[i] == 0:
                return False
        return True

    # --- Precomputation ---
    
    # Total number of words in the space
    total_words = q ** n
    
    # Generate all indices
    all_indices = list(range(total_words))
    
    # Precompute balls (coverage sets) for every possible word
    # balls[i] is a set of indices covered by word i
    balls = [None] * total_words
    for i in range(total_words):
        balls[i] = get_neighbors_idx(i, radius)
        
    # --- Construction Strategies ---

    def greedy_cover(random_seed=None):
        """
        Greedy set cover: start with empty set, add codeword that covers most uncovered words.
        Uses random tie-breaking if random_seed is provided.
        """
        if random_seed is not None:
            random.seed(random_seed)
            
        uncovered = set(all_indices)
        codewords = []
        
        while uncovered:
            if time.time() > deadline:
                break
            
            best_idx = -1
            best_count = 0
            candidates = []
            
            # Iterate through all possible codewords to find the one that covers the most uncovered words
            # Optimization: we can stop early if we find a codeword that covers all uncovered? Unlikely.
            # We can also skip codewords that are already in the set? 
            # Actually, if a codeword is in the set, its ball is already subtracted, so overlap is 0.
            # So we don't need to explicitly skip, but checking `if idx in chosen_set` might be faster 
            # than computing intersection if the set is small? 
            # Intersection of sets is O(min(len(a), len(b))). 
            # Let's just compute intersection.
            
            for idx in all_indices:
                # Skip if we've already chosen this codeword? 
                # It won't add new coverage, so overlap will be 0. 
                # But computing intersection takes time. 
                # Let's maintain a set of chosen codewords to skip them quickly.
                # However, maintaining a set of chosen codewords and checking membership is O(1).
                # Intersection is O(size of ball).
                # So checking membership first is a good optimization.
                
                # We don't have a chosen_set here in the loop scope easily without passing it.
                # Let's just compute overlap. It's fast enough for 729 words.
                overlap = len(balls[idx] & uncovered)
                
                if overlap > best_count:
                    best_count = overlap
                    best_idx = idx
                    candidates = [idx]
                elif overlap == best_count and overlap > 0:
                    candidates.append(idx)
            
            if not candidates:
                break # Should not happen if uncovered is not empty
                
            # Pick randomly from candidates to break ties
            chosen_idx = random.choice(candidates)
            codewords.append(chosen_idx)
            uncovered -= balls[chosen_idx]
            
        return codewords

    def remove_redundant(codewords):
        """
        Remove redundant codewords from the list.
        A codeword is redundant if its removal does not uncover any words.
        """
        if not codewords:
            return codewords
            
        # Convert to list of indices
        # codewords is a list of indices
        # We need to check if removing a codeword leaves any word uncovered.
        
        # Precompute coverage counts for the current set
        coverage_counts = [0] * total_words
        for cw in codewords:
            for idx in balls[cw]:
                coverage_counts[idx] += 1
                
        removed = [False] * len(codewords)
        improved = True
        
        while improved:
            improved = False
            for i in range(len(codewords)):
                if removed[i]:
                    continue
                if time.time() > deadline:
                    break
                
                cw = codewords[i]
                can_remove = True
                for idx in balls[cw]:
                    if coverage_counts[idx] <= 1:
                        can_remove = False
                        break
                
                if can_remove:
                    removed[i] = True
                    improved = True
                    for idx in balls[cw]:
                        coverage_counts[idx] -= 1
        
        return [codewords[i] for i in range(len(codewords)) if not removed[i]]

    def local_search_swap(codewords):
        """
        Try to improve the solution by swapping one codeword for another.
        This is a heuristic local search.
        """
        if not codewords:
            return codewords
            
        # Create a set of current codewords for quick lookup
        current_set = set(codewords)
        best_solution = codewords[:]
        
        # We can try to remove one and add another that covers more?
        # Or just try removing one if it's redundant (already handled by remove_redundant).
        # A better local search: try to replace a codeword with another that allows removal of another?
        # This is complex. Let's stick to simple removal for now, as greedy + removal is strong.
        
        # Actually, let's try a "move" heuristic:
        # Pick a codeword c1. Pick another codeword c2 not in the set.
        # Replace c1 with c2. Check if still covered.
        # This is expensive. 
        # Given the time limit, the greedy + removal approach with multiple restarts is likely sufficient.
        
        return best_solution

    # --- Main Execution ---

    # 1. Start with a known good construction: direct sum of tetracode
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        # Convert tetracode words to indices for the first 4 coordinates
        # We need to map the full 6-length word to an index.
        # The direct sum construction creates words: [tetracode_word] + [tail]
        # We can generate these indices directly.
        
        initial_codewords = []
        for t in tetracode:
            for tail in itertools.product(range(q), repeat=n - 4):
                full_word = t + list(tail)
                idx = word_to_idx(full_word)
                initial_codewords.append(idx)
    else:
        # Fallback for other q, n: start with empty and greedy
        initial_codewords = []

    # 2. Remove redundant codewords from the initial construction
    best_solution = remove_redundant(initial_codewords)
    best_size = len(best_solution)
    
    # 3. Try greedy cover with different random seeds
    # We run multiple iterations of greedy cover to find a better solution
    # Greedy cover is deterministic if no ties, but we use random tie-breaking.
    
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        random.seed(i)
        greedy_sol = greedy_cover(random_seed=i)
        
        # Remove redundant from the greedy solution
        greedy_sol_clean = remove_redundant(greedy_sol)
        
        if len(greedy_sol_clean) < best_size:
            best_size = len(greedy_sol_clean)
            best_solution = greedy_sol_clean
            
        # If we found the known optimal, we can stop early
        if q == 3 and n == 6 and best_size == 73:
            break

    # 4. Final polish: try to remove more redundant codewords from the best solution
    # This might find additional redundancies missed in the previous step
    best_solution = remove_redundant(best_solution)
    
    # 5. Convert indices back to words (lists of integers)
    result_words = [idx_to_word(idx) for idx in best_solution]
    
    # Ensure distinctness (should be guaranteed by set logic, but just in case)
    # The problem states codewords must be distinct.
    # Our indices are distinct, so words are distinct.
    
    return result_words
# EVOLVE-BLOCK-END
