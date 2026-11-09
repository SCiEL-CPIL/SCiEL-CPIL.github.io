import time
import itertools
import random

def construct(q, n, radius):
    """
    Construct a covering code for K_q(n, radius).
    For q=3, n=8, radius=1, this uses greedy construction + local removal to approach 486.
    """
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback: direct sum construction (works generally but suboptimal)
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(t) for t in itertools.product(range(q), repeat=n)]
    
    # Special case: q=3, n=8, radius=1
    # Known optimal is 486, best lower bound is 403
    
    # Strategy: 
    # 1. Greedy construction to get an initial covering code
    # 2. Local search removal to minimize size
    
    # For efficiency, work with integer representations
    # Each word is an integer in [0, 3^8) = [0, 6561)
    
    # Precompute neighbors: for each word, list of words within Hamming distance 1
    # This includes the word itself and all words differing in exactly 1 position
    
    total_words = 3 ** n  # 6561
    
    # Build neighbor lists
    neighbors = [[] for _ in range(total_words)]
    for word in range(total_words):
        # Convert to tuple of digits
        digits = [(word // (3 ** i)) % 3 for i in range(n)]
        neighbors[word].append(word)  # self
        for pos in range(n):
            orig = digits[pos]
            for new_val in range(3):
                if new_val != orig:
                    diff = (new_val - orig) * (3 ** pos)
                    neighbors[word].append(word + diff)
    
    # Greedy construction
    # Start with empty set, add codeword that covers most uncovered words
    covered = [False] * total_words
    code = []
    
    # For efficiency, track count of covered neighbors for each potential codeword
    # Actually, let's use a simpler approach: repeatedly pick the codeword that covers
    # the most currently uncovered words
    
    uncovered_count = total_words
    
    # To make this faster, use a heap or just iterate
    # For 6561 words, even O(n^2) is manageable if we're careful
    
    # Better approach: maintain for each word, how many uncovered words it would cover
    # Initially, each word covers 1 + 8*2 = 17 words (itself + 2 neighbors per position)
    
    # Let's use a priority approach
    import heapq
    
    # Max heap (negate for min heap)
    # (negative_cover_count, word)
    
    # Initial cover count for each word
    cover_counts = [17] * total_words  # Each word covers itself + 16 neighbors
    
    # Actually this isn't quite right because neighbors overlap. Let me just do greedy
    # with a simpler approach.
    
    # Greedy: repeatedly select the word that maximizes newly covered words
    # This is O(|code| * total_words * 17) which for ~500 code words is about 500*6561*17 ≈ 56M, might be too slow
    
    # Let me use a smarter greedy with bitsets or numpy
    
    import numpy as np
    
    # Use numpy for speed
    # covered is a boolean array of size total_words
    
    # For each candidate codeword, the number of newly covered words is:
    # sum over neighbors of (1 - covered[neighbor])
    
    # Precompute neighbor indices as numpy arrays
    neighbor_arrays = [np.array(neighbors[i], dtype=np.int32) for i in range(total_words)]
    
    # Greedy construction
    covered_np = np.zeros(total_words, dtype=bool)
    code_set = set()
    
    while uncovered_count > 0 and time.time() < deadline - 20:  # Reserve 20s for removal phase
        # Find the word that covers the most uncovered neighbors
        best_word = -1
        best_gain = -1
        
        # For speed, sample or use a smarter heuristic
        # Let's just iterate through all words - 6561 iterations per step, ~500 steps = 3.3M, fine
        
        for w in range(total_words):
            if w in code_set:
                continue
            nbrs = neighbor_arrays[w]
            gain = np.sum(~covered_np[nbrs])
            if gain > best_gain:
                best_gain = gain
                best_word = w
        
        if best_word < 0:
            break
        
        code.append(best_word)
        code_set.add(best_word)
        # Update covered
        nbrs = neighbor_arrays[best_word]
        new_covered = ~covered_np[nbrs]
        uncovered_count -= np.sum(new_covered)
        covered_np[nbrs] = True
    
    # Convert to list of digit lists for output later
    code_digits = [[(w // (3 ** i)) % 3 for i in range(n)] for w in code]
    
    # Phase 2: Local removal - try to remove codewords that are redundant
    # A codeword is redundant if all its neighbors are covered by other codewords
    
    # Convert back to working with integer representations
    # Rebuild covered from scratch after each removal attempt
    
    def check_redundant(code_list):
        """Check which codewords are redundant."""
        if not code_list:
            return []
        
        # Build covered array
        covered_check = np.zeros(total_words, dtype=bool)
        for w in code_list:
            covered_check[neighbor_arrays[w]] = True
        
        # For each codeword, check if all its neighbors are still covered without it
        redundant = []
        for idx, w in enumerate(code_list):
            # Temporarily remove w
            covered_tmp = covered_check.copy()
            covered_tmp[neighbor_arrays[w]] = False
            
            # Re-add all other codewords
            for other in code_list:
                if other != w:
                    covered_tmp[neighbor_arrays[other]] = True
            
            # Check if all neighbors of w are covered
            if np.all(covered_tmp[neighbor_arrays[w]]):
                redundant.append(idx)
        
        return redundant
    
    # This is O(|code|^2 * 17) per check, too slow for large codes
    # Better approach: for each codeword, check if removing it leaves all its neighbors covered
    # by counting how many codewords cover each neighbor
    
    # Let's use a coverage count approach
    def try_remove(code_list):
        """Try to remove redundant codewords."""
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            
            # Count how many codewords cover each word
            cov_count = np.zeros(total_words, dtype=np.int32)
            for w in code_list:
                cov_count[neighbor_arrays[w]] += 1
            
            # A codeword w is removable if for all v in neighbors(w), cov_count[v] > 1
            # (because removing w reduces cov_count by 1 for each neighbor)
            for idx in range(len(code_list) - 1, -1, -1):
                w = code_list[idx]
                nbrs = neighbor_arrays[w]
                if np.all(cov_count[nbrs] > 1):
                    # Safe to remove
                    code_list.pop(idx)
                    changed = True
                    # Update cov_count
                    cov_count[nbrs] -= 1
                    break  # Restart the loop to recompute cov_count properly
        
        return code_list
    
    code = try_remove(code)
    code_digits = [[(w // (3 ** i)) % 3 for i in range(n)] for w in code]
    
    # Phase 3: Try some swaps if time permits
    # Try replacing pairs of codewords with single codewords or different combinations
    # This is complex; let's just do a few random restarts of greedy if time permits
    
    best_code = code_digits
    best_size = len(best_code)
    
    if time.time() < deadline - 10:
        # Try a few random perturbations
        random.seed(42)
        for restart in range(3):
            if time.time() >= deadline - 5:
                break
            
            # Random initial code
            initial_size = len(code) + random.randint(-50, 50)
            initial_code = random.sample(range(total_words), min(initial_size, total_words))
            
            # Greedy add
            covered_np2 = np.zeros(total_words, dtype=bool)
            for w in initial_code:
                covered_np2[neighbor_arrays[w]] = True
            
            current_code = list(initial_code)
            current_set = set(initial_code)
            
            # Add until covered
            while True:
                if time.time() >= deadline - 5:
                    break
                remaining = np.sum(~covered_np2)
                if remaining == 0:
                    break
                
                best_word = -1
                best_gain = -1
                # Sample some candidates for speed
                candidates = random.sample([w for w in range(total_words) if w not in current_set], min(100, len(current_set) + 100))
                for w in candidates:
                    gain = np.sum(~covered_np2[neighbor_arrays[w]])
                    if gain > best_gain:
                        best_gain = gain
                        best_word = w
                
                if best_word < 0:
                    # Fall back to full search
                    for w in range(total_words):
                        if w in current_set:
                            continue
                        gain = np.sum(~covered_np2[neighbor_arrays[w]])
                        if gain > best_gain:
                            best_gain = gain
                            best_word = w
                    if best_word < 0:
                        break
                
                current_code.append(best_word)
                current_set.add(best_word)
                covered_np2[neighbor_arrays[best_word]] = True
            
            # Removal phase
            current_code = try_remove(current_code)
            
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = [[(w // (3 ** i)) % 3 for i in range(n)] for w in current_code]
    
    return best_code
