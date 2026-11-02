import time
import itertools
import random

def construct(q, n, radius):
    """
    Constructs a covering code using a Greedy Set Cover approach.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 110s deadline to be safe within 120s
    
    # Handle small cases or fallback to seed if something goes wrong
    if q <= 1 or n <= 0:
        return []
    
    # For very small n, we can just return all words if radius is large enough or just use brute force
    # But for the specific task (q=3, n=7, r=1), we use the greedy algorithm.
    
    # Generate all possible words
    all_words = list(itertools.product(range(q), repeat=n))
    N = len(all_words)
    
    # Map word to index for faster lookup if needed, but sets of tuples are fine for N=2187
    # Precompute coverage sets for each word
    # C(w) = {v : d(w, v) <= radius}
    coverage_sets = []
    for w in all_words:
        cov = set()
        # Add w itself
        cov.add(w)
        # Add neighbors at distance 1 (since radius is usually 1 for this specific problem type, 
        # but let's generalize for radius 1 as per task contract hints, though it says radius is passed)
        # The task says "differs from some chosen word in at most one position", implying radius=1.
        # However, the function takes 'radius'. We assume radius=1 for the optimization strategy,
        # but we should handle general radius if possible or restrict to radius=1.
        # Given the specific problem K_3(7,1), we optimize for radius=1.
        
        if radius == 1:
            for i in range(n):
                for val in range(q):
                    if val != w[i]:
                        new_w = list(w)
                        new_w[i] = val
                        cov.add(tuple(new_w))
        else:
            # General case: check all words
            # This is expensive, but N is small.
            for v in all_words:
                dist = sum(1 for a, b in zip(w, v) if a != b)
                if dist <= radius:
                    cov.add(v)
        
        coverage_sets.append(cov)
    
    # Initialize uncovered set
    uncovered = set(all_words)
    code = []
    
    # Indices of words not yet chosen as codewords? 
    # Actually, a codeword can be any word in the space.
    # We iterate until uncovered is empty.
    
    # Optimization: We can precompute the list of coverage sets.
    # In each step, we find the word w that covers the most uncovered words.
    
    while uncovered:
        if time.time() > deadline:
            # Return best effort. If not complete, it's invalid for the "valid when every word..." check.
            # But for N=2187, it should finish in < 1 sec.
            # If it doesn't, we might return an incomplete set which is invalid.
            # We trust it finishes.
            break
            
        best_w = None
        best_count = -1
        best_cov = None
        
        # Iterate through all possible words to find the best one
        for i, w in enumerate(all_words):
            # Calculate intersection size with uncovered
            # cov = coverage_sets[i]
            # count = len(cov & uncovered)
            
            # Optimization: If best_count is already high, we can skip some?
            # Max possible is len(uncovered).
            
            # Using a loop might be faster than set intersection for large sets?
            # Set intersection is implemented in C, so it's fast.
            count = len(coverage_sets[i] & uncovered)
            
            if count > best_count:
                best_count = count
                best_w = w
                best_cov = coverage_sets[i]
                
                if best_count == len(uncovered):
                    # Perfect cover found in this step
                    break
        
        if best_w is None:
            break # Should not happen if uncovered is not empty
            
        code.append(list(best_w))
        uncovered -= best_cov
        
    return code

# Fallback if the specific parameters don't match the optimized path or if we want to be safe
# The above logic is generic enough for small N.
