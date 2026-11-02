import itertools
import time

def construct(q, n, radius):
    start_time = time.time()
    limit = 140.0 # Safety margin for 150s limit
    
    total_words = q ** n
    
    # Precompute powers of q for index conversion
    powers = [q**i for i in range(n)]
    
    def get_digits(idx):
        digits = [0] * n
        for i in range(n):
            digits[i] = idx % q
            idx //= q
        return digits

    # Precompute neighbor masks
    # mask[idx] is an integer with bits set for all words within radius of idx
    neighbor_masks = [0] * total_words
    
    for idx in range(total_words):
        mask = 1 << idx
        # Radius 1 neighbors
        for i in range(n):
            original = (idx // powers[i]) % q
            base = idx - original * powers[i]
            for new_val in range(q):
                if new_val != original:
                    neighbor_idx = base + new_val * powers[i]
                    mask |= (1 << neighbor_idx)
        neighbor_masks[idx] = mask
        
    # Greedy Set Cover using bitmasks
    uncovered = (1 << total_words) - 1
    solution_indices = []
    
    while uncovered:
        if time.time() - start_time > limit:
            break
            
        best_word = -1
        max_covered = 0
        
        # Find the word that covers the most uncovered words
        # We iterate through all words. 
        # Optimization: We can skip words that are already in solution? 
        # No, a word in solution might still cover uncovered words if we didn't pick it optimally, 
        # but actually, if it's in solution, its neighbors were removed from 'uncovered' when it was added.
        # Wait, 'uncovered' is updated by removing the neighbors of the *chosen* word.
        # So if a word is in solution, its neighbors are no longer in 'uncovered'.
        # However, a word *not* in solution might cover many uncovered words.
        # A word *in* solution covers 0 uncovered words (because they were removed).
        # So we can skip words in solution? 
        # Actually, `neighbor_masks[w] & uncovered` will be 0 for w in solution.
        # So the count will be 0. We can just iterate all and it will naturally pick non-solution words.
        
        for w in range(total_words):
            # Count bits in (neighbor_masks[w] & uncovered)
            # Python 3.10+ has int.bit_count(). 
            # If older, bin(x).count('1'). Let's assume modern Python or use bin.
            # The environment usually supports recent Python.
            count = (neighbor_masks[w] & uncovered).bit_count()
            if count > max_covered:
                max_covered = count
                best_word = w
                
        if best_word == -1:
            break
            
        solution_indices.append(best_word)
        uncovered &= ~neighbor_masks[best_word]
        
    # Local Search: Try to remove redundant codewords
    # A codeword is redundant if all words it covers are also covered by other codewords.
    # This is equivalent to checking if the union of neighbors of (solution \ {w}) covers the neighbors of w.
    # Or simpler: Re-check coverage.
    # Given time constraints, a simple pass to remove any codeword that doesn't add unique coverage is good.
    
    # Let's do a single pass of removal.
    # We need to know if removing w leaves any gap.
    # This is expensive if done naively.
    # Alternative: Just return the greedy result. The greedy result with bitmasks should be very fast and likely better than the slow set-based one.
    
    # Let's check if we have time for a simple local search.
    # The greedy phase for 6561 words with bitmasks is very fast (< 1s).
    # We have ~139s left.
    
    # Simple Local Search:
    # For each codeword in solution (in reverse order of addition?), check if it can be removed.
    # To check if w can be removed:
    # 1. Compute union of neighbors of all other codewords.
    # 2. Check if neighbors[w] is a subset of that union.
    # This is O(|solution|^2 * bit_ops). |solution| ~ 400. 160k ops. Very fast.
    
    new_solution = list(solution_indices)
    changed = True
    while changed and (time.time() - start_time < limit):
        changed = False
        for i in range(len(new_solution) - 1, -1, -1):
            if time.time() - start_time > limit:
                break
            w = new_solution[i]
            # Check if w is redundant
            # Build mask of all other codewords
            others_mask = 0
            for j in range(len(new_solution)):
                if j != i:
                    others_mask |= neighbor_masks[new_solution[j]]
            
            if (neighbor_masks[w] & ~others_mask) == 0:
                # w is redundant
                new_solution.pop(i)
                changed = True
                break # Restart loop as indices changed
        
    # Convert indices back to lists of digits
    return [get_digits(idx) for idx in new_solution]
