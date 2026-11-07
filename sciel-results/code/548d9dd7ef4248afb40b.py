"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    """
    Construct 40 distinct subsets of vertices 0..11 to maximize comparable pairs.
    Strategy:
    1. Use a deterministic initial construction based on size distribution.
       Sets of size 0 and 12 are comparable to almost everything.
       Sets of size 1 and 11 are comparable to many.
       A good strategy is to use many small sets and many large sets.
       Let's try a distribution that maximizes the sum of C(n, k_i) * C(n-k_i, k_j-k_i) for i<j.
       
       Actually, a simpler structural insight:
       If we pick sets such that they are all subsets of a smaller universe, or supersets of a larger core, they are comparable.
       
       Let's use a greedy approach with a specific initialization:
       Initialize with 40 sets chosen to be highly comparable.
       For example, take all sets of size 0, 1, 2, 10, 11, 12.
       Number of sets:
       Size 0: 1
       Size 1: 12
       Size 2: 66
       Size 10: 66
       Size 11: 12
       Size 12: 1
       Total available: 158. We need 40.
       
       Let's pick 1 of size 0, 12 of size 1, 12 of size 11, 1 of size 12, and 14 of size 2 or 10.
       Actually, let's just use a fast local search starting from a good random initialization 
       biased towards small and large sets, as random search from scratch is slow.
       
       But the directive asks for a DIFFERENT strategy. 
       The inspirations used Simulated Annealing and Greedy+Local Search.
       Let's try a **Greedy Construction** from scratch, adding one set at a time 
       that maximizes the number of comparable pairs with the already selected sets.
       This is a "Greedy Set Selection" approach.
    """
    n = 12
    m = 40
    deadline = time.time() + 80.0  # Safety margin
    
    # Precompute all possible subsets as masks
    # There are 2^12 = 4096 possible subsets.
    all_masks = list(range(4096))
    
    # Helper to count comparable pairs with existing sets
    # For a new mask 'new_mask', count how many existing masks are comparable.
    # Comparable if (a & b) == a or (a & b) == b.
    
    selected_masks = []
    
    # Greedy initialization
    # To speed up, we can precompute comparability matrix? 4096x4096 is ~16M entries. 
    # In Python, a list of lists of bytes might be too slow/memory heavy.
    # But we can compute on the fly. 40 * 4096 * (bitwise ops) is fast.
    
    # We need to ensure distinct sets.
    available = set(all_masks)
    
    for _ in range(m):
        if time.time() > deadline:
            break
        best_mask = -1
        best_score = -1
        # Randomly sample a subset of candidates to speed up? 
        # Or just iterate all? 40 * 4096 = 163k iterations. 
        # Each iteration checks against up to 40 selected masks. 163k * 40 = 6.5M ops. 
        # This is fast in Python (<1s).
        
        # To ensure diversity and avoid getting stuck in a local optimum of the greedy method,
        # we can add a small random perturbation to the score or randomize the candidate order.
        
        # Let's iterate all available masks
        for mask in available:
            # Calculate score: number of comparable pairs with selected_masks
            score = 0
            for sm in selected_masks:
                if (mask & sm) == mask or (mask & sm) == sm:
                    score += 1
            # Add a small random tie-breaker to avoid deterministic bias
            score += random.random() * 0.1
            if score > best_score:
                best_score = score
                best_mask = mask
        
        if best_mask == -1:
            # Fallback: pick a random available mask
            best_mask = random.choice(list(available))
        
        selected_masks.append(best_mask)
        available.remove(best_mask)
        
    # Convert masks to lists of ints
    result = []
    for mask in selected_masks:
        s = []
        for b in range(n):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)
        
    return result

# EVOLVE-BLOCK-END
