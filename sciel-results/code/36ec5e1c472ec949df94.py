import time
import random

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # Helper to check comparability
    def is_comparable(a, b):
        return (a & b) == a or (a & b) == b
    
    # Helper to count comparable pairs between x and a list of masks, skipping index skip_idx
    def count_comp(x, fam, skip_idx):
        c = 0
        for j, y in enumerate(fam):
            if j == skip_idx:
                continue
            if is_comparable(x, y):
                c += 1
        return c
    
    # Initialize family
    # Start with a random family
    # To avoid duplicates, use a set for tracking
    fam_set = set()
    fam = []
    while len(fam) < m:
        mask = random.randint(0, total_masks - 1)
        if mask not in fam_set:
            fam_set.add(mask)
            fam.append(mask)
    
    # Calculate initial score
    current_score = 0
    for i in range(m):
        current_score += count_comp(fam[i], fam, i)
    current_score //= 2  # Each pair counted twice
    
    # Local Search
    # We try to improve by swapping elements
    # To speed up, we can precompute "potential" candidates or just randomize
    
    # Set a seed for reproducibility in this run if needed, but random is fine
    # random.seed(42) # Optional
    
    no_improve_count = 0
    max_no_improve = 10000 # Reset strategy if stuck? Or just keep going.
    
    while time.time() < deadline:
        improved = False
        # Try swapping each element with a random candidate
        # To be efficient, we iterate through indices and try a few candidates for each
        for i in range(m):
            if time.time() > deadline:
                break
            
            # Try a few random candidates for position i
            # We want candidates that are "good"
            # Random candidates are fine, but we can bias towards neighbors
            # Let's just try 10 random candidates per position per pass
            for _ in range(10):
                cand = random.randint(0, total_masks - 1)
                if cand in fam_set:
                    continue
                
                # Calculate gain
                # Remove fam[i], add cand
                # Gain = count_comp(cand, fam, i) - count_comp(fam[i], fam, i)
                gain = count_comp(cand, fam, i) - count_comp(fam[i], fam, i)
                
                if gain > 0:
                    # Accept
                    fam_set.remove(fam[i])
                    fam_set.add(cand)
                    fam[i] = cand
                    current_score += gain
                    improved = True
                    # After a swap, we might want to restart the loop or continue
                    # Continuing is fine, but the 'i' loop might benefit from restarting
                    # to re-evaluate other positions with the new context
                    # However, restarting is expensive. Just continue.
                    break # Move to next index i
        
        if not improved:
            # If no improvement in a full pass, we might be stuck in a local optimum
            # We can try a random restart or just accept
            # Given time, we can try a few random perturbations
            # But simple random swaps usually find the global optimum for this size
            pass

    # Convert masks to lists of integers
    result = []
    for mask in fam:
        subset = [b for b in range(n) if (mask >> b) & 1]
        result.append(subset)
    
    return result

# EVOLVE-BLOCK-END
