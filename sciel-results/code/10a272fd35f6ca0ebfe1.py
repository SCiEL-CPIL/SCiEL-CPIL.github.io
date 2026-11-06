import math
import time

def construct():
    N = 512
    max_a = N - 1
    
    # Precompute weights and compatibility masks
    # weight[a] = 1 / (512 - a)
    # compatible_mask[a] has bit b set if gcd(a, b) == 1
    
    weights = [0.0] * (max_a + 1)
    for a in range(1, max_a + 1):
        weights[a] = 1.0 / (N - a)
        
    # Bitmasks for compatibility
    # Bit i corresponds to number i
    compatible_masks = [0] * (max_a + 1)
    
    # Initialize masks with all bits set, then clear conflicts
    # Or just build from scratch
    for a in range(1, max_a + 1):
        mask = 0
        for b in range(1, max_a + 1):
            if math.gcd(a, b) == 1:
                mask |= (1 << b)
        compatible_masks[a] = mask
        
    # Greedy initialization
    def run_greedy(initial_set, excluded):
        """
        initial_set: list of numbers already selected
        excluded: set of numbers that cannot be selected (e.g., the one we removed)
        Returns the final set
        """
        S = list(initial_set)
        mask_S = 0
        for a in S:
            mask_S |= (1 << a)
            
        # Candidates are all numbers not in S and not excluded
        # Sort by weight descending
        candidates = [a for a in range(1, max_a + 1) if a not in S and a not in excluded]
        candidates.sort(key=lambda x: weights[x], reverse=True)
        
        for b in candidates:
            # Check if b is compatible with all in S
            # compatible_masks[b] has 1s for coprimes
            # We need all bits in mask_S to be 1 in compatible_masks[b]
            if (compatible_masks[b] & mask_S) == mask_S:
                S.append(b)
                mask_S |= (1 << b)
                
        return S

    def score(S):
        return sum(weights[a] for a in S)
        
    # Initial Greedy
    S = run_greedy([], set())
    current_score = score(S)
    
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    while time.time() < deadline:
        improved = False
        # Try removing each element
        for a in list(S):
            if time.time() > deadline:
                break
            excluded = {a}
            S_temp = [x for x in S if x != a]
            S_new = run_greedy(S_temp, excluded)
            new_score = score(S_new)
            if new_score > current_score + 1e-9:
                S = S_new
                current_score = new_score
                improved = True
                break # Restart the for loop to check the new set
        
        if not improved:
            break
            
    return S
