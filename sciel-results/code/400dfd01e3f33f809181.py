import time
import random
import numpy as np

def construct():
    N = 512
    start_time = time.time()
    # Safety margin: evaluator kills at 90s. We stop at 85s to be safe.
    deadline = start_time + 85.0
    
    # 1. Precompute Conflict Structure
    # For each element i (0-indexed), we store a list of "masks".
    # A mask represents the other 3 elements in a 4-AP.
    # If (S & mask) == mask, then adding i creates a 4-AP.
    # Actually, we need to check if the other 3 are present.
    # Let's precompute for each i, the list of tuples (j, k, l) such that {i,j,k,l} is a 4-AP.
    # To make it faster for SA, we can precompute a list of masks for each i.
    # Mask is an integer with bits set at j, k, l.
    
    conflict_masks = [[] for _ in range(N)]
    
    # Iterate through all 4-APs
    # a, a+d, a+2d, a+3d
    # Indices: 0 to N-1
    for a in range(N):
        max_d = (N - 1 - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            
            # For a: others are b, c, e
            mask_a = (1 << b) | (1 << c) | (1 << e)
            conflict_masks[a].append(mask_a)
            
            # For b: others are a, c, e
            mask_b = (1 << a) | (1 << c) | (1 << e)
            conflict_masks[b].append(mask_b)
            
            # For c: others are a, b, e
            mask_c = (1 << a) | (1 << b) | (1 << e)
            conflict_masks[c].append(mask_c)
            
            # For e: others are a, b, c
            mask_e = (1 << a) | (1 << b) | (1 << c)
            conflict_masks[e].append(mask_e)
            
    # 2. Fast Validity Check
    def is_valid_state(S):
        """
        Check if bitmask S contains any 4-AP.
        Returns True if valid (no 4-AP), False otherwise.
        """
        # We can check by iterating through set bits and checking conflicts?
        # Or iterate through all 4-APs and check if all 4 bits are set.
        # The latter might be faster if we precompute all 4-AP masks.
        # Let's precompute all 4-AP masks globally for O(1) check per AP.
        # There are ~N^2/6 4-APs. 512^2/6 ~ 43k. This is very fast.
        pass

    # Precompute global list of 4-AP masks
    all_ap_masks = []
    for a in range(N):
        max_d = (N - 1 - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            mask = (1 << a) | (1 << b) | (1 << c) | (1 << e)
            all_ap_masks.append(mask)
            
    def count_violations(S):
        """Count how many 4-APs are fully contained in S."""
        count = 0
        for mask in all_ap_masks:
            if (S & mask) == mask:
                count += 1
        return count

    def has_violation(S):
        """Check if any 4-AP exists in S."""
        for mask in all_ap_masks:
            if (S & mask) == mask:
                return True
        return False

    # 3. Objective Function for SA
    # We want to maximize cardinality, but strictly avoid 4-APs.
    # Standard approach:
    # If valid: Score = Cardinality
    # If invalid: Score = Cardinality - Penalty * Violations
    # We use a large penalty to ensure validity is prioritized, but allow
    # transitions to invalid states if they lead to higher cardinality later,
    # or simply use a "hard" constraint with restarts?
    # The inspiration says "density + penalty" allows escaping local optima.
    # Let's define:
    # Energy = -Cardinality + Penalty * Violations
    # We minimize Energy.
    
    P = 1000  # Penalty weight
    
    def energy(S):
        card = bin(S).count('1')
        viol = count_violations(S)
        return -card + P * viol

    # 4. Initial Solution
    # Start with a greedy construction to get a good starting point.
    # Greedy: add elements in random order if they don't create a 4-AP.
    
    def greedy_init():
        S = 0
        order = list(range(N))
        random.shuffle(order)
        for i in order:
            # Check if adding i creates a 4-AP
            # We can check conflicts for i
            valid = True
            for mask in conflict_masks[i]:
                if (S & mask) == mask:
                    valid = False
                    break
            if valid:
                S |= (1 << i)
        return S

    # 5. Simulated Annealing
    # State: Bitmask S
    # Move: Flip a bit (add or remove an element)
    # Or: Swap (remove one, add another) to keep cardinality similar?
    # Flipping is simpler. Removing reduces card, adding increases.
    # If we just flip, we might drop into low cardinality.
    # Better move: 
    #   - If valid: Try adding a random element not in S. If it creates violation, reject? 
    #     No, SA allows it.
    #   - Try removing a random element in S.
    #   - Try swapping: remove x, add y.
    
    # Let's stick to simple bit flips for robustness, but weighted towards additions.
    # Or use "kick" moves.
    
    # Let's try a hybrid: 
    # 1. Greedy init
    # 2. SA with moves: 
    #    - Pick random i.
    #    - If i in S: remove it.
    #    - If i not in S: add it.
    #    - Calculate delta E.
    #    - Accept if delta E < 0 or with prob exp(-delta E / T).
    
    best_S = greedy_init()
    best_card = bin(best_S).count('1')
    current_S = best_S
    current_E = energy(current_S)
    
    T = 10.0  # Initial Temperature
    T_min = 0.1
    cooling = 0.999
    
    # Precompute list of indices for faster random selection?
    # Just use random.randint
    
    while time.time() < deadline - 2.0:
        # Generate neighbor
        i = random.randint(0, N - 1)
        bit = 1 << i
        
        if current_S & bit:
            # Remove
            new_S = current_S ^ bit
        else:
            # Add
            new_S = current_S | bit
            
        new_E = energy(new_S)
        delta_E = new_E - current_E
        
        # Acceptance criterion
        if delta_E < 0 or random.random() < np.exp(-delta_E / T):
            current_S = new_S
            current_E = new_E
            
            # Update best if valid and larger
            if current_E < 0: # Valid state (violations=0 -> E = -card < 0)
                card = bin(current_S).count('1')
                if card > best_card:
                    best_card = card
                    best_S = current_S
                    # Optional: if we hit a very good score, cool down faster?
        
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
            # If stuck at low temp, maybe reheat or restart?
            # For now, just keep going.
            # To avoid infinite loop at T_min, we can periodically reheat slightly.
            if random.random() < 0.01:
                T = 5.0

    # Final Local Search / Cleanup on best_S
    # Try to add any remaining elements that fit
    S = best_S
    changed = True
    while changed:
        changed = False
        for i in range(N):
            if not (S & (1 << i)):
                valid = True
                for mask in conflict_masks[i]:
                    if (S & mask) == mask:
                        valid = False
                        break
                if valid:
                    S |= (1 << i)
                    changed = True
    
    # Verify final result
    if has_violation(S):
        # Fallback to a known valid small set or greedy
        S = greedy_init()
        # Ensure greedy result is valid
        while has_violation(S):
             # This should not happen for greedy, but just in case
             # Remove random element
             bits = [i for i in range(N) if S & (1<<i)]
             if not bits: break
             idx = random.choice(bits)
             S ^= (1 << idx)
    else:
        best_S = S
        best_card = bin(S).count('1')

    # Convert to list
    result = []
    for i in range(N):
        if best_S & (1 << i):
            result.append(i + 1)
            
    return sorted(result)

# EVOLVE-BLOCK-END
