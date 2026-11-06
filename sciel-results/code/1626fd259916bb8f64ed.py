import time
import random

N = 512
DEADLINE_SECONDS = 55

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # 1. Precomputation
    # bit_mask[i] represents the bit for element i+1
    bit_mask = [1 << i for i in range(N)]
    
    # conflicts[x] is a list of bitmasks. 
    # Each mask represents the other 3 elements in a 4-AP containing x.
    # If (S & mask) == mask, then adding x creates a 4-AP.
    conflicts = [[] for _ in range(N)] # Index 0 for element 1, ... Index N-1 for element N
    
    # Generate all 4-APs
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e > N:
                break
            
            # Elements are a, b, c, e (1-indexed)
            # Indices are a-1, b-1, c-1, e-1
            idx_a = a - 1
            idx_b = b - 1
            idx_c = c - 1
            idx_e = e - 1
            
            mask_all = bit_mask[idx_a] | bit_mask[idx_b] | bit_mask[idx_c] | bit_mask[idx_e]
            
            # For each element in the AP, the conflict mask is the others
            # For a: mask is b|c|e
            # For b: mask is a|c|e
            # etc.
            
            mask_bce = mask_all ^ bit_mask[idx_a]
            mask_ace = mask_all ^ bit_mask[idx_b]
            mask_abe = mask_all ^ bit_mask[idx_c]
            mask_abc = mask_all ^ bit_mask[idx_e]
            
            conflicts[idx_a].append(mask_bce)
            conflicts[idx_b].append(mask_ace)
            conflicts[idx_c].append(mask_abe)
            conflicts[idx_e].append(mask_abc)
            
    def is_valid_add(x_idx, S):
        """Check if element x_idx (0-indexed) can be added to S."""
        for m in conflicts[x_idx]:
            if (S & m) == m:
                return False
        return True

    def greedy_add(S, order):
        """Add elements from order to S if valid."""
        for x_idx in order:
            if not (S & bit_mask[x_idx]):
                if is_valid_add(x_idx, S):
                    S |= bit_mask[x_idx]
        return S

    # 2. Initial Greedy Construction
    # Standard greedy (1 to N)
    S = 0
    initial_order = list(range(N))
    S = greedy_add(S, initial_order)
    best_S = S
    best_size = S.bit_count()
    
    # 3. Local Search with Perturbation
    while time.time() < deadline:
        # Try to fill gaps
        # We can randomize the order of trying to add elements
        # But first, let's see if we can add anything in standard order
        # Actually, random order is better for diversity
        
        # Perturb current best or current S?
        # Strategy: Keep a 'current' S. Perturb it. If result is better than 'best', update 'best'.
        # If 'current' is improved by perturbation, keep it.
        
        # Let's try a few perturbation strengths
        # Remove k random elements from S
        k = random.randint(1, 6)
        
        # Get indices of set bits in S
        set_indices = [i for i in range(N) if (S & bit_mask[i])]
        
        if len(set_indices) < k:
            continue
            
        to_remove = random.sample(set_indices, k)
        
        S_perturbed = S
        for idx in to_remove:
            S_perturbed &= ~bit_mask[idx]
            
        # Re-greedy with random order
        # We want to prioritize adding elements that "fit well"
        # Random order is a good baseline
        order = list(range(N))
        random.shuffle(order)
        
        S_new = greedy_add(S_perturbed, order)
        
        # Check if we can add more in a second pass with a different order?
        # Maybe not necessary if greedy_add is thorough.
        
        size_new = S_new.bit_count()
        
        if size_new > best_size:
            best_S = S_new
            best_size = size_new
            S = S_new # Continue from this new best
        else:
            # If not better than best, maybe keep S as is or revert?
            # Usually, we stick with the current trajectory unless it's better.
            # Or we can accept "sideways" moves? 
            # For maximization, sticking to best-so-far is safer.
            # But to explore, we should probably restart from best_S occasionally.
            # Let's say: if S_new is not better than best_S, we might still use S_new 
            # if it's close? No, let's just restart from best_S if we stagnate?
            # Simpler: Just update S if it's better than current S, 
            # and track global best.
            
            if size_new > S.bit_count():
                S = S_new
            else:
                # Revert to best_S to avoid drifting into bad local optima?
                # Or keep S? 
                # Let's try: if we haven't improved best in a while, restart from best_S.
                # For now, just keep S as the base for next perturbation.
                pass

    # Convert best_S to sorted list
    result = [i + 1 for i in range(N) if (best_S & bit_mask[i])]
    return result
