import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    N = 256
    
    # 1. Precompute 4-APs and 3-APs as bit masks
    # A is represented as an integer where bit i is 1 if i is in A.
    # Note: Python ints are arbitrary precision, so this is safe.
    
    masks_4ap = []
    elem_to_4ap_masks = {i: [] for i in range(N + 1)}
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            # 4-AP: a, a+d, a+2d, a+3d
            mask = (1 << a) | (1 << (a + d)) | (1 << (a + 2 * d)) | (1 << (a + 3 * d))
            masks_4ap.append(mask)
            # Map each element in the AP to this mask
            elem_to_4ap_masks[a].append(mask)
            elem_to_4ap_masks[a + d].append(mask)
            elem_to_4ap_masks[a + 2 * d].append(mask)
            elem_to_4ap_masks[a + 3 * d].append(mask)
            
    masks_3ap = []
    elem_to_3ap_masks = {i: [] for i in range(N + 1)}
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            # 3-AP: a, a+d, a+2d
            mask = (1 << a) | (1 << (a + d)) | (1 << (a + 2 * d))
            masks_3ap.append(mask)
            # Map each element in the AP to this mask
            elem_to_3ap_masks[a].append(mask)
            elem_to_3ap_masks[a + d].append(mask)
            elem_to_3ap_masks[a + 2 * d].append(mask)

    # Helper functions using bit masks
    
    def is_valid_add(A, x):
        """Check if adding x to A creates any 4-AP."""
        # x must not be in A
        if A & (1 << x):
            return False
        # Check all 4-APs containing x
        for mask in elem_to_4ap_masks[x]:
            # If the other 3 bits are already set in A, then adding x completes the 4-AP
            # mask has 4 bits set. A has 3 of them set.
            # (A | (1<<x)) & mask == mask is equivalent to (A & mask) == (mask ^ (1<<x))
            # Simpler: check if (A & mask) has 3 bits set.
            # Since x is not in A, A & mask will have at most 3 bits.
            # If it has exactly 3, adding x makes it 4.
            if (A & mask).bit_count() == 3:
                return False
        return True

    def is_valid_remove(A, x):
        """Removing x never creates a 4-AP, so always valid."""
        return True

    def is_valid_swap(A, x, y):
        """Check if swapping x (out) and y (in) creates any 4-AP."""
        # x in A, y not in A
        # New set A' = A ^ (1<<x) ^ (1<<y)
        # We only need to check 4-APs containing y, because 4-APs not containing y 
        # were either already present (impossible) or lost (impossible to gain).
        # Actually, a 4-AP in A' must contain y.
        for mask in elem_to_4ap_masks[y]:
            # Check if the other 3 elements are in A'
            # A' contains y. So we need the other 3 bits of mask to be in A \ {x}.
            # Let mask_without_y = mask ^ (1<<y)
            # We need (A' & mask_without_y) == mask_without_y
            # A' = A ^ (1<<x) ^ (1<<y)
            # A' & mask_without_y = (A ^ (1<<x) ^ (1<<y)) & mask_without_y
            # Since y is not in mask_without_y, ^ (1<<y) has no effect on mask_without_y bits.
            # So we need (A ^ (1<<x)) & mask_without_y == mask_without_y
            # This is equivalent to:
            # 1. x is not in mask_without_y (if x is, we removed it, so it's fine if it was there? No.)
            # Let's just construct A_prime locally for the check? No, that's slow.
            # Let's use the bit count trick.
            # We need exactly 3 other elements in A'.
            # A' has y. The other 3 must be from mask_without_y.
            # Let other_bits = mask_without_y
            # We need (A' & other_bits) == other_bits
            # A' & other_bits = (A & other_bits) ^ (1<<x if x in other_bits else 0)
            # Case 1: x is not in other_bits. Then A' & other_bits = A & other_bits.
            #   We need A & other_bits == other_bits.
            # Case 2: x is in other_bits. Then A' & other_bits = (A & other_bits) ^ (1<<x).
            #   Since x is in A, A & other_bits has bit x set.
            #   (A & other_bits) ^ (1<<x) clears bit x.
            #   We need this result to be other_bits.
            #   But other_bits has bit x set (by definition of Case 2).
            #   So (A & other_bits) ^ (1<<x) cannot be equal to other_bits (which has x set).
            #   Therefore, if x is in the 4-AP mask (and x != y), the swap is INVALID?
            #   Wait. If x is in the 4-AP, and we remove x, then the 4-AP is broken.
            #   So we don't need to worry about 4-APs containing x?
            #   Correct. Any 4-AP in A' must contain y.
            #   If a 4-AP contains both x and y, removing x breaks it.
            #   So we only care about 4-APs containing y but NOT x.
            
            if mask & (1 << x):
                # This 4-AP contains x. Removing x breaks it. So it's safe.
                continue
            
            # x is not in this 4-AP.
            # So A' & mask = (A & mask) | (1<<y)
            # We need A' & mask == mask
            # This requires (A & mask) == mask ^ (1<<y)
            # Let's check if the other 3 bits are in A.
            other_bits = mask ^ (1 << y)
            if (A & other_bits) == other_bits:
                return False
        return True

    def count_3aps(A):
        """Count 3-APs in A."""
        count = 0
        for mask in masks_3ap:
            if (A & mask) == mask:
                count += 1
        return count

    def gain_add(A, x):
        """Calculate gain in 3-AP count if adding x."""
        gain = 0
        for mask in elem_to_3ap_masks[x]:
            # Check if other 2 bits are in A
            other_bits = mask ^ (1 << x)
            if (A & other_bits) == other_bits:
                gain += 1
        return gain

    def loss_remove(A, x):
        """Calculate loss in 3-AP count if removing x."""
        loss = 0
        for mask in elem_to_3ap_masks[x]:
            # Check if other 2 bits are in A
            other_bits = mask ^ (1 << x)
            if (A & other_bits) == other_bits:
                loss += 1
        return loss

    def delta_swap(A, x, y):
        """Calculate change in 3-AP count if swapping x (out) and y (in)."""
        # Loss from removing x
        loss = 0
        for mask in elem_to_3ap_masks[x]:
            other_bits = mask ^ (1 << x)
            # The other 2 bits must be in A \ {y} (y is not in A yet)
            # So just check if they are in A.
            if (A & other_bits) == other_bits:
                loss += 1
        
        # Gain from adding y
        gain = 0
        for mask in elem_to_3ap_masks[y]:
            other_bits = mask ^ (1 << y)
            # The other 2 bits must be in A \ {x}
            # So check if they are in A, and not x.
            # If one of the other bits is x, then it's not in A \ {x}.
            if (A & other_bits) == other_bits:
                # Check if x is in other_bits
                if not (other_bits & (1 << x)):
                    gain += 1
                else:
                    # x is one of the other bits.
                    # We need the OTHER bit (besides x) to be in A.
                    # other_bits has 2 bits. One is x. The other is z.
                    # We need z to be in A.
                    # Since (A & other_bits) == other_bits, z is in A.
                    # So this 3-AP is completed by y, provided x is NOT in A (which is true after removal).
                    # Wait, the 3-AP is {x, y, z}.
                    # In A', we have y and z. x is removed.
                    # So the 3-AP is NOT completed.
                    # So gain is 0 for this mask.
                    pass
                    
        return gain - loss

    # 2. Greedy Initialization
    A = 0
    current_score = 0
    remaining = list(range(1, N + 1))
    random.shuffle(remaining) # Randomize order for tie-breaking or just iterate
    
    # To make it faster, we can just iterate 1..N
    for x in range(1, N + 1):
        if not (A & (1 << x)):
            if is_valid_add(A, x):
                g = gain_add(A, x)
                if g > 0:
                    A |= (1 << x)
                    current_score += g
                # Even if g==0, we might add it to allow future gains? 
                # Greedy usually adds if valid. Let's add if valid.
                # But we want to maximize 3-APs. Adding an element with 0 gain doesn't hurt 
                # unless it blocks future gains.
                # Let's stick to adding only if g > 0 for a tighter start, 
                # or just add all valid ones.
                # Let's add all valid ones.
                else:
                    A |= (1 << x)

    best_A = A
    best_score = current_score
    
    # 3. Local Search (Hill Climbing)
    def local_search(A, score, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try all swaps, adds, removes
            # Swaps are most likely to improve.
            
            # Collect all possible moves
            # To speed up, we can just iterate.
            
            # 1. Try Adds
            for x in range(1, N + 1):
                if not (A & (1 << x)):
                    if is_valid_add(A, x):
                        g = gain_add(A, x)
                        if g > 0:
                            A |= (1 << x)
                            score += g
                            improved = True
                            # Restart loop to recalculate? 
                            # Or just continue. First improvement strategy.
                            # Let's restart to ensure we pick the best next move.
                            break
            if improved:
                continue
                
            # 2. Try Removes
            # Removing usually decreases score, so only useful if it enables a better swap?
            # In pure hill climbing, we only accept if score increases.
            # So we skip pure removes unless we are doing SA.
            # For ILS, we rely on swaps.
            
            # 3. Try Swaps
            # Iterate x in A, y not in A
            # This is O(N^2). 256^2 = 65536. 
            # Each swap check is O(degree). Degree is small.
            # This might be slow in Python if done naively.
            # Optimization: Only check promising swaps?
            # Or just do it. 65k * 50 ops = 3M ops per local search step.
            # If we do 100 local search steps, 300M ops. Might be too slow.
            
            # Better: Randomized Hill Climbing?
            # Pick random x in A, random y not in A.
            # If valid and score increases, accept.
            # Repeat until no improvement in K random tries.
            
            no_improve_count = 0
            max_random_tries = 1000
            while no_improve_count < max_random_tries and time.time() < time_limit:
                # Pick random x in A
                # We need a list of active elements
                active = [i for i in range(1, N+1) if A & (1<<i)]
                inactive = [i for i in range(1, N+1) if not (A & (1<<i))]
                
                if not active or not inactive:
                    break
                    
                x = random.choice(active)
                y = random.choice(inactive)
                
                if is_valid_swap(A, x, y):
                    d = delta_swap(A, x, y)
                    if d > 0:
                        # Apply swap
                        A ^= (1 << x)
                        A ^= (1 << y)
                        score += d
                        no_improve_count = 0
                        improved = True
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
                    
            # If we exhausted random tries without improvement, try a full scan?
            # Or just stop local search.
            # Full scan is expensive. Let's try a limited full scan if time permits.
            if not improved:
                # Try a few deterministic swaps?
                pass
                
        return A, score

    # 4. Iterated Local Search
    # We will run local search, then perturb, then local search again.
    
    # Initial Local Search
    A, score = local_search(A, current_score, deadline - 5.0)
    if score > best_score:
        best_score = score
        best_A = A
        
    # Perturbation function
    def perturb(A, k):
        """Perform k random valid moves."""
        for _ in range(k):
            if time.time() > deadline - 5.0:
                break
            active = [i for i in range(1, N+1) if A & (1<<i)]
            inactive = [i for i in range(1, N+1) if not (A & (1<<i))]
            
            if not active or not inactive:
                break
                
            move_type = random.random()
            if move_type < 0.7:
                # Swap
                x = random.choice(active)
                y = random.choice(inactive)
                if is_valid_swap(A, x, y):
                    A ^= (1 << x)
                    A ^= (1 << y)
            elif move_type < 0.85:
                # Add
                y = random.choice(inactive)
                if is_valid_add(A, y):
                    A |= (1 << y)
            else:
                # Remove
                x = random.choice(active)
                A ^= (1 << x)
        return A

    # Main ILS Loop
    while time.time() < deadline - 2.0:
        # Perturb
        A_perturbed = perturb(A, 5)
        
        # Recalculate score for perturbed set
        score_perturbed = count_3aps(A_perturbed)
        
        # Local Search on perturbed set
        A_ls, score_ls = local_search(A_perturbed, score_perturbed, deadline - 2.0)
        
        if score_ls > best_score:
            best_score = score_ls
            best_A = A_ls
            A = A_ls # Update current best for next perturbation
        
        # If no improvement for a long time, increase perturbation strength?
        # For now, fixed perturbation size.

    # Final Polish: Run local search on best_A one more time
    A_final, score_final = local_search(best_A, best_score, deadline)
    if score_final > best_score:
        best_A = A_final
        best_score = score_final

    # Convert best_A to list
    res = [i for i in range(1, N + 1) if best_A & (1 << i)]
    
    if not res:
        return [1]
        
    return res

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Result size: {len(res)}")
    # print(res)
