import time
import random

N = 256
DEADLINE = 58.0  # Safety margin for 60s limit

def construct():
    start_time = time.time()
    
    # 1. Precomputation
    # 3-APs: (a, a+d, a+2d)
    three_aps = []
    # For incremental score, we might need to know which 3-APs contain which elements?
    # Or just iterate through all 3-APs? 
    # Number of 3-APs is ~ N^2/4 = 16k. Iterating 16k masks is fast.
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1))
            three_aps.append(mask)
            
    # 4-APs: (a, a+d, a+2d, a+3d)
    # For each v, store list of (mask_rest, mask_full) where mask_rest = mask_full ^ (1 << (v-1))
    # Actually, we just need mask_rest.
    four_aps_by_v = {i: [] for i in range(1, N+1)}
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1)) | (1 << (e-1))
            
            # For each element in the AP, store the rest of the mask
            for v in [a, b, c, e]:
                rest_mask = mask ^ (1 << (v-1))
                four_aps_by_v[v].append(rest_mask)
                
    # Helper to get elements from mask
    def get_elements(mask):
        return [i+1 for i in range(N) if (mask >> i) & 1]
        
    # Helper to count 3-APs
    def get_score(mask):
        return sum(1 for ap_mask in three_aps if (mask & ap_mask) == ap_mask)
        
    # Helper to check if swap u->v is valid
    # u is in S, v is not in S
    # S' = S \ {u} U {v}
    # Invalid if exists 4-AP M containing v such that M \ {v} is subset of S \ {u}
    def is_valid_swap(S_mask, u, v):
        u_bit = 1 << (u-1)
        # Check all 4-APs containing v
        for rest_mask in four_aps_by_v[v]:
            # Condition: rest_mask subset of S \ {u}
            # This implies:
            # 1. u is NOT in rest_mask
            # 2. rest_mask is subset of S
            if (rest_mask & u_bit) != 0:
                continue
            if (S_mask & rest_mask) == rest_mask:
                return False
        return True

    # 2. Initial Greedy Set
    current_mask = 0
    for x in range(1, N+1):
        bit = 1 << (x-1)
        new_mask = current_mask | bit
        # Check validity of adding x
        valid = True
        for rest_mask in four_aps_by_v[x]:
            # Adding x creates 4-AP if rest_mask is subset of current_mask
            if (current_mask & rest_mask) == rest_mask:
                valid = False
                break
        if valid:
            current_mask = new_mask
            
    current_score = get_score(current_mask)
    best_mask = current_mask
    best_score = current_score
    
    # 3. Local Search
    # We will try to improve the score by swapping elements.
    # To make it more effective, we can use a "first improvement" or "best improvement" strategy.
    # Given the small N, first improvement is usually faster per step, allowing more steps.
    
    # Precompute list of elements for faster iteration?
    # We will maintain a list of elements in S and not in S.
    
    while time.time() - start_time < DEADLINE:
        elements = get_elements(current_mask)
        non_elements = [i for i in range(1, N+1) if not (current_mask >> (i-1)) & 1]
        
        if not elements or not non_elements:
            break
            
        # Shuffle to randomize order
        random.shuffle(elements)
        random.shuffle(non_elements)
        
        improved = False
        for u in elements:
            if time.time() - start_time > DEADLINE:
                break
            u_bit = 1 << (u-1)
            for v in non_elements:
                if time.time() - start_time > DEADLINE:
                    break
                # Check validity
                if not is_valid_swap(current_mask, u, v):
                    continue
                
                # If valid, calculate new score
                # Optimization: We can estimate the score change.
                # But get_score is fast enough (16k ops).
                new_mask = (current_mask & ~u_bit) | (1 << (v-1))
                new_score = get_score(new_mask)
                
                if new_score > current_score:
                    current_mask = new_mask
                    current_score = new_score
                    improved = True
                    # Update best
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
                    # Break inner loop to restart with new set
                    break
            if improved:
                break
                
        if not improved:
            # Local optimum reached.
            # Try a random kick (accept a worse or equal move to escape)
            # Pick random u, v
            if random.random() < 0.5:
                # Try adding an element if possible? (If set is not maximal)
                # Greedy usually makes it maximal.
                pass
            
            # Random swap kick
            u = random.choice(elements)
            v = random.choice(non_elements)
            if is_valid_swap(current_mask, u, v):
                new_mask = (current_mask & ~(1 << (u-1))) | (1 << (v-1))
                new_score = get_score(new_mask)
                # Accept if better, or with some probability if worse (Simulated Annealing style)
                # For simplicity, just accept if it doesn't drop too much, or just accept to explore.
                # Let's accept if new_score >= current_score - 2 (allow small dip)
                if new_score >= current_score - 2:
                    current_mask = new_mask
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
            else:
                # If even random kick is invalid, we might be stuck in a very tight spot.
                # Try another random pair
                for _ in range(10):
                    u = random.choice(elements)
                    v = random.choice(non_elements)
                    if is_valid_swap(current_mask, u, v):
                        new_mask = (current_mask & ~(1 << (u-1))) | (1 << (v-1))
                        new_score = get_score(new_mask)
                        if new_score >= current_score - 2:
                            current_mask = new_mask
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best_mask = current_mask
                        break

    return get_elements(best_mask)
