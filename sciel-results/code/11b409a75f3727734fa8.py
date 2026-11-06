import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    N = 256

    # Precompute 3-APs and 4-APs
    three_aps = []
    four_aps = []
    
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            three_aps.append((a, a + d, a + 2 * d))
            
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            four_aps.append((a, a + d, a + 2 * d, a + 3 * d))

    # Map element to indices of APs it belongs to
    # Also store the "other" elements for fast check
    elem_to_3aps = [[] for _ in range(N + 1)]
    for idx, (a, b, c) in enumerate(three_aps):
        elem_to_3aps[a].append((idx, b, c))
        elem_to_3aps[b].append((idx, a, c))
        elem_to_3aps[c].append((idx, a, b))

    elem_to_4aps = [[] for _ in range(N + 1)]
    for idx, (a, b, c, d) in enumerate(four_aps):
        elem_to_4aps[a].append((idx, b, c, d))
        elem_to_4aps[b].append((idx, a, c, d))
        elem_to_4aps[c].append((idx, a, b, d))
        elem_to_4aps[d].append((idx, a, b, c))

    def count_3aps(S):
        """Count 3-APs in bitmask S"""
        count = 0
        for a, b, c in three_aps:
            if (S >> (a-1)) & 1 and (S >> (b-1)) & 1 and (S >> (c-1)) & 1:
                count += 1
        return count

    def is_safe_add(S, x):
        """Check if adding x to S creates any 4-AP"""
        for _, u, v, w in elem_to_4aps[x]:
            if ((S >> (u-1)) & 1) and ((S >> (v-1)) & 1) and ((S >> (w-1)) & 1):
                return False
        return True

    def gain_3aps(S, x):
        """Calculate number of new 3-APs formed by adding x to S"""
        gain = 0
        for _, u, v in elem_to_3aps[x]:
            if ((S >> (u-1)) & 1) and ((S >> (v-1)) & 1):
                gain += 1
        return gain

    def loss_3aps(S, x):
        """Calculate number of 3-APs lost by removing x from S"""
        # x is currently in S
        loss = 0
        for _, u, v in elem_to_3aps[x]:
            if ((S >> (u-1)) & 1) and ((S >> (v-1)) & 1):
                loss += 1
        return loss

    def greedy_fill(S, max_time=None):
        """Greedily add safe elements to S, prioritizing high gain. Returns (new_S, new_score)."""
        current_S = S
        current_score = count_3aps(current_S)
        
        # Get all elements not in S
        candidates = [i for i in range(1, N+1) if not ((current_S >> (i-1)) & 1)]
        
        # Sort by potential gain (approximate: count how many pairs in S could form 3-AP with candidate)
        # We'll do a few passes: first try high-gain elements, then fill the rest
        for _ in range(3):  # Multiple passes to handle interactions
            if max_time is not None and time.time() > max_time:
                break
            # Sort candidates by gain
            scored = []
            for y in candidates:
                if (current_S >> (y-1)) & 1:
                    continue
                g = gain_3aps(current_S, y)
                scored.append((g, y))
            scored.sort(reverse=True)
            
            added = set()
            for g, y in scored:
                if y in added:
                    continue
                if (current_S >> (y-1)) & 1:
                    continue
                if is_safe_add(current_S, y):
                    current_S |= (1 << (y-1))
                    current_score += gain_3aps(current_S ^ (1 << (y-1)), y)
                    added.add(y)
                    if max_time is not None and time.time() > max_time:
                        break
            
            # Update candidates
            candidates = [i for i in range(1, N+1) if not ((current_S >> (i-1)) & 1)]
            if not candidates:
                break
        
        return current_S, current_score

    # Local Search with swap moves
    def local_search(S, time_limit):
        current_S = S
        current_score = count_3aps(current_S)
        best_S = current_S
        best_score = current_score
        
        T = 15.0
        alpha = 0.998
        
        while time.time() < time_limit:
            # Choose move type: remove+fill (50%), swap (30%), remove only (20%)
            move_type = random.random()
            
            bits = [i+1 for i in range(N) if (current_S >> i) & 1]
            if not bits:
                break
            
            if move_type < 0.5:
                # Remove one element, then greedily fill
                x = random.choice(bits)
                new_S = current_S & ~(1 << (x-1))
                loss = loss_3aps(current_S, x)
                new_score = current_score - loss
                
                # Greedy fill
                new_S, new_score = greedy_fill(new_S, max_time=time_limit)
                
            elif move_type < 0.8:
                # Swap: remove x, add y
                x = random.choice(bits)
                new_S = current_S & ~(1 << (x-1))
                loss = loss_3aps(current_S, x)
                
                # Try to find a good y to add
                candidates = [i for i in range(1, N+1) if not ((new_S >> (i-1)) & 1)]
                if not candidates:
                    continue
                
                # Pick y with highest gain that is safe
                best_y = -1
                best_gain = -1
                for y in candidates:
                    if is_safe_add(new_S, y):
                        g = gain_3aps(new_S, y)
                        if g > best_gain:
                            best_gain = g
                            best_y = y
                
                if best_y == -1:
                    # No safe element to add, just do remove
                    new_S, new_score = greedy_fill(new_S, max_time=time_limit)
                else:
                    new_S |= (1 << (best_y-1))
                    new_score = current_score - loss + best_gain
                    # Further greedy fill
                    new_S, new_score = greedy_fill(new_S, max_time=time_limit)
            else:
                # Remove only (no fill) - explore sparse regions
                x = random.choice(bits)
                new_S = current_S & ~(1 << (x-1))
                loss = loss_3aps(current_S, x)
                new_score = current_score - loss
            
            # Compare
            if new_score > best_score:
                best_S = new_S
                best_score = new_score
                current_S = new_S
                current_score = new_score
            else:
                # Simulated Annealing acceptance
                delta = new_score - current_score
                if delta < 0:
                    prob = math.exp(delta / T)
                    if random.random() < prob:
                        current_S = new_S
                        current_score = new_score
                else:
                    current_S = new_S
                    current_score = new_score
            
            T *= alpha
            if T < 0.1:
                T = 0.1

        return best_S, best_score

    # Initialize with randomized greedy
    def generate_initial_set():
        S = 0
        elements = list(range(1, N + 1))
        random.shuffle(elements)
        for x in elements:
            if is_safe_add(S, x):
                S |= (1 << (x-1))
        return S

    # Run multiple restarts
    best_global_S = 0
    best_global_score = 0
    
    # Strategy: 8 quick restarts, then 1 long final run
    num_quick = 8
    quick_time_each = 5.0
    final_time = deadline - start_time - num_quick * quick_time_each
    
    for i in range(num_quick):
        if time.time() > deadline - 10.0:
            break
            
        S_init = generate_initial_set()
        
        slice_deadline = start_time + (i+1)*quick_time_each
        if slice_deadline > deadline - 10.0:
            slice_deadline = deadline - 10.0
        S_opt, score_opt = local_search(S_init, slice_deadline)
        
        if score_opt > best_global_score:
            best_global_score = score_opt
            best_global_S = S_opt
    
    # Final long run
    if time.time() < deadline - 5.0:
        S_init = generate_initial_set()
        S_opt, score_opt = local_search(S_init, deadline - 2.0)
        
        if score_opt > best_global_score:
            best_global_score = score_opt
            best_global_S = S_opt
    
    # Also try starting from best_global_S and do one more polish
    if time.time() < deadline - 5.0 and best_global_S != 0:
        S_opt, score_opt = local_search(best_global_S, deadline - 2.0)
        if score_opt > best_global_score:
            best_global_score = score_opt
            best_global_S = S_opt

    # Convert bitmask to list
    result = [i for i in range(1, N+1) if (best_global_S >> (i-1)) & 1]
    
    # Final safety check: Ensure no 4-APs
    # If somehow invalid, fallback to a simple greedy
    valid = True
    for a, b, c, d in four_aps:
        if (best_global_S >> (a-1)) & 1 and (best_global_S >> (b-1)) & 1 and \
           (best_global_S >> (c-1)) & 1 and (best_global_S >> (d-1)) & 1:
            valid = False
            break
            
    if not valid or not result:
        # Fallback: Simple greedy
        S = 0
        for x in range(1, N+1):
            if is_safe_add(S, x):
                S |= (1 << (x-1))
        result = [i for i in range(1, N+1) if (S >> (i-1)) & 1]

    return result
