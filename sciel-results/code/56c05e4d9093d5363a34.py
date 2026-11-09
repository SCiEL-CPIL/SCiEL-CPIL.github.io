import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 256
    
    # Precompute 4-APs
    # For each 4-AP (a, a+d, a+2d, a+3d), store the bitmask of the 4 elements.
    # Also, for each element x, store a list of 4-AP masks that contain x.
    four_ap_by_elem = [[] for _ in range(N)] # 0-indexed for bits
    
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            a0 = a - 1
            b0 = a + d - 1
            c0 = a + 2 * d - 1
            d0 = a + 3 * d - 1
            mask = (1 << a0) | (1 << b0) | (1 << c0) | (1 << d0)
            four_ap_by_elem[a0].append(mask)
            four_ap_by_elem[b0].append(mask)
            four_ap_by_elem[c0].append(mask)
            four_ap_by_elem[d0].append(mask)
            
    # Precompute 3-AP neighbors for fast delta calculation
    # three_ap_by_elem[x] contains list of (u, v) such that (x, u, v) form a 3-AP
    three_ap_by_elem = [[] for _ in range(N)]
    for d in range(1, (N - 1) // 2 + 1):
        max_a = N - 2 * d
        for a in range(1, max_a + 1):
            a0 = a - 1
            b0 = a + d - 1
            c0 = a + 2 * d - 1
            # a0, b0, c0 are 0-indexed
            three_ap_by_elem[a0].append((b0, c0))
            three_ap_by_elem[b0].append((a0, c0))
            three_ap_by_elem[c0].append((a0, b0))

    def can_add(state, x):
        # Check if adding x creates a 4-AP
        # Only 4-APs containing x are at risk.
        for mask in four_ap_by_elem[x]:
            # If all OTHER bits in mask are already in state, then adding x completes the 4-AP.
            # (state & mask) == (mask ^ (1 << x))
            if (state & mask) == (mask ^ (1 << x)):
                return False
        return True

    def count_3aps(state):
        count = 0
        # We can sum the contributions, but each 3-AP is counted 3 times (once for each element).
        # So count = sum(deg(x) for x in state) / 3
        # deg(x) is number of 3-APs containing x where the other 2 are in state.
        total = 0
        for x in range(N):
            if (state >> x) & 1:
                for u, v in three_ap_by_elem[x]:
                    if ((state >> u) & 1) and ((state >> v) & 1):
                        total += 1
        return total // 3

    def calc_delta_add(state, x):
        delta = 0
        for u, v in three_ap_by_elem[x]:
            if ((state >> u) & 1) and ((state >> v) & 1):
                delta += 1
        return delta

    def calc_delta_remove(state, x):
        # Number of 3-APs lost by removing x
        delta = 0
        for u, v in three_ap_by_elem[x]:
            if ((state >> u) & 1) and ((state >> v) & 1):
                delta += 1
        return delta

    # --- Phase 1: Best-First Greedy Construction ---
    # Start with empty set.
    # At each step, pick the element x not in state that maximizes calc_delta_add(state, x)
    # subject to can_add(state, x) being True.
    # Tie-breaking: Random or by index? Random to diversify restarts.
    
    best_state = 0
    best_score = 0
    
    num_restarts = 5
    
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        state = 0
        current_score = 0
        available = [i for i in range(N)]
        
        while available and time.time() < deadline - 5.0:
            best_x = -1
            best_gain = -1
            best_rand = -1.0
            
            # To speed up, we don't necessarily need to check ALL available elements every step if N is large,
            # but N=256 is small enough. 256 * 50 steps * 5 restarts = 64k checks. Fast.
            
            for x in available:
                if not can_add(state, x):
                    continue
                gain = calc_delta_add(state, x)
                # Add small random noise to break ties and allow exploration
                noise = random.random() * 0.5
                if gain > best_gain or (gain == best_gain and noise > best_rand):
                    best_gain = gain
                    best_rand = noise
                    best_x = x
            
            if best_x == -1:
                break
                
            # Commit to best_x
            state |= (1 << best_x)
            current_score += best_gain
            available.remove(best_x)
            
        if current_score > best_score:
            best_score = current_score
            best_state = state

    # --- Phase 2: Targeted Hill Climbing (Local Search) ---
    # We perform local search on the best state found.
    # Moves: Add, Remove, Swap.
    # Accept only if score improves (or equal with some probability? No, strict improvement for stability).
    
    current_state = best_state
    current_score = best_score
    
    # Precompute list of indices in state for faster sampling?
    # Or just iterate.
    
    improved = True
    max_local_iters = 100000
    iter_count = 0
    
    while improved and iter_count < max_local_iters and time.time() < deadline - 2.0:
        improved = False
        iter_count += 1
        
        # Try Add
        # Pick a random subset of candidates to avoid O(N^2) full scan every time?
        # With N=256, full scan is ~256 * 256 ops per pass if we check all adds.
        # Let's do a full scan for adds first, then swaps.
        
        # 1. Try to add any element that improves score
        for x in range(N):
            if (current_state >> x) & 1:
                continue
            if not can_add(current_state, x):
                continue
            delta = calc_delta_add(current_state, x)
            if delta > 0:
                current_state |= (1 << x)
                current_score += delta
                improved = True
                break # Restart scan after change? Or continue?
                # Continuing might be faster if we don't restart, but state changes.
                # Let's just break and restart the while loop to re-evaluate everything.
        
        if improved:
            continue
            
        # 2. Try to remove any element that improves score (i.e., reduces "waste")
        # Removing x loses delta_remove 3-APs. We want to remove if delta_remove < 0?
        # No, delta_remove is positive (number of APs lost).
        # Score change = -delta_remove.
        # We want -delta_remove > 0 => delta_remove < 0. Impossible.
        # So we never remove an element in a pure maximization context unless it allows
        # adding MORE elements later. This is handled by Swaps.
        
        # 3. Try Swaps
        # Pick x in state, y not in state.
        # Check if swap is valid and improves score.
        
        # To make this efficient, we can iterate x in state, then y not in state.
        # But that's O(N^2) checks. 256*256 = 65k checks.
        # Each check involves can_add logic and delta calc.
        # This might be slow in Python if done 1000 times.
        
        # Optimization: Only try a random subset of swaps per iteration.
        # Or, do a full sweep of swaps once per "epoch" of local search.
        
        # Let's do a full sweep of swaps.
        state_indices = [i for i in range(N) if (current_state >> i) & 1]
        non_state_indices = [i for i in range(N) if not (current_state >> i) & 1]
        
        # Randomize order to avoid bias
        random.shuffle(state_indices)
        random.shuffle(non_state_indices)
        
        for x in state_indices:
            if time.time() > deadline - 2.0:
                break
            for y in non_state_indices:
                if time.time() > deadline - 2.0:
                    break
                
                # Check validity of swap: remove x, add y
                # Valid if y can be added to state - {x}
                valid = True
                for mask in four_ap_by_elem[y]:
                    other_mask = mask ^ (1 << y)
                    if (current_state & other_mask) == other_mask:
                        # All other 3 are in current_state.
                        # If x is NOT in other_mask, then they are in state-{x} too.
                        if not (other_mask & (1 << x)):
                            valid = False
                            break
                
                if not valid:
                    continue
                
                # Calculate delta
                # Loss: 3-APs containing x in current_state
                loss = 0
                for u, v in three_ap_by_elem[x]:
                    if ((current_state >> u) & 1) and ((current_state >> v) & 1):
                        loss += 1
                        
                # Gain: 3-APs containing y in state - {x}
                gain = 0
                state_minus_x = current_state & ~(1 << x)
                for u, v in three_ap_by_elem[y]:
                    if ((state_minus_x >> u) & 1) and ((state_minus_x >> v) & 1):
                        gain += 1
                        
                delta = gain - loss
                if delta > 0:
                    # Apply swap
                    current_state = (current_state & ~(1 << x)) | (1 << y)
                    current_score += delta
                    improved = True
                    # Update indices? 
                    # x is now not in state, y is now in state.
                    # For simplicity, break out of loops and restart the outer while loop.
                    break
            
            if improved:
                break
        
        # If we finished a full sweep without improvement, we are at a local optimum.
        # To escape, we could do a "kick" (random swap regardless of delta) and re-optimize.
        # Given time constraints, we'll stop if no improvement in a full sweep.
        
    # Final polish: Add any remaining beneficial elements
    changed = True
    while changed and time.time() < deadline - 0.5:
        changed = False
        for x in range(N):
            if (current_state >> x) & 1:
                continue
            if not can_add(current_state, x):
                continue
            delta = calc_delta_add(current_state, x)
            if delta > 0:
                current_state |= (1 << x)
                current_score += delta
                changed = True

    # Convert to list of 1-based integers
    result = []
    for i in range(N):
        if (current_state >> i) & 1:
            result.append(i + 1)
            
    return sorted(result)

# Ensure math is imported if used (it's not strictly needed here but good practice)
# import math 

# The function construct is defined above.
# No global state is modified.
