import time
import random
import math

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit

    # Precompute all 4-term arithmetic progressions (a, a+d, a+2d, a+3d) with 1 <= a <= 256
    # Indices are 0-based: a-1, a+d-1, a+2d-1, a+3d-1
    aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            i1 = a - 1
            i2 = a + d - 1
            i3 = a + 2 * d - 1
            i4 = a + 3 * d - 1
            aps.append((i1, i2, i3, i4))
    
    total_aps = len(aps)
    
    # Precompute for each index, which APs it belongs to
    # ap_list[i] is a list of (ap_id, other_indices_tuple)
    ap_list = [[] for _ in range(N)]
    for ap_id, ap in enumerate(aps):
        for idx in ap:
            others = tuple(j for j in ap if j != idx)
            ap_list[idx].append((ap_id, others))

    def count_mono(colors):
        count = 0
        for ap in aps:
            c = colors[ap[0]]
            if c == colors[ap[1]] and c == colors[ap[2]] and c == colors[ap[3]]:
                count += 1
        return count

    def get_delta_single_flip(colors, idx):
        """
        Calculate change in mono count if we flip colors[idx].
        """
        delta = 0
        c = colors[idx]
        for ap_id, others in ap_list[idx]:
            o0 = colors[others[0]]
            if o0 == colors[others[1]] and o0 == colors[others[2]]:
                if c == o0:
                    delta -= 1
                else:
                    delta += 1
        return delta

    def get_delta_block_flip(colors, start, end):
        """
        Calculate change in mono count if we reverse the block colors[start..end].
        Note: This is expensive to calculate exactly from scratch, but we can approximate 
        or calculate precisely. Given N=256, calculating the full count for a block flip 
        might be too slow if done naively for every step. 
        However, we can use a simpler heuristic: 
        Actually, let's stick to single flips for the core SA loop for speed, 
        but use a 'kick' mechanism that performs a random block flip and accepts 
        unconditionally or based on a high temperature, effectively restarting 
        the local search from a new point.
        
        Wait, the prompt suggests a structural change. 
        Let's implement a "2-opt" style move: swap two bits? No, that's just two flips.
        Let's implement a "segment reversal" as a distinct move type in SA.
        
        To make it efficient, we can precompute the contribution of each AP.
        Actually, for N=256, count_mono takes ~1-2ms. 
        If we do 10,000 steps, that's 10-20 seconds. We have 118s.
        So we can afford to calculate the exact delta for a block flip by 
        re-evaluating only the APs that intersect the flipped block?
        No, an AP can span the whole array.
        
        Alternative Strategy: 
        Use Single Bit Flips for SA, but add a "Perturbation" step every K iterations 
        that performs a random block flip and resets the temperature or accepts 
        the new state regardless of cost (hard restart).
        
        Let's refine: 
        1. Standard SA with single bit flips.
        2. Every 500 steps, perform a "Block Flip" (reverse random segment) and 
           accept it with a probability based on the cost difference, or just 
           accept it to escape deep local minima.
        """
        # For now, let's just use single flips in the main loop for speed,
        # but we will implement a "Kick" that does a random block flip and 
        # updates the state, accepting the new cost.
        pass

    # We will use a hybrid: 
    # 1. Simulated Annealing with single bit flips.
    # 2. Periodic "Kicks": Randomly reverse a block of length L (e.g., 10-50).
    #    This is a structural change from pure greedy.

    best_count = float('inf')
    best_colors = [0] * N
    
    # SA Parameters
    T_start = 5.0
    T_min = 0.01
    alpha = 0.9995  # Cooling rate
    # We want to run for as long as possible.
    # Estimate steps: 110 seconds / (time per step)
    # Time per step ~ 0.05 ms (delta calc is fast) + overhead. Let's say 100k steps is safe.
    
    colors = [random.randint(0, 1) for _ in range(N)]
    current_count = count_mono(colors)
    
    if current_count < best_count:
        best_count = current_count
        best_colors = colors[:]
        
    T = T_start
    step = 0
    
    # Precompute random indices for speed? No, just use random module.
    
    while time.time() < deadline:
        # Check if we should do a "Kick" (Block Flip)
        # Do a kick every 200 steps to escape local minima
        if step % 200 == 0:
            # Pick a random block to reverse
            block_len = random.randint(5, 30)
            if block_len < N:
                start_idx = random.randint(0, N - block_len)
                end_idx = start_idx + block_len - 1
                
                # Perform the flip
                # Reverse the slice
                colors[start_idx:end_idx+1] = colors[start_idx:end_idx+1][::-1]
                
                # Recalculate count (expensive, but infrequent)
                new_count = count_mono(colors)
                
                # Accept if better, or with some probability if worse (since T is high initially)
                delta = new_count - current_count
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_count = new_count
                    if current_count < best_count:
                        best_count = current_count
                        best_colors = colors[:]
                else:
                    # Revert
                    colors[start_idx:end_idx+1] = colors[start_idx:end_idx+1][::-1]
        
        # Standard SA Step: Single Bit Flip
        idx = random.randint(0, N - 1)
        delta = get_delta_single_flip(colors, idx)
        
        # Acceptance criterion
        if delta < 0 or random.random() < math.exp(-delta / T):
            colors[idx] = 1 - colors[idx]
            current_count += delta
            
            if current_count < best_count:
                best_count = current_count
                best_colors = colors[:]
        
        # Cool down
        T *= alpha
        if T < T_min:
            # Reheat or restart? 
            # Let's just reset T to a moderate level to continue searching
            T = T_start * 0.1
            
        step += 1
        
        # Early exit if we hit a very low count (unlikely to be 0, but <10 is good)
        if best_count <= 5:
            break

    return best_colors
