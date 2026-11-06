import time
import random
import math

def construct():
    N = 256
    deadline = time.time() + 110.0  # Safety margin
    
    # 1. Precompute all 4-term APs (0-indexed)
    # a, a+d, a+2d, a+3d
    # a in [0, N-1], d >= 1, a+3d < N
    aps = []
    # For each d, a can range from 0 to N - 1 - 3d
    for d in range(1, N):
        max_a = N - 1 - 3 * d
        if max_a < 0:
            break
        for a in range(max_a + 1):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(aps)
    
    # Map index to list of AP indices containing it
    # This allows fast delta calculation
    aps_by_index = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            aps_by_index[pos].append(idx)
    
    # Helper to calculate total cost
    def calc_cost(x):
        count = 0
        for ap in aps:
            if x[ap[0]] == x[ap[1]] == x[ap[2]] == x[ap[3]]:
                count += 1
        return count

    # Helper to calculate delta cost if we flip index i
    def calc_delta(x, i):
        delta = 0
        for ap_idx in aps_by_index[i]:
            ap = aps[ap_idx]
            # Check if mono before
            # Note: x[i] is the current value
            b0 = x[ap[0]]
            b1 = x[ap[1]]
            b2 = x[ap[2]]
            b3 = x[ap[3]]
            
            was_mono = (b0 == b1 == b2 == b3)
            
            # Simulate flip of i
            # If i is ap[0], b0 flips. etc.
            # We can just check the new values
            # Let's create a temporary check
            # Actually, simpler: 
            # If was_mono, and we flip one bit, it becomes non-mono (delta -1)
            # If not was_mono, and we flip one bit, it might become mono (delta +1)
            
            if was_mono:
                # Flipping any bit in a mono AP makes it non-mono
                delta -= 1
            else:
                # Check if flipping i makes it mono
                # The other 3 bits must be equal to each other, 
                # and the flipped bit must equal them.
                # Let's find the other 3 positions
                others = [pos for pos in ap if pos != i]
                val_o1 = x[others[0]]
                val_o2 = x[others[1]]
                val_o3 = x[others[2]]
                
                if val_o1 == val_o2 == val_o3:
                    # The others are mono. 
                    # If the flipped bit becomes equal to them, it becomes mono.
                    # Current x[i] is NOT equal to them (since it wasn't mono).
                    # So flipping x[i] WILL make it equal to them.
                    delta += 1
        
        return delta

    # Local Search with Simulated Annealing
    def local_search(initial_x, max_time):
        x = initial_x[:]
        current_cost = calc_cost(x)
        best_x = x[:]
        best_cost = current_cost
        
        # Cooling schedule
        # Start high, end low
        # We don't know exact steps, so use time-based or fixed steps?
        # Fixed steps is easier to control.
        # N=256. Let's do many steps.
        # Estimate steps per second: ~100k-1M depending on overhead.
        # Let's aim for 200,000 steps per run, or until time runs out.
        
        start_time = time.time()
        
        # Simple linear cooling
        # T_start = 1.0, T_end = 0.01
        # But we don't know number of steps in advance if we break on time.
        # Let's use a fixed number of steps per "epoch" and cool down.
        
        # Alternatively, just use Hill Climbing with random restarts?
        # SA is better for escaping local minima.
        
        # Let's try a simple SA loop
        T = 1.0
        cooling_rate = 0.9999
        min_T = 0.01
        
        while time.time() < max_time:
            if T < min_T:
                break
            
            # Pick random index
            i = random.randint(0, N - 1)
            
            delta = calc_delta(x, i)
            
            # Acceptance criterion
            if delta < 0 or random.random() < math.exp(-delta / T):
                # Flip
                x[i] = 1 - x[i]
                current_cost += delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_x = x[:]
            
            T *= cooling_rate
            
            # Check time periodically to avoid long calc_delta if possible?
            # calc_delta is O(1) amortized if aps_by_index is small?
            # Max APs per index:
            # An index i can be a, a+d, a+2d, a+3d.
            # For a=1, d=1: (0,1,2,3). 0 is in 1 AP.
            # For middle elements, more APs.
            # Max APs for i is roughly O(N).
            # 256 elements. Max APs ~ 10000.
            # Average APs per index ~ 10000 * 4 / 256 = 156.
            # So calc_delta is O(156).
            # 100k steps/s is feasible.
            
        return best_x, best_cost

    # Initial Candidates
    candidates = []
    
    # 1. Thue-Morse
    def thue_morse(n):
        # parity of popcount
        return bin(n).count('1') % 2
    
    tm = [thue_morse(i) for i in range(N)]
    candidates.append(tm)
    
    # 2. Random initializations
    # We will generate a few random starts
    # But we need to be careful with time.
    # Let's do 5 random starts.
    for _ in range(5):
        candidates.append([random.randint(0, 1) for _ in range(N)])
    
    # 3. Alternating blocks of 2 (Seed)
    candidates.append([(i // 2) % 2 for i in range(N)])
    
    best_overall_x = None
    best_overall_cost = float('inf')
    
    # Run local search on each candidate
    # Allocate time evenly? Or just run until deadline?
    # Better to run each for a fixed portion of time, or just loop and break if time is up.
    # Let's give each candidate ~15 seconds.
    # Total 110s. 7 candidates * 15s = 105s.
    
    time_per_candidate = 15.0
    
    for cand in candidates:
        if time.time() > deadline - 5.0: # 5s buffer for final return
            break
        
        end_time = time.time() + time_per_candidate
        # Ensure we don't exceed global deadline
        end_time = min(end_time, deadline - 1.0)
        
        res_x, res_cost = local_search(cand, end_time)
        
        if res_cost < best_overall_cost:
            best_overall_cost = res_cost
            best_overall_x = res_x
    
    # If we still have time, do more random restarts or polish the best one?
    # If best_overall_x is not None, we can try to polish it further with the remaining time.
    if best_overall_x is not None and time.time() < deadline - 2.0:
        # Run local search again on the best found
        end_time = deadline - 1.0
        res_x, res_cost = local_search(best_overall_x, end_time)
        if res_cost < best_overall_cost:
            best_overall_cost = res_cost
            best_overall_x = res_x

    if best_overall_x is None:
        # Fallback to seed
        return [(i // 2) % 2 for i in range(N)]
        
    return best_overall_x

# EVOLVE-BLOCK-END
