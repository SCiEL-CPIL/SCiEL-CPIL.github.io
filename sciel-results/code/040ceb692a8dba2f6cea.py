import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin (evaluator kills at 150s, but we want buffer)

    def solve_n(n):
        """
        Attempt to find a valid 2-coloring of triples on n vertices.
        Returns (colors_list, success_bool).
        """
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        k4_list = list(combinations(range(n), 4))
        k5_list = list(combinations(range(n), 5))
        
        num_k4 = len(k4_list)
        num_k5 = len(k5_list)
        
        # Mapping: triple index -> list of k4 indices containing it
        triple_to_k4s = [[] for _ in range(num_triples)]
        # Mapping: triple index -> list of k5 indices containing it
        triple_to_k5s = [[] for _ in range(num_triples)]
        
        # Precompute k4 triples for faster delta calc? No, just use the lists.
        # For n=34, num_triples=5984, num_k4=46376, num_k5=278256.
        
        for i, k4 in enumerate(k4_list):
            for j in range(4):
                # k4 is (a,b,c,d) sorted. The 4 triples are (a,b,c), (a,b,d), (a,c,d), (b,c,d)
                # Since k4 is sorted, k4[:j] + k4[j+1:] gives the triple
                t = tuple(k4[:j] + k4[j+1:])
                idx = triple_to_idx[t]
                triple_to_k4s[idx].append(i)
                
        for i, k5 in enumerate(k5_list):
            # k5 is (a,b,c,d,e) sorted. 10 triples.
            # Using combinations on range(5) is fast enough for setup
            for combo in combinations(range(5), 3):
                t = tuple(k5[j] for j in combo)
                idx = triple_to_idx[t]
                triple_to_k5s[idx].append(i)

        # Initial random assignment
        # We can bias towards blue or red?
        # Red K4 is size 4, Blue K5 is size 5.
        # Probability of random red K4 is (1/2)^4 = 1/16.
        # Probability of random blue K5 is (1/2)^10 = 1/1024.
        # So red is much more likely to be a violation.
        # We should start with more Blue edges to reduce red K4s?
        # Or just random. Let's try random first, maybe biased if needed.
        
        # State
        colors = np.random.randint(0, 2, size=num_triples).astype(np.int8)
        
        # Counts
        # red_count_k4[i] = number of red edges in K4 i
        # blue_count_k5[i] = number of blue edges in K5 i
        red_count_k4 = np.zeros(num_k4, dtype=np.int16)
        blue_count_k5 = np.zeros(num_k5, dtype=np.int16)
        
        # Initialize counts
        for t_idx in range(num_triples):
            if colors[t_idx] == 0: # Red
                for k4_idx in triple_to_k4s[t_idx]:
                    red_count_k4[k4_idx] += 1
            else: # Blue
                for k5_idx in triple_to_k5s[t_idx]:
                    blue_count_k5[k5_idx] += 1
                    
        # Initial violations
        red_viol = np.sum(red_count_k4 == 4)
        blue_viol = np.sum(blue_count_k5 == 10)
        total_viol = red_viol + blue_viol
        
        if total_viol == 0:
            return colors.tolist(), True

        # Simulated Annealing
        T = 10.0
        T_end = 0.01
        # We want to run for a specific amount of time
        # Let's use a fixed number of steps or time-based.
        # Time-based is safer.
        
        # Optimization: Pre-compute "neighborhood" lists? 
        # triple_to_k4s and triple_to_k5s are already lists.
        
        # To speed up, we can use a "delta" calculation that only looks at the specific K4s and K5s affected.
        
        # Let's run SA loops
        # We will restart if stuck.
        
        while time.time() < deadline:
            # If we found a solution, return it
            if total_viol == 0:
                return colors.tolist(), True
            
            # Pick random triple
            t_idx = random.randint(0, num_triples - 1)
            old_color = colors[t_idx]
            new_color = 1 - old_color
            
            delta_v = 0
            
            # Update Red K4 counts
            # If old was Red (0), it becomes Blue. Red count decreases.
            # If old was Blue (1), it becomes Red. Red count increases.
            
            if old_color == 0: # Red -> Blue
                # Red count in K4s decreases by 1
                for k4_idx in triple_to_k4s[t_idx]:
                    old_c = red_count_k4[k4_idx]
                    new_c = old_c - 1
                    red_count_k4[k4_idx] = new_c
                    if old_c == 4:
                        delta_v -= 1 # Removed a violation
                    # new_c == 4 is impossible since old_c was at least 1 (it contained the red edge)
                    # Wait, if old_c was 4, it was a violation. Now 3.
                    # If old_c was 3, now 2. No change in violation status.
            else: # Blue -> Red
                # Red count in K4s increases by 1
                for k4_idx in triple_to_k4s[t_idx]:
                    old_c = red_count_k4[k4_idx]
                    new_c = old_c + 1
                    red_count_k4[k4_idx] = new_c
                    if old_c == 3:
                        delta_v += 1 # Created a violation (3->4)
                    # If old_c was 4, it was violation. Now 5. Still "all red"? 
                    # No, K4 has exactly 4 triples. So count can be 0..4.
                    # So old_c cannot be 4 if we are adding? 
                    # Wait, if old_c was 4, that means all 4 were red. 
                    # But we are flipping one from Blue to Red. 
                    # So the edge was Blue. So it wasn't counted in red_count.
                    # So old_c was at most 3.
                    # So if old_c==3, new_c==4, violation created.
                    
            # Update Blue K5 counts
            if old_color == 1: # Blue -> Red
                # Blue count in K5s decreases by 1
                for k5_idx in triple_to_k5s[t_idx]:
                    old_c = blue_count_k5[k5_idx]
                    new_c = old_c - 1
                    blue_count_k5[k5_idx] = new_c
                    if old_c == 10:
                        delta_v -= 1 # Removed a violation
            else: # Red -> Blue
                # Blue count in K5s increases by 1
                for k5_idx in triple_to_k5s[t_idx]:
                    old_c = blue_count_k5[k5_idx]
                    new_c = old_c + 1
                    blue_count_k5[k5_idx] = new_c
                    if old_c == 9:
                        delta_v += 1 # Created a violation (9->10)
            
            # Acceptance criterion
            accepted = False
            if delta_v < 0:
                accepted = True
            elif delta_v == 0:
                # Accept neutral moves with some probability to explore
                if random.random() < 0.5:
                    accepted = True
            else:
                # delta_v > 0
                if T > T_end:
                    # exp(-delta / T)
                    # If delta is large and T is small, prob is 0
                    # Use log to avoid underflow? 
                    # log_prob = -delta_v / T
                    # if log_prob < -50, prob is ~0
                    if delta_v / T < 50:
                        prob = np.exp(-delta_v / T)
                        if random.random() < prob:
                            accepted = True
            
            if accepted:
                colors[t_idx] = new_color
                total_viol += delta_v
                # Cool down
                T *= 0.99995 # Slow cooling
                # Occasionally reheat?
            else:
                # Revert counts
                if old_color == 0:
                    for k4_idx in triple_to_k4s[t_idx]:
                        red_count_k4[k4_idx] += 1
                else:
                    for k4_idx in triple_to_k4s[t_idx]:
                        red_count_k4[k4_idx] -= 1
                if old_color == 1:
                    for k5_idx in triple_to_k5s[t_idx]:
                        blue_count_k5[k5_idx] += 1
                else:
                    for k5_idx in triple_to_k5s[t_idx]:
                        blue_count_k5[k5_idx] -= 1
            
            # Restart logic
            # If T is too low or stuck for too long, restart
            # Let's check time.
            # A simple restart every few seconds if no improvement?
            # We don't track improvement here easily without extra state.
            # Let's just restart if T < T_end * 10.
            if T < T_end * 10:
                # Restart with new random colors
                colors = np.random.randint(0, 2, size=num_triples).astype(np.int8)
                red_count_k4 = np.zeros(num_k4, dtype=np.int16)
                blue_count_k5 = np.zeros(num_k5, dtype=np.int16)
                for t_idx in range(num_triples):
                    if colors[t_idx] == 0:
                        for k4_idx in triple_to_k4s[t_idx]:
                            red_count_k4[k4_idx] += 1
                    else:
                        for k5_idx in triple_to_k5s[t_idx]:
                            blue_count_k5[k5_idx] += 1
                total_viol = np.sum(red_count_k4 == 4) + np.sum(blue_count_k5 == 10)
                T = 10.0
                if total_viol == 0:
                    return colors.tolist(), True

        return colors.tolist(), False

    # Try n=34 first
    # If it fails, try n=33, etc.
    # But n=34 is the target.
    
    # To increase chances, we can run multiple attempts within the time limit.
    # The solve_n function runs until deadline. 
    # We should modify it to take a time_budget or check a global deadline.
    
    # Let's restructure:
    # We have ~110s.
    # Let's try n=34.
    # If we don't find it, we might not have time for n=33?
    # Actually, if we don't find n=34, we should return a valid smaller n.
    # But the score is n. So 33 is better than 0.
    
    # However, the current solve_n runs until deadline.
    # We need to be careful not to exceed deadline.
    
    # Let's try n=34.
    colors, success = solve_n(34)
    if success:
        return {'n': 34, 'colors': colors}
    
    # If failed, try n=33?
    # We have little time left if solve_n ran for 110s.
    # Let's allocate time.
    # 100s for n=34, 10s for n=33?
    
    # Let's rewrite to accept a time limit.
    
    def solve_n_limited(n, time_limit):
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        k4_list = list(combinations(range(n), 4))
        k5_list = list(combinations(range(n), 5))
        
        num_k4 = len(k4_list)
        num_k5 = len(k5_list)
        
        triple_to_k4s = [[] for _ in range(num_triples)]
        triple_to_k5s = [[] for _ in range(num_triples)]
        
        for i, k4 in enumerate(k4_list):
            for j in range(4):
                t = tuple(k4[:j] + k4[j+1:])
                idx = triple_to_idx[t]
                triple_to_k4s[idx].append(i)
                
        for i, k5 in enumerate(k5_list):
            for combo in combinations(range(5), 3):
                t = tuple(k5[j] for j in combo)
                idx = triple_to_idx[t]
                triple_to_k5s[idx].append(i)

        colors = np.random.randint(0, 2, size=num_triples).astype(np.int8)
        red_count_k4 = np.zeros(num_k4, dtype=np.int16)
        blue_count_k5 = np.zeros(num_k5, dtype=np.int16)
        
        for t_idx in range(num_triples):
            if colors[t_idx] == 0:
                for k4_idx in triple_to_k4s[t_idx]:
                    red_count_k4[k4_idx] += 1
            else:
                for k5_idx in triple_to_k5s[t_idx]:
                    blue_count_k5[k5_idx] += 1
                    
        total_viol = np.sum(red_count_k4 == 4) + np.sum(blue_count_k5 == 10)
        
        if total_viol == 0:
            return colors.tolist(), True

        T = 10.0
        T_end = 0.01
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            if total_viol == 0:
                return colors.tolist(), True
            
            t_idx = random.randint(0, num_triples - 1)
            old_color = colors[t_idx]
            new_color = 1 - old_color
            
            delta_v = 0
            
            if old_color == 0:
                for k4_idx in triple_to_k4s[t_idx]:
                    old_c = red_count_k4[k4_idx]
                    new_c = old_c - 1
                    red_count_k4[k4_idx] = new_c
                    if old_c == 4:
                        delta_v -= 1
            else:
                for k4_idx in triple_to_k4s[t_idx]:
                    old_c = red_count_k4[k4_idx]
                    new_c = old_c + 1
                    red_count_k4[k4_idx] = new_c
                    if old_c == 3:
                        delta_v += 1
                    
            if old_color == 1:
                for k5_idx in triple_to_k5s[t_idx]:
                    old_c = blue_count_k5[k5_idx]
                    new_c = old_c - 1
                    blue_count_k5[k5_idx] = new_c
                    if old_c == 10:
                        delta_v -= 1
            else:
                for k5_idx in triple_to_k5s[t_idx]:
                    old_c = blue_count_k5[k5_idx]
                    new_c = old_c + 1
                    blue_count_k5[k5_idx] = new_c
                    if old_c == 9:
                        delta_v += 1
            
            accepted = False
            if delta_v < 0:
                accepted = True
            elif delta_v == 0:
                if random.random() < 0.5:
                    accepted = True
            else:
                if T > T_end:
                    if delta_v / T < 50:
                        prob = np.exp(-delta_v / T)
                        if random.random() < prob:
                            accepted = True
            
            if accepted:
                colors[t_idx] = new_color
                total_viol += delta_v
                T *= 0.99995
            else:
                if old_color == 0:
                    for k4_idx in triple_to_k4s[t_idx]:
                        red_count_k4[k4_idx] += 1
                else:
                    for k4_idx in triple_to_k4s[t_idx]:
                        red_count_k4[k4_idx] -= 1
                if old_color == 1:
                    for k5_idx in triple_to_k5s[t_idx]:
                        blue_count_k5[k5_idx] += 1
                else:
                    for k5_idx in triple_to_k5s[t_idx]:
                        blue_count_k5[k5_idx] -= 1
            
            if T < T_end * 10:
                colors = np.random.randint(0, 2, size=num_triples).astype(np.int8)
                red_count_k4 = np.zeros(num_k4, dtype=np.int16)
                blue_count_k5 = np.zeros(num_k5, dtype=np.int16)
                for t_idx in range(num_triples):
                    if colors[t_idx] == 0:
                        for k4_idx in triple_to_k4s[t_idx]:
                            red_count_k4[k4_idx] += 1
                    else:
                        for k5_idx in triple_to_k5s[t_idx]:
                            blue_count_k5[k5_idx] += 1
                total_viol = np.sum(red_count_k4 == 4) + np.sum(blue_count_k5 == 10)
                T = 10.0
                if total_viol == 0:
                    return colors.tolist(), True

        return colors.tolist(), False

    # Try n=34 with most of the time
    time_left = deadline - time.time()
    if time_left > 10:
        colors, success = solve_n_limited(34, time_left - 5)
        if success:
            return {'n': 34, 'colors': colors}
    
    # Try n=33
    time_left = deadline - time.time()
    if time_left > 5:
        colors, success = solve_n_limited(33, time_left - 2)
        if success:
            return {'n': 33, 'colors': colors}
            
    # Fallback: trivial valid coloring
    # n=9 is safe?
    # We need to ensure the fallback is valid.
    # The algebraic one: 0 if (a+b+c)%3==0 else 1.
    # Let's verify n=9 validity quickly in head?
    # It's a known construction for small n.
    # But to be absolutely safe, we can just return n=1 or n=2.
    # n=1: no triples. Valid. Score 1.
    # n=2: no triples. Valid. Score 2.
    # n=3: 1 triple. Valid. Score 3.
    # n=4: 4 triples. If all red, K4 red violation. If 1 blue, 3 red. No K4 red. No K5. Valid.
    # So we can easily construct n=4.
    # Let's return n=4 with colors [1, 0, 0, 0] (one blue, three red).
    # Triples: (0,1,2), (0,1,3), (0,2,3), (1,2,3).
    # If we set (0,1,2) to Blue, others Red.
    # Red K4? Needs all 4 red. We have 1 blue. So no.
    # Blue K5? Needs 5 vertices. We have 4. So no.
    # So n=4 is always valid if we avoid monochromatic K4.
    
    # Even better, we can try to find a valid n=5, 6, 7, 8 quickly?
    # n=5: 10 triples.
    # We can just return n=4. It's better than 0.
    
    # But wait, if we failed n=34 and n=33, it means we spent all time.
    # So we have very little time left.
    # Just return n=4.
    
    n = 4
    # Triples: (0,1,2), (0,1,3), (0,2,3), (1,2,3)
    # Colors: 1, 0, 0, 0
    colors = [1, 0, 0, 0]
    return {'n': n, 'colors': colors}

# Entry point
if __name__ == "__main__":
    pass # Not needed, evaluator calls construct()
