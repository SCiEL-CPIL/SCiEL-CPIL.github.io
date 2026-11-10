import itertools
import random
import time
import math
import sys

def construct():
    # Time limit management
    start_time = time.time()
    time_limit = 110.0 # Safety margin
    
    # We will try to maximize n
    # Start with known good bound 34, try to push to 35
    
    def find_coloring(n, time_budget):
        """
        Finds a valid coloring for n vertices.
        Returns colors list if found, None otherwise.
        """
        if n < 4:
            # Trivial cases
            triples = list(itertools.combinations(range(n), 3))
            return [0] * len(triples)
            
        triples = list(itertools.combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Precompute 4-sets and 5-sets
        four_sets = list(itertools.combinations(range(n), 4))
        five_sets = list(itertools.combinations(range(n), 5))
        
        # Map triple -> list of indices of 4-sets containing it
        triple_to_fours = [[] for _ in range(num_triples)]
        for idx, quad in enumerate(four_sets):
            t1 = (quad[0], quad[1], quad[2])
            t2 = (quad[0], quad[1], quad[3])
            t3 = (quad[0], quad[2], quad[3])
            t4 = (quad[1], quad[2], quad[3])
            triple_to_fours[triple_to_idx[t1]].append(idx)
            triple_to_fours[triple_to_idx[t2]].append(idx)
            triple_to_fours[triple_to_idx[t3]].append(idx)
            triple_to_fours[triple_to_idx[t4]].append(idx)
            
        # Map triple -> list of indices of 5-sets containing it
        triple_to_fives = [[] for _ in range(num_triples)]
        for idx, quint in enumerate(five_sets):
            for t in itertools.combinations(quint, 3):
                triple_to_fives[triple_to_idx[t]].append(idx)
                
        # Helper to compute initial counts and cost
        def init_state():
            colors = [random.randint(0, 1) for _ in range(num_triples)]
            red_counts = [0] * len(four_sets)
            blue_counts = [0] * len(five_sets)
            
            cost = 0
            for i, quad in enumerate(four_sets):
                t1 = triple_to_idx[(quad[0], quad[1], quad[2])]
                t2 = triple_to_idx[(quad[0], quad[1], quad[3])]
                t3 = triple_to_idx[(quad[0], quad[2], quad[3])]
                t4 = triple_to_idx[(quad[1], quad[2], quad[3])]
                rc = (1 if colors[t1]==0 else 0) + (1 if colors[t2]==0 else 0) + (1 if colors[t3]==0 else 0) + (1 if colors[t4]==0 else 0)
                red_counts[i] = rc
                if rc == 4:
                    cost += 1
                    
            for i, quint in enumerate(five_sets):
                bc = 0
                for t in itertools.combinations(quint, 3):
                    if colors[triple_to_idx[t]] == 1:
                        bc += 1
                blue_counts[i] = bc
                if bc == 10:
                    cost += 1
                    
            return colors, red_counts, blue_counts, cost

        # Local Search with Restarts
        best_cost_found = float('inf')
        best_colors = None
        
        while time.time() - start_time < time_limit - 5.0: # Leave 5s for return logic
            colors, red_counts, blue_counts, cost = init_state()
            
            if cost == 0:
                return colors
                
            if cost < best_cost_found:
                best_cost_found = cost
                best_colors = colors[:]
                
            # Simulated Annealing parameters
            T = 1.0
            cooling = 0.999
            min_T = 0.01
            no_improve_steps = 0
            max_no_improve = 10000 # Restart if stuck
            
            while no_improve_steps < max_no_improve and time.time() - start_time < time_limit - 5.0:
                if cost == 0:
                    return colors
                    
                t_idx = random.randint(0, num_triples - 1)
                current_color = colors[t_idx]
                
                delta_cost = 0
                
                # Update Red K4 violations
                for quad_idx in triple_to_fours[t_idx]:
                    rc = red_counts[quad_idx]
                    if current_color == 0: # Red -> Blue
                        # rc decreases by 1
                        if rc == 4:
                            delta_cost -= 1 # Violation removed
                        # rc becomes 3, no new violation
                    else: # Blue -> Red
                        # rc increases by 1
                        if rc == 3:
                            delta_cost += 1 # Violation added
                        # rc becomes 4, violation added? No, if rc was 3, adding 1 makes it 4.
                        # Wait, if rc was 3, it was NOT a violation. After flip, it IS. So +1.
                        # If rc was < 3, no change in violation status.
                        
                # Update Blue K5 violations
                for quint_idx in triple_to_fives[t_idx]:
                    bc = blue_counts[quint_idx]
                    if current_color == 1: # Blue -> Red
                        # bc decreases by 1
                        if bc == 10:
                            delta_cost -= 1 # Violation removed
                    else: # Red -> Blue
                        # bc increases by 1
                        if bc == 9:
                            delta_cost += 1 # Violation added
                        
                # Acceptance
                accept = False
                if delta_cost <= 0:
                    accept = True
                else:
                    if T > min_T:
                        prob = math.exp(-delta_cost / T)
                        if random.random() < prob:
                            accept = True
                            
                if accept:
                    # Apply flip
                    colors[t_idx] = 1 - current_color
                    cost += delta_cost
                    
                    # Update counts
                    for quad_idx in triple_to_fours[t_idx]:
                        if current_color == 0:
                            red_counts[quad_idx] -= 1
                        else:
                            red_counts[quad_idx] += 1
                            
                    for quint_idx in triple_to_fives[t_idx]:
                        if current_color == 1:
                            blue_counts[quint_idx] -= 1
                        else:
                            blue_counts[quint_idx] += 1
                            
                    if cost < best_cost_found:
                        best_cost_found = cost
                        best_colors = colors[:]
                        no_improve_steps = 0
                    else:
                        no_improve_steps += 1
                        
                    T *= cooling
                else:
                    no_improve_steps += 1
                    T *= cooling

        return best_colors if best_cost_found == 0 else None

    # Try n=34
    colors_34 = find_coloring(34, 60.0)
    if colors_34:
        # Try n=35 with remaining time
        remaining = time_limit - (time.time() - start_time)
        if remaining > 10.0:
            colors_35 = find_coloring(35, remaining - 2.0)
            if colors_35:
                return {'n': 35, 'colors': colors_35}
        return {'n': 34, 'colors': colors_34}
    else:
        # Fallback: try smaller n if 34 fails (unlikely)
        for n in range(33, 8, -1):
            colors_n = find_coloring(n, 5.0)
            if colors_n:
                return {'n': n, 'colors': colors_n}
        
        # Last resort: trivial n=9
        n = 9
        triples = list(itertools.combinations(range(n), 3))
        colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]
        return {'n': n, 'colors': colors}
