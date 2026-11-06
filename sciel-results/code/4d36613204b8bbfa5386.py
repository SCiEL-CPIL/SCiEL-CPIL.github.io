import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin for 120s limit
    
    # The task asks to maximize n. The known bound is 34.
    # We will try to find a valid coloring for n=34.
    # If it fails, we try n=33, etc.
    # Given the difficulty, we focus heavily on n=34.
    
    # Precompute structures for n=34 once? No, n varies.
    # But we can structure the code to be efficient.
    
    for n in [34, 33, 32, 31, 30]:
        if time.time() > deadline - 5:
            break
        result = solve_sa(n, deadline)
        if result is not None:
            return result
            
    # Fallback: return a small valid coloring
    # e.g. n=1 is always valid
    return {'n': 1, 'colors': []}

def solve_sa(n, deadline):
    # Precompute triples
    triples = list(itertools.combinations(range(n), 3))
    m = len(triples)
    
    # Map triple to index
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    # Precompute 4-sets and their triples
    four_sets = list(itertools.combinations(range(n), 4))
    four_set_triples = []
    for s in four_sets:
        trs = [triple_to_idx[t] for t in itertools.combinations(s, 3)]
        four_set_triples.append(trs)
        
    # Precompute 5-sets and their triples
    five_sets = list(itertools.combinations(range(n), 5))
    five_set_triples = []
    for s in five_sets:
        trs = [triple_to_idx[t] for t in itertools.combinations(s, 3)]
        five_set_triples.append(trs)
        
    # For each triple, which 4-sets and 5-sets does it belong to?
    # This allows incremental updates.
    triple_to_four_sets = [[] for _ in range(m)]
    triple_to_five_sets = [[] for _ in range(m)]
    
    for i, trs in enumerate(four_set_triples):
        for t_idx in trs:
            triple_to_four_sets[t_idx].append(i)
            
    for j, trs in enumerate(five_set_triples):
        for t_idx in trs:
            triple_to_five_sets[t_idx].append(j)
            
    # We will use Simulated Annealing
    # State: colors array (0 or 1)
    # Cost: Number of red K4 + Number of blue K5
    # We want Cost == 0.
    
    # Initialize
    colors = np.random.randint(0, 2, m)
    
    # Initialize counts
    # red_count_4[i] = number of red triples in 4-set i
    red_count_4 = np.zeros(len(four_sets), dtype=np.int8)
    # blue_count_5[j] = number of blue triples in 5-set j
    blue_count_5 = np.zeros(len(five_sets), dtype=np.int8)
    
    # Initial calculation
    for i, trs in enumerate(four_set_triples):
        red_count_4[i] = np.sum(colors[trs] == 0)
        
    for j, trs in enumerate(five_set_triples):
        blue_count_5[j] = np.sum(colors[trs] == 1)
        
    # Calculate initial cost
    # Cost = sum(1 for c in red_count_4 if c==4) + sum(1 for c in blue_count_5 if c==10)
    # We can maintain this incrementally.
    cost = np.sum(red_count_4 == 4) + np.sum(blue_count_5 == 10)
    
    if cost == 0:
        return {'n': n, 'colors': colors.tolist()}
        
    # SA Parameters
    T = 10.0
    T_min = 0.1
    cooling = 0.999
    max_steps = 500000
    step = 0
    
    best_cost = cost
    best_colors = colors.copy()
    
    while time.time() < deadline - 1:
        # Pick a random triple to flip
        t_idx = random.randint(0, m - 1)
        
        old_color = colors[t_idx]
        new_color = 1 - old_color
        
        # Calculate Delta Cost
        # Flipping t_idx from old to new.
        # If old==0 (Red) -> new==1 (Blue):
        #   Red count in 4-sets containing t_idx decreases by 1.
        #   Blue count in 5-sets containing t_idx increases by 1.
        # If old==1 (Blue) -> new==0 (Red):
        #   Red count in 4-sets containing t_idx increases by 1.
        #   Blue count in 5-sets containing t_idx decreases by 1.
        
        delta_cost = 0
        
        # Update 4-sets
        for i in triple_to_four_sets[t_idx]:
            if old_color == 0:
                # Red -> Blue. Red count decreases.
                old_rc = red_count_4[i]
                new_rc = old_rc - 1
                # If old_rc was 4, cost decreases by 1.
                # If new_rc becomes 4, cost increases by 1.
                if old_rc == 4:
                    delta_cost -= 1
                if new_rc == 4:
                    delta_cost += 1
                red_count_4[i] = new_rc
            else:
                # Blue -> Red. Red count increases.
                old_rc = red_count_4[i]
                new_rc = old_rc + 1
                if old_rc == 4:
                    delta_cost -= 1
                if new_rc == 4:
                    delta_cost += 1
                red_count_4[i] = new_rc
                
        # Update 5-sets
        for j in triple_to_five_sets[t_idx]:
            if old_color == 1:
                # Blue -> Red. Blue count decreases.
                old_bc = blue_count_5[j]
                new_bc = old_bc - 1
                if old_bc == 10:
                    delta_cost -= 1
                if new_bc == 10:
                    delta_cost += 1
                blue_count_5[j] = new_bc
            else:
                # Red -> Blue. Blue count increases.
                old_bc = blue_count_5[j]
                new_bc = old_bc + 1
                if old_bc == 10:
                    delta_cost -= 1
                if new_bc == 10:
                    delta_cost += 1
                blue_count_5[j] = new_bc
                
        cost += delta_cost
        
        # Acceptance criterion
        if delta_cost <= 0:
            # Always accept better or equal
            colors[t_idx] = new_color
        else:
            # Accept worse with probability exp(-delta_cost / T)
            if random.random() < np.exp(-delta_cost / T):
                colors[t_idx] = new_color
            else:
                # Reject: revert changes
                colors[t_idx] = old_color
                # Revert counts
                for i in triple_to_four_sets[t_idx]:
                    if old_color == 0:
                        red_count_4[i] += 1
                    else:
                        red_count_4[i] -= 1
                for j in triple_to_five_sets[t_idx]:
                    if old_color == 1:
                        blue_count_5[j] += 1
                    else:
                        blue_count_5[j] -= 1
                cost -= delta_cost
        
        if cost < best_cost:
            best_cost = cost
            best_colors = colors.copy()
            if best_cost == 0:
                return {'n': n, 'colors': best_colors.tolist()}
                
        # Cool down
        T *= cooling
        if T < T_min:
            # Restart with higher temperature if stuck? Or just stop?
            # Let's do a restart with a new random coloring if we haven't found a solution.
            if time.time() > deadline - 10:
                break
            T = 10.0
            colors = np.random.randint(0, 2, m)
            # Recalculate counts
            for i, trs in enumerate(four_set_triples):
                red_count_4[i] = np.sum(colors[trs] == 0)
            for j, trs in enumerate(five_set_triples):
                blue_count_5[j] = np.sum(colors[trs] == 1)
            cost = np.sum(red_count_4 == 4) + np.sum(blue_count_5 == 10)
            best_cost = cost
            best_colors = colors.copy()
            
        step += 1
        
    # If we didn't find 0 cost, return best found if it's 0? No, must be 0.
    # If best_cost is not 0, return None.
    if best_cost == 0:
        return {'n': n, 'colors': best_colors.tolist()}
        
    return None
