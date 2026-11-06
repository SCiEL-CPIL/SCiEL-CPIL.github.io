import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Try to find a valid coloring for n=34 down to n=1
    # The known lower bound is 34.
    
    for n in range(34, 0, -1):
        if time.time() > deadline - 15:
            break
        success, colors = find_valid_coloring(n, deadline)
        if success:
            return {'n': n, 'colors': colors}
            
    # Fallback: return a valid small coloring if we ran out of time
    # n=9 from seed is valid.
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}

def find_valid_coloring(n, deadline):
    # Precompute structures for fast energy calculation
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    triple_index = {t: i for i, t in enumerate(triples)}
    
    # Precompute K4s and K5s as lists of triple indices
    k4s = []
    for quad in combinations(range(n), 4):
        idxs = [triple_index[t] for t in combinations(quad, 3)]
        k4s.append(idxs)
        
    k5s = []
    for quint in combinations(range(n), 5):
        idxs = [triple_index[t] for t in combinations(quint, 3)]
        k5s.append(idxs)
    
    num_k4s = len(k4s)
    num_k5s = len(k5s)
    
    # For each triple, precompute which K4s and K5s it belongs to
    triple_to_k4s = [[] for _ in range(num_triples)]
    triple_to_k5s = [[] for _ in range(num_triples)]
    
    for ki, idxs in enumerate(k4s):
        for idx in idxs:
            triple_to_k4s[idx].append(ki)
            
    for ki, idxs in enumerate(k5s):
        for idx in idxs:
            triple_to_k5s[idx].append(ki)
    
    # Weighted penalty: use smooth surrogate to guide search
    # Penalty for K4: sum of (4 - red_count)^2 when close to violation
    # Penalty for K5: sum of (10 - blue_count)^2 when close to violation
    # This creates a gradient that guides away from violations
    
    best_energy = float('inf')
    best_colors = None
    best_soft = float('inf')
    
    def compute_soft_energy(k4_red_count, k5_blue_count):
        """Compute smooth surrogate energy"""
        soft = 0.0
        for c in k4_red_count:
            # Penalize being close to 4 reds
            if c >= 2:
                soft += (4 - c) ** 2
        for c in k5_blue_count:
            # Penalize being close to 10 blues
            if c >= 7:
                soft += (10 - c) ** 2
        return soft
    
    def hard_energy(k4_red_count, k5_blue_count):
        """Compute hard constraint violations"""
        return sum(1 for c in k4_red_count if c == 4) + sum(1 for c in k5_blue_count if c == 10)
    
    # Multiple restarts with different strategies
    max_restarts = 50
    restart_count = 0
    
    while time.time() < deadline - 5 and restart_count < max_restarts:
        restart_count += 1
        
        # Different initialization strategies
        strategy = restart_count % 3
        
        if strategy == 0:
            # Random initialization
            colors = [random.randint(0, 1) for _ in range(num_triples)]
        elif strategy == 1:
            # Bias toward blue for K5 avoidance (more blue means fewer all-blue K5s)
            # Actually, we want to avoid all-blue K5s, so bias toward red
            # But we also need to avoid all-red K4s
            # Use balanced initialization with slight red bias
            colors = [0 if random.random() < 0.45 else 1 for _ in range(num_triples)]
        else:
            # Structured initialization based on vertex parity
            colors = []
            for a, b, c in triples:
                # Use a pattern that avoids monochromatic cliques
                if (a + b + c) % 3 == 0:
                    colors.append(0)
                elif (a + b + c) % 3 == 1:
                    colors.append(1)
                else:
                    colors.append(random.randint(0, 1))
        
        # Compute initial energy and red/blue counts per K4 and K5
        k4_red_count = [0] * num_k4s
        k5_blue_count = [0] * num_k5s
        
        for ki, idxs in enumerate(k4s):
            for idx in idxs:
                if colors[idx] == 0:
                    k4_red_count[ki] += 1
                    
        for ki, idxs in enumerate(k5s):
            for idx in idxs:
                if colors[idx] == 1:
                    k5_blue_count[ki] += 1
        
        # Initial hard and soft energy
        energy = hard_energy(k4_red_count, k5_blue_count)
        soft = compute_soft_energy(k4_red_count, k5_blue_count)
        
        if energy == 0:
            return True, colors
            
        if energy < best_energy or (energy == best_energy and soft < best_soft):
            best_energy = energy
            best_soft = soft
            best_colors = colors[:]
        
        # Simulated annealing with both hard and soft objectives
        # Use soft energy for acceptance, hard energy for tracking
        temperature = 3.0
        cooling = 0.9998
        no_improvement = 0
        max_no_improvement = 8000
        
        while no_improvement < max_no_improvement and time.time() < deadline - 5:
            # Pick random triple to flip
            i = random.randint(0, num_triples - 1)
            old_color = colors[i]
            new_color = 1 - old_color
            
            # Calculate delta for hard and soft energy
            delta_hard = 0
            delta_soft = 0.0
            
            # Update K4s containing triple i
            for ki in triple_to_k4s[i]:
                old_red = k4_red_count[ki]
                if old_color == 0:
                    # Was red, now blue: red count decreases
                    new_red = old_red - 1
                    # Hard energy change
                    if old_red == 4:
                        delta_hard += 1
                    if new_red == 4:
                        delta_hard -= 1
                    # Soft energy change
                    if old_red >= 2:
                        delta_soft += (4 - new_red) ** 2 - (4 - old_red) ** 2
                    elif new_red >= 2:
                        delta_soft += (4 - new_red) ** 2
                    k4_red_count[ki] = new_red
                else:
                    # Was blue, now red: red count increases
                    new_red = old_red + 1
                    # Hard energy change
                    if old_red == 4:
                        delta_hard -= 1
                    if new_red == 4:
                        delta_hard += 1
                    # Soft energy change
                    if old_red >= 2:
                        delta_soft += (4 - new_red) ** 2 - (4 - old_red) ** 2
                    elif new_red >= 2:
                        delta_soft += (4 - new_red) ** 2
                    k4_red_count[ki] = new_red
                    
            # Update K5s containing triple i
            for ki in triple_to_k5s[i]:
                old_blue = k5_blue_count[ki]
                if old_color == 1:
                    # Was blue, now red: blue count decreases
                    new_blue = old_blue - 1
                    # Hard energy change
                    if old_blue == 10:
                        delta_hard += 1
                    if new_blue == 10:
                        delta_hard -= 1
                    # Soft energy change
                    if old_blue >= 7:
                        delta_soft += (10 - new_blue) ** 2 - (10 - old_blue) ** 2
                    elif new_blue >= 7:
                        delta_soft += (10 - new_blue) ** 2
                    k5_blue_count[ki] = new_blue
                else:
                    # Was red, now blue: blue count increases
                    new_blue = old_blue + 1
                    # Hard energy change
                    if old_blue == 10:
                        delta_hard -= 1
                    if new_blue == 10:
                        delta_hard += 1
                    # Soft energy change
                    if old_blue >= 7:
                        delta_soft += (10 - new_blue) ** 2 - (10 - old_blue) ** 2
                    elif new_blue >= 7:
                        delta_soft += (10 - new_blue) ** 2
                    k5_blue_count[ki] = new_blue
            
            energy += delta_hard
            soft += delta_soft
            
            if energy < 0:
                energy = 0
            if soft < 0:
                soft = 0
            
            # Accept based on combined objective: prioritize hard, then soft
            # Use soft energy for the acceptance decision
            if delta_soft <= 0 or random.random() < math.exp(-delta_soft / max(temperature, 0.001)):
                colors[i] = new_color
                if delta_hard < 0 or (delta_hard == 0 and delta_soft < 0):
                    no_improvement = 0
                    if energy < best_energy or (energy == best_energy and soft < best_soft):
                        best_energy = energy
                        best_soft = soft
                        best_colors = colors[:]
                else:
                    no_improvement += 1
            else:
                # Reject: revert the counts
                for ki in triple_to_k4s[i]:
                    if old_color == 0:
                        k4_red_count[ki] += 1
                    else:
                        k4_red_count[ki] -= 1
                for ki in triple_to_k5s[i]:
                    if old_color == 1:
                        k5_blue_count[ki] += 1
                    else:
                        k5_blue_count[ki] -= 1
                energy -= delta_hard
                soft -= delta_soft
                no_improvement += 1
            
            if energy == 0:
                return True, colors
                
            temperature *= cooling
        
        # After local search, check if we found a solution
        if energy == 0:
            return True, colors
            
    # Final polish: if we have a good solution, try to improve it further
    if best_colors is not None and best_energy > 0 and time.time() < deadline - 3:
        colors = best_colors[:]
        k4_red_count = [0] * num_k4s
        k5_blue_count = [0] * num_k5s
        
        for ki, idxs in enumerate(k4s):
            for idx in idxs:
                if colors[idx] == 0:
                    k4_red_count[ki] += 1
                    
        for ki, idxs in enumerate(k5s):
            for idx in idxs:
                if colors[idx] == 1:
                    k5_blue_count[ki] += 1
        
        energy = hard_energy(k4_red_count, k5_blue_count)
        soft = compute_soft_energy(k4_red_count, k5_blue_count)
        
        # Aggressive local search with low temperature
        temperature = 0.5
        cooling = 0.999
        no_improvement = 0
        max_no_improvement = 10000
        
        while no_improvement < max_no_improvement and time.time() < deadline - 3:
            # Target triples in violating structures
            # Find triples that are part of near-violations
            candidate_triples = []
            
            for ki, idxs in enumerate(k4s):
                if k4_red_count[ki] >= 3:
                    for idx in idxs:
                        if colors[idx] == 1:  # Blue triple in a mostly-red K4
                            candidate_triples.append(idx)
                            
            for ki, idxs in enumerate(k5s):
                if k5_blue_count[ki] >= 8:
                    for idx in idxs:
                        if colors[idx] == 0:  # Red triple in a mostly-blue K5
                            candidate_triples.append(idx)
            
            if not candidate_triples:
                i = random.randint(0, num_triples - 1)
            else:
                i = random.choice(candidate_triples)
                
            old_color = colors[i]
            new_color = 1 - old_color
            
            # Calculate delta for hard and soft energy
            delta_hard = 0
            delta_soft = 0.0
            
            # Update K4s containing triple i
            for ki in triple_to_k4s[i]:
                old_red = k4_red_count[ki]
                if old_color == 0:
                    new_red = old_red - 1
                    if old_red == 4:
                        delta_hard += 1
                    if new_red == 4:
                        delta_hard -= 1
                    if old_red >= 2:
                        delta_soft += (4 - new_red) ** 2 - (4 - old_red) ** 2
                    elif new_red >= 2:
                        delta_soft += (4 - new_red) ** 2
                    k4_red_count[ki] = new_red
                else:
                    new_red = old_red + 1
                    if old_red == 4:
                        delta_hard -= 1
                    if new_red == 4:
                        delta_hard += 1
                    if old_red >= 2:
                        delta_soft += (4 - new_red) ** 2 - (4 - old_red) ** 2
                    elif new_red >= 2:
                        delta_soft += (4 - new_red) ** 2
                    k4_red_count[ki] = new_red
                    
            # Update K5s containing triple i
            for ki in triple_to_k5s[i]:
                old_blue = k5_blue_count[ki]
                if old_color == 1:
                    new_blue = old_blue - 1
                    if old_blue == 10:
                        delta_hard += 1
                    if new_blue == 10:
                        delta_hard -= 1
                    if old_blue >= 7:
                        delta_soft += (10 - new_blue) ** 2 - (10 - old_blue) ** 2
                    elif new_blue >= 7:
                        delta_soft += (10 - new_blue) ** 2
                    k5_blue_count[ki] = new_blue
                else:
                    new_blue = old_blue + 1
                    if old_blue == 10:
                        delta_hard -= 1
                    if new_blue == 10:
                        delta_hard += 1
                    if old_blue >= 7:
                        delta_soft += (10 - new_blue) ** 2 - (10 - old_blue) ** 2
                    elif new_blue >= 7:
                        delta_soft += (10 - new_blue) ** 2
                    k5_blue_count[ki] = new_blue
            
            energy += delta_hard
            soft += delta_soft
            
            if energy < 0:
                energy = 0
            if soft < 0:
                soft = 0
            
            if delta_soft <= 0 or random.random() < math.exp(-delta_soft / max(temperature, 0.001)):
                colors[i] = new_color
                if delta_hard < 0 or (delta_hard == 0 and delta_soft < 0):
                    no_improvement = 0
                else:
                    no_improvement += 1
            else:
                for ki in triple_to_k4s[i]:
                    if old_color == 0:
                        k4_red_count[ki] += 1
                    else:
                        k4_red_count[ki] -= 1
                for ki in triple_to_k5s[i]:
                    if old_color == 1:
                        k5_blue_count[ki] += 1
                    else:
                        k5_blue_count[ki] -= 1
                energy -= delta_hard
                soft -= delta_soft
                no_improvement += 1
            
            if energy == 0:
                return True, colors
                
            temperature *= cooling
        
        if energy == 0:
            return True, colors
            
    # If we didn't find 0, return False
    return False, None
