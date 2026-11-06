import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # We aim for n=34, the known lower bound.
    # If we can't find 34, we fall back to smaller n.
    
    for n in range(34, 0, -1):
        if time.time() > deadline - 10:
            break
        success, colors = find_valid_coloring_walksat(n, deadline)
        if success:
            return {'n': n, 'colors': colors}
            
    # Fallback: return a valid small coloring if we ran out of time
    # n=9 from seed is valid.
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}

def find_valid_coloring_walksat(n, deadline):
    # Precompute structures
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    triple_index = {t: i for i, t in enumerate(triples)}
    
    # Precompute K4s and K5s
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
            
    # State variables
    # colors: 0 for red, 1 for blue
    # k4_red_count: number of red edges in each K4
    # k5_blue_count: number of blue edges in each K5
    
    # Heuristic Initialization:
    # Use a pattern that balances colors.
    # For R(4,5;3), a common base is related to Z_2^5 or similar.
    # Let's try a simple modular pattern first, then refine.
    # If (a+b+c) % 3 == 0, Red (0), else Blue (1).
    # This creates a 3-coloring of vertices effectively? No, it's a coloring of edges.
    
    def init_colors():
        colors = []
        for a, b, c in triples:
            # This pattern avoids small monochromatic structures in some contexts
            if (a + b + c) % 3 == 0:
                colors.append(0)
            else:
                colors.append(1)
        return colors

    def calculate_counts(colors):
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
        return k4_red_count, k5_blue_count

    def get_violations(k4_red_count, k5_blue_count):
        red_violations = [i for i, c in enumerate(k4_red_count) if c == 4]
        blue_violations = [i for i, c in enumerate(k5_blue_count) if c == 10]
        return red_violations, blue_violations

    def soft_energy(k4_red_count, k5_blue_count):
        # Penalty for being close to violation
        # K4: red count 3 is "1 away", 2 is "2 away"
        # K5: blue count 9 is "1 away", 8 is "2 away"
        e = 0
        for c in k4_red_count:
            if c >= 3:
                e += (4 - c) ** 2 # 1 for 3, 0 for 4 (but 4 is hard violation)
                # Actually, let's penalize distance to 4.
                # If c=3, dist=1. If c=2, dist=2.
                # We want to minimize distance.
                # So penalty = (4-c)^2 is good.
        for c in k5_blue_count:
            if c >= 8:
                e += (10 - c) ** 2
        return e

    best_energy = float('inf')
    best_colors = None
    
    max_restarts = 10
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        colors = init_colors()
        
        # Add some random noise to break symmetry
        if restart > 0:
            for i in range(num_triples):
                if random.random() < 0.1:
                    colors[i] = random.randint(0, 1)
                    
        k4_red_count, k5_blue_count = calculate_counts(colors)
        
        # WalkSAT Loop
        # Parameters
        walk_prob = 0.2 # Probability of taking a random walk step
        max_steps = 5000
        step = 0
        
        while step < max_steps and time.time() < deadline - 5:
            step += 1
            red_violations, blue_violations = get_violations(k4_red_count, k5_blue_count)
            
            if not red_violations and not blue_violations:
                return True, colors
                
            # Pick a violation to fix
            # Prioritize the one with the most "slack" or just random
            if red_violations and (not blue_violations or random.random() < 0.5):
                viol_idx = random.choice(red_violations)
                viol_type = 'red'
                clique_triples = k4s[viol_idx]
                # We need to flip a RED triple in this K4 to BLUE to break it
                # Candidates are triples in clique_triples that are currently RED
                candidates = [i for i in clique_triples if colors[i] == 0]
            else:
                viol_idx = random.choice(blue_violations)
                viol_type = 'blue'
                clique_triples = k5s[viol_idx]
                # We need to flip a BLUE triple in this K5 to RED to break it
                candidates = [i for i in clique_triples if colors[i] == 1]
                
            if not candidates:
                # Should not happen if violation is correctly identified
                break
                
            # Heuristic: Choose the candidate that minimizes the increase in soft energy
            # or maximizes the decrease in hard violations (if any)
            
            # If walk_prob is triggered, pick random
            if random.random() < walk_prob:
                flip_idx = random.choice(candidates)
            else:
                best_candidate = None
                best_score = float('inf')
                
                for idx in candidates:
                    # Simulate flip
                    old_color = colors[idx]
                    new_color = 1 - old_color
                    
                    # Calculate delta soft energy
                    delta_soft = 0.0
                    # Calculate delta hard violations (count of new violations - old violations)
                    delta_hard = 0
                    
                    # Update K4s
                    for ki in triple_to_k4s[idx]:
                        old_red = k4_red_count[ki]
                        if old_color == 0: # Red -> Blue
                            new_red = old_red - 1
                            # Hard: if old was 4, -1. if new is 4, +1 (impossible here)
                            if old_red == 4: delta_hard -= 1
                            if new_red == 4: delta_hard += 1
                            # Soft
                            if old_red >= 3:
                                delta_soft += (4 - new_red)**2 - (4 - old_red)**2
                            elif new_red >= 3:
                                delta_soft += (4 - new_red)**2
                        else: # Blue -> Red
                            new_red = old_red + 1
                            if old_red == 4: delta_hard -= 1
                            if new_red == 4: delta_hard += 1
                            if old_red >= 3:
                                delta_soft += (4 - new_red)**2 - (4 - old_red)**2
                            elif new_red >= 3:
                                delta_soft += (4 - new_red)**2
                                
                    # Update K5s
                    for ki in triple_to_k5s[idx]:
                        old_blue = k5_blue_count[ki]
                        if old_color == 1: # Blue -> Red
                            new_blue = old_blue - 1
                            if old_blue == 10: delta_hard -= 1
                            if new_blue == 10: delta_hard += 1
                            if old_blue >= 8:
                                delta_soft += (10 - new_blue)**2 - (10 - old_blue)**2
                            elif new_blue >= 8:
                                delta_soft += (10 - new_blue)**2
                        else: # Red -> Blue
                            new_blue = old_blue + 1
                            if old_blue == 10: delta_hard -= 1
                            if new_blue == 10: delta_hard += 1
                            if old_blue >= 8:
                                delta_soft += (10 - new_blue)**2 - (10 - old_blue)**2
                            elif new_blue >= 8:
                                delta_soft += (10 - new_blue)**2
                                
                    # We prefer flips that reduce hard violations.
                    # Since we picked a violation, flipping a member usually reduces hard violations by 1.
                    # But it might create new ones.
                    # Score: delta_hard * 100 + delta_soft
                    score = delta_hard * 100 + delta_soft
                    
                    if score < best_score:
                        best_score = score
                        best_candidate = idx
                        
                flip_idx = best_candidate
                
            # Apply flip
            old_color = colors[flip_idx]
            new_color = 1 - old_color
            colors[flip_idx] = new_color
            
            # Update counts
            for ki in triple_to_k4s[flip_idx]:
                if old_color == 0:
                    k4_red_count[ki] -= 1
                else:
                    k4_red_count[ki] += 1
                    
            for ki in triple_to_k5s[flip_idx]:
                if old_color == 1:
                    k5_blue_count[ki] -= 1
                else:
                    k5_blue_count[ki] += 1
                    
            # Track best
            red_v, blue_v = get_violations(k4_red_count, k5_blue_count)
            current_hard = len(red_v) + len(blue_v)
            current_soft = soft_energy(k4_red_count, k5_blue_count)
            total_score = current_hard * 1000 + current_soft
            
            if total_score < best_energy:
                best_energy = total_score
                best_colors = colors[:]
                
    if best_colors is not None:
        # Verify best_colors
        k4_red_count, k5_blue_count = calculate_counts(best_colors)
        red_v, blue_v = get_violations(k4_red_count, k5_blue_count)
        if not red_v and not blue_v:
            return True, best_colors
            
    return False, None
