import time
import random
import math
from itertools import combinations

def construct():
    """
    Find a valid 2-colouring of triples on n vertices with no red K4^(3) and no blue K5^(3).
    Maximise n. The known construction has n=34.
    """
    start_time = time.time()
    deadline = start_time + 145  # 145s safety margin before 150s limit
    
    target_n = 34
    
    # Precompute all triples in lexicographic order
    triples = list(combinations(range(target_n), 3))
    num_triples = len(triples)
    
    # Map triple -> index
    triple_index = {}
    for i, t in enumerate(triples):
        triple_index[t] = i
    
    # For each 4-set, the 4 triples that form K4^(3)
    # For each 5-set, the 10 triples that form K5^(3)
    k4_triple_indices = []
    k5_triple_indices = []
    
    for quad in combinations(range(target_n), 4):
        a, b, c, d = quad
        idxs = [
            triple_index[(a, b, c)],
            triple_index[(a, b, d)],
            triple_index[(a, c, d)],
            triple_index[(b, c, d)]
        ]
        k4_triple_indices.append(idxs)
        
    for quint in combinations(range(target_n), 5):
        a, b, c, d, e = quint
        idxs = [
            triple_index[(a, b, c)],
            triple_index[(a, b, d)],
            triple_index[(a, b, e)],
            triple_index[(a, c, d)],
            triple_index[(a, c, e)],
            triple_index[(a, d, e)],
            triple_index[(b, c, d)],
            triple_index[(b, c, e)],
            triple_index[(b, d, e)],
            triple_index[(c, d, e)]
        ]
        k5_triple_indices.append(idxs)
    
    num_k4 = len(k4_triple_indices)
    num_k5 = len(k5_triple_indices)
    
    # Reverse maps: triple -> list of k4 indices, triple -> list of k5 indices
    triple_to_k4s = [[] for _ in range(num_triples)]
    for k4_i, idxs in enumerate(k4_triple_indices):
        for tri_idx in idxs:
            triple_to_k4s[tri_idx].append(k4_i)
            
    triple_to_k5s = [[] for _ in range(num_triples)]
    for k5_i, idxs in enumerate(k5_triple_indices):
        for tri_idx in idxs:
            triple_to_k5s[tri_idx].append(k5_i)

    def greedy_initialization():
        """Greedy sequential construction: assign colors in lexicographic order."""
        colors = [-1] * num_triples  # -1 means unassigned
        
        # For each triple, we need to know which k4s and k5s it belongs to
        # When assigning a color, check if it would create a violation
        
        red_k4_count = [0] * num_k4
        blue_k5_count = [0] * num_k5
        
        for tri_idx in range(num_triples):
            # Try red first, then blue
            # Check if assigning red would create a red K4
            can_be_red = True
            for k4_i in triple_to_k4s[tri_idx]:
                if red_k4_count[k4_i] == 3:
                    # This k4 already has 3 red edges, adding this would make it 4
                    can_be_red = False
                    break
            
            # Check if assigning blue would create a blue K5
            can_be_blue = True
            for k5_i in triple_to_k5s[tri_idx]:
                if blue_k5_count[k5_i] == 9:
                    # This k5 already has 9 blue edges, adding this would make it 10
                    can_be_blue = False
                    break
            
            if can_be_red and not can_be_blue:
                colors[tri_idx] = 0  # red
                for k4_i in triple_to_k4s[tri_idx]:
                    red_k4_count[k4_i] += 1
            elif can_be_blue and not can_be_red:
                colors[tri_idx] = 1  # blue
                for k5_i in triple_to_k5s[tri_idx]:
                    blue_k5_count[k5_i] += 1
            elif can_be_red and can_be_blue:
                # Both work, choose randomly with slight bias
                if random.random() < 0.5:
                    colors[tri_idx] = 0  # red
                    for k4_i in triple_to_k4s[tri_idx]:
                        red_k4_count[k4_i] += 1
                else:
                    colors[tri_idx] = 1  # blue
                    for k5_i in triple_to_k5s[tri_idx]:
                        blue_k5_count[k5_i] += 1
            else:
                # Neither works, choose the one that creates fewer future problems
                # Count how many k4s would be at risk if red
                red_risk = 0
                for k4_i in triple_to_k4s[tri_idx]:
                    if red_k4_count[k4_i] >= 2:
                        red_risk += 1
                
                blue_risk = 0
                for k5_i in triple_to_k5s[tri_idx]:
                    if blue_k5_count[k5_i] >= 7:
                        blue_risk += 1
                
                if red_risk <= blue_risk:
                    colors[tri_idx] = 0  # red
                    for k4_i in triple_to_k4s[tri_idx]:
                        red_k4_count[k4_i] += 1
                else:
                    colors[tri_idx] = 1  # blue
                    for k5_i in triple_to_k5s[tri_idx]:
                        blue_k5_count[k5_i] += 1
        
        return colors

    def count_violations(colors):
        """Count the number of violations."""
        red_k4_count = [0] * num_k4
        blue_k5_count = [0] * num_k5
        
        for tri_idx in range(num_triples):
            if colors[tri_idx] == 0:  # red
                for k4_i in triple_to_k4s[tri_idx]:
                    red_k4_count[k4_i] += 1
            else:  # blue
                for k5_i in triple_to_k5s[tri_idx]:
                    blue_k5_count[k5_i] += 1
        
        violations = sum(1 for c in red_k4_count if c == 4) + \
                     sum(1 for c in blue_k5_count if c == 10)
        return violations

    def run_local_search(colors, max_time):
        """Local search: flip triples involved in violations to try to reduce them."""
        red_k4_count = [0] * num_k4
        blue_k5_count = [0] * num_k5
        
        for tri_idx in range(num_triples):
            if colors[tri_idx] == 0:  # red
                for k4_i in triple_to_k4s[tri_idx]:
                    red_k4_count[k4_i] += 1
            else:  # blue
                for k5_i in triple_to_k5s[tri_idx]:
                    blue_k5_count[k5_i] += 1
        
        current_violations = sum(1 for c in red_k4_count if c == 4) + \
                             sum(1 for c in blue_k5_count if c == 10)
        
        if current_violations == 0:
            return colors[:]
        
        best_violations = current_violations
        best_colors = colors[:]
        
        T = 3.0
        T_min = 0.01
        cooling = 0.9999
        
        no_improve_steps = 0
        max_no_improve = 50000
        
        while time.time() < max_time and no_improve_steps < max_no_improve:
            # Find triples involved in violations
            candidate_triples = set()
            
            # Triples in red K4 violations
            for k4_i, count in enumerate(red_k4_count):
                if count == 4:
                    for tri_idx in k4_triple_indices[k4_i]:
                        candidate_triples.add(tri_idx)
            
            # Triples in blue K5 violations
            for k5_i, count in enumerate(blue_k5_count):
                if count == 10:
                    for tri_idx in k5_triple_indices[k5_i]:
                        candidate_triples.add(tri_idx)
            
            if not candidate_triples:
                # No violations, but let's double check
                if current_violations == 0:
                    return colors[:]
                # Pick random triple
                t = random.randrange(num_triples)
            else:
                t = random.choice(list(candidate_triples))
            
            old_color = colors[t]
            
            # Compute delta violation
            delta_v = 0
            
            if old_color == 0:  # red -> blue
                for k4_i in triple_to_k4s[t]:
                    if red_k4_count[k4_i] == 4:
                        delta_v -= 1
                for k5_i in triple_to_k5s[t]:
                    if blue_k5_count[k5_i] == 10:
                        delta_v -= 1
            else:  # blue -> red
                for k4_i in triple_to_k4s[t]:
                    if red_k4_count[k4_i] == 3:
                        delta_v += 1
                for k5_i in triple_to_k5s[t]:
                    if blue_k5_count[k5_i] == 9:
                        delta_v += 1
            
            # Acceptance decision
            accept = False
            if delta_v <= 0:
                accept = True
            else:
                if T > T_min:
                    if random.random() < math.exp(-delta_v / T):
                        accept = True
            
            if accept:
                # Apply the flip
                colors[t] = 1 - old_color
                if old_color == 0:  # was red, now blue
                    for k4_i in triple_to_k4s[t]:
                        red_k4_count[k4_i] -= 1
                    for k5_i in triple_to_k5s[t]:
                        blue_k5_count[k5_i] -= 1
                else:  # was blue, now red
                    for k4_i in triple_to_k4s[t]:
                        red_k4_count[k4_i] += 1
                    for k5_i in triple_to_k5s[t]:
                        blue_k5_count[k5_i] += 1
                
                current_violations += delta_v
                
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = colors[:]
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
                
                # Cool down
                T *= cooling
            else:
                no_improve_steps += 1
                T *= cooling
            
            # Early exit if solved
            if best_violations == 0:
                return best_colors[:]
                
        return best_colors if best_violations == 0 else None

    # Strategy: Try greedy initialization + local search
    num_attempts = 30
    time_per_attempt = 4.0  # 4 seconds per attempt, 120s total
    
    for i in range(num_attempts):
        if time.time() > deadline:
            break
        
        end_time = time.time() + time_per_attempt
        if end_time > deadline:
            end_time = deadline
            
        # Greedy initialization
        colors = greedy_initialization()
        
        # Check if already valid
        violations = count_violations(colors)
        if violations == 0:
            return {'n': target_n, 'colors': colors}
        
        # Local search
        result = run_local_search(colors, end_time)
        if result is not None:
            return {'n': target_n, 'colors': result}
        
        # If we have time, try a few more with different random seeds
        if time.time() > deadline - 5:
            break

    # Fallback: try smaller n values
    # n=13 is known to work with simple constructions
    for n in [33, 32, 31, 30, 25, 20, 15, 10, 9]:
        if time.time() > deadline - 2:
            break
        
        triples_n = list(combinations(range(n), 3))
        colors = [random.randint(0, 1) for _ in triples_n]
        
        # Quick check for small n
        if n <= 15:
            # Verify
            valid = True
            for quad in combinations(range(n), 4):
                a, b, c, d = quad
                red_count = 0
                for triple in [(a,b,c), (a,b,d), (a,c,d), (b,c,d)]:
                    idx = triple_index.get(triple, -1)
                    if idx >= 0 and colors[idx] == 0:
                        red_count += 1
                if red_count == 4:
                    valid = False
                    break
            
            if valid:
                for quint in combinations(range(n), 5):
                    a, b, c, d, e = quint
                    blue_count = 0
                    for triple in [(a,b,c), (a,b,d), (a,b,e), (a,c,d), (a,c,e), 
                                   (a,d,e), (b,c,d), (b,c,e), (b,d,e), (c,d,e)]:
                        idx = triple_index.get(triple, -1)
                        if idx >= 0 and colors[idx] == 1:
                            blue_count += 1
                    if blue_count == 10:
                        valid = False
                        break
            
            if valid:
                return {'n': n, 'colors': colors}

    # Last resort: n=9 with sum mod 3
    n = 9
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
    return {'n': n, 'colors': colors}
