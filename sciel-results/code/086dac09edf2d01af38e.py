import numpy as np
import time
from itertools import combinations
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin before 150s kill
    
    # We need to find a valid 2-coloring of triples on n vertices
    # such that no 4 vertices form a red K4^(3) (all 4 triples red)
    # and no 5 vertices form a blue K5^(3) (all 10 triples blue)
    #
    # The task says a valid coloring on n points certifies R(4,5;3) > n
    # and the published construction has 34 points.
    # So we want to maximize n, and the target is n=34 (or higher if possible).
    #
    # Strategy: Use constraint propagation / local search.
    # Start with n=34, try to find a valid coloring via simulated annealing
    # with careful initialization and urgency-based variable selection.
    
    def count_violations(n, colors):
        """Count red K4 violations and blue K5 violations."""
        # colors is a list indexed by triple (a,b,c) with a<b<c
        # We need a fast way to look up the color of any triple
        # Build a lookup: for a given set of 3 vertices, what's the color?
        # Use a dict or array indexed by (a*n + b) * n + c or similar
        
        # For efficiency, precompute all K4s and K5s
        red_k4_violations = 0
        blue_k5_violations = 0
        
        # Precompute color lookup as a 3D array or nested dict
        # Since n <= 40, we can use a numpy array of shape (n, n, n)
        color3d = np.zeros((n, n, n), dtype=np.uint8)
        idx = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    color3d[a, b, c] = colors[idx]
                    idx += 1
        
        # Count red K4s: 4 vertices where all 4 triples are red
        # C(40,4) = 91,390 - manageable
        # Count blue K5s: C(40,5) = 658,008 - also manageable but heavier
        
        # For speed, let's count only violations
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if (color3d[a,b,c] == 0 and color3d[a,b,d] == 0 and 
                color3d[a,c,d] == 0 and color3d[b,c,d] == 0):
                red_k4_violations += 1
        
        for quint in combinations(range(n), 5):
            a, b, c, d, e = quint
            # All 10 triples must be blue
            if (color3d[a,b,c] == 1 and color3d[a,b,d] == 1 and 
                color3d[a,b,e] == 1 and color3d[a,c,d] == 1 and
                color3d[a,c,e] == 1 and color3d[a,d,e] == 1 and
                color3d[b,c,d] == 1 and color3d[b,c,e] == 1 and
                color3d[b,d,e] == 1 and color3d[c,d,e] == 1):
                blue_k5_violations += 1
        
        return red_k4_violations, blue_k5_violations
    
    def count_violations_fast(n, colors, color3d):
        """Faster violation count using precomputed color3d."""
        red_k4_violations = 0
        blue_k5_violations = 0
        
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if (color3d[a,b,c] == 0 and color3d[a,b,d] == 0 and 
                color3d[a,c,d] == 0 and color3d[b,c,d] == 0):
                red_k4_violations += 1
            if time.time() > deadline:
                break
        
        if time.time() <= deadline:
            for quint in combinations(range(n), 5):
                a, b, c, d, e = quint
                if (color3d[a,b,c] == 1 and color3d[a,b,d] == 1 and 
                    color3d[a,b,e] == 1 and color3d[a,c,d] == 1 and
                    color3d[a,c,e] == 1 and color3d[a,d,e] == 1 and
                    color3d[b,c,d] == 1 and color3d[b,c,e] == 1 and
                    color3d[b,d,e] == 1 and color3d[c,d,e] == 1):
                    blue_k5_violations += 1
                if time.time() > deadline:
                    break
        
        return red_k4_violations, blue_k5_violations
    
    def build_color3d(n, colors):
        """Build 3D numpy array from flat colors list."""
        color3d = np.zeros((n, n, n), dtype=np.uint8)
        idx = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    color3d[a, b, c] = colors[idx]
                    idx += 1
        return color3d
    
    def initial_coloring(n, p_red):
        """Generate initial coloring with probability p_red for red."""
        num_triples = len(list(combinations(range(n), 3)))
        # More efficient:
        num_triples = n * (n-1) * (n-2) // 6
        colors = [1 if random.random() < (1 - p_red) else 0 for _ in range(num_triples)]
        return colors
    
    def simulated_annealing(n, max_iter, deadline, p_red_init=0.6):
        """Run simulated annealing to find valid coloring."""
        random.seed(42)
        colors = initial_coloring(n, p_red_init)
        num_triples = n * (n-1) * (n-2) // 6
        
        # Precompute for each triple, which K4s and K5s it belongs to
        # For urgency-based selection
        triple_to_k4s = {}
        triple_to_k5s = {}
        
        # For each K4, its 4 triples
        k4_triples = []
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            triples = [(a,b,c), (a,b,d), (a,c,d), (b,c,d)]
            k4_triples.append((quad, triples))
        
        k5_triples = []
        for quint in combinations(range(n), 5):
            a, b, c, d, e = quint
            triples = list(combinations((a,b,c,d,e), 3))
            k5_triples.append((quint, triples))
        
        # Map each triple to the K4s and K5s it belongs to
        for quad, triples in k4_triples:
            for t in triples:
                triple_to_k4s.setdefault(t, []).append(quad)
        
        for quint, triples in k5_triples:
            for t in triples:
                triple_to_k5s.setdefault(t, []).append(quint)
        
        color3d = build_color3d(n, colors)
        
        # Precompute triple indices
        triple_idx = {}
        idx = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    triple_idx[(a,b,c)] = idx
                    idx += 1
        
        # Get initial violation count
        r_viol, b_viol = count_violations_fast(n, colors, color3d)
        total_viol = r_viol + b_viol
        best_viol = total_viol
        best_colors = colors[:]
        
        if total_viol == 0:
            return colors, True
        
        temperature = 2.0
        cooling = 0.999
        min_temp = 0.01
        
        iter_count = 0
        while time.time() < deadline:
            if temperature < min_temp:
                break
            
            # Select a triple to flip with urgency-based bias
            # Urgency: number of near-complete monochromatic cliques
            # For simplicity, pick a random triple but weighted by violations
            # Actually, let's just pick a random triple for speed
            t_idx = random.randint(0, num_triples - 1)
            
            # Flip this triple
            old_color = colors[t_idx]
            new_color = 1 - old_color
            colors[t_idx] = new_color
            color3d_flat = colors  # We'll rebuild color3d if needed
            
            # Update color3d
            # Find the (a,b,c) for t_idx
            # We need to map index back to (a,b,c)
            # Precompute this mapping
            pass
        
        return best_colors, (best_viol == 0)
    
    # Precompute triple index to (a,b,c) mapping
    def build_triple_map(n):
        triple_map = []
        idx = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    triple_map.append((a, b, c))
                    idx += 1
        return triple_map
    
    def sa_with_urgency(n, deadline, p_red_init=0.6, seed=42):
        """Simulated annealing with urgency-based triple selection."""
        random.seed(seed)
        num_triples = n * (n-1) * (n-2) // 6
        triple_map = build_triple_map(n)
        
        colors = initial_coloring(n, p_red_init)
        color3d = build_color3d(n, colors)
        
        # Precompute K4 and K5 lists with their triples
        k4_list = []
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            triples_idx = [
                triple_map.index((a,b,c)),
                triple_map.index((a,b,d)),
                triple_map.index((a,c,d)),
                triple_map.index((b,c,d))
            ]
            k4_list.append((quad, triples_idx))
        
        k5_list = []
        for quint in combinations(range(n), 5):
            triples_idx = []
            for t in combinations(quint, 3):
                triples_idx.append(triple_map.index(t))
            k5_list.append((quint, triples_idx))
        
        # For each triple, list of K4s and K5s containing it
        triple_k4s = [[] for _ in range(num_triples)]
        for quad, t_idx_list in k4_list:
            for ti in t_idx_list:
                triple_k4s[ti].append(quad)
        
        triple_k5s = [[] for _ in range(num_triples)]
        for quint, t_idx_list in k5_list:
            for ti in t_idx_list:
                triple_k5s[ti].append(quint)
        
        # Compute current violations
        def compute_violations():
            r_v = 0
            for quad, t_idx_list in k4_list:
                if all(colors[ti] == 0 for ti in t_idx_list):
                    r_v += 1
            b_v = 0
            for quint, t_idx_list in k5_list:
                if all(colors[ti] == 1 for ti in t_idx_list):
                    b_v += 1
            return r_v, b_v
        
        r_viol, b_viol = compute_violations()
        total_viol = r_viol + b_viol
        best_viol = total_viol
        best_colors = colors[:]
        
        if total_viol == 0:
            return colors, True
        
        temperature = 3.0
        cooling = 0.9995
        min_temp = 0.001
        
        # Precompute for each K4: count of red edges
        k4_red_counts = [sum(1 for ti in t_idx_list if colors[ti] == 0) for _, t_idx_list in k4_list]
        k5_blue_counts = [sum(1 for ti in t_idx_list if colors[ti] == 1) for _, t_idx_list in k5_list]
        
        # Urgency for a triple: how many K4s are close to all-red (3 out of 4 red)
        # and how many K5s are close to all-blue (9 out of 10 blue)
        # When we flip a triple, we change these counts.
        
        while time.time() < deadline:
            if temperature < min_temp:
                break
            
            # Compute urgency scores for all triples
            # A triple is "urgent" if flipping it would reduce violations
            # For now, pick randomly but bias toward triples in violated cliques
            
            # Select triple: with some probability pick from violated K4s or K5s
            if r_viol > 0:
                # Pick a random violated K4, then pick one of its triples
                violated_k4s = [i for i, c in enumerate(k4_red_counts) if c == 4]
                if violated_k4s:
                    k4_idx = random.choice(violated_k4s)
                    t_idx = random.choice(k4_list[k4_idx][1])
                else:
                    t_idx = random.randint(0, num_triples - 1)
            elif b_viol > 0:
                violated_k5s = [i for i, c in enumerate(k5_blue_counts) if c == 10]
                if violated_k5s:
                    k5_idx = random.choice(violated_k5s)
                    t_idx = random.choice(k5_list[k5_idx][1])
                else:
                    t_idx = random.randint(0, num_triples - 1)
            else:
                t_idx = random.randint(0, num_triples - 1)
            
            # Flip the triple
            old_color = colors[t_idx]
            new_color = 1 - old_color
            
            # Update k4_red_counts and k5_blue_counts
            # For each K4 containing t_idx:
            for k4_idx in triple_k4s[t_idx]:
                if old_color == 0:
                    k4_red_counts[k4_idx] -= 1
                else:
                    k4_red_counts[k4_idx] += 1
            
            for k5_idx in triple_k5s[t_idx]:
                if old_color == 1:
                    k5_blue_counts[k5_idx] -= 1
                else:
                    k5_blue_counts[k5_idx] += 1
            
            colors[t_idx] = new_color
            color3d[triple_map[t_idx]] = new_color  # This doesn't work as expected
            
            # Compute new violations
            r_viol = sum(1 for c in k4_red_counts if c == 4)
            b_viol = sum(1 for c in k5_blue_counts if c == 10)
            new_total = r_viol + b_viol
            
            if new_total < total_viol:
                total_viol = new_total
                if total_viol < best_viol:
                    best_viol = total_viol
                    best_colors = colors[:]
                    if best_viol == 0:
                        return best_colors, True
            else:
                # Metropolis acceptance
                delta = new_total - total_viol
                if delta < 0:
                    total_viol = new_total
                    if total_viol < best_viol:
                        best_viol = total_viol
                        best_colors = colors[:]
                        if best_viol == 0:
                            return best_colors, True
                else:
                    accept_prob = np.exp(-delta / temperature) if temperature > 0 else 0
                    if random.random() < accept_prob:
                        total_viol = new_total
                    else:
                        # Reject: flip back
                        colors[t_idx] = old_color
                        for k4_idx in triple_k4s[t_idx]:
                            if new_color == 0:
                                k4_red_counts[k4_idx] -= 1
                            else:
                                k4_red_counts[k4_idx] += 1
                        for k5_idx in triple_k5s[t_idx]:
                            if new_color == 1:
                                k5_blue_counts[k5_idx] -= 1
                            else:
                                k5_blue_counts[k5_idx] += 1
                        # Restore original counts
                        for k4_idx in triple_k4s[t_idx]:
                            if old_color == 0:
                                k4_red_counts[k4_idx] -= 1
                            else:
                                k4_red_counts[k4_idx] += 1
                        for k5_idx in triple_k5s[t_idx]:
                            if old_color == 1:
                                k5_blue_counts[k5_idx] -= 1
                            else:
                                k5_blue_counts[k5_idx] += 1
            
            temperature *= cooling
        
        return best_colors, (best_viol == 0)
    
    # Try n=34 first, then smaller values
    for n in [34, 33, 32, 31, 30]:
        if time.time() > deadline:
            break
        
        # Try multiple random seeds
        for seed in range(20):
            if time.time() > deadline:
                break
            
            colors, valid = sa_with_urgency(n, deadline, p_red_init=0.55, seed=seed)
            if valid:
                return {'n': n, 'colors': colors}
        
        # If we've used too much time, break
        if time.time() > deadline - 5:
            break
    
    # Fallback: return a small valid coloring
    # n=3 is always valid (no K4 or K5 possible)
    n = 3
    colors = [0, 0, 0]  # All red, no K4^(3) possible with 3 vertices
    return {'n': n, 'colors': colors}
