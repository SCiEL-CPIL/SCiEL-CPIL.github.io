import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for overhead/return
    
    def prepare_data(n):
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # For each edge (triple), find which K4s and K5s it belongs to
        # A triple (a,b,c) is in K4 (a,b,c,d) for all d != a,b,c
        # A triple (a,b,c) is in K5 (a,b,c,d,e) for all pairs d<e, d,e not in {a,b,c}
        
        # Precompute for each edge index, the list of K4 indices and K5 indices it belongs to
        # But we need K4 and K5 to be enumerated first
        
        # Red K4s: all 4-subsets
        red_k4s = []
        for quad in combinations(range(n), 4):
            edges = []
            for tri in combinations(quad, 3):
                edges.append(triple_to_idx[tri])
            red_k4s.append(edges)
            
        # Blue K5s: all 5-subsets
        blue_k5s = []
        for quint in combinations(range(n), 5):
            edges = []
            for tri in combinations(quint, 3):
                edges.append(triple_to_idx[tri])
            blue_k5s.append(edges)
            
        # For each edge, which K4s and K5s does it belong to?
        edge_to_k4s = [[] for _ in range(num_triples)]
        edge_to_k5s = [[] for _ in range(num_triples)]
        
        for i, edges in enumerate(red_k4s):
            for e in edges:
                edge_to_k4s[e].append(i)
                
        for i, edges in enumerate(blue_k5s):
            for e in edges:
                edge_to_k5s[e].append(i)
        
        return triples, red_k4s, blue_k5s, edge_to_k4s, edge_to_k5s

    def greedy_construct(n, seed_val=0):
        """
        Greedy sequential construction: assign colors to triples in lex order.
        For each triple, choose the color that minimizes the number of 
        newly completed monochromatic cliques.
        """
        rng = random.Random(seed_val)
        triples, red_k4s, blue_k5s, edge_to_k4s, edge_to_k5s = prepare_data(n)
        num_triples = len(triples)
        
        colors = [-1] * num_triples  # -1 means uncolored
        
        # Track counts of red edges in each K4, blue edges in each K5
        red_counts = [0] * len(red_k4s)
        blue_counts = [0] * len(blue_k5s)
        
        # For efficiency, track for each edge how many K4s/K5s it's in
        # When we color an edge, we update counts for all K4s/K5s containing it
        
        # But the key insight: when coloring edge e, we only care about K4s/K5s
        # where ALL OTHER edges are already colored (i.e., the clique is "complete" 
        # except for this edge).
        
        # Let's precompute: for each edge e, which K4s have all other 3 edges already 
        # assigned when we get to e? This is complex because it depends on order.
        
        # Simpler approach: when coloring edge e, for each K4 containing e, check if
        # the other 3 edges are colored. If so, count how many are red. If 3 red and 
        # we color e red, we complete a red K4. Similarly for blue K5s.
        
        # To make this efficient, let's precompute for each edge e, the list of (k4_idx, other_edges)
        # and (k5_idx, other_edges)
        
        # Actually, let's just do it naively first and see if it's fast enough.
        
        for e_idx in range(num_triples):
            a, b, c = triples[e_idx]
            
            # Try color 0 (red) and color 1 (blue)
            best_color = rng.choice([0, 1])
            best_score = float('inf')
            
            for color in [0, 1]:
                # Count how many new monochromatic cliques would be formed
                new_red_k4 = 0
                new_blue_k5 = 0
                
                # Check all K4s containing this edge
                for k4_idx in edge_to_k4s[e_idx]:
                    edges = red_k4s[k4_idx]
                    # Check if all other edges are colored
                    all_colored = True
                    red_count = 0
                    for ee in edges:
                        if ee == e_idx:
                            continue
                        if colors[ee] == -1:
                            all_colored = False
                            break
                        if colors[ee] == 0:
                            red_count += 1
                    if all_colored:
                        if color == 0 and red_count == 3:
                            new_red_k4 += 1
                
                # Check all K5s containing this edge
                for k5_idx in edge_to_k5s[e_idx]:
                    edges = blue_k5s[k5_idx]
                    all_colored = True
                    blue_count = 0
                    for ee in edges:
                        if ee == e_idx:
                            continue
                        if colors[ee] == -1:
                            all_colored = False
                            break
                        if colors[ee] == 1:
                            blue_count += 1
                    if all_colored:
                        if color == 1 and blue_count == 9:
                            new_blue_k5 += 1
                
                # Weight: red K4 violations are more critical (4 edges vs 10)
                # But both are equally bad. Let's use equal weight.
                score = new_red_k4 + new_blue_k5
                
                if score < best_score:
                    best_score = score
                    best_color = color
                elif score == best_score:
                    # Tie-break randomly
                    if rng.random() < 0.5:
                        best_color = color
            
            colors[e_idx] = best_color
            
            # Update counts
            if best_color == 0:
                for k4_idx in edge_to_k4s[e_idx]:
                    red_counts[k4_idx] += 1
            else:
                for k5_idx in edge_to_k5s[e_idx]:
                    blue_counts[k5_idx] += 1
        
        # Verify no violations
        red_violations = 0
        blue_violations = 0
        for i, edges in enumerate(red_k4s):
            cnt = sum(1 for e in edges if colors[e] == 0)
            if cnt == 4:
                red_violations += 1
        for i, edges in enumerate(blue_k5s):
            cnt = sum(1 for e in edges if colors[e] == 1)
            if cnt == 10:
                blue_violations += 1
        
        if red_violations == 0 and blue_violations == 0:
            return colors
        else:
            return None

    def local_search_refine(n, initial_colors, time_limit=5.0):
        """
        Refine a coloring using local search to fix any violations.
        """
        triples, red_k4s, blue_k5s, edge_to_k4s, edge_to_k5s = prepare_data(n)
        num_triples = len(triples)
        
        colors = initial_colors[:]
        
        red_counts = [0] * len(red_k4s)
        blue_counts = [0] * len(blue_k5s)
        
        for i, edges in enumerate(red_k4s):
            red_counts[i] = sum(1 for e in edges if colors[e] == 0)
        for i, edges in enumerate(blue_k5s):
            blue_counts[i] = sum(1 for e in edges if colors[e] == 1)
        
        red_violations = sum(1 for c in red_counts if c == 4)
        blue_violations = sum(1 for c in blue_counts if c == 10)
        total_violations = red_violations + blue_violations
        
        if total_violations == 0:
            return colors
        
        start = time.time()
        stuck = 0
        
        while time.time() - start < time_limit:
            if total_violations == 0:
                return colors
            
            # Pick a random edge involved in a violation
            # Find a violating K4 or K5
            if red_violations > 0 and (blue_violations == 0 or random.random() < red_violations / total_violations):
                # Pick random K4 until we find a violation
                while True:
                    k4_idx = random.randint(0, len(red_k4s) - 1)
                    if red_counts[k4_idx] == 4:
                        break
                # Pick a random edge in this K4 to flip
                e = random.choice(red_k4s[k4_idx])
            else:
                while True:
                    k5_idx = random.randint(0, len(blue_k5s) - 1)
                    if blue_counts[k5_idx] == 10:
                        break
                e = random.choice(blue_k5s[k5_idx])
            
            current_color = colors[e]
            new_color = 1 - current_color
            
            delta_red = 0
            delta_blue = 0
            
            for k4_idx in edge_to_k4s[e]:
                cnt = red_counts[k4_idx]
                if current_color == 0:
                    if cnt == 4:
                        delta_red -= 1
                else:
                    if cnt == 3:
                        delta_red += 1
                        
            for k5_idx in edge_to_k5s[e]:
                cnt = blue_counts[k5_idx]
                if current_color == 1:
                    if cnt == 10:
                        delta_blue -= 1
                else:
                    if cnt == 9:
                        delta_blue += 1
                        
            total_delta = delta_red + delta_blue
            
            # Accept if it reduces violations, or with some probability if neutral
            accepted = False
            if total_delta < 0:
                accepted = True
                stuck = 0
            elif total_delta == 0:
                if random.random() < 0.5:
                    accepted = True
                else:
                    stuck += 1
            else:
                # Only accept worsening if small and we're stuck
                if total_delta <= 2 and stuck > 50 and random.random() < 0.2:
                    accepted = True
                    stuck += 1
                else:
                    stuck += 1
            
            if accepted:
                colors[e] = new_color
                for k4_idx in edge_to_k4s[e]:
                    if current_color == 0:
                        red_counts[k4_idx] -= 1
                    else:
                        red_counts[k4_idx] += 1
                for k5_idx in edge_to_k5s[e]:
                    if current_color == 1:
                        blue_counts[k5_idx] -= 1
                    else:
                        blue_counts[k5_idx] += 1
                red_violations += delta_red
                blue_violations += delta_blue
                total_violations += total_delta
            else:
                stuck += 1
                
            if stuck > 200:
                break  # Restart with new random init
        
        return None

    best_n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]
    
    # Try increasing n from 10 to 34
    for n in range(10, 35):
        if time.time() > deadline - 5:
            break
        
        # Time budget for this n
        remaining = deadline - time.time() - 2
        if remaining < 1:
            break
        
        # Try greedy construction with a few different seeds
        found = False
        max_attempts = min(5, int(remaining / 2) + 1)
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 2:
                break
                
            t_start = time.time()
            colors = greedy_construct(n, seed_val=attempt * 12345 + n)
            
            if colors is not None:
                best_n = n
                best_colors = colors
                found = True
                break
            
            # If greedy didn't work, try local search refinement
            # First, get a greedy coloring (even if invalid) and refine
            # Actually, greedy_construct returns None if invalid, so let's modify
            # to always return the coloring and check validity separately
            
            # For now, just try a few more greedy attempts with different seeds
        
        if not found and time.time() < deadline - 3:
            # Try local search from a random start
            t_start = time.time()
            time_limit = min(remaining - 2, 5.0)
            
            for attempt in range(3):
                if time.time() > deadline - 2:
                    break
                
                random_colors = [random.randint(0, 1) for _ in range(len(list(combinations(range(n), 3))))]
                refined = local_search_refine(n, random_colors, time_limit=time_limit)
                if refined is not None:
                    best_n = n
                    best_colors = refined
                    found = True
                    break
    
    return {'n': best_n, 'colors': best_colors}
