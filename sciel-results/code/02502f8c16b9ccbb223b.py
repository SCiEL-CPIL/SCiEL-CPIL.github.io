import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_n = 0
    best_colors = []
    
    # Try n from 34 down to 10
    for n in range(34, 9, -1):
        if time.time() > deadline - 5.0:
            break
        
        # Try up to 5 restarts for this n
        found = False
        for attempt in range(5):
            if time.time() > deadline - 2.0:
                break
                
            # Initialize random coloring
            triples = list(combinations(range(n), 3))
            num_triples = len(triples)
            triple_to_idx = {t: i for i, t in enumerate(triples)}
            
            # Use different seed for each attempt
            random.seed(attempt * 1000 + n)
            colors = [random.randint(0, 1) for _ in range(num_triples)]
            
            # Precompute adjacency for updates
            # This is expensive, so we do it once per n, not per attempt?
            # But adj depends on n.
            # Let's move precomputation outside the attempt loop.
            
            # Actually, precomputation is ~1s. 5 attempts = 5s.
            # We can precompute once for n, then reuse for 5 attempts.
            pass
            
        # Refactor: Precompute once per n
        if time.time() > deadline - 2.0:
            continue
            
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        four_sets = list(combinations(range(n), 4))
        five_sets = list(combinations(range(n), 5))
        
        adj_4 = [[] for _ in range(num_triples)]
        adj_5 = [[] for _ in range(num_triples)]
        
        for s4 in four_sets:
            t4 = [triple_to_idx[t] for t in combinations(s4, 3)]
            for i in range(4):
                others = [t4[j] for j in range(4) if j != i]
                adj_4[t4[i]].append(others)
                
        for s5 in five_sets:
            t5 = [triple_to_idx[t] for t in combinations(s5, 3)]
            for i in range(10):
                others = [t5[j] for j in range(10) if j != i]
                adj_5[t5[i]].append(others)
                
        # Now try multiple random inits
        found = False
        for attempt in range(5):
            if time.time() > deadline - 2.0:
                break
                
            random.seed(attempt * 1000 + n)
            colors = [random.randint(0, 1) for _ in range(num_triples)]
            
            red_conflicts = [0] * num_triples
            blue_conflicts = [0] * num_triples
            
            for s4 in four_sets:
                t4 = [triple_to_idx[t] for t in combinations(s4, 3)]
                if all(colors[i] == 0 for i in t4):
                    for i in t4:
                        red_conflicts[i] += 1
                        
            for s5 in five_sets:
                t5 = [triple_to_idx[t] for t in combinations(s5, 3)]
                if all(colors[i] == 1 for i in t5):
                    for i in t5:
                        blue_conflicts[i] += 1
                        
            # Local Search
            while time.time() < deadline - 1.0:
                max_conf = -1
                max_idx = -1
                for i in range(num_triples):
                    c = red_conflicts[i] + blue_conflicts[i]
                    if c > max_conf:
                        max_conf = c
                        max_idx = i
                
                if max_conf == 0:
                    found = True
                    break
                
                old_color = colors[max_idx]
                new_color = 1 - old_color
                colors[max_idx] = new_color
                
                if old_color == 0:
                    for others in adj_4[max_idx]:
                        if all(colors[o] == 0 for o in others):
                            red_conflicts[max_idx] -= 1
                            for o in others:
                                red_conflicts[o] -= 1
                    for others in adj_5[max_idx]:
                        if all(colors[o] == 1 for o in others):
                            blue_conflicts[max_idx] += 1
                            for o in others:
                                blue_conflicts[o] += 1
                else:
                    for others in adj_5[max_idx]:
                        if all(colors[o] == 1 for o in others):
                            blue_conflicts[max_idx] -= 1
                            for o in others:
                                blue_conflicts[o] -= 1
                    for others in adj_4[max_idx]:
                        if all(colors[o] == 0 for o in others):
                            red_conflicts[max_idx] += 1
                            for o in others:
                                red_conflicts[o] += 1
            
            if found:
                best_n = n
                best_colors = colors
                break # Found valid for this n, break attempt loop
        
        if found:
            break # Found valid for max n, break n loop
            
    if best_n == 0:
        n = 9
        return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
        
    return {'n': best_n, 'colors': best_colors}
