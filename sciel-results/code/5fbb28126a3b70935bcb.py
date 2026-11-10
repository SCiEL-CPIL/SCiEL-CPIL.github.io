import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def solve_for_n(n, time_budget):
        if n < 4:
            return {'n': n, 'colors': [0] * (n*(n-1)*(n-2)//6)}
        
        # Generate n distinct vectors in GF(3)^k
        # Use k=4 for n up to 81
        k = 4
        # Generate random distinct vectors
        vectors = []
        seen = set()
        while len(vectors) < n:
            v = tuple(random.randint(0, 2) for _ in range(k))
            if v not in seen:
                seen.add(v)
                vectors.append(v)
        
        # Initialize colors based on sum=0
        # Color 0 (Red) if sum is (0,0,0,0)
        # Color 1 (Blue) otherwise
        colors = [0] * (n*(n-1)*(n-2)//6)
        
        # Precompute indices for triples
        # We need to map (a,b,c) to index in colors
        # And for each triple, which K4s and K5s it belongs to
        
        # This preprocessing might be heavy for n=40?
        # C(40,3) = 9880.
        # C(40,4) = 91390.
        # C(40,5) = 658008.
        # Total subgraphs ~ 750k.
        # Building the mapping:
        # For each K4, add its 4 triples to the list of that triple.
        # For each K5, add its 10 triples to the list of that triple.
        
        # To save time, we only do this if we are serious about solving.
        # For n=34, it's fast.
        
        if time.time() > time_budget:
            return None
            
        # Map triple (a,b,c) to index
        def get_triple_idx(a, b, c):
            # Lexicographic order
            # We can precompute a dictionary
            pass

        # Let's use a simpler approach:
        # Store colors in a 3D array or dictionary for easy access?
        # Or just a flat list and a dict mapping (a,b,c) -> idx.
        
        idx_map = {}
        idx = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    idx_map[(a,b,c)] = idx
                    # Init color
                    s = (vectors[a][0] + vectors[b][0] + vectors[c][0]) % 3
                    s1 = (vectors[a][1] + vectors[b][1] + vectors[c][1]) % 3
                    s2 = (vectors[a][2] + vectors[b][2] + vectors[c][2]) % 3
                    s3 = (vectors[a][3] + vectors[b][3] + vectors[c][3]) % 3
                    colors[idx] = 0 if (s==0 and s1==0 and s2==0 and s3==0) else 1
                    idx += 1
                    
        # Precompute K4s and K5s
        k4s = []
        k5s = []
        
        # For each K4, store the 4 triple indices
        # For each K5, store the 10 triple indices
        
        # Also, for each triple index, store list of K4 indices and K5 indices
        tri_to_k4 = [[] for _ in range(len(colors))]
        tri_to_k5 = [[] for _ in range(len(colors))]
        
        # Generate K4s
        k4_idx = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    for d in range(c+1, n):
                        t1 = idx_map[(a,b,c)]
                        t2 = idx_map[(a,b,d)]
                        t3 = idx_map[(a,c,d)]
                        t4 = idx_map[(b,c,d)]
                        k4s.append((t1, t2, t3, t4))
                        tri_to_k4[t1].append(k4_idx)
                        tri_to_k4[t2].append(k4_idx)
                        tri_to_k4[t3].append(k4_idx)
                        tri_to_k4[t4].append(k4_idx)
                        k4_idx += 1
                        
        # Generate K5s
        k5_idx = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    for d in range(c+1, n):
                        for e in range(d+1, n):
                            # 10 triples
                            tris = [
                                idx_map[(a,b,c)], idx_map[(a,b,d)], idx_map[(a,b,e)],
                                idx_map[(a,c,d)], idx_map[(a,c,e)], idx_map[(a,d,e)],
                                idx_map[(b,c,d)], idx_map[(b,c,e)], idx_map[(b,d,e)],
                                idx_map[(c,d,e)]
                            ]
                            k5s.append(tris)
                            for t in tris:
                                tri_to_k5[t].append(k5_idx)
                            k5_idx += 1
                            
        # Initialize counts
        # red_count[k4_idx] = number of red triples in this K4
        # blue_count[k5_idx] = number of blue triples in this K5
        
        red_counts = [0] * len(k4s)
        blue_counts = [0] * len(k5s)
        
        # Calculate initial counts
        for i, k4 in enumerate(k4s):
            cnt = 0
            for t in k4:
                if colors[t] == 0:
                    cnt += 1
            red_counts[i] = cnt
            
        for i, k5 in enumerate(k5s):
            cnt = 0
            for t in k5:
                if colors[t] == 1:
                    cnt += 1
            blue_counts[i] = cnt
            
        # Initial cost
        cost = sum(1 for c in red_counts if c == 4) + sum(1 for c in blue_counts if c == 10)
        
        best_cost = cost
        best_colors = colors.copy()
        
        # Simulated Annealing
        T = 1.0
        cooling = 0.9999
        min_T = 0.01
        
        while time.time() < time_budget:
            if cost == 0:
                return {'n': n, 'colors': colors}
                
            # Pick random triple
            t_idx = random.randint(0, len(colors)-1)
            
            # Calculate delta
            old_color = colors[t_idx]
            new_color = 1 - old_color
            
            delta_cost = 0
            
            # Update K4s
            for k4_i in tri_to_k4[t_idx]:
                old_c = red_counts[k4_i]
                if old_color == 0:
                    # Was red, now blue. Red count decreases.
                    new_c = old_c - 1
                    if old_c == 4:
                        delta_cost -= 1
                    if new_c == 4: # Impossible since 3!=4
                        delta_cost += 1
                    red_counts[k4_i] = new_c
                else:
                    # Was blue, now red. Red count increases.
                    new_c = old_c + 1
                    if old_c == 4:
                        delta_cost -= 1 # Impossible
                    if new_c == 4:
                        delta_cost += 1
                    red_counts[k4_i] = new_c
                    
            # Update K5s
            for k5_i in tri_to_k5[t_idx]:
                old_c = blue_counts[k5_i]
                if old_color == 1:
                    # Was blue, now red. Blue count decreases.
                    new_c = old_c - 1
                    if old_c == 10:
                        delta_cost -= 1
                    if new_c == 10:
                        delta_cost += 1
                    blue_counts[k5_i] = new_c
                else:
                    # Was red, now blue. Blue count increases.
                    new_c = old_c + 1
                    if old_c == 10:
                        delta_cost -= 1
                    if new_c == 10:
                        delta_cost += 1
                    blue_counts[k5_i] = new_c
                    
            # Decide accept
            if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                colors[t_idx] = new_color
                cost += delta_cost
                if cost < best_cost:
                    best_cost = cost
                    best_colors = colors.copy()
            else:
                # Revert counts
                for k4_i in tri_to_k4[t_idx]:
                    if old_color == 0:
                        red_counts[k4_i] += 1
                    else:
                        red_counts[k4_i] -= 1
                for k5_i in tri_to_k5[t_idx]:
                    if old_color == 1:
                        blue_counts[k5_i] += 1
                    else:
                        blue_counts[k5_i] -= 1
                        
            T *= cooling
            if T < min_T:
                T = min_T
                
        # Return best found
        if best_cost == 0:
            return {'n': n, 'colors': best_colors}
        return None

    # Try n=34 first
    best_result = None
    for n in [34, 35, 33, 36]:
        if time.time() > deadline - 10:
            break
        res = solve_for_n(n, deadline - 5)
        if res:
            best_result = res
            break # Found a valid one, return immediately? Or try to maximize?
            # Task says maximize n. If 34 works, try 35?
            # But 35 might take longer.
            # Let's just return the first valid one found in the order 34, 35...
            # Actually, if 34 works, we can try 35 with remaining time.
            
    if best_result:
        return best_result
        
    # Fallback: return a small valid coloring
    # n=9 from seed
    n = 9
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in __import__('itertools').combinations(range(n), 3)]
    return {'n': n, 'colors': colors}

# EVOLVE-BLOCK-END
