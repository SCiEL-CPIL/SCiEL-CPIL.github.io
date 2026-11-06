import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 140.0  # Keep 10s margin
    
    best_n = 0
    best_colors = None
    
    # We will try to increase n
    # Start with a small n and try to grow, or just try high n directly?
    # Growing is safer to ensure we have a valid base.
    
    # Let's start with n=5, which is easy to satisfy.
    # R(4,5;3) > 5 is trivial.
    
    current_n = 5
    # Initialize a valid coloring for n=5?
    # Any coloring works as long as no red K4 (impossible for n<4? No, n=4 is K4).
    # For n=5, we need no red K4 and no blue K5.
    # Blue K5 requires all 10 edges blue.
    # Random init usually works for small n.
    
    # Let's define a function to run SA for a given n
    def run_sa(n, max_iters, time_budget):
        # Precomputation
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Colors array
        colors = np.random.randint(0, 2, size=num_triples)
        
        # Precompute K4s and K5s
        # K4: 4 vertices, 4 triples
        k4s = []
        for verts in combinations(range(n), 4):
            trs = [triple_to_idx[(a,b,c)] for a,b,c in combinations(verts, 3)]
            k4s.append(trs)
            
        # K5: 5 vertices, 10 triples
        k5s = []
        for verts in combinations(range(n), 5):
            trs = [triple_to_idx[(a,b,c)] for a,b,c in combinations(verts, 3)]
            k5s.append(trs)
            
        # For each triple, list of K4 indices and K5 indices containing it
        triple_to_k4s = [[] for _ in range(num_triples)]
        for i, trs in enumerate(k4s):
            for t in trs:
                triple_to_k4s[t].append(i)
                
        triple_to_k5s = [[] for _ in range(num_triples)]
        for i, trs in enumerate(k5s):
            for t in trs:
                triple_to_k5s[t].append(i)
                
        # Calculate initial energy
        def calc_energy(colors):
            e = 0
            for trs in k4s:
                if all(colors[t] == 0 for t in trs):
                    e += 1
            for trs in k5s:
                if all(colors[t] == 1 for t in trs):
                    e += 1
            return e
            
        energy = calc_energy(colors)
        
        if energy == 0:
            return colors.copy()
            
        temp = 1.0
        cooling = 0.999
        
        for it in range(max_iters):
            if time.time() - start_time > time_budget:
                break
                
            # Pick random triple
            t = random.randint(0, num_triples - 1)
            
            # Calculate delta energy
            # Current color
            c = colors[t]
            
            # Affected K4s
            delta_e = 0
            for k4_idx in triple_to_k4s[t]:
                trs = k4s[k4_idx]
                # Count reds in this K4
                # We are flipping t from c to 1-c
                reds = 0
                for tt in trs:
                    if tt == t:
                        if c == 0: reds += 1 # It was red, will be blue
                        else: pass # It was blue, will be red
                    else:
                        if colors[tt] == 0: reds += 1
                        
                # Before:
                if reds == 4:
                    delta_e -= 1 # Was a red K4, now not
                # After:
                new_reds = reds
                if c == 0:
                    new_reds -= 1 # t becomes blue
                else:
                    new_reds += 1 # t becomes red
                    
                if new_reds == 4:
                    delta_e += 1 # Now a red K4
            
            # Affected K5s
            for k5_idx in triple_to_k5s[t]:
                trs = k5s[k5_idx]
                blues = 0
                for tt in trs:
                    if tt == t:
                        if c == 1: blues += 1
                    else:
                        if colors[tt] == 1: blues += 1
                        
                if blues == 10:
                    delta_e -= 1
                    
                new_blues = blues
                if c == 1:
                    new_blues -= 1
                else:
                    new_blues += 1
                    
                if new_blues == 10:
                    delta_e += 1
            
            # Metropolis step
            if delta_e < 0 or random.random() < np.exp(-delta_e / temp):
                colors[t] = 1 - c
                energy += delta_e
                
            temp *= cooling
            
            if energy == 0:
                return colors.copy()
                
        return None

    # Main loop: try to increase n
    # Start with n=10, try to go up to 35
    # If we fail at n, we might restart with a better random seed or higher temp
    
    for n in range(10, 36):
        if time.time() - start_time > time_limit - 10:
            break
            
        # Try a few restarts for this n
        found = None
        for restart in range(5):
            if time.time() - start_time > time_limit - 10:
                break
            # Increase max_iters as n grows?
            # Or just use time budget
            res = run_sa(n, 10000, time_limit - (time.time() - start_time) - 5)
            if res is not None:
                found = res
                best_n = n
                best_colors = res
                break # Found valid coloring for n, try n+1
        
        if found is None:
            # If we can't solve n, maybe we can't solve n+1 either?
            # But sometimes n is hard, n+1 is easy? Unlikely.
            # Break if we failed for n, assuming monotonicity of difficulty?
            # Actually, if we fail n, we should probably stop or try harder.
            # Given time, let's stop.
            break
            
    if best_colors is None:
        # Fallback to seed
        n = 9
        colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
        return {'n': n, 'colors': colors}
        
    return {'n': best_n, 'colors': best_colors.tolist()}
