import random
import math

def construct():
    N = 128
    # We suspect 3 colors are sufficient. 2 colors are likely not.
    # We will search for a valid coloring using at most 3 colors.
    # If we find one with 3, we return it.
    # We can also try to see if 2 is possible, but it's very unlikely.
    
    def count_violations(colors):
        v = 0
        # Check all 4-term APs: x, x+d, x+2d, x+3d
        # x is 0-indexed: 0 to N-1
        # d is 1 to (N-1-x)//3
        for x in range(N):
            max_d = (N - 1 - x) // 3
            for d in range(1, max_d + 1):
                c1 = colors[x]
                c2 = colors[x + d]
                c3 = colors[x + 2 * d]
                c4 = colors[x + 3 * d]
                # Violation if the set of colors has size < 3
                # i.e., size 1 or 2
                s = {c1, c2, c3, c4}
                if len(s) < 3:
                    v += 1
        return v

    def unique_colors(colors):
        return len(set(colors))

    # Initialize with 3 colors in a pattern that breaks many APs
    # Pattern: 0, 1, 2, 0, 1, 2 ...
    colors = [i % 3 for i in range(N)]
    
    # Calculate initial violations
    current_violations = count_violations(colors)
    current_unique = unique_colors(colors)
    
    # If already valid, return
    if current_violations == 0:
        return colors

    # Simulated Annealing
    # We want to minimize violations first, then unique colors.
    # Energy = violations * 1000 + unique_colors
    # We will use a penalty for violations to drive them to 0.
    
    T = 10.0
    T_min = 0.01
    cooling = 0.995
    
    best_colors = colors[:]
    best_energy = current_violations * 1000 + current_unique
    
    for step in range(50000):
        # Pick a random index
        i = random.randint(0, N - 1)
        old_color = colors[i]
        
        # Pick a new color. 
        # We restrict to colors currently in use + maybe one new one?
        # To minimize unique colors, we should mostly pick from existing ones.
        # But if we are stuck, we might need to introduce a new one?
        # Actually, if we start with 3, we should stick to 3 unless we find 2 works.
        # Let's try to swap to any of the 3 colors.
        new_color = random.choice([0, 1, 2])
        
        if new_color == old_color:
            continue
            
        colors[i] = new_color
        
        # Calculate delta violations
        # Only APs involving index i change.
        # i can be x, x+d, x+2d, x+3d
        delta_v = 0
        
        # Case 1: i is x
        # x = i, x+d = i+d, x+2d = i+2d, x+3d = i+3d
        # d >= 1, i+3d < N
        for d in range(1, (N - 1 - i) // 3 + 1):
            c1 = colors[i]
            c2 = colors[i + d]
            c3 = colors[i + 2 * d]
            c4 = colors[i + 3 * d]
            s_new = {c1, c2, c3, c4}
            # We need old violation status. 
            # This is expensive to track perfectly without storing old state.
            # For simplicity and given N=128 is small, we can just recalculate total violations.
            # 50k steps * (128*128/3) ops is too slow in Python if we do full recount.
            # Let's optimize: N=128, total APs is ~ sum_{x=0}^{127} (127-x)//3 ~ 128*64/3 ~ 2700 APs.
            # 50k * 2700 = 135M ops. Might be too slow in Python (60s limit).
            # We need incremental update or fewer steps.
            pass
        
        # Let's use a simpler approach: 
        # Since N is small, we can just do a local search with full recount but fewer iterations, 
        # or use a smarter heuristic.
        
        # Alternative: Random Restart with Greedy Local Search
        # Let's try a different strategy: 
        # 1. Try to find a 3-coloring.
        # 2. If found, check if we can reduce to 2.
        
        # Let's restart the loop with a more efficient method.
        # We'll use a simple hill climbing on violations, then minimize unique colors.
        
        break # Exit this inefficient loop structure

    # Efficient Local Search
    # We will use a list of all APs to make violation counting faster? 
    # No, iterating APs is fine if we do it smartly.
    
    # Let's try a few random initializations and local search.
    for attempt in range(10):
        # Random init with 3 colors
        colors = [random.randint(0, 2) for _ in range(N)]
        
        # Local Search to minimize violations
        improved = True
        while improved:
            improved = False
            for i in range(N):
                old_c = colors[i]
                best_c = old_c
                best_v = count_violations(colors)
                
                for new_c in [0, 1, 2]:
                    if new_c == old_c:
                        continue
                    colors[i] = new_c
                    v = count_violations(colors)
                    if v < best_v:
                        best_v = v
                        best_c = new_c
                    colors[i] = old_c
                
                if best_c != old_c:
                    colors[i] = best_c
                    improved = True
        
        if count_violations(colors) == 0:
            # Valid 3-coloring found. 
            # Can we reduce to 2?
            # Try to map one color to another?
            # Or just return this. The task says "Minimize the number of labels".
            # If 3 is the minimum, we return 3.
            # If 2 is possible, we should find it.
            # Let's try to see if 2 colors are possible by checking if we can merge two color classes.
            
            # For now, return the valid 3-coloring.
            # We can try to optimize the labels to be 0,1,2.
            unique = sorted(set(colors))
            mapping = {u: i for i, u in enumerate(unique)}
            final_colors = [mapping[c] for c in colors]
            return final_colors
            
    # Fallback: Return the best effort
    # This should not happen if 3 colors are sufficient.
    return colors

# The above logic is a bit mixed. Let's refine the "Efficient Local Search" part.
# Recounting violations for every swap in a while loop is O(N * #APs).
# #APs ~ 2700. N=128. 128 * 2700 = 345,600 ops per full pass.
# If we do 100 passes, that's 34M ops. Python can do ~10-50M simple ops/sec.
# So 100 passes might take 1-3 seconds. 10 attempts = 10-30s. Acceptable.

# Let's rewrite the construct function cleanly.

def construct():
    N = 128
    
    def count_violations(colors):
        v = 0
        for x in range(N):
            max_d = (N - 1 - x) // 3
            for d in range(1, max_d + 1):
                c1 = colors[x]
                c2 = colors[x + d]
                c3 = colors[x + 2 * d]
                c4 = colors[x + 3 * d]
                if len({c1, c2, c3, c4}) < 3:
                    v += 1
        return v

    # Try to find a valid coloring with 3 colors
    # If found, try to reduce to 2.
    
    # Strategy 1: Find a 3-coloring
    found_3 = None
    for _ in range(20):
        # Random init
        colors = [random.randint(0, 2) for _ in range(N)]
        
        # Hill Climbing
        prev_v = count_violations(colors)
        improved = True
        while improved:
            improved = False
            # Random order to avoid cycles
            indices = list(range(N))
            random.shuffle(indices)
            for i in indices:
                old_c = colors[i]
                best_c = old_c
                best_v = prev_v
                
                for new_c in [0, 1, 2]:
                    if new_c == old_c:
                        continue
                    colors[i] = new_c
                    v = count_violations(colors)
                    if v < best_v:
                        best_v = v
                        best_c = new_c
                    colors[i] = old_c
                
                if best_c != old_c:
                    colors[i] = best_c
                    prev_v = best_v
                    improved = True
        
        if prev_v == 0:
            found_3 = colors
            break

    if found_3:
        # Check if 2 colors are possible
        # We can try to see if we can map the 3-coloring to 2 colors.
        # This is equivalent to checking if the graph of conflicts is 2-colorable?
        # No, the constraint is on 4-APs.
        # Let's just try to find a 2-coloring directly with a similar search.
        # If we find one, it's better.
        
        found_2 = None
        for _ in range(10):
            colors = [random.randint(0, 1) for _ in range(N)]
            prev_v = count_violations(colors)
            improved = True
            while improved:
                improved = False
                indices = list(range(N))
                random.shuffle(indices)
                for i in indices:
                    old_c = colors[i]
                    best_c = old_c
                    best_v = prev_v
                    
                    for new_c in [0, 1]:
                        if new_c == old_c:
                            continue
                        colors[i] = new_c
                        v = count_violations(colors)
                        if v < best_v:
                            best_v = v
                            best_c = new_c
                        colors[i] = old_c
                    
                    if best_c != old_c:
                        colors[i] = best_c
                        prev_v = best_v
                        improved = True
            
            if prev_v == 0:
                found_2 = colors
                break

        if found_2:
            return found_2
        else:
            # Return the 3-coloring, normalized
            unique = sorted(set(found_3))
            mapping = {u: i for i, u in enumerate(unique)}
            return [mapping[c] for c in found_3]

    # Fallback if no 3-coloring found (should not happen)
    # Return a random 3-coloring
    colors = [i % 3 for i in range(N)]
    return colors

