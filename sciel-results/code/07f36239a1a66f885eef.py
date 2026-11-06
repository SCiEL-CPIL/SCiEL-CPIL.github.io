import random
import time

def construct():
    N = 128
    deadline = time.time() + 85.0
    
    # Precompute APs for each position
    # ap_s[i] = list of (indices_tuple) where i is part of the AP
    ap_s = [[] for _ in range(N)]
    for a in range(N):
        for d in range(1, (N - 1 - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e >= N:
                break
            ap = (a, b, c, e)
            ap_s[a].append(ap)
            ap_s[b].append(ap)
            ap_s[c].append(ap)
            ap_s[e].append(ap)

    def is_valid_local(coloring, x):
        for ap in ap_s[x]:
            if len(set(coloring[i] for i in ap)) < 3:
                return False
        return True

    def try_eliminate_color(coloring, color_to_eliminate):
        positions = [i for i in range(N) if coloring[i] == color_to_eliminate]
        if not positions:
            return True
            
        other_colors = [c for c in range(128) if c != color_to_eliminate and any(coloring[i] == c for i in range(N))]
        
        # Order positions by number of APs (most constrained first)
        positions.sort(key=lambda i: len(ap_s[i]), reverse=True)
        
        def backtrack(idx):
            if idx == len(positions):
                return True
            x = positions[idx]
            # Try each other color
            # Randomize order for diversity?
            random.shuffle(other_colors)
            for c in other_colors:
                coloring[x] = c
                if is_valid_local(coloring, x):
                    if backtrack(idx + 1):
                        return True
            coloring[x] = color_to_eliminate
            return False
            
        return backtrack(0)

    best_coloring = None
    best_colors = 128
    
    for seed in range(5000):
        if time.time() > deadline:
            break
            
        random.seed(seed)
        
        # Start with all distinct colors
        coloring = list(range(N))
        current_colors = set(coloring)
        
        # Greedy elimination
        while True:
            if time.time() > deadline:
                break
                
            freqs = {}
            for c in current_colors:
                freqs[c] = sum(1 for i in range(N) if coloring[i] == c)
            
            eliminated = False
            # Try to eliminate the color with smallest frequency
            for c in sorted(freqs, key=lambda x: freqs[x]):
                if time.time() > deadline:
                    break
                if c not in current_colors:
                    continue
                if freqs[c] == 0:
                    continue
                
                temp_coloring = coloring[:]
                if try_eliminate_color(temp_coloring, c):
                    coloring = temp_coloring
                    current_colors.remove(c)
                    eliminated = True
                    
                    if len(current_colors) < best_colors:
                        best_colors = len(current_colors)
                        best_coloring = coloring[:]
                    break
            
            if not eliminated:
                # Stuck. Try a random perturbation
                # Pick a random position and change its color to a random other color
                x = random.randint(0, N - 1)
                old_c = coloring[x]
                other_c = random.choice([c for c in current_colors if c != old_c])
                coloring[x] = other_c
                if is_valid_local(coloring, x):
                    # Accepted perturbation
                    pass
                else:
                    coloring[x] = old_c
                    # If perturbation failed, restart
                    break

    if best_coloring is None:
        return list(range(N))
        
    return best_coloring
