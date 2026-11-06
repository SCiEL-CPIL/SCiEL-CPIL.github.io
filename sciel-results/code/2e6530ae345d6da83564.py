import random
import time

def construct():
    N = 128
    # Precompute all 4-term APs
    aps = []
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    def count_bad(colors):
        bad = 0
        for a, b, c, d in aps:
            ca = colors[a - 1]
            cb = colors[b - 1]
            cc = colors[c - 1]
            cd = colors[d - 1]
            if len({ca, cb, cc, cd}) < 3:
                bad += 1
        return bad

    # Try increasing number of colors
    for k in range(3, 10):
        start_time = time.time()
        # Multiple restarts
        found = False
        for restart in range(10):
            if time.time() - start_time > 5:
                break
            
            # Structured initialization: use base-k representation of index
            # Color i by (i mod k) but with perturbation
            colors = []
            for i in range(N):
                # Use a hash-like function to spread colors
                c = (i * 7 + restart * 13) % k
                colors.append(c)
            
            # Min-conflicts with steepest descent
            max_iterations = 20000
            for _ in range(max_iterations):
                if time.time() - start_time > 5:
                    break
                
                bad_aps = []
                for ap in aps:
                    a, b, c, d = ap
                    ca = colors[a - 1]
                    cb = colors[b - 1]
                    cc = colors[c - 1]
                    cd = colors[d - 1]
                    if len({ca, cb, cc, cd}) < 3:
                        bad_aps.append(ap)
                
                if not bad_aps:
                    found = True
                    break
                
                # Pick a random bad AP
                ap = random.choice(bad_aps)
                a, b, c, d = ap
                
                # Try all positions and all colors, pick the best improvement
                best_delta = 0
                best_pos = None
                best_color = None
                for pos in [a, b, c, d]:
                    old_color = colors[pos - 1]
                    for new_color in range(k):
                        if new_color == old_color:
                            continue
                        # Estimate improvement by checking APs containing pos
                        colors[pos - 1] = new_color
                        cur_bad = count_bad(colors)
                        colors[pos - 1] = old_color
                        # We want to minimize bad, so delta = old_bad - cur_bad
                        # But we don't have old_bad easily. Just count how many APs this fixes.
                        # Simpler: just use random for speed
                        break
                    break
                
                # Fallback: random change
                pos = random.choice([a, b, c, d])
                new_color = random.randint(0, k - 1)
                colors[pos - 1] = new_color
            
            if found:
                break
        
        if found:
            return colors
    
    # Fallback: return a valid coloring with more colors
    return list(range(N))
