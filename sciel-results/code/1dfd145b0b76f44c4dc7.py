import time
import random

def construct():
    n = 256
    
    def count_mono_4ap(color):
        count = 0
        for d in range(1, n // 3 + 1):
            for a in range(1, n - 3 * d + 1):
                if color[a-1] == color[a+d-1] == color[a+2*d-1] == color[a+3*d-1]:
                    count += 1
        return count

    def delta_from_flip(color, p, n):
        """Compute change in monochromatic 4-AP count when flipping position p (0-indexed)."""
        delta = 0
        c = color[p]
        for d in range(1, n // 3 + 1):
            # p as first element: (p, p+d, p+2d, p+3d) 0-indexed
            if p + 3*d < n:
                others = [color[p+d], color[p+2*d], color[p+3*d]]
                if all(o == c for o in others):
                    delta -= 1
                elif all(o != c for o in others):
                    delta += 1
            # p as second: (p-d, p, p+d, p+2d)
            if p - d >= 0 and p + 2*d < n:
                others = [color[p-d], color[p+d], color[p+2*d]]
                if all(o == c for o in others):
                    delta -= 1
                elif all(o != c for o in others):
                    delta += 1
            # p as third: (p-2d, p-d, p, p+d)
            if p - 2*d >= 0 and p + d < n:
                others = [color[p-2*d], color[p-d], color[p+d]]
                if all(o == c for o in others):
                    delta -= 1
                elif all(o != c for o in others):
                    delta += 1
            # p as fourth: (p-3d, p-2d, p-d, p)
            if p - 3*d >= 0:
                others = [color[p-3*d], color[p-2*d], color[p-d]]
                if all(o == c for o in others):
                    delta -= 1
                elif all(o != c for o in others):
                    delta += 1
        return delta

    def hill_climb(color):
        """Perform hill climbing until no single flip improves the cost."""
        current_cost = count_mono_4ap(color)
        improved = True
        while improved:
            improved = False
            for p in range(n):
                d = delta_from_flip(color, p, n)
                if d < 0:
                    color[p] = 1 - color[p]
                    current_cost += d
                    improved = True
                    # Restart loop to catch new improvements enabled by this flip
                    break
        return current_cost, color

    # 1. Generate candidate initializations using modular patterns
    candidates = []
    
    # Pattern 1: Random
    candidates.append([random.randint(0, 1) for _ in range(n)])
    
    # Pattern 2: Alternating (bad for APs but a baseline)
    candidates.append([i % 2 for i in range(n)])
    
    # Pattern 3: Modular patterns
    for m in range(3, 15):
        # Try all subsets of size roughly m/2
        # To keep it fast, we try a few specific sets
        # e.g., {0, 1} mod 4, {0, 1, 2} mod 5, etc.
        if m == 4:
            sets = [[0, 1], [0, 2]]
        elif m == 5:
            sets = [[0, 1, 2], [0, 1]]
        elif m == 6:
            sets = [[0, 1, 2], [0, 2, 4]]
        elif m == 7:
            sets = [[0, 1, 2, 3]]
        else:
            sets = [list(range(m // 2))]
        
        for s in sets:
            col = [1 if (i % m) in s else 0 for i in range(n)]
            candidates.append(col)
            
    # 2. Evaluate candidates and pick the best few
    scored_candidates = []
    for c in candidates:
        cost = count_mono_4ap(c)
        scored_candidates.append((cost, c))
        
    scored_candidates.sort(key=lambda x: x[0])
    
    best_cost = scored_candidates[0][0]
    best_color = scored_candidates[0][1][:]
    
    # 3. Iterated Local Search
    start_time = time.time()
    time_limit = 110.0
    
    # Take the top 5 initializations and run ILS on each
    for i in range(min(5, len(scored_candidates))):
        current_color = scored_candidates[i][1][:]
        current_cost = scored_candidates[i][0]
        
        # Run Hill Climbing
        current_cost, current_color = hill_climb(current_color)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_color = current_color[:]
            
        # If we have time, try to perturb and hill climb again
        while time.time() - start_time < time_limit:
            # Perturbation: flip k random bits
            k = random.randint(5, 15)
            perturbed_color = current_color[:]
            for _ in range(k):
                p = random.randint(0, n - 1)
                perturbed_color[p] = 1 - perturbed_color[p]
                
            # Hill Climb the perturbed color
            perturbed_cost, perturbed_color = hill_climb(perturbed_color)
            
            if perturbed_cost < current_cost:
                current_cost = perturbed_cost
                current_color = perturbed_color[:]
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_color = current_color[:]
            else:
                # If perturbation didn't help, we might be stuck in a local min
                # Try a larger perturbation or reset to best known
                # For simplicity, just continue trying
                pass

    return best_color
