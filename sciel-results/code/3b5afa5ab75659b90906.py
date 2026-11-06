import time
import random
import math

def construct():
    n = 256
    
    # 1. Thue-Morse Initialization
    # The Thue-Morse sequence t(n) is the parity of the number of 1s in the binary representation of n.
    # It has the property that it avoids long runs of identical symbols, which helps minimize small-d APs.
    def thue_morse(n):
        # t(n) = sum(bits of n) % 2
        return bin(n).count('1') % 2
        
    color = [thue_morse(i) for i in range(n)]

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

    def local_search(color):
        """Perform deterministic hill climbing until no single flip improves the cost."""
        current_cost = count_mono_4ap(color)
        improved = True
        while improved:
            improved = False
            # Iterate in a fixed order to ensure determinism within a pass
            for p in range(n):
                delta = delta_from_flip(color, p, n)
                if delta < 0:
                    color[p] = 1 - color[p]
                    current_cost += delta
                    improved = True
        return current_cost

    # 2. Main Loop: Random Restarts with Local Search
    best_cost = local_search(color)
    best_color = color[:]
    
    start_time = time.time()
    time_limit = 110.0
    
    # Parameters for restarts
    num_restarts = 50
    perturbation_rate = 0.05 # 5% of bits flipped per perturbation
    
    for _ in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Create a candidate by perturbing the best known solution
        candidate = best_color[:]
        
        # Flip a few random bits
        num_flips = max(1, int(n * perturbation_rate))
        for _ in range(num_flips):
            p = random.randint(0, n - 1)
            candidate[p] = 1 - candidate[p]
            
        # Run local search on the candidate
        candidate_cost = local_search(candidate)
        
        if candidate_cost < best_cost:
            best_cost = candidate_cost
            best_color = candidate[:]
            
    return best_color
