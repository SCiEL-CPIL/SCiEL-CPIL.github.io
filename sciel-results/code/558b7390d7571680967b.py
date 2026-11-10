"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    max_val = 1000000
    
    def get_cost(x):
        return max(x) - min(x)
        
    def greedy_init():
        # Sort indices by divisor size (larger first, as they are more constrained)
        # Actually, smaller divisors have more options, so larger divisors are more constrained?
        # No, larger divisors have fewer multiples in a small window.
        # So assign larger divisors first?
        # Let's just pick a central value and expand.
        # Start with a random center
        center = random.randint(100, 5000)
        x = [0] * n
        used = set()
        
        # Assign in order of decreasing divisor (more constrained first)
        indices = sorted(range(n), key=lambda i: divisors[i], reverse=True)
        
        for i in indices:
            d = divisors[i]
            # Find multiple closest to center, not in used
            # Try k such that k*d is close to center
            k = center // d
            # Check k, k+1, k-1, k+2, k-2...
            found = False
            for delta in range(0, 100):
                for k_try in [k + delta, k - delta]:
                    if k_try <= 0: continue
                    val = k_try * d
                    if val > max_val: continue
                    if val not in used:
                        x[i] = val
                        used.add(val)
                        found = True
                        break
                if found:
                    break
            if not found:
                # Fallback: random
                while True:
                    val = random.randint(1, 10000) // d * d
                    if val not in used and val > 0:
                        x[i] = val
                        used.add(val)
                        break
        return x

    best_x = greedy_init()
    best_cost = get_cost(best_x)
    
    # Simulated Annealing
    T = 100.0
    cooling = 0.995
    iters = 50000
    
    current_x = best_x.copy()
    current_cost = best_cost
    
    for _ in range(iters):
        # Pick random index
        i = random.randint(0, n - 1)
        d = divisors[i]
        current_val = current_x[i]
        
        # Propose move
        # 80% chance: local move (step +/- 1, 2, 5)
        # 20% chance: random jump in [1, 10000]
        if random.random() < 0.8:
            step = random.choice([-5, -2, -1, 1, 2, 5])
            new_val = current_val + step * d
        else:
            max_k = 10000 // d
            if max_k == 0: max_k = 1
            new_val = random.randint(1, max_k) * d
            
        if new_val < 1 or new_val > max_val:
            continue
            
        # Check collision
        # If new_val is in current_x and not at i, it's a collision
        # We need to check if new_val is already used by another index
        collision = False
        for j in range(n):
            if j != i and current_x[j] == new_val:
                collision = True
                break
        
        if collision:
            continue
            
        new_x = current_x.copy()
        new_x[i] = new_val
        new_cost = get_cost(new_x)
        
        delta = new_cost - current_cost
        if delta < 0:
            current_x = new_x
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_x = new_x.copy()
        else:
            if T > 0:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_x = new_x
                    current_cost = new_cost
        
        T *= cooling
        if T < 0.1:
            break
            
    # Final local search
    improved = True
    while improved:
        improved = False
        for i in range(n):
            d = divisors[i]
            current_val = best_x[i]
            for step in [-3, -2, -1, 1, 2, 3]:
                new_val = current_val + step * d
                if new_val < 1 or new_val > max_val:
                    continue
                # Check collision
                collision = False
                for j in range(n):
                    if j != i and best_x[j] == new_val:
                        collision = True
                        break
                if collision:
                    continue
                new_x = best_x.copy()
                new_x[i] = new_val
                new_cost = get_cost(new_x)
                if new_cost < best_cost:
                    best_x = new_x
                    best_cost = new_cost
                    current_val = new_val
                    improved = True
                    break
                    
    return [int(x) for x in best_x]
# EVOLVE-BLOCK-END
