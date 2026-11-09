import time
import random
import math

N = 512
DEADLINE = 55.0  # Safety margin

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Precompute conflicts for faster checks?
    # Actually, bitmask checks are quite fast. 
    # Let's stick to bitmask but optimize the can_add loop.
    
    def can_add(mask, x):
        """Check if x can be added to mask without creating a 4-AP."""
        # x is 1st: x, x+d, x+2d, x+3d
        max_d = (N - x) // 3
        for d in range(1, max_d + 1):
            if (mask >> (x + d) & 1) and (mask >> (x + 2*d) & 1) and (mask >> (x + 3*d) & 1):
                return False
        # x is 2nd: x-d, x, x+d, x+2d
        max_d2 = min(x - 1, (N - x) // 2)
        for d in range(1, max_d2 + 1):
            if (mask >> (x - d) & 1) and (mask >> (x + d) & 1) and (mask >> (x + 2*d) & 1):
                return False
        # x is 3rd: x-2d, x-d, x, x+d
        max_d3 = min((x - 1) // 2, N - x)
        for d in range(1, max_d3 + 1):
            if (mask >> (x - 2*d) & 1) and (mask >> (x - d) & 1) and (mask >> (x + d) & 1):
                return False
        # x is 4th: x-3d, x-2d, x-d, x
        max_d4 = (x - 1) // 3
        for d in range(1, max_d4 + 1):
            if (mask >> (x - 3*d) & 1) and (mask >> (x - 2*d) & 1) and (mask >> (x - d) & 1):
                return False
        return True

    def fast_greedy(rng):
        """Fast greedy construction."""
        mask = 0
        order = list(range(1, N + 1))
        rng.shuffle(order)
        for x in order:
            if can_add(mask, x):
                mask |= (1 << x)
        return mask

    def get_size(mask):
        return bin(mask).count('1')

    def polish_pass(mask, deadline):
        """Final polish: try adding elements in multiple orders."""
        S_size = get_size(mask)
        blocked = [x for x in range(1, N + 1) if not (mask >> x & 1)]
        
        # Try natural order
        for x in blocked:
            if time.time() >= deadline:
                break
            if can_add(mask, x):
                mask |= (1 << x)
                S_size += 1
        
        # Try reverse order
        for x in reversed(blocked):
            if time.time() >= deadline:
                break
            if not (mask >> x & 1) and can_add(mask, x):
                mask |= (1 << x)
                S_size += 1
        
        # Try a few random passes
        rng = random.Random(int(time.time() * 1000) % 10000)
        for _ in range(10):
            blocked = [x for x in range(1, N + 1) if not (mask >> x & 1)]
            if not blocked:
                break
            rng.shuffle(blocked)
            for x in blocked:
                if time.time() >= deadline:
                    break
                if can_add(mask, x):
                    mask |= (1 << x)
                    S_size += 1
        
        return mask, S_size

    def simulated_annealing(initial_mask, deadline, rng):
        """Simulated Annealing to escape local optima."""
        current_mask = initial_mask
        current_size = get_size(current_mask)
        best_mask = current_mask
        best_size = current_size
        
        T = 10.0
        T_min = 0.01
        cooling_rate = 0.999
        
        # Precompute list of all elements for random selection
        all_elements = list(range(1, N + 1))
        
        steps = 0
        no_improve_steps = 0
        max_no_improve = 5000
        
        while time.time() < deadline and no_improve_steps < max_no_improve:
            steps += 1
            
            # Choose a move type: Add, Remove, Swap
            move_type = rng.random()
            
            new_mask = current_mask
            new_size = current_size
            valid_move = False
            
            if move_type < 0.3:
                # Try to add a random element
                x = rng.choice(all_elements)
                if not (current_mask >> x & 1):
                    if can_add(current_mask, x):
                        new_mask = current_mask | (1 << x)
                        new_size = current_size + 1
                        valid_move = True
            elif move_type < 0.5:
                # Try to remove a random element
                # Only remove if we have some elements
                if current_size > 10:
                    # Pick a random element from current mask
                    # Efficient way: find a set bit
                    # Or just pick random x and check if it's in mask
                    x = rng.choice(all_elements)
                    if (current_mask >> x & 1):
                        new_mask = current_mask & ~(1 << x)
                        new_size = current_size - 1
                        valid_move = True
            else:
                # Try to swap: remove r, add a
                # Pick random a (not in mask) and r (in mask)
                a = rng.choice(all_elements)
                r = rng.choice(all_elements)
                
                if (current_mask >> a & 1) or not (current_mask >> r & 1):
                    continue # Invalid swap candidates
                
                # Check if removing r and adding a is valid
                # We need to check if a can be added to (mask - r)
                mask_without_r = current_mask & ~(1 << r)
                if can_add(mask_without_r, a):
                    new_mask = mask_without_r | (1 << a)
                    new_size = current_size # Size remains same
                    valid_move = True
            
            if not valid_move:
                # If move was invalid (e.g. conflict), we just stay put or count as step
                # To avoid wasting time on invalid moves, we could retry, 
                # but for simplicity, we just accept the "no change" as a step.
                # However, SA needs to explore. 
                # Let's just continue to next iteration.
                continue
                
            delta = new_size - current_size
            
            if delta >= 0:
                # Accept improvement
                current_mask = new_mask
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
            else:
                # Accept worse move with probability
                # delta is negative
                prob = math.exp(delta / T)
                if rng.random() < prob:
                    current_mask = new_mask
                    current_size = new_size
                    no_improve_steps += 1
                else:
                    no_improve_steps += 1
            
            # Cool down
            T *= cooling_rate
            if T < T_min:
                T = T_min
        
        return best_mask, best_size

    # Main
    best_mask = 0
    best_size = 0
    
    rng = random.Random(42)
    
    # Initial Greedy
    current_mask = fast_greedy(rng)
    current_size = get_size(current_mask)
    
    # Initial Polish
    current_mask, current_size = polish_pass(current_mask, deadline - 20)
    
    if current_size > best_size:
        best_size = current_size
        best_mask = current_mask
    
    # Iterated Simulated Annealing
    # We run SA multiple times, restarting from the best found or a new greedy if stuck
    iterations = 0
    max_iterations = 20 # Limit iterations to ensure we finish
    
    while time.time() < deadline - 5 and iterations < max_iterations:
        iterations += 1
        
        # Perturb the best solution
        perturb_mask = best_mask
        S_list = [x for x in range(1, N + 1) if (perturb_mask >> x & 1)]
        
        # Remove k random elements
        k = rng.randint(5, 15)
        if len(S_list) > k:
            to_remove = rng.sample(S_list, k)
            for r in to_remove:
                perturb_mask &= ~(1 << r)
        
        # Run SA from this perturbed state
        # Allocate remaining time, but keep a buffer
        iter_deadline = min(deadline - 2, time.time() + (deadline - 2 - time.time()) / (max_iterations - iterations))
        
        new_mask, new_size = simulated_annealing(perturb_mask, iter_deadline, rng)
        
        if new_size > best_size:
            best_size = new_size
            best_mask = new_mask
            
            # If we improved significantly, maybe restart SA from this new best
            # The loop naturally does this by perturbing best_mask
        
    # Final Polish on the best solution
    if best_size > 0 and time.time() < deadline - 1:
        best_mask, best_size = polish_pass(best_mask, deadline - 0.5)
    
    # Convert to list
    result = [x for x in range(1, N + 1) if (best_mask >> x & 1)]
    
    return result

# Quick test
if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    t1 = time.time()
    print(f"Size: {len(res)}, Time: {t1-t0:.2f}s")
