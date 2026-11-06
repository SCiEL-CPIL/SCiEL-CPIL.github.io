import time
import random

N = 12
FULL_MASK = (1 << N) - 1
DEADLINE = 58.0  # Safety margin before 60s kill

def construct():
    start_time = time.time()
    
    def build_next(trans):
        """Precompute subset transitions for a single letter."""
        size = 1 << N
        next_sub = [0] * size
        # next_sub[0] is 0
        for mask in range(1, size):
            # Use lowest set bit trick
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            rest = mask ^ lsb
            next_sub[mask] = next_sub[rest] | (1 << trans[idx])
        return next_sub

    def compute_threshold(next_a, next_b):
        """BFS on subset automaton to find shortest reset word length."""
        size = 1 << N
        dist = [-1] * size
        dist[FULL_MASK] = 0
        queue = [FULL_MASK]
        head = 0
        
        while head < len(queue):
            s = queue[head]
            head += 1
            d = dist[s]
            
            # Try 'a'
            n1 = next_a[s]
            if n1 == 1: # Single state 0
                return d + 1
            if dist[n1] == -1:
                dist[n1] = d + 1
                queue.append(n1)
                
            # Try 'b'
            n2 = next_b[s]
            if n2 == 1:
                return d + 1
            if dist[n2] == -1:
                dist[n2] = d + 1
                queue.append(n2)
                
        return 0 # Not synchronizing

    def evaluate(fa, fb):
        next_a = build_next(fa)
        next_b = build_next(fb)
        return compute_threshold(next_a, next_b)

    rng = random.Random(20240517)
    
    best_score = 0
    best_fa = None
    best_fb = None
    
    # Strategy: 
    # 1. Use a permutation for 'a' (cycle).
    # 2. Initialize 'b' with a structure known to have high threshold (e.g., pinwheel).
    # 3. Simulated Annealing on 'b' (and occasionally 'a') to escape local minima.
    
    def pinwheel_b(cycle_a):
        """Create a 'b' transition that works well with a cycle 'a'.
        Standard pinwheel: b(i) = i-1 mod N, except for one state which maps to 0 or similar.
        """
        fb = [0] * N
        for i in range(N):
            fb[i] = (i - 1) % N
        # Ensure it's not just a cycle (which wouldn't sync)
        # Force one state to map to 0 to break the cycle.
        fb[0] = 0
        return fb

    def random_b():
        return [rng.randrange(N) for _ in range(N)]

    def simulated_annealing(fa, fb, max_iter=20000):
        nonlocal best_score, best_fa, best_fb
        
        curr_fa = fa[:]
        curr_fb = fb[:]
        curr_score = evaluate(curr_fa, curr_fb)
        
        if curr_score > best_score:
            best_score = curr_score
            best_fa = curr_fa[:]
            best_fb = curr_fb[:]
            
        # Annealing parameters
        # Start with high temperature, cool down exponentially
        # T0 = 10.0, alpha = 0.999
        # We want to explore broadly at first, then refine.
        # Since score is integer, a small change in score is significant.
        # Let's say T starts at 5.0 and cools to 0.1.
        T = 5.0
        alpha = 0.9995 # Slow cooling to allow more exploration
        min_T = 0.01
        
        next_a = build_next(curr_fa)
        
        for i in range(max_iter):
            if time.time() - start_time > DEADLINE:
                return
            
            # Decide what to mutate
            # 90% mutate b, 10% mutate a
            if rng.random() < 0.9:
                # Mutate b
                old_b = curr_fb[:]
                state = rng.randrange(N)
                new_dest = rng.randrange(N)
                curr_fb[state] = new_dest
                
                next_b = build_next(curr_fb)
                new_score = compute_threshold(next_a, next_b)
                
                delta = new_score - curr_score
                
                if delta > 0 or rng.random() < pow(2.718, delta / max(T, 0.001)):
                    # Accept
                    curr_score = new_score
                    if curr_score > best_score:
                        best_score = curr_score
                        best_fa = curr_fa[:]
                        best_fb = curr_fb[:]
                else:
                    # Revert
                    curr_fb = old_b
                
            else:
                # Mutate a (Permutation)
                old_fa = curr_fa[:]
                # Swap two destinations in a to keep it a permutation
                s1, s2 = rng.sample(range(N), 2)
                curr_fa[s1], curr_fa[s2] = curr_fa[s2], curr_fa[s1]
                
                next_a = build_next(curr_fa)
                next_b = build_next(curr_fb)
                new_score = compute_threshold(next_a, next_b)
                
                delta = new_score - curr_score
                
                if delta > 0 or rng.random() < pow(2.718, delta / max(T, 0.001)):
                    # Accept
                    curr_score = new_score
                    if curr_score > best_score:
                        best_score = curr_score
                        best_fa = curr_fa[:]
                        best_fb = curr_fb[:]
                else:
                    curr_fa = old_fa
                    # Rebuild next_a as it was reverted
                    next_a = build_next(curr_fa)
            
            # Cool down
            T *= alpha
            if T < min_T:
                T = min_T

    # Main Loop: Multiple restarts with different initial structures
    while time.time() - start_time < DEADLINE:
        # Generate a random cycle for 'a'
        fa = list(range(N))
        rng.shuffle(fa)
        
        # Generate initial 'b'
        # Try different pinwheel variants
        variant = rng.randrange(3)
        if variant == 0:
            fb = pinwheel_b(fa)
        elif variant == 1:
            # b(i) = (i+1)%N for i!=0, b(0)=0.
            fb = [0]*N
            for i in range(1, N):
                fb[i] = (i+1) % N
            fb[0] = 0
        else:
            fb = random_b()
        
        # Run simulated annealing for a short duration
        # We limit iterations to allow restarts, but SA is more robust than hill climbing.
        # 20000 iterations is a good balance.
        simulated_annealing(fa, fb, max_iter=20000)
        
        # If we are close to deadline, do one final long climb on the best found so far
        if time.time() - start_time > DEADLINE - 10:
            if best_fa:
                simulated_annealing(best_fa, best_fb, max_iter=100000) # Run until deadline
            break

    if best_fa is None:
        # Fallback: return a known good automaton or random
        # Just return a random one, though it might not sync well
        return [rng.randrange(N) for _ in range(N)], [rng.randrange(N) for _ in range(N)]

    return best_fa, best_fb
