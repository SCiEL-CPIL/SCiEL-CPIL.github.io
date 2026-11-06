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
    # 3. Hill climb on 'b' (and occasionally 'a').
    
    def pinwheel_b(cycle_a):
        """Create a 'b' transition that works well with a cycle 'a'.
        Standard pinwheel: b(i) = i-1 mod N, except for one state which maps to 0 or similar.
        Actually, a common high-threshold structure is:
        a: 0->1->2->...->N-1->0
        b: 0->0, 1->0, 2->1, ..., N-1->N-2 (This is a specific comb)
        Let's try: b(i) = (i-1) % N for all i, but b(0) = 0.
        """
        fb = [0] * N
        for i in range(N):
            fb[i] = (i - 1) % N
        # Ensure it's not just a cycle (which wouldn't sync)
        # The above is a cycle 0->N-1->N-2...->0. 
        # We need a "sink" or contraction.
        # Let's force one state to map to 0 to break the cycle.
        # Usually mapping the "start" of the cycle back to itself or 0 helps.
        # Let's try: b(i) = i-1 mod N, but b(0)=0.
        fb[0] = 0
        return fb

    def random_b():
        return [rng.randrange(N) for _ in range(N)]

    def hill_climb(fa, fb, max_iter=5000):
        nonlocal best_score, best_fa, best_fb
        
        curr_fa = fa[:]
        curr_fb = fb[:]
        curr_score = evaluate(curr_fa, curr_fb)
        
        if curr_score > best_score:
            best_score = curr_score
            best_fa = curr_fa[:]
            best_fb = curr_fb[:]
            
        no_improve_count = 0
        # Precompute next_a once if fa is fixed during this phase
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
                
                if new_score > curr_score:
                    curr_score = new_score
                    no_improve_count = 0
                    if curr_score > best_score:
                        best_score = curr_score
                        best_fa = curr_fa[:]
                        best_fb = curr_fb[:]
                else:
                    # Revert
                    curr_fb = old_b
                    no_improve_count += 1
                    
                    # If stuck for too long, try a bigger jump
                    if no_improve_count > 100:
                        # Swap two destinations in b
                        s1, s2 = rng.sample(range(N), 2)
                        curr_fb[s1], curr_fb[s2] = curr_fb[s2], curr_fb[s1]
                        next_b = build_next(curr_fb)
                        new_score = compute_threshold(next_a, next_b)
                        if new_score > curr_score:
                            curr_score = new_score
                            no_improve_count = 0
                            if curr_score > best_score:
                                best_score = curr_score
                                best_fa = curr_fa[:]
                                best_fb = curr_fb[:]
                        else:
                            # Revert swap
                            curr_fb[s1], curr_fb[s2] = curr_fb[s2], curr_fb[s1]
                            # If still stuck, restart b with a different pinwheel variant
                            if no_improve_count > 200:
                                curr_fb = pinwheel_b(curr_fa)
                                curr_fb[rng.randrange(N)] = rng.randrange(N) # Perturb
                                next_b = build_next(curr_fb)
                                curr_score = compute_threshold(next_a, next_b)
                                no_improve_count = 0
                                if curr_score > best_score:
                                    best_score = curr_score
                                    best_fa = curr_fa[:]
                                    best_fb = curr_fb[:]
            else:
                # Mutate a (Permutation)
                old_fa = curr_fa[:]
                # Swap two destinations in a to keep it a permutation? 
                # Or just random change? If we break permutation, we might lose the "cycle" property 
                # which is good for high threshold. 
                # Let's try swapping two images in a to keep it a permutation.
                s1, s2 = rng.sample(range(N), 2)
                curr_fa[s1], curr_fa[s2] = curr_fa[s2], curr_fa[s1]
                
                next_a = build_next(curr_fa)
                next_b = build_next(curr_fb)
                new_score = compute_threshold(next_a, next_b)
                
                if new_score > curr_score:
                    curr_score = new_score
                    no_improve_count = 0
                    if curr_score > best_score:
                        best_score = curr_score
                        best_fa = curr_fa[:]
                        best_fb = curr_fb[:]
                else:
                    curr_fa = old_fa
                    no_improve_count += 1
                    if no_improve_count > 50:
                         # Rebuild next_a as it was reverted
                         next_a = build_next(curr_fa)

    # Main Loop: Multiple restarts with different initial structures
    while time.time() - start_time < DEADLINE:
        # Generate a random cycle for 'a'
        fa = list(range(N))
        rng.shuffle(fa)
        
        # Generate initial 'b'
        # Mix of pinwheel and random to ensure diversity
        if rng.random() < 0.5:
            fb = pinwheel_b(fa)
            # Perturb slightly
            fb[rng.randrange(N)] = rng.randrange(N)
        else:
            fb = random_b()
        
        hill_climb(fa, fb)

    if best_fa is None:
        # Fallback: return a known synchronizing automaton
        fa = list(range(N))
        fb = [0] * N
        return fa, fb

    return best_fa, best_fb
