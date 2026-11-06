import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 110.0  # 10s margin
    
    # Try different limits, starting high and going down if too slow/memory error
    # We'll estimate the number of powerful numbers to decide L
    # P(L) ~ C * L^0.5
    # Let's try L = 10^8 first.
    
    limits = [10**8, 5*10**7, 10**7]
    
    best_N = 2
    best_pairs = []
    best_count = 0
    
    for L in limits:
        if time.time() > deadline:
            break
            
        # Generate powerful numbers up to L
        powerful_set = set()
        # m = u^2 v^3
        # v^3 <= L => v <= L^(1/3)
        max_v = int(L ** (1/3)) + 1
        for v in range(1, max_v + 1):
            v3 = v ** 3
            if v3 > L:
                break
            # u^2 <= L / v^3 => u <= sqrt(L/v^3)
            max_u = int((L // v3) ** 0.5) + 1
            for u in range(1, max_u + 1):
                m = u * u * v3
                if m <= L:
                    powerful_set.add(m)
        
        powerful_list = sorted(powerful_set)
        n_pow = len(powerful_list)
        
        # Estimate pairs
        # If n_pow is too large, skip?
        # 10^8 -> ~10000 pow -> 50M pairs. OK.
        # 10^9 -> ~30000 pow -> 450M pairs. Might be slow.
        
        counts = defaultdict(int)
        # Store pairs for the best N found so far in this iteration?
        # We need to retrieve pairs for the global best.
        # We can store all pairs for the current best candidate.
        
        current_best_N = 0
        current_best_count = 0
        current_best_pairs = []
        
        # To avoid storing all pairs for all N, we only store for the current best.
        # But we need to update if we find a better one.
        
        for i in range(n_pow):
            a = powerful_list[i]
            # b must be >= a
            # b = N - a
            # We iterate j from i to n_pow-1
            for j in range(i, n_pow):
                b = powerful_list[j]
                s = a + b
                counts[s] += 1
                # Update current best
                if counts[s] > current_best_count:
                    current_best_count = counts[s]
                    current_best_N = s
                    # Reconstruct pairs for this N?
                    # We can just store the count and N, and re-derive pairs at the end.
                    # But re-deriving requires iterating again.
                    # Or we can store the pairs incrementally?
                    # Storing pairs for the best N is efficient if we clear when updated.
                    current_best_pairs = [] # Clear old
                    # We don't know all pairs yet, only that this is the best so far.
                    # We can just store the N and count.
        
        # After loop, we have current_best_N and current_best_count
        if current_best_count > best_count:
            best_count = current_best_count
            best_N = current_best_N
            # Find pairs for best_N
            best_pairs = []
            pow_set = set(powerful_list)
            for a in powerful_list:
                if a > best_N // 2:
                    break
                b = best_N - a
                if b in pow_set:
                    best_pairs.append([a, b])
                    
        # Check time
        if time.time() > deadline:
            break
            
    return {'N': best_N, 'pairs': best_pairs}
