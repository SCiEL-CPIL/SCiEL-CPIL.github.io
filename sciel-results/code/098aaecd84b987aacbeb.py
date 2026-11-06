import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    def try_construct(n, max_trials):
        max_rem = 38
        
        for trial in range(max_trials):
            if time.time() > deadline:
                return None
            
            random.seed(trial) # Reproducible per trial
            
            # Initialize complete graph
            adj = [0] * n
            for i in range(n):
                mask = (1 << n) - 1
                adj[i] = mask ^ (1 << i)
            
            rem = [0] * n
            valid = True
            
            # Shuffle pair order for this trial
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(pairs)
            
            while True:
                if time.time() > deadline:
                    valid = False
                    break
                
                # Find a C4
                c4_found = False
                best_edge = None
                best_cost = float('inf')
                
                for i, j in pairs:
                    # Check if i and j are adjacent
                    if not (adj[i] >> j) & 1:
                        continue
                    
                    common = adj[i] & adj[j]
                    if common == 0:
                        continue
                    k_bit = common & -common
                    k = k_bit.bit_length() - 1
                    common ^= k_bit
                    if common == 0:
                        continue
                    l_bit = common & -common
                    l = l_bit.bit_length() - 1
                    
                    # C4 found: i-k-j-l-i
                    # Edges: (i,k), (j,k), (j,l), (i,l)
                    c4_found = True
                    
                    # Collect candidate edges and their costs
                    candidates = []
                    for u, v in [(i, k), (j, k), (j, l), (i, l)]:
                        cost = max(rem[u], rem[v])
                        candidates.append((cost, u, v))
                    
                    # Pick the best edge (lowest cost)
                    # To add randomness, pick randomly from those with cost <= best_cost + 1
                    candidates.sort()
                    min_c = candidates[0][0]
                    good = [c for c in candidates if c[0] <= min_c + 1]
                    _, u, v = random.choice(good)
                    
                    if rem[u] >= max_rem or rem[v] >= max_rem:
                        valid = False
                        break
                    
                    # Remove edge (u, v)
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    rem[u] += 1
                    rem[v] += 1
                    break # Restart pair search
                
                if not c4_found:
                    break
                if not valid:
                    break
            
            if valid:
                min_deg = min(n - 1 - rem[i] for i in range(n))
                if min_deg >= n - 39:
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if (adj[i] >> j) & 1:
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
        
        return None

    # Try n=47
    res = try_construct(47, 20)
    if res:
        return res

    # Try n=46
    res = try_construct(46, 20)
    if res:
        return res

    # Try n=45
    res = try_construct(45, 20)
    if res:
        return res

    # Fallback
    return {'n': 39, 'edges': []}
