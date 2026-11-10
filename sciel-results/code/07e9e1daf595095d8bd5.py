import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    def check_violation(n, adj):
        for k in range(6, n + 1):
            for combo in itertools.combinations(range(n), k):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                degs = [(adj[v] & mask).bit_count() for v in combo]
                if len(set(degs)) == 1:
                    return False, combo
        return True, None

    for n in [20, 19, 18]:
        if time.time() > deadline:
            break
            
        for restart in range(1000):
            if time.time() > deadline:
                break
                
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            for step in range(200):
                if time.time() > deadline:
                    break
                    
                is_valid, violation = check_violation(n, adj)
                if is_valid:
                    # Construct edges
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if (adj[i] >> j) & 1:
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
                
                if violation is None:
                    break
                    
                # Repair
                v_set = set(violation)
                candidates = []
                for i, u in enumerate(violation):
                    for v in violation[i+1:]:
                        candidates.append((u, v))
                
                if not candidates:
                    break
                    
                u, v = random.choice(candidates)
                if (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)

    return {'n': 5, 'edges': []}
