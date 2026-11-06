import math
import random
import time

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    # Precompute bad pairs
    bad_pairs = set()
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g = math.gcd(a, b)
            u = a // g
            v = b // g
            if g % (u + v) == 0:
                bad_pairs.add((a, b))
    
    adj = [set() for _ in range(N + 1)]
    for a, b in bad_pairs:
        adj[a].add(b)
        adj[b].add(a)
        
    def can_add(v, S):
        for u in adj[v]:
            if u in S:
                return False
        return True

    def can_swap(v_in, v_out, S):
        for u in adj[v_out]:
            if u in S and u != v_in:
                return False
        return True

    best_S = set()
    best_size = 0
    
    # Try several random restarts
    while time.time() < deadline:
        # Greedy init
        S = set()
        # Randomize order
        vertices = list(range(1, N + 1))
        random.shuffle(vertices)
        
        for v in vertices:
            if can_add(v, S):
                S.add(v)
                
        if len(S) > best_size:
            best_S = set(S)
            best_size = len(S)
            
        # Local Search
        # We will try to improve S
        no_improve_count = 0
        while time.time() < deadline and no_improve_count < 100:
            improved = False
            candidates = [v for v in range(1, N + 1) if v not in S]
            if not candidates:
                break
                
            # Try adding
            random.shuffle(candidates)
            for v in candidates:
                if can_add(v, S):
                    S.add(v)
                    improved = True
                    if len(S) > best_size:
                        best_S = set(S)
                        best_size = len(S)
                    break
                    
            if improved:
                no_improve_count = 0
                continue
                
            # Try swapping
            # Pick random v_out
            v_out = random.choice(candidates)
            neighbors_in_S = [u for u in adj[v_out] if u in S]
            
            if not neighbors_in_S:
                # Should have been added
                S.add(v_out)
                improved = True
                if len(S) > best_size:
                    best_S = set(S)
                    best_size = len(S)
                no_improve_count = 0
                continue
                
            # Try to swap with a neighbor
            v_in = random.choice(neighbors_in_S)
            if can_swap(v_in, v_out, S):
                S.remove(v_in)
                S.add(v_out)
                improved = True
                no_improve_count = 0
                continue
                
            no_improve_count += 1

    return sorted(best_S)
