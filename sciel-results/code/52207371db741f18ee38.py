import time
import random
import math

N = 64

def dist2(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def get_sidon_points(N):
    xs, used = [], set()
    x = 0
    while x < N:
        diffs = {abs(x - y) for y in xs}
        if not (diffs & used):
            used |= diffs
            xs.append(x)
        x += 1
    return [(x, 0) for x in xs]

def construct():
    deadline = time.time() + 290
    all_points = [(x, y) for x in range(N) for y in range(N)]
    best_set = []
    
    sidon_x = get_sidon_points(N)
    sidon_y = [(y, x) for (x, y) in sidon_x] # Wait, get_sidon_points returns (x,0). So (y, x) is wrong.
    # sidon_x is [(x, 0)].
    # For Y axis, we want (0, y).
    # So sidon_y = [(0, x) for x in [p[0] for p in sidon_x]]
    
    seeds = [
        [],
        sidon_x,
        [(0, x) for x in [p[0] for p in sidon_x]]
    ]
    
    while time.time() < deadline:
        seed_choice = random.choice(seeds)
        
        S = []
        S_set = set()
        D = {}
        
        # Initialize with seed
        for P in seed_choice:
            S.append(P)
            S_set.add(P)
        
        # Build D for seed
        for i in range(len(S)):
            for j in range(len(S)):
                if i != j:
                    D.setdefault(S[i], set()).add(dist2(S[i], S[j]))
        
        # List of remaining points
        remaining = [p for p in all_points if p not in S_set]
        random.shuffle(remaining)
        
        # Greedy add
        for P in remaining:
            if time.time() > deadline:
                break
            if P in S_set:
                continue
            
            valid = True
            dists_to_existing = []
            
            for Q in S:
                d = dist2(P, Q)
                dists_to_existing.append(d)
                if d in D[Q]:
                    valid = False
                    break
            
            if valid:
                if len(set(dists_to_existing)) != len(dists_to_existing):
                    valid = False
            
            if valid:
                S.append(P)
                S_set.add(P)
                D[P] = set(dists_to_existing)
                for Q in S[:-1]:
                    D[Q].add(dist2(P, Q))
        
        if len(S) > len(best_set):
            best_set = S.copy()
            
    return best_set
