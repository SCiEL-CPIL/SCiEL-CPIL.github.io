import random
import time
import itertools
import numpy as np

N = 16
GRID_SIZE = N * N

def construct():
    # 1. Setup
            points = list(itertools.product(range(N), repeat=2))
            idx_to_point = {i: p for i, p in enumerate(points)}
            
            # Precompute dist2
            dist2 = [[0]*GRID_SIZE for _ in range(GRID_SIZE)]
            for i in range(GRID_SIZE):
                xi, yi = points[i]
                for j in range(GRID_SIZE):
                    xj, yj = points[j]
                    dist2[i][j] = (xi-xj)**2 + (yi-yj)**2
            
            # Precompute iso_with_pair
            # iso_with_pair[u][v] is a set of indices q such that (u,v,q) is isosceles
            iso_with_pair = [[set() for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
            for u in range(GRID_SIZE):
                for v in range(u+1, GRID_SIZE):
                    d_uv = dist2[u][v]
                    for q in range(GRID_SIZE):
                        if q == u or q == v: continue
                        d_uq = dist2[u][q]
                        d_vq = dist2[v][q]
                        if d_uv == d_uq or d_uv == d_vq or d_uq == d_vq:
                            iso_with_pair[u][v].add(q)
                            iso_with_pair[v][u].add(q)
                            
            # Helper: Calculate block counts for a set S
            def calc_block_counts(S):
                counts = [0] * GRID_SIZE
                S_list = list(S)
                for i in range(len(S_list)):
                    u = S_list[i]
                    for j in range(i+1, len(S_list)):
                        v = S_list[j]
                        for q in iso_with_pair[u][v]:
                            counts[q] += 1
                return counts

            # Helper: Check if S is maximal
            def is_maximal(S, counts):
                S_set = set(S)
                for q in range(GRID_SIZE):
                    if q in S_set:
                        continue
                    if counts[q] == 0:
                        return False
                return True

            # Helper: Greedy build
            def greedy_build(order):
                S = []
                S_set = set()
                counts = [0] * GRID_SIZE
                for p in order:
                    if counts[p] == 0:
                        # Add p
                        S.append(p)
                        S_set.add(p)
                        # Update counts
                        for u in S_set:
                            if u == p: continue
                            for q in iso_with_pair[p][u]:
                                counts[q] += 1
                return S, counts

            # Helper: Minimize
            def minimize(S, counts):
                S = list(S)
                changed = True
                while changed:
                    changed = False
                    for i in range(len(S)):
                        # Try removing S[i]
                        trial_S = S[:i] + S[i+1:]
                        trial_counts = calc_block_counts(trial_S)
                        if is_maximal(trial_S, trial_counts):
                            S = trial_S
                            counts = trial_counts
                            changed = True
                            break
                return S, counts

            # Main Search
            random.seed(42)
            start_time = time.time()
            deadline = start_time + 280
            
            best_S = None
            best_size = GRID_SIZE + 1
            
            # Initial heuristic: Most constrained first?
            # Calculate degree (number of isosceles triangles involving point)
            degrees = [0] * GRID_SIZE
            for u in range(GRID_SIZE):
                for v in range(u+1, GRID_SIZE):
                    for q in iso_with_pair[u][v]:
                        degrees[u] += 1
                        degrees[v] += 1
                        degrees[q] += 1
                        # Note: this counts each triangle 3 times? 
                        # We just want a heuristic weight.
            
            # Try different strategies
            strategies = [
                lambda: list(range(GRID_SIZE)), # Random
                lambda: sorted(range(GRID_SIZE), key=lambda x: -degrees[x]), # High degree first
                lambda: sorted(range(GRID_SIZE), key=lambda x: degrees[x]), # Low degree first
            ]
            
            for attempt in range(20000):
                if time.time() > deadline:
                    break
                
                # Choose strategy
                if attempt % 3 == 0:
                    order = list(range(GRID_SIZE))
                    random.shuffle(order)
                elif attempt % 3 == 1:
                    order = sorted(range(GRID_SIZE), key=lambda x: -degrees[x])
                    # Add some randomness to break ties
                    random.shuffle(order) # Wait, this destroys the sort.
                    # Better: sort, then swap a few?
                    # Or just use the sorted order as is for deterministic runs?
                    # Let's stick to random for most, and specific heuristics for some.
                    order = sorted(range(GRID_SIZE), key=lambda x: (degrees[x], random.random()))
                else:
                    order = sorted(range(GRID_SIZE), key=lambda x: (degrees[x], random.random()))
                    
                S, counts = greedy_build(order)
                
                # Minimize
                S, counts = minimize(S, counts)
                
                if len(S) < best_size:
                    best_size = len(S)
                    best_S = S
                    if best_size <= 10:
                        # Keep searching for smaller?
                        pass
                        # break # Don't break, we want minimum maximal.
                        
            return [idx_to_point[i] for i in best_S]
