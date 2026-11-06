import time
import numpy as np
from itertools import combinations
from math import comb

def construct():
    start_time = time.time()
    deadline = start_time + 112  # safety margin before 150s kill

    def search_for_n(n, time_limit, seed=0):
        M = comb(n, 3)
        triples = list(combinations(range(n), 3))
        tri_to_idx = {tri: i for i, tri in enumerate(triples)}

        # Precompute incidence
        quad_triples = []
        for quad in combinations(range(n), 4):
            idxs = [tri_to_idx[tuple(x for j, x in enumerate(quad) if j != i)] for i in range(4)]
            quad_triples.append(idxs)
        quint_triples = []
        for quint in combinations(range(n), 5):
            idxs = [tri_to_idx[t] for t in combinations(quint, 3)]
            quint_triples.append(idxs)

        n_quads = len(quad_triples)
        n_quints = len(quint_triples)

        # Inverse incidence
        triple_to_quads = [[] for _ in range(M)]
        for qidx, tidxs in enumerate(quad_triples):
            for t in tidxs:
                triple_to_quads[t].append(qidx)
        
        triple_to_quints = [[] for _ in range(M)]
        for qidx, tidxs in enumerate(quint_triples):
            for t in tidxs:
                triple_to_quints[t].append(qidx)

        rng = np.random.RandomState(seed)

        # Initialize colors
        colors = np.zeros(M, dtype=np.int8)
        for i, (a, b, c) in enumerate(triples):
            colors[i] = 0 if (a + b + c) % 3 == 0 else 1

        # Initialize counts
        red_counts = np.zeros(n_quads, dtype=np.int32)
        blue_counts = np.zeros(n_quints, dtype=np.int32)
        
        for t in range(M):
            if colors[t] == 0:
                for q in triple_to_quads[t]:
                    red_counts[q] += 1
            else:
                for k in triple_to_quints[t]:
                    blue_counts[k] += 1
        
        total_violations = np.sum(red_counts == 4) + np.sum(blue_counts == 10)

        def flip(t):
            nonlocal total_violations
            old_color = colors[t]
            
            if old_color == 0:
                for q in triple_to_quads[t]:
                    old_c = red_counts[q]
                    if old_c == 4:
                        total_violations -= 1
                    red_counts[q] = old_c - 1
                for k in triple_to_quints[t]:
                    old_c = blue_counts[k]
                    blue_counts[k] = old_c + 1
                    if old_c + 1 == 10:
                        total_violations += 1
                colors[t] = 1
            else:
                for k in triple_to_quints[t]:
                    old_c = blue_counts[k]
                    if old_c == 10:
                        total_violations -= 1
                    blue_counts[k] = old_c - 1
                for q in triple_to_quads[t]:
                    old_c = red_counts[q]
                    red_counts[q] = old_c + 1
                    if old_c + 1 == 4:
                        total_violations += 1
                colors[t] = 0

        # Greedy Conflict Elimination
        # We maintain a list of "active" triples involved in violations
        # To be efficient, we just scan for the best move each iteration.
        
        # Optimization: Instead of scanning all M triples, we can maintain a set of candidates.
        # But M is only ~6000, so scanning is fast.
        
        while time.time() < start_time + time_limit:
            if total_violations == 0:
                return colors.tolist()
            
            # Find the triple that reduces violations the most
            best_delta = 0
            best_t = -1
            
            # We only care about triples that are part of a violation
            # A triple t is part of a violation if:
            # - It's Red and in a Quad with 4 Red (red_counts[q]==4)
            # - It's Blue and in a Quint with 10 Blue (blue_counts[k]==10)
            # - OR, flipping it might *create* a violation?
            #   We want to minimize total violations.
            #   Delta = NewViolations - OldViolations.
            #   We want Delta < 0.
            
            # Let's just evaluate all triples? 6000 * 500 = 3M ops.
            # 3M ops in Python is ~0.1-0.3s.
            # If we do this 100 times, that's 30s.
            # We need to be smarter.
            
            # Only evaluate triples in active violations.
            candidates = set()
            
            # Add triples from violating quads
            for q in range(n_quads):
                if red_counts[q] == 4:
                    for t in quad_triples[q]:
                        candidates.add(t)
                        
            # Add triples from violating quints
            for k in range(n_quints):
                if blue_counts[k] == 10:
                    for t in quint_triples[k]:
                        candidates.add(t)
            
            if not candidates:
                # Should not happen if total_violations > 0
                break
                
            # Evaluate candidates
            for t in candidates:
                old_color = colors[t]
                delta = 0
                if old_color == 0:
                    # Red -> Blue
                    for q in triple_to_quads[t]:
                        if red_counts[q] == 4: delta -= 1
                    for k in triple_to_quints[t]:
                        if blue_counts[k] == 9: delta += 1
                else:
                    # Blue -> Red
                    for k in triple_to_quints[t]:
                        if blue_counts[k] == 10: delta -= 1
                    for q in triple_to_quads[t]:
                        if red_counts[q] == 3: delta += 1
                
                if delta < best_delta:
                    best_delta = delta
                    best_t = t
            
            if best_t != -1:
                flip(best_t)
            else:
                # No improving move among candidates.
                # Try a random move from candidates (WalkSAT)
                t = rng.choice(list(candidates))
                flip(t)
                
        return None

    best_n = 1
    best_colors = [0]

    # Try to find a solution for n=34 directly, as it's the target.
    # If we fail, try smaller n.
    # But the task says "Maximize n".
    # We should try 34, then 33, etc.
    
    # However, 34 is the known maximum.
    # Let's try 34 with multiple seeds.
    
    for seed in range(10):
        if time.time() > deadline - 5:
            break
        res = search_for_n(34, 10.0, seed=seed)
        if res is not None:
            best_n = 34
            best_colors = res
            break

    if best_n < 34:
        for n in range(33, 1, -1):
            if time.time() > deadline - 5:
                break
            res = search_for_n(n, 10.0, seed=n)
            if res is not None:
                best_n = n
                best_colors = res
                break

    return {'n': best_n, 'colors': best_colors}
