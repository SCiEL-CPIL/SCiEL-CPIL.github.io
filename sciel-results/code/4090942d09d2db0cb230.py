import time
import random
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    def get_structures(n):
        triples = list(combinations(range(n), 3))
        triple_idx = {t: i for i, t in enumerate(triples)}
        num_triples = len(triples)
        
        # Build K4 and K5 structures
        k4s = list(combinations(range(n), 4))
        k4_triples = []
        for k4 in k4s:
            ts = []
            for i in range(4):
                for j in range(i+1, 4):
                    for k in range(j+1, 4):
                        t = (k4[i], k4[j], k4[k])
                        ts.append(triple_idx[t])
            k4_triples.append(np.array(ts, dtype=np.int32))
        num_k4 = len(k4_triples)
        
        k5s = list(combinations(range(n), 5))
        k5_triples = []
        for k5 in k5s:
            ts = []
            for t in combinations(k5, 3):
                ts.append(triple_idx[t])
            k5_triples.append(np.array(ts, dtype=np.int32))
        num_k5 = len(k5_triples)
        
        # For each triple, which K4s and K5s does it belong to?
        triple_to_k4 = [[] for _ in range(num_triples)]
        for i, ts in enumerate(k4_triples):
            for t_idx in ts:
                triple_to_k4[t_idx].append(i)
                
        triple_to_k5 = [[] for _ in range(num_triples)]
        for i, ts in enumerate(k5_triples):
            for t_idx in ts:
                triple_to_k5[t_idx].append(i)
                
        return triples, triple_idx, k4_triples, k5_triples, triple_to_k4, triple_to_k5, num_triples, num_k4, num_k5

    def solve_with_constraint_propagation(n, time_limit, rng_seed=42):
        rng = random.Random(rng_seed)
        triples, triple_idx, k4_triples, k5_triples, triple_to_k4, triple_to_k5, num_triples, num_k4, num_k5 = get_structures(n)
        
        # Use numpy for vectorized operations
        colors = np.array([rng.randint(0, 1) for _ in range(num_triples)], dtype=np.int8)
        
        # Precompute K4 and K5 as numpy arrays for fast indexing
        k4_arr = np.array(k4_triples, dtype=np.int32)  # shape (num_k4, 4)
        k5_arr = np.array(k5_triples, dtype=np.int32)  # shape (num_k5, 10)
        
        # Compute initial violation counts
        # Red K4 violation: all 4 triples red (color 0)
        red_k4_violations = np.all(k4_arr == 0, axis=1)  # shape (num_k4,)
        # Blue K5 violation: all 10 triples blue (color 1)
        blue_k5_violations = np.all(k5_arr == 1, axis=1)  # shape (num_k5,)
        
        num_violations = np.sum(red_k4_violations) + np.sum(blue_k5_violations)
        
        if num_violations == 0:
            return colors.tolist()
        
        # Constraint propagation: identify triples that are "critical" for violations
        # A triple is critical if flipping it would resolve at least one violation
        # without creating new ones
        
        # For efficiency, we'll use a greedy approach with constraint awareness
        # Track which triples appear in violations
        violation_triples_k4 = set()
        for i in range(num_k4):
            if red_k4_violations[i]:
                for t_idx in k4_triples[i]:
                    violation_triples_k4.add(t_idx)
                    
        violation_triples_k5 = set()
        for i in range(num_k5):
            if blue_k5_violations[i]:
                for t_idx in k5_triples[i]:
                    violation_triples_k5.add(t_idx)
        
        # Main search loop with constraint propagation
        step = 0
        while time.time() < time_limit:
            # Recompute violations periodically for accuracy
            if step % 10 == 0:
                red_k4_violations = np.all(k4_arr == 0, axis=1)
                blue_k5_violations = np.all(k5_arr == 1, axis=1)
                num_violations = np.sum(red_k4_violations) + np.sum(blue_k5_violations)
                if num_violations == 0:
                    return colors.tolist()
                
                # Update violation triple sets
                violation_triples_k4 = set()
                for i in range(num_k4):
                    if red_k4_violations[i]:
                        for t_idx in k4_triples[i]:
                            violation_triples_k4.add(t_idx)
                            
                violation_triples_k5 = set()
                for i in range(num_k5):
                    if blue_k5_violations[i]:
                        for t_idx in k5_triples[i]:
                            violation_triples_k5.add(t_idx)
            
            # Choose which type of violation to fix
            num_k4_viol = np.sum(red_k4_violations)
            num_k5_viol = np.sum(blue_k5_violations)
            
            if num_k4_viol == 0 and num_k5_viol == 0:
                return colors.tolist()
            
            # Bias toward the more common violation type
            if num_k4_viol > 0 and (num_k5_viol == 0 or rng.random() < num_k4_viol / (num_k4_viol + num_k5_viol)):
                # Fix a red K4 violation
                viol_idx = rng.choice(np.where(red_k4_violations)[0])
                ts = k4_triples[viol_idx]
                
                # Constraint propagation: prefer triples that appear in fewer other potential violations
                # Score each triple by how many other K4s/K5s it belongs to
                best_t_idx = None
                best_score = float('inf')
                for t_idx in ts:
                    if colors[t_idx] != 0:
                        continue
                    # Count how many other violations this triple is part of
                    score = len(triple_to_k4[t_idx]) + len(triple_to_k5[t_idx])
                    # Also penalize if flipping would create new violations
                    # (approximate by checking neighbors)
                    if score < best_score:
                        best_score = score
                        best_t_idx = t_idx
                
                if best_t_idx is None:
                    best_t_idx = rng.choice(ts)
                
                # Flip 0 -> 1
                t_idx = best_t_idx
                colors[t_idx] = 1
                
                # Update violation arrays incrementally (approximate)
                # For K4s containing t_idx:
                for k4_i in triple_to_k4[t_idx]:
                    # If this K4 was all red, it's no longer all red
                    if red_k4_violations[k4_i]:
                        red_k4_violations[k4_i] = False
                # For K5s containing t_idx:
                # Flipping to blue might create a new blue K5 if all others were blue
                for k5_i in triple_to_k5[t_idx]:
                    if not blue_k5_violations[k5_i]:
                        # Check if all other triples in this K5 are blue
                        ts_k5 = k5_triples[k5_i]
                        all_others_blue = True
                        for other_t in ts_k5:
                            if other_t != t_idx and colors[other_t] != 1:
                                all_others_blue = False
                                break
                        if all_others_blue:
                            blue_k5_violations[k5_i] = True
                            
            else:
                # Fix a blue K5 violation
                viol_idx = rng.choice(np.where(blue_k5_violations)[0])
                ts = k5_triples[viol_idx]
                
                best_t_idx = None
                best_score = float('inf')
                for t_idx in ts:
                    if colors[t_idx] != 1:
                        continue
                    score = len(triple_to_k4[t_idx]) + len(triple_to_k5[t_idx])
                    if score < best_score:
                        best_score = score
                        best_t_idx = t_idx
                        
                if best_t_idx is None:
                    best_t_idx = rng.choice(ts)
                    
                t_idx = best_t_idx
                colors[t_idx] = 0
                
                # Update violations
                for k5_i in triple_to_k5[t_idx]:
                    if blue_k5_violations[k5_i]:
                        blue_k5_violations[k5_i] = False
                for k4_i in triple_to_k4[t_idx]:
                    if not red_k4_violations[k4_i]:
                        ts_k4 = k4_triples[k4_i]
                        all_others_red = True
                        for other_t in ts_k4:
                            if other_t != t_idx and colors[other_t] != 0:
                                all_others_red = False
                                break
                        if all_others_red:
                            red_k4_violations[k4_i] = True
            
            step += 1
            
            # Periodic restart if stuck
            if step % 500 == 0:
                # Check if we've made progress
                current_viol = np.sum(red_k4_violations) + np.sum(blue_k5_violations)
                if current_viol >= num_violations:
                    # No progress, restart with new random seed
                    colors = np.array([rng.randint(0, 1) for _ in range(num_triples)], dtype=np.int8)
                    red_k4_violations = np.all(k4_arr == 0, axis=1)
                    blue_k5_violations = np.all(k5_arr == 1, axis=1)
                    num_violations = np.sum(red_k4_violations) + np.sum(blue_k5_violations)
                    if num_violations == 0:
                        return colors.tolist()
                    step = 0
        
        return None

    # Try n=34 first with multiple restarts
    best_result = None
    
    # Allocate time for different n values
    # n=34: most time, n=33: less, n=32: fallback
    
    # Try n=34
    t_start = time.time()
    time_for_34 = min(100, deadline - t_start - 20)
    
    for attempt in range(5):
        remaining = deadline - time.time() - 5
        if remaining < 10:
            break
        alloc = min(remaining, max(15, time_for_34 / (5 - attempt)))
        result = solve_with_constraint_propagation(34, time.time() + alloc, rng_seed=42 + attempt)
        if result:
            return {'n': 34, 'colors': result}
    
    # Try n=33
    t_start = time.time()
    remaining = deadline - t_start - 10
    if remaining > 15:
        for attempt in range(3):
            alloc = min(remaining / (3 - attempt), 20)
            result = solve_with_constraint_propagation(33, time.time() + alloc, rng_seed=100 + attempt)
            if result:
                return {'n': 33, 'colors': result}
    
    # Try n=32
    t_start = time.time()
    remaining = deadline - t_start - 5
    if remaining > 10:
        result = solve_with_constraint_propagation(32, time.time() + remaining - 3, rng_seed=200)
        if result:
            return {'n': 32, 'colors': result}
    
    # Fallback: use seed solution for n=9
    n = 9
    seed_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
    return {'n': n, 'colors': seed_colors}
