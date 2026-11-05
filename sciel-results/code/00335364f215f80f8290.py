"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 64
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 285  # Safety margin
    
    # Grid points
    points = [(x, y) for y in range(N) for x in range(N)]
    num_points = N * N
    
    # Precompute squared distances
    xs = np.array([p[0] for p in points], dtype=np.int32)
    ys = np.array([p[1] for p in points], dtype=np.int32)
    
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist2 = (dx*dx + dy*dy).astype(np.int32)
    
    def greedy_construct(order, time_limit=None):
        S = []
        S_set = set()
        used_dists = {}
        
        for idx_P in order:
            if idx_P in S_set:
                continue
            if time_limit and time.time() > time_limit:
                break
            
            valid = True
            new_dists = set()
            
            for idx_Q in S:
                d = int(dist2[idx_P, idx_Q])
                if d in new_dists:
                    valid = False
                    break
                new_dists.add(d)
                if d in used_dists[idx_Q]:
                    valid = False
                    break
            
            if valid:
                S.append(idx_P)
                S_set.add(idx_P)
                used_dists[idx_P] = new_dists
                for idx_Q in S:
                    if idx_Q != idx_P:
                        used_dists[idx_Q].add(int(dist2[idx_Q, idx_P]))
        
        return S, used_dists

    def simulated_annealing(initial_S, time_limit):
        rng = random.Random(42)
        
        S_list = initial_S[:]
        S_set = set(S_list)
        
        # State: dist_counts[P] = {d: count}, v_conflicts[P] = num_duplicates
        dist_counts = {p: {} for p in S_list}
        v_conflicts = {p: 0 for p in S_list}
        total_v = 0
        
        for u in S_list:
            for v in S_list:
                if u == v: continue
                d = int(dist2[u, v])
                dist_counts[u][d] = dist_counts[u].get(d, 0) + 1
        
        for u in S_list:
            for d, cnt in dist_counts[u].items():
                if cnt > 1:
                    v_conflicts[u] += (cnt - 1)
                    total_v += (cnt - 1)
        
        best_S = S_list[:]
        best_v = total_v
        best_size = len(S_list)
        
        current_score = len(S_list) - 1000 * total_v
        best_valid_score = best_size if best_v == 0 else -1
        
        T = 5.0
        T_min = 0.1
        cooling_rate = 0.9995
        
        while time.time() < time_limit:
            if T < T_min:
                break
            
            # Save state for potential revert
            S_copy = S_list[:]
            S_set_copy = S_set.copy()
            dist_counts_copy = {k: v.copy() for k, v in dist_counts.items()}
            v_conflicts_copy = v_conflicts.copy()
            total_v_copy = total_v
            
            move_type = rng.random()
            
            if move_type < 0.5:
                # Remove
                if not S_list:
                    continue
                u = rng.choice(S_list)
                
                # Calculate new state
                new_S = [p for p in S_list if p != u]
                new_S_set = S_set_copy - {u}
                
                # Update dist_counts for remaining points
                new_dist_counts = {k: v.copy() for k, v in dist_counts_copy.items() if k != u}
                for v in new_S:
                    d = int(dist2[u, v])
                    if d in new_dist_counts[v]:
                        new_dist_counts[v][d] -= 1
                        if new_dist_counts[v][d] == 0:
                            del new_dist_counts[v][d]
                
                # Recalculate v_conflicts for remaining points and total_v
                new_v_conflicts = {}
                new_total_v = 0
                for p in new_S:
                    cnts = new_dist_counts[p]
                    conf = 0
                    for d, cnt in cnts.items():
                        if cnt > 1:
                            conf += (cnt - 1)
                    new_v_conflicts[p] = conf
                    new_total_v += conf
                
                # Apply
                S_list = new_S
                S_set = new_S_set
                dist_counts = new_dist_counts
                v_conflicts = new_v_conflicts
                total_v = new_total_v
                
                delta_score = -1 - 1000 * (new_total_v - total_v_copy)
                
            else:
                # Add
                u = rng.randint(0, num_points - 1)
                if u in S_set:
                    continue
                
                new_S = S_list + [u]
                new_S_set = S_set_copy | {u}
                
                new_dist_counts = {k: v.copy() for k, v in dist_counts_copy.items()}
                new_dist_counts[u] = {}
                
                for v in S_list:
                    d = int(dist2[u, v])
                    new_dist_counts[u][d] = new_dist_counts[u].get(d, 0) + 1
                    new_dist_counts[v][d] = new_dist_counts[v].get(d, 0) + 1
                
                new_v_conflicts = {}
                new_total_v = 0
                for p in new_S:
                    cnts = new_dist_counts[p]
                    conf = 0
                    for d, cnt in cnts.items():
                        if cnt > 1:
                            conf += (cnt - 1)
                    new_v_conflicts[p] = conf
                    new_total_v += conf
                
                S_list = new_S
                S_set = new_S_set
                dist_counts = new_dist_counts
                v_conflicts = new_v_conflicts
                total_v = new_total_v
                
                delta_score = 1 - 1000 * (new_total_v - total_v_copy)
            
            new_score = current_score + delta_score
            
            if new_score > current_score or rng.random() < np.exp((new_score - current_score) / T):
                current_score = new_score
                if total_v == 0 and len(S_list) > best_valid_score:
                    best_valid_score = len(S_list)
                    best_S = S_list[:]
                    best_v = total_v
            else:
                # Revert
                S_list = S_copy
                S_set = S_set_copy
                dist_counts = dist_counts_copy
                v_conflicts = v_conflicts_copy
                total_v = total_v_copy
            
            T *= cooling_rate
        
        return best_S

    # Phase 1: Greedy
    rng = random.Random(42)
    order = list(range(num_points))
    rng.shuffle(order)
    S_init, _ = greedy_construct(order, time_limit=deadline - 60)
    
    # Phase 2: SA
    S_best = simulated_annealing(S_init, deadline - 10)
    
    return [points[i] for i in S_best]
# EVOLVE-BLOCK-END
