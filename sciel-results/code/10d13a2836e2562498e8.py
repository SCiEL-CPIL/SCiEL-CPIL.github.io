import numpy as np
import time
from itertools import combinations
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s kill

    def build_indices(n):
        """Precompute K4 and K5 index tuples and reverse mappings."""
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        k4s = []
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            k4s.append((
                triple_to_idx[(a, b, c)],
                triple_to_idx[(a, b, d)],
                triple_to_idx[(a, c, d)],
                triple_to_idx[(b, c, d)]
            ))
        
        k5s = []
        for quint in combinations(range(n), 5):
            idxs = []
            for triple in combinations(quint, 3):
                idxs.append(triple_to_idx[triple])
            k5s.append(tuple(idxs))
            
        # Convert to numpy arrays for vectorized operations
        k4s_arr = np.array(k4s, dtype=np.int32)
        k5s_arr = np.array(k5s, dtype=np.int32)
        
        # Precompute reverse mappings: triple_idx -> list of clique indices
        triple_to_k4s = [[] for _ in range(num_triples)]
        triple_to_k5s = [[] for _ in range(num_triples)]
        
        for i in range(len(k4s)):
            for t in k4s[i]:
                triple_to_k4s[t].append(i)
                
        for i in range(len(k5s)):
            for t in k5s[i]:
                triple_to_k5s[t].append(i)

        return num_triples, k4s, k5s, k4s_arr, k5s_arr, triple_to_k4s, triple_to_k5s

    def get_violations(colors, k4s_arr, k5s_arr):
        """
        Vectorized check for violations.
        Returns:
        red_k4_mask: boolean array of which K4s are all red
        blue_k5_mask: boolean array of which K5s are all blue
        """
        if len(k4s_arr) == 0:
            red_k4_mask = np.array([], dtype=bool)
        else:
            k4_colors = colors[k4s_arr]
            red_k4_mask = np.all(k4_colors == 0, axis=1)
            
        if len(k5s_arr) == 0:
            blue_k5_mask = np.array([], dtype=bool)
        else:
            k5_colors = colors[k5s_arr]
            blue_k5_mask = np.all(k5_colors == 1, axis=1)
            
        return red_k4_mask, blue_k5_mask

    def local_search_greedy(n, time_limit):
        """
        Greedy Local Search focusing on conflict-involved triples.
        """
        num_triples, k4s, k5s, k4s_arr, k5s_arr, triple_to_k4s, triple_to_k5s = build_indices(n)
        
        best_colors = None
        best_cost = float('inf')
        
        max_restarts = 20
        
        for restart in range(max_restarts):
            if time.time() > time_limit:
                break
                
            # Initialize with random
            colors = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
            
            # Initial cost
            red_k4_mask, blue_k5_mask = get_violations(colors, k4s_arr, k5s_arr)
            current_cost = np.sum(red_k4_mask) + np.sum(blue_k5_mask)
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors.copy()
                if best_cost == 0:
                    return best_colors, True
            
            # Local Search Phase
            no_improve_steps = 0
            max_no_improve = 500
            step_deadline = time.time() + max(1.0, (time_limit - time.time()) * 0.4)
            
            while time.time() < step_deadline and no_improve_steps < max_no_improve:
                # Get current violations
                red_k4_mask, blue_k5_mask = get_violations(colors, k4s_arr, k5s_arr)
                current_cost = np.sum(red_k4_mask) + np.sum(blue_k5_mask)
                
                if current_cost == 0:
                    if current_cost < best_cost:
                        best_cost = 0
                        best_colors = colors.copy()
                    return best_colors, True
                
                # Identify candidate triples: those involved in any violation
                # We can find indices of violating K4s and K5s, then map to triples
                viol_k4_indices = np.where(red_k4_mask)[0]
                viol_k5_indices = np.where(blue_k5_mask)[0]
                
                candidates = set()
                
                # Add triples from violating K4s
                if len(viol_k4_indices) > 0:
                    # k4s_arr[viol_k4_indices] gives shape (num_viol_k4, 4)
                    # Flatten and get unique
                    viol_k4_triples = k4s_arr[viol_k4_indices].flatten()
                    candidates.update(viol_k4_triples.tolist())
                    
                # Add triples from violating K5s
                if len(viol_k5_indices) > 0:
                    viol_k5_triples = k5s_arr[viol_k5_indices].flatten()
                    candidates.update(viol_k5_triples.tolist())
                
                if not candidates:
                    # Should not happen if cost > 0, but just in case
                    break
                
                # If too many candidates, sample a subset to speed up
                # For n=34, cost might be high, candidates might be large.
                # Sampling 100-200 candidates is a good trade-off.
                if len(candidates) > 200:
                    candidates = random.sample(list(candidates), 200)
                else:
                    candidates = list(candidates)
                
                best_delta = 0
                best_t = -1
                
                # Evaluate each candidate
                for t in candidates:
                    old_color = colors[t]
                    new_color = 1 - old_color
                    
                    delta = 0
                    
                    # Check K4s containing t
                    for k4_idx in triple_to_k4s[t]:
                        idxs = k4s[k4_idx]
                        # Check if it's a red K4 before
                        is_red_before = True
                        for i in idxs:
                            if colors[i] != 0:
                                is_red_before = False
                                break
                        
                        if old_color == 0:
                            # Flipping 0->1
                            if is_red_before:
                                delta -= 1
                        else:
                            # Flipping 1->0
                            if not is_red_before:
                                # Check if it becomes red
                                # Other 3 must be 0
                                is_red_after = True
                                for i in idxs:
                                    if i == t:
                                        continue
                                    if colors[i] != 0:
                                        is_red_after = False
                                        break
                                if is_red_after:
                                    delta += 1
                                    
                    # Check K5s containing t
                    for k5_idx in triple_to_k5s[t]:
                        idxs = k5s[k5_idx]
                        # Check if it's a blue K5 before
                        is_blue_before = True
                        for i in idxs:
                            if colors[i] != 1:
                                is_blue_before = False
                                break
                                
                        if old_color == 1:
                            # Flipping 1->0
                            if is_blue_before:
                                delta -= 1
                        else:
                            # Flipping 0->1
                            if not is_blue_before:
                                # Check if it becomes blue
                                # Other 9 must be 1
                                is_blue_after = True
                                for i in idxs:
                                    if i == t:
                                        continue
                                    if colors[i] != 1:
                                        is_blue_after = False
                                        break
                                if is_blue_after:
                                    delta += 1
                                        
                    if delta < best_delta:
                        best_delta = delta
                        best_t = t
                        
                if best_t != -1:
                    colors[best_t] = 1 - colors[best_t]
                    no_improve_steps = 0
                else:
                    # No improving move found among candidates
                    # Accept a random move to escape local minimum
                    # Or just break and restart?
                    # Let's accept a random move from candidates with 50% prob, else break
                    if random.random() < 0.5:
                        t = random.choice(candidates)
                        colors[t] = 1 - colors[t]
                    else:
                        break
                    no_improve_steps += 1

        return best_colors, (best_cost == 0)

    # Main loop
    best_n = 9
    # Seed coloring for n=9: simple modular
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]
    
    # Targets to try
    # We want to maximize n.
    # n=34 is the goal.
    # Time allocation:
    # Try 34 for a while. If fail, try 33, etc.
    
    targets = [34, 33, 32, 31, 30, 28, 25]
    
    # We will allocate time dynamically.
    # If we find a valid n, we return it.
    # If we don't find 34, we try lower.
    
    # To be safe, let's try 34 with 60s, 33 with 30s, 32 with 20s, 31 with 10s, 30 with 5s.
    time_alloc = {
        34: 60.0,
        33: 30.0,
        32: 20.0,
        31: 10.0,
        30: 5.0,
        28: 3.0,
        25: 2.0
    }
    
    for n in targets:
        if time.time() > deadline - 5:
            break
            
        t_limit = time.time() + time_alloc.get(n, 5.0)
        
        try:
            colors, is_valid = local_search_greedy(n, t_limit)
            if is_valid:
                best_n = n
                best_colors = colors.tolist()
                # If we found a valid one, should we try higher?
                # The loop goes from high to low.
                # So if we find 34, we break.
                # If we find 33, we continue to 32? No, 33 is better than 32.
                # But we started with 34. If 34 failed, we try 33.
                # If 33 succeeds, we should return 33.
                # Wait, the loop is `for n in [34, 33, ...]`.
                # If 34 fails, we go to 33. If 33 succeeds, we set best_n=33 and break?
                # Yes, because we are going down. The first success is the highest n.
                break
        except Exception as e:
            pass

    return {'n': best_n, 'colors': best_colors}
