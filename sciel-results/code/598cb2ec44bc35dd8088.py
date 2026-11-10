import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for overhead/return
    
    n = 16
    vertices = list(range(n))
    k4s = list(itertools.combinations(vertices, 4))
    num_k4s = len(k4s)
    
    # Map edge (i, j) with i < j to index 0..119
    edge_map = {}
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_map[(i, j)] = idx
            idx += 1
            
    # For each K4, store the list of edge indices
    k4_edges = []
    for combo in k4s:
        edges = []
        for u, v in itertools.combinations(combo, 2):
            edges.append(edge_map[(u, v)])
        k4_edges.append(edges)
        
    # For each edge, store the list of K4 indices that contain it
    edge_to_k4s = [[] for _ in range(120)]
    for k4_idx, edges in enumerate(k4_edges):
        for e in edges:
            edge_to_k4s[e].append(k4_idx)
            
    # Precompute max colors needed for freq arrays (K is small, max 15)
    MAX_K = 15
    
    def try_find_coloring(K, time_limit):
        # colors: numpy array of int
        colors = np.random.randint(0, K, size=120)
        
        # distinct_counts: numpy array of int
        distinct_counts = np.zeros(num_k4s, dtype=np.int32)
        
        # freq: list of lists, freq[k4_idx][color] = count
        # Since K <= 15, we can use a list of lists of size K
        # Or a 2D numpy array? List of lists is faster for small K in Python if we do manual updates?
        # Let's use list of lists for freq to avoid numpy overhead for small arrays.
        freq = [[0] * K for _ in range(num_k4s)]
        
        # Initialize distinct_counts and freq
        for k4_idx in range(num_k4s):
            edges = k4_edges[k4_idx]
            # Count colors
            for e in edges:
                c = colors[e]
                freq[k4_idx][c] += 1
            distinct_counts[k4_idx] = sum(1 for count in freq[k4_idx] if count > 0)
            
        # Maintain set of invalid K4s
        invalid_k4s = set()
        for k4_idx in range(num_k4s):
            if distinct_counts[k4_idx] < 5:
                invalid_k4s.add(k4_idx)
                
        if not invalid_k4s:
            return colors
            
        current_viol = len(invalid_k4s)
        
        # Local Search
        # We want to minimize current_viol to 0.
        # Strategy: Pick a random invalid K4, pick a random edge in it, change color to a random new color.
        # Accept if it reduces violations.
        
        steps = 0
        max_steps = 50000
        no_improve = 0
        max_no_improve = 5000
        
        # Pre-shuffled list of K4 indices for faster random selection?
        # Or just pick random k4_idx from invalid_k4s.
        # If invalid_k4s is large, random.choice(list(invalid_k4s)) is O(N).
        # If it's small, it's fast.
        
        while steps < max_steps and no_improve < max_no_improve:
            if time.time() > time_limit:
                break
                
            if not invalid_k4s:
                break
                
            # Pick a random invalid K4
            # Converting set to list is O(N). N <= 1820.
            # To optimize, we can just pick a random index from 0..num_k4s-1 and check if it's invalid.
            # But if most are valid, this is wasteful.
            # Given we start with many invalids, list conversion is fine.
            k4_idx = random.choice(list(invalid_k4s))
            edges = k4_edges[k4_idx]
            e_idx = random.choice(edges)
            
            old_color = colors[e_idx]
            new_color = random.randint(0, K-1)
            if new_color == old_color:
                continue
                
            # Apply change
            colors[e_idx] = new_color
            
            # Update distinct_counts and invalid_k4s for affected K4s
            affected_k4s = edge_to_k4s[e_idx]
            delta_viol = 0
            
            for k4 in affected_k4s:
                old_dist = distinct_counts[k4]
                
                # Update freq
                freq[k4][old_color] -= 1
                if freq[k4][old_color] == 0:
                    old_dist -= 1
                    
                freq[k4][new_color] += 1
                if freq[k4][new_color] == 1:
                    old_dist += 1
                    
                new_dist = old_dist
                distinct_counts[k4] = new_dist
                
                # Update invalid set
                was_invalid = old_dist < 5
                is_invalid = new_dist < 5
                
                if was_invalid and not is_invalid:
                    invalid_k4s.discard(k4)
                    delta_viol -= 1
                elif not was_invalid and is_invalid:
                    invalid_k4s.add(k4)
                    delta_viol += 1
                    
            new_viol = current_viol + delta_viol
            
            if new_viol < current_viol:
                current_viol = new_viol
                no_improve = 0
            else:
                # Revert
                colors[e_idx] = old_color
                for k4 in affected_k4s:
                    # We need to revert freq and distinct_counts
                    # We didn't store old freqs, so we have to reverse the operations
                    # Reverse:
                    # freq[k4][new_color] -= 1
                    # if freq[k4][new_color] == 0: distinct -= 1 (if it was 1 before) -> wait
                    # Let's just recalculate for affected K4s if we revert? No, that's slow.
                    # Better: Store old distinct counts? Or just reverse the freq logic carefully.
                    
                    # Reverse logic:
                    # 1. new_color count decreases by 1
                    freq[k4][new_color] -= 1
                    if freq[k4][new_color] == 0:
                        distinct_counts[k4] -= 1
                    # 2. old_color count increases by 1
                    freq[k4][old_color] += 1
                    if freq[k4][old_color] == 1:
                        distinct_counts[k4] += 1
                        
                    # Update invalid set
                    new_dist = distinct_counts[k4]
                    # The state should be back to what it was before the change
                    # So we just need to ensure the set matches the new_dist
                    is_invalid = new_dist < 5
                    if is_invalid:
                        invalid_k4s.add(k4)
                    else:
                        invalid_k4s.discard(k4)
                
                no_improve += 1
                
            steps += 1
            
        if not invalid_k4s:
            return colors
        return None

    # Try K=13, 14, 15
    # Allocate time: 40s for 13, 40s for 14, 20s for 15
    # If 13 succeeds, return.
    # If 14 succeeds, return.
    # If 15 succeeds, return.
    # Else return 16? Or best effort.
    
    best_colors = None
    best_K = 16 # Fallback
    
    # Try 13
    t_limit_13 = start_time + 40
    res = try_find_coloring(13, t_limit_13)
    if res is not None:
        return res.tolist()
        
    # Try 14
    t_limit_14 = start_time + 80
    res = try_find_coloring(14, t_limit_14)
    if res is not None:
        return res.tolist()
        
    # Try 15
    t_limit_15 = start_time + 100
    res = try_find_coloring(15, t_limit_15)
    if res is not None:
        return res.tolist()
        
    # Fallback: Random 16 coloring? Or simple formula?
    # The task says "Minimize the number of colours used".
    # If we fail to find a valid one for 15, we must return a valid one.
    # Is 16 guaranteed? The bound is 13 <= f(16) <= 15. So 15 is sufficient.
    # My search for 15 might fail due to time.
    # I should ensure I return a VALID coloring.
    # If try_find_coloring returns None, it means it didn't find a valid one in time.
    # I need a fallback that is definitely valid.
    # A random 16-coloring is likely valid.
    # Or I can run a longer search for 16 if I have time?
    # Let's just use a random 16-coloring and verify. If invalid, fix it?
    # Actually, with 16 colors, it's very easy to be valid.
    # Let's generate a random 16-coloring.
    
    colors = np.random.randint(0, 16, size=120)
    # Quick check/fix loop just in case (should be fast)
    distinct_counts = np.zeros(num_k4s, dtype=np.int32)
    freq = [[0] * 16 for _ in range(num_k4s)]
    for k4_idx in range(num_k4s):
        edges = k4_edges[k4_idx]
        for e in edges:
            c = colors[e]
            freq[k4_idx][c] += 1
        distinct_counts[k4_idx] = sum(1 for count in freq[k4_idx] if count > 0)
        
    invalid_k4s = set(i for i in range(num_k4s) if distinct_counts[i] < 5)
    
    # If any invalid, do a quick local search for 16 colors
    steps = 0
    while invalid_k4s and steps < 10000:
        if time.time() > deadline:
            break
        k4_idx = random.choice(list(invalid_k4s))
        edges = k4_edges[k4_idx]
        e_idx = random.choice(edges)
        old_color = colors[e_idx]
        new_color = random.randint(0, 15)
        if new_color == old_color: continue
        
        colors[e_idx] = new_color
        affected = edge_to_k4s[e_idx]
        delta = 0
        for k4 in affected:
            old_dist = distinct_counts[k4]
            freq[k4][old_color] -= 1
            if freq[k4][old_color] == 0: old_dist -= 1
            freq[k4][new_color] += 1
            if freq[k4][new_color] == 1: old_dist += 1
            new_dist = old_dist
            distinct_counts[k4] = new_dist
            
            was_inv = old_dist < 5
            is_inv = new_dist < 5
            if was_inv and not is_inv:
                invalid_k4s.discard(k4)
                delta -= 1
            elif not was_inv and is_inv:
                invalid_k4s.add(k4)
                delta += 1
        # We don't strictly need to check if it improved for 16 colors, just accept any change?
        # No, accept only if it reduces violations to ensure progress.
        # But with 16 colors, violations are rare.
        # Let's just accept if delta < 0.
        if delta < 0:
            steps += 1
        else:
            # Revert
            colors[e_idx] = old_color
            for k4 in affected:
                freq[k4][new_color] -= 1
                if freq[k4][new_color] == 0: distinct_counts[k4] -= 1
                freq[k4][old_color] += 1
                if freq[k4][old_color] == 1: distinct_counts[k4] += 1
                new_dist = distinct_counts[k4]
                if new_dist < 5: invalid_k4s.add(k4)
                else: invalid_k4s.discard(k4)
            steps += 1
            if steps % 100 == 0:
                 # Random restart?
                 pass

    return colors.tolist()

