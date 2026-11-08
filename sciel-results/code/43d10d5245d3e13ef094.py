import random
import math
import time

def score_set(U):
    if not U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    
    # Calculate U+U and U-U
    # Using sets for simplicity, N is small
    sum_set = set()
    diff_set = set()
    
    # Optimization: Iterate over unique pairs to avoid redundant additions?
    # Actually, for sets, a+b is commutative, but we need all unique sums.
    # The double loop is O(N^2). For N=10-20, this is fine.
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
            
    size_sum = len(sum_set)
    size_diff = len(diff_set)
    
    if size_sum == 0:
        return 0.0
        
    ratio = size_diff / size_sum
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    
    if log_denom == 0:
        return 0.0
        
    score = 1 + log_ratio / log_denom
    return score

def construct():
    # Initialize with diverse candidates based on known good structures
    # We want to maximize |U-U|/|U+U|. 
    # |U-U| is at most 2*max(U)+1. |U+U| is at least 2*|U|-1.
    # To maximize ratio, we want |U-U| to be large (close to max possible) and |U+U| to be small (close to min possible).
    # This suggests sets with large gaps (to keep sums distinct/small count? No, large gaps make sums distinct, increasing |U+U|).
    # Wait. 
    # If U is an arithmetic progression {0, 1, ..., k}, |U+U| = 2k+1, |U-U| = 2k+1. Ratio = 1. Score = 1.
    # If U is geometric {1, 2, 4, ...}, sums are distinct. |U+U| = N(N+1)/2. |U-U| is also large.
    # Let's test the seed: [0, 1, 2, 3, 7].
    # Sums: 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14. (15 elements)
    # Diffs: -7..7. (15 elements). Ratio 1. Score 1.
    # Seed score is 1.0? The prompt says current score is 1.0240.
    # Let's check a set with larger gaps.
    # U = {0, 1, 10, 11}.
    # Sums: 0,1,2, 10,11,12, 20,21,22. (9 elements).
    # Diffs: -11..11. (23 elements). Ratio 23/9 = 2.55.
    # Max U = 11. Log(23)/Log(23) = 1. Score = 1 + 1 = 2? No.
    # Score = 1 + log(23/9) / log(23).
    # log(2.55) ~ 0.93. log(23) ~ 3.13. 0.93/3.13 ~ 0.3. Score ~ 1.3.
    # So sets with "clusters" separated by large gaps are good.
    
    candidates = [
        [0, 1, 2, 3, 7],       # Current seed
        [0, 1, 10, 11],        # Cluster
        [0, 1, 2, 10, 11, 12], # Two clusters
        [0, 1, 2, 3, 4, 10, 11, 12, 13, 14], # Larger clusters
        [0, 1, 5, 6, 10, 11],  # Spaced clusters
        [0, 1, 2, 5, 6, 7, 10, 11, 12], # 3 clusters
    ]
    
    best_set = candidates[0]
    best_score = score_set(best_set)
    
    for c in candidates:
        s = score_set(c)
        if s > best_score:
            best_score = s
            best_set = c
            
    # Simulated Annealing Parameters
    # We represent the set as a sorted list of integers.
    # To speed up, we restrict the search to a reasonable range.
    max_val = 100
    max_size = 12
    start_temp = 1.0
    cool_rate = 0.995
    iterations = 200000
    start_time = time.time()
    time_limit = 110 # Use most of the available time
    
    current_set = list(best_set)
    current_score = best_score
    temp = start_temp
    
    no_improve_count = 0
    restart_interval = 5000
    
    for i in range(iterations):
        if time.time() - start_time > time_limit:
            break
            
        # Generate move
        move_type = random.random()
        
        if move_type < 0.4:
            # Toggle add/remove
            if len(current_set) < max_size:
                # Add random element
                # Bias towards creating clusters or gaps
                if random.random() < 0.7:
                    # Add near an existing element (cluster)
                    base = random.choice(current_set)
                    offset = random.randint(1, 3) * random.choice([1, -1])
                    idx = base + offset
                else:
                    # Add far away (gap)
                    idx = random.randint(1, max_val)
                
                if 0 < idx <= max_val and idx not in current_set:
                    new_set = current_set + [idx]
                    new_set.sort()
                else:
                    new_set = current_set
            else:
                # Remove random element (keep 0)
                if len(current_set) > 1:
                    idx = random.choice([x for x in current_set if x != 0])
                    new_set = [x for x in current_set if x != idx]
                else:
                    new_set = current_set
        elif move_type < 0.7:
            # Swap
            if len(current_set) > 1:
                removable = [x for x in current_set if x != 0]
                if removable:
                    old_val = random.choice(removable)
                    # Try to swap with a value that maintains structure
                    if random.random() < 0.5:
                         # Small change
                         delta = random.randint(-2, 2)
                         new_val = old_val + delta
                    else:
                         # Random change
                         new_val = random.randint(1, max_val)
                    
                    if 0 < new_val <= max_val and new_val not in current_set:
                        new_set = [x if x != old_val else new_val for x in current_set]
                        new_set.sort()
                    else:
                        new_set = current_set
                else:
                    new_set = current_set
            else:
                new_set = current_set
        else:
            # Shift a cluster
            # Find a contiguous block and shift it
            if len(current_set) > 2:
                # Pick a random element to be the start of a potential cluster
                start_idx = random.randint(0, len(current_set)-1)
                # Determine cluster size
                cluster_size = random.randint(1, min(3, len(current_set)-start_idx))
                cluster = current_set[start_idx:start_idx+cluster_size]
                
                # Check if it's actually a cluster (consecutive or close)
                # For simplicity, just shift the first element of the segment
                # Or just shift one random element by a small amount
                old_val = random.choice([x for x in current_set if x != 0])
                delta = random.randint(-3, 3)
                new_val = old_val + delta
                
                if 0 < new_val <= max_val and new_val not in current_set:
                    new_set = [x if x != old_val else new_val for x in current_set]
                    new_set.sort()
                else:
                    new_set = current_set
            else:
                new_set = current_set
        
        if new_set == current_set:
            continue
            
        new_score = score_set(new_set)
        
        delta = new_score - current_score
        
        if delta > 0 or random.random() < math.exp(delta / temp):
            current_set = new_set
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_set = list(current_set)
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
            
        # Cooling
        temp *= cool_rate
        if temp < 1e-6:
            temp = 1e-6
            
        # Restart if stuck
        if no_improve_count > restart_interval:
            # Reset to a random candidate or best so far with perturbation
            if random.random() < 0.5:
                current_set = list(best_set)
            else:
                # Generate a new random structured set
                # Create 2-3 clusters
                num_clusters = random.randint(2, 3)
                cluster_sizes = [random.randint(1, 3) for _ in range(num_clusters)]
                while sum(cluster_sizes) > max_size - 1: # Reserve 1 for 0
                    cluster_sizes[random.randint(0, len(cluster_sizes)-1)] -= 1
                    if cluster_sizes[random.randint(0, len(cluster_sizes)-1)] == 0:
                         cluster_sizes.pop()
                         break
                
                current_set = [0]
                current_pos = 0
                for cs in cluster_sizes:
                    gap = random.randint(5, 20)
                    current_pos += gap
                    for j in range(cs):
                        if current_pos <= max_val and current_pos not in current_set:
                            current_set.append(current_pos)
                        current_pos += 1
                current_set.sort()
                if not current_set:
                    current_set = [0, 1]
            current_score = score_set(current_set)
            no_improve_count = 0
            # Reheat slightly
            temp = max(temp, 0.5)

    return best_set
