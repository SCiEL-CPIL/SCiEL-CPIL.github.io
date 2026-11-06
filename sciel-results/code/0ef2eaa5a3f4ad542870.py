import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    N = 1000
    best_set = []
    best_size = 0
    
    # Helper to check if adding x to set S (with counts) is valid
    def is_valid_addition(x, chosen, counts):
        if counts.get(2*x, 0) >= 2:
            return False
        for y in chosen:
            s = x + y
            if counts.get(s, 0) >= 2:
                return False
        return True

    def run_greedy(noise=0.0, time_limit=None, prefer_small=True):
        """
        Greedy construction.
        Picks the element that causes the least "tightness" (sums reaching count 2).
        """
        chosen = []
        counts = defaultdict(int)
        candidates = set(range(1, N + 1))
        
        if time_limit is None:
            time_limit = deadline
        
        while candidates:
            if time.time() > time_limit:
                break
            
            best_x = None
            best_score = float('inf')
            
            # To speed up, we can sort candidates or just iterate.
            # Iterating 1000 items is fast enough.
            # We want to minimize tightness.
            # Tie-breaking: prefer smaller x (to pack low numbers?) or larger x?
            # Usually packing low numbers is harder for B2 sets?
            # Actually, for B2 sets, small numbers have fewer sums.
            # Let's stick to tightness, with random tie-breaking.
            
            # Optimization: Pre-compute potential sums? No, counts change.
            
            for x in candidates:
                valid = True
                tight = 0
                
                c_2x = counts.get(2*x, 0)
                if c_2x >= 2:
                    continue
                if c_2x == 1:
                    tight += 1
                    
                for y in chosen:
                    s = x + y
                    c = counts.get(s, 0)
                    if c >= 2:
                        valid = False
                        break
                    if c == 1:
                        tight += 1
                        
                if not valid:
                    continue
                
                # Add small random noise to break ties and explore
                score = tight + random.random() * noise
                if score < best_score:
                    best_score = score
                    best_x = x
            
            if best_x is None:
                break
                
            # Add best_x
            counts[2*best_x] = counts.get(2*best_x, 0) + 1
            for y in chosen:
                s = best_x + y
                counts[s] = counts.get(s, 0) + 1
            chosen.append(best_x)
            candidates.remove(best_x)
            
        return chosen

    def local_search(initial_set, time_limit):
        """
        Tries to grow the set by adding elements.
        If stuck, tries swaps to find a better configuration that allows growth.
        Includes a "kick" mechanism: remove k random elements and refill greedily.
        """
        chosen = list(initial_set)
        counts = defaultdict(int)
        
        # Rebuild counts
        for x in chosen:
            counts[2*x] = counts.get(2*x, 0) + 1
        for i in range(len(chosen)):
            for j in range(i+1, len(chosen)):
                s = chosen[i] + chosen[j]
                counts[s] = counts.get(s, 0) + 1
                
        candidates = set(range(1, N + 1))
        for x in chosen:
            candidates.discard(x)
            
        if not candidates:
            return chosen
            
        while time.time() < time_limit:
            # 1. Try to add elements
            # Shuffle candidates to avoid bias
            cand_list = list(candidates)
            random.shuffle(cand_list)
            
            added_any = False
            for x in cand_list:
                if time.time() > time_limit:
                    break
                if is_valid_addition(x, chosen, counts):
                    counts[2*x] = counts.get(2*x, 0) + 1
                    for y in chosen:
                        s = x + y
                        counts[s] = counts.get(s, 0) + 1
                    chosen.append(x)
                    candidates.discard(x)
                    added_any = True
                    break # Restart loop to pick next best candidate
            
            if added_any:
                continue
                
            # 2. If we couldn't add, try a swap
            # Pick a random x from candidates, random y from chosen
            # If S - {y} + {x} is valid, do it.
            
            if len(candidates) == 0:
                break
                
            x_to_try = random.choice(list(candidates))
            chosen_list = list(chosen)
            random.shuffle(chosen_list)
            
            swap_success = False
            for y in chosen_list:
                # Check if removing y and adding x is valid
                # We need to check sums involving x and (chosen - {y})
                
                # Save state
                old_counts = dict(counts)
                
                # Remove y
                counts[2*y] -= 1
                if counts[2*y] == 0: del counts[2*y]
                for z in chosen:
                    if z != y:
                        s = y + z
                        counts[s] -= 1
                        if counts[s] == 0: del counts[s]
                        
                # Check x
                if is_valid_addition(x, [z for z in chosen if z != y], counts):
                    # Add x
                    counts[2*x] = counts.get(2*x, 0) + 1
                    for z in chosen:
                        if z != y:
                            s = x + z
                            counts[s] = counts.get(s, 0) + 1
                            
                    # Update sets
                    chosen.remove(y)
                    chosen.append(x)
                    candidates.discard(x)
                    candidates.add(y)
                    
                    swap_success = True
                    break
                else:
                    # Revert counts
                    counts.clear()
                    counts.update(old_counts)
                    
            if swap_success:
                continue
                
            # 3. If we can't even swap, try a "kick"
            # Remove 1-3 random elements, then greedily refill
            kick_size = random.randint(1, 3)
            if len(chosen) <= kick_size:
                break
                
            # Remove random elements
            removed = random.sample(chosen, kick_size)
            for y in removed:
                # Update counts
                counts[2*y] -= 1
                if counts[2*y] == 0: del counts[2*y]
                for z in chosen:
                    if z != y:
                        s = y + z
                        counts[s] -= 1
                        if counts[s] == 0: del counts[s]
                chosen.remove(y)
                candidates.add(y)
                
            # Greedily refill
            # We can use a simplified greedy here
            refill_time = time.time() + 2.0
            while candidates and time.time() < refill_time:
                best_x = None
                best_score = float('inf')
                
                # Sample a subset of candidates to speed up
                sample_size = min(50, len(candidates))
                sample = random.sample(list(candidates), sample_size)
                
                for x in sample:
                    valid = True
                    tight = 0
                    
                    c_2x = counts.get(2*x, 0)
                    if c_2x >= 2:
                        continue
                    if c_2x == 1:
                        tight += 1
                        
                    for y in chosen:
                        s = x + y
                        c = counts.get(s, 0)
                        if c >= 2:
                            valid = False
                            break
                        if c == 1:
                            tight += 1
                            
                    if not valid:
                        continue
                    
                    score = tight + random.random() * 0.5
                    if score < best_score:
                        best_score = score
                        best_x = x
                        
                if best_x is None:
                    break
                    
                counts[2*best_x] = counts.get(2*best_x, 0) + 1
                for y in chosen:
                    s = best_x + y
                    counts[s] = counts.get(s, 0) + 1
                chosen.append(best_x)
                candidates.discard(best_x)
                
        return chosen

    # Main loop
    # Strategy:
    # 1. Run greedy with different noise levels to get diverse starting points.
    # 2. Refine each with local search.
    # 3. Keep the best.
    
    # We will allocate time to multiple restarts.
    # Each restart: Greedy (fast) + Local Search (slower).
    
    num_restarts = 10 # We will run up to 10 restarts, or until time is up.
    
    for i in range(num_restarts):
        if time.time() > deadline - 5: # Reserve 5s for safety
            break
            
        # Vary noise to explore different greedy paths
        # Lower noise = more deterministic, higher noise = more random
        noise = random.uniform(0.0, 2.0)
        
        # Run greedy
        # Limit greedy time to 5s
        greedy_time_limit = time.time() + 5.0
        baseline = run_greedy(noise=noise, time_limit=greedy_time_limit)
        
        if len(baseline) > best_size:
            best_size = len(baseline)
            best_set = baseline
            
        # Refine with Local Search
        # Allocate remaining time for this round
        remaining = deadline - time.time()
        if remaining < 1:
            break
            
        # Split time: 80% local search, keep 20% for next restart
        # But cap local search time to allow multiple restarts
        ls_time = min(remaining * 0.7, 10.0) # Cap LS time per round to 10s
        refined = local_search(best_set if len(best_set) > len(baseline) else baseline, time.time() + ls_time)
        
        if len(refined) > best_size:
            best_size = len(refined)
            best_set = refined
            
    return best_set
