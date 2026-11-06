import time
import random
import math
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s buffer for safety

    def get_block_sums(seq):
        """Calculate all consecutive block sums."""
        sums = []
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                sums.append(s)
        return sums

    def count_conflicts(seq):
        """Return the number of duplicate sums."""
        sums = get_block_sums(seq)
        return len(sums) - len(set(sums))

    def is_valid(seq):
        return count_conflicts(seq) == 0

    # 1. Greedy Baseline
    values, sums = [], set()
    for x in range(1, 257):
        tails = []
        total = x
        tails.append(total)
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)

    best_seq = values[:]
    best_len = len(values)

    # 2. Backtracking Search with Pruning
    # We try to find a valid sequence of length L for L in [max_L_to_try, ..., best_len + 1]
    # If we find one, we return it.
    
    # Estimate max L. 
    # The greedy baseline is likely around 10-12.
    # Let's try up to 15 or 16.
    
    max_L_to_try = 16
    
    # We will try to find a sequence of length L for L in [max_L_to_try, ..., best_len + 1]
    # If we find one, we return it.
    
    # To make it robust, we can try multiple random restarts for each L.
    
    for L in range(max_L_to_try, best_len, -1):
        if time.time() > deadline:
            break
            
        # Time budget for this L
        # Allocate more time to larger L? Or equal?
        # Let's give 3 seconds per L attempt.
        t_start_L = time.time()
        t_limit_L = t_start_L + 3.0
        
        found = False
        attempts = 0
        while time.time() < t_limit_L and not found:
            attempts += 1
            # Initialize random increasing sequence
            # Pick L distinct values from 1..256
            # Sort them
            seq = sorted(random.sample(range(1, 257), L))
            
            # Simulated Annealing
            current_cost = count_conflicts(seq)
            if current_cost == 0:
                best_seq = seq
                best_len = L
                found = True
                break
                
            T = 10.0  # Initial temperature
            min_T = 0.01
            cooling = 0.95
            
            # Run SA for a fixed number of steps or time
            # Since each step is fast, let's do a fixed number of steps
            # or check time inside loop.
            
            steps = 10000
            for step in range(steps):
                if time.time() > t_limit_L:
                    break
                
                # Generate neighbor
                i = random.randint(0, L - 1)
                delta = random.randint(-5, 5)
                new_val = seq[i] + delta
                
                # Check bounds and monotonicity
                if new_val < 1 or new_val > 256:
                    continue
                if i > 0 and new_val <= seq[i-1]:
                    continue
                if i < L - 1 and new_val >= seq[i+1]:
                    continue
                
                old_val = seq[i]
                seq[i] = new_val
                
                new_cost = count_conflicts(seq)
                delta_cost = new_cost - current_cost
                
                if delta_cost <= 0:
                    current_cost = new_cost
                    if current_cost == 0:
                        best_seq = seq
                        best_len = L
                        found = True
                        break
                else:
                    # Accept with probability exp(-delta_cost / T)
                    if random.random() < math.exp(-delta_cost / T):
                        current_cost = new_cost
                    else:
                        seq[i] = old_val
                        
                T *= cooling
                if T < min_T:
                    T = min_T
        
        if found:
            # We found a valid sequence of length L.
            # Since we are iterating L downwards, this is the max length found so far.
            # We can return immediately? 
            # Yes, because we are looking for the max length. 
            # If we found L=12, we don't need to check L=11.
            # But wait, we started from max_L_to_try. 
            # If we tried L=15 and failed, then L=14 and failed, then L=13 and found it.
            # Then 13 is the best.
            return best_seq

    # 3. Backtracking with Pruning (Structural Change)
    # We try to find a valid sequence of length L for L in [best_len + 1, ...]
    # We use a recursive backtracking approach with pruning based on distinct sums.
    
    def backtrack(seq, used_sums, target_len):
        """
        Recursively build a sequence of length target_len.
        seq: current sequence
        used_sums: set of sums already used
        target_len: desired length
        """
        if len(seq) == target_len:
            return seq[:]
        
        last_val = seq[-1] if seq else 0
        
        # Try next values from last_val + 1 to 256
        for x in range(last_val + 1, 257):
            # Calculate new sums
            new_sums = set()
            total = x
            new_sums.add(total)
            for a in reversed(seq):
                total += a
                new_sums.add(total)
            
            # Check if any new sum conflicts with used sums
            if new_sums.intersection(used_sums):
                continue
            
            # Pruning: If the number of possible remaining sums is less than the number of required sums, prune.
            # This is complex, so we skip it for now.
            
            # Recursive call
            result = backtrack(seq + [x], used_sums.union(new_sums), target_len)
            if result is not None:
                return result
        
        return None

    # Try to find a sequence of length best_len + 1
    target_len = best_len + 1
    while target_len <= 20:  # Arbitrary upper bound
        if time.time() > deadline:
            break
            
        # Reset for backtracking
        # We can start with an empty sequence or a partial one?
        # Starting empty is most general.
        # But we can use the greedy sequence as a hint?
        # No, backtracking is independent.
        
        # We can add a heuristic: prioritize smaller values? Or larger?
        # Smaller values keep the sums smaller, allowing more room for more elements.
        # So iterating x from last_val + 1 upwards is good.
        
        # We can also limit the search depth or width.
        # But let's try pure backtracking first.
        
        # To make it faster, we can use a smarter initialization or pruning.
        # Pruning: If the number of possible remaining sums is less than the number of required sums, prune.
        # But that's complex.
        
        # Let's just run it.
        # We need to be careful with recursion depth.
        sys.setrecursionlimit(1000)
        
        found = backtrack([], set(), target_len)
        if found:
            best_seq = found
            best_len = target_len
            target_len += 1
        else:
            # If we didn't find a sequence of length target_len, maybe we can't find a longer one?
            # Not necessarily, but it's likely.
            # We can break or continue.
            # Let's break to save time, as longer sequences are harder.
            break

    return best_seq

construct()
