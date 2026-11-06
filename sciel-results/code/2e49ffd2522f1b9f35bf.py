import time
import random
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin for 60s limit
    
    n = 256
    
    # Precompute 3-APs
    # A 3-AP is (a, b, c) with a < b < c and a+c = 2b.
    # We represent the set S as a boolean array or bitmask.
    # For fast checking:
    # If we have a bitmask `mask` where bit i is 1 if i is in S (0-indexed, so bit 0 is 1, bit 1 is 2, etc.)
    # Actually, Python integers are arbitrary precision, so we can use a single int as a bitmask for n=256.
    # bit i corresponds to number i+1.
    
    # Precompute for each number x (1..n), the list of pairs (y, z) such that {x, y, z} is an AP.
    # If we add x to S, we must ensure that for no (y, z) in pairs[x], both y and z are in S.
    # Let's precompute a bitmask for each x: conflict_mask[x]
    # conflict_mask[x] will have bits set for y and z? No, that's not enough to check "both y and z".
    # We need to check: for all (y, z) in pairs[x], not (bit_y AND bit_z).
    
    # Alternative: Precompute for each x, a list of pairs (y, z).
    # To check if x can be added to mask M:
    # For each (y, z) in pairs[x]:
    #    if (M >> y) & 1 and (M >> z) & 1: return False
    # This is O(deg(x)). deg(x) is small (~10-20).
    
    # To make it faster with NumPy?
    # We can precompute a 2D array or just use the list.
    # Given n=256, pure Python loop over ~20 pairs is very fast.
    # Let's stick to Python lists for pairs, but optimize the search loop.
    
    conflict_pairs = {i: [] for i in range(1, n+1)}
    for b in range(2, n):
        for a in range(1, b):
            c = 2*b - a
            if c <= n:
                # AP: a, b, c
                conflict_pairs[a].append((b, c))
                conflict_pairs[b].append((a, c))
                conflict_pairs[c].append((a, b))
    
    # Helper to check if x can be added to mask M
    # M is an int where bit (i-1) is 1 if i is in S.
    def can_add(x, M):
        for y, z in conflict_pairs[x]:
            # Check if y and z are in M
            # y, z are 1-indexed
            if (M & (1 << (y-1))) and (M & (1 << (z-1))):
                return False
        return True

    def sum_recip(M):
        s = 0.0
        # Iterate through bits set in M
        # n is small, we can just loop 1..n
        # Or use bit trick.
        # Loop is fine for n=256.
        for i in range(1, n+1):
            if M & (1 << (i-1)):
                s += 1.0 / i
        return s

    # Multi-start Simulated Annealing
    
    best_score = 0.0
    best_mask = 0
    
    # We will run multiple SA instances.
    # Each instance starts from a random valid set or greedy set.
    
    # Greedy initialization
    def greedy_init():
        M = 0
        # Randomize order? No, greedy by 1/x means picking smallest available.
        # To get diverse starts, we can shuffle the order of candidates?
        # Or just start with empty set and add random valid elements?
        # Let's do a randomized greedy: pick random x, add if valid.
        # Or better: Start with a few small numbers, then fill.
        
        # Let's just do a standard greedy (smallest first) for one instance,
        # and random restarts for others.
        for x in range(1, n+1):
            if can_add(x, M):
                M |= (1 << (x-1))
        return M

    # Random valid set generation
    def random_valid_init():
        M = 0
        # Add random elements
        # To ensure validity, we add one by one.
        # To make it dense, we try to add small numbers.
        candidates = list(range(1, n+1))
        random.shuffle(candidates)
        for x in candidates:
            if can_add(x, M):
                M |= (1 << (x-1))
        return M

    # Simulated Annealing Move
    # Moves:
    # 1. Add x (if valid)
    # 2. Remove x
    # 3. Swap x with y
    # 4. Remove set R, Add set A (multi-move)
    
    # For SA, we need to evaluate delta.
    # If we add x: delta = 1/x. Validity check required.
    # If we remove x: delta = -1/x. Always valid.
    # If we swap x (in) with y (out): delta = 1/y - 1/x. Check validity of adding y to M\x.
    
    # To escape local minima, we allow "worse" moves with probability exp(-delta/T).
    # But we must maintain validity. So "worse" means lower score, but still valid.
    
    def run_sa(initial_mask, time_limit):
        M = initial_mask
        current_score = sum_recip(M)
        best_local_score = current_score
        best_local_mask = M
        
        T = 1.0 # Initial temperature
        cooling = 0.999
        
        # Precompute 1/x for all x
        inv = [0.0] + [1.0/i for i in range(1, n+1)]
        
        steps = 0
        while time.time() < time_limit:
            steps += 1
            move_type = random.random()
            
            if move_type < 0.3:
                # Add a random element
                # Pick random x not in M
                # To bias towards small x (high value), we can pick from a weighted list
                # But random is okay for exploration.
                # Let's pick x uniformly from [1..n]
                x = random.randint(1, n)
                if not (M & (1 << (x-1))):
                    if can_add(x, M):
                        new_M = M | (1 << (x-1))
                        delta = inv[x]
                        # Always accept if improving (delta > 0)
                        # If delta < 0 (impossible for add), reject
                        # Here delta is always positive.
                        # Wait, adding an element always increases sum.
                        # So we should always accept valid adds?
                        # In SA, we might want to add a large x (small 1/x) to enable future swaps?
                        # No, we want to maximize sum. Adding any valid x increases sum.
                        # So "Add" move is always an improvement in score.
                        # The constraint is the bottleneck.
                        # So we should aggressively try to add small x.
                        # If we can add x, we do it.
                        M = new_M
                        current_score += delta
                        if current_score > best_local_score:
                            best_local_score = current_score
                            best_local_mask = M
            elif move_type < 0.6:
                # Remove a random element
                # Pick random x in M
                # To pick random set bit, we can use random.sample on indices?
                # Or just random x and check if in M.
                x = random.randint(1, n)
                if M & (1 << (x-1)):
                    new_M = M & ~(1 << (x-1))
                    delta = -inv[x]
                    # Removing always decreases score.
                    # Accept with prob exp(delta/T)
                    if random.random() < np.exp(delta / T):
                        M = new_M
                        current_score += delta
            elif move_type < 0.85:
                # Swap x (in) with y (out)
                # Pick x in M, y not in M
                # To bias towards improving swaps, we might prefer y < x.
                # But for SA, random is better for exploration.
                # Let's pick x uniformly from M, y uniformly from [1..n]\M.
                
                # Get list of elements in M?
                # For n=256, we can just loop or use a precomputed list of bits.
                # Let's just pick random x, y.
                x = random.randint(1, n)
                y = random.randint(1, n)
                
                if (M & (1 << (x-1))) and not (M & (1 << (y-1))):
                    # Check if y can be added to M \ {x}
                    M_temp = M & ~(1 << (x-1))
                    if can_add(y, M_temp):
                        new_M = M_temp | (1 << (y-1))
                        delta = inv[y] - inv[x]
                        # Accept if delta > 0 or with prob exp(delta/T)
                        if delta > 0 or random.random() < np.exp(delta / T):
                            M = new_M
                            current_score += delta
                            if current_score > best_local_score:
                                best_local_score = current_score
                                best_local_mask = M
            else:
                # Multi-move: Remove 2 elements, Add 1 element
                # This helps escape traps where a small y is blocked by two large x's.
                # Pick x1, x2 in M, y not in M.
                x1 = random.randint(1, n)
                x2 = random.randint(1, n)
                y = random.randint(1, n)
                
                if x1 == x2: continue
                if (M & (1 << (x1-1))) and (M & (1 << (x2-1))) and not (M & (1 << (y-1))):
                    M_temp = M & ~(1 << (x1-1)) & ~(1 << (x2-1))
                    if can_add(y, M_temp):
                        new_M = M_temp | (1 << (y-1))
                        delta = inv[y] - inv[x1] - inv[x2]
                        if delta > 0 or random.random() < np.exp(delta / T):
                            M = new_M
                            current_score += delta
                            if current_score > best_local_score:
                                best_local_score = current_score
                                best_local_mask = M
            
            T *= cooling
            if T < 0.01:
                break # Too cold, restart or stop
                
        return best_local_mask, best_local_score

    # Run multiple instances
    # 1. Greedy start
    M0 = greedy_init()
    best_mask, best_score = run_sa(M0, deadline - 5.0)
    
    # 2. Random starts
    # Use remaining time
    remaining = deadline - time.time()
    if remaining > 2.0:
        num_restarts = int(remaining / 2.0)
        for i in range(num_restarts):
            M_rand = random_valid_init()
            # Run short SA
            m, s = run_sa(M_rand, min(1.0, deadline - time.time()))
            if s > best_score:
                best_score = s
                best_mask = m
            if time.time() > deadline - 1.0:
                break

    # Convert best_mask to list
    result = []
    for i in range(1, n+1):
        if best_mask & (1 << (i-1)):
            result.append(i)
            
    return result

# EVOLVE-BLOCK-END
