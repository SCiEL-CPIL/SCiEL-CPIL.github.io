"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 55  # safety margin before 60s kill
    
    N = 256
    
    # Precompute conflicts for fast validity checking
    # For each x, we need to know if adding x to a bitmask is valid.
    # x is invalid if there exist a, c in mask such that a+c=2x (x is middle)
    # or x, b, c in mask such that x+c=2b (x is first)
    # or a, b, x in mask such that a+x=2b (x is last)
    
    # Instead of complex pair lists, we can precompute for each x, the list of 
    # (a, c) pairs that form an AP with x.
    # But checking "are a and c in mask" is O(1) with bitmasks.
    # Number of pairs per x is O(N). Total precomp is O(N^2).
    
    # Let's precompute:
    # mid_pairs[x]: list of (a, c) such that a < x < c and a+c=2x
    # first_pairs[x]: list of (b, c) such that x < b < c and x+c=2b => c=2b-x
    # last_pairs[x]: list of (a, b) such that a < b < x and a+x=2b => a=2b-x
    
    mid_pairs = [[] for _ in range(N + 1)]
    first_pairs = [[] for _ in range(N + 1)]
    last_pairs = [[] for _ in range(N + 1)]
    
    for x in range(1, N + 1):
        # x as middle
        for a in range(1, x):
            c = 2 * x - a
            if c <= N and c > x:
                mid_pairs[x].append((a, c))
        
        # x as first
        for b in range(x + 1, N + 1):
            c = 2 * b - x
            if c <= N and c > b:
                first_pairs[x].append((b, c))
        
        # x as last
        for b in range(1, x):
            a = 2 * b - x
            if a >= 1 and a < b:
                last_pairs[x].append((a, b))
    
    def is_valid_addition(bitmask, x):
        """Check if x can be added to set represented by bitmask."""
        if (bitmask >> (x - 1)) & 1:
            return False
        
        # Check mid conflicts: if a and c are in mask, x cannot be added
        for a, c in mid_pairs[x]:
            if ((bitmask >> (a - 1)) & 1) and ((bitmask >> (c - 1)) & 1):
                return False
        
        # Check first conflicts: if b and c are in mask, x cannot be added
        for b, c in first_pairs[x]:
            if ((bitmask >> (b - 1)) & 1) and ((bitmask >> (c - 1)) & 1):
                return False
        
        # Check last conflicts: if a and b are in mask, x cannot be added
        for a, b in last_pairs[x]:
            if ((bitmask >> (a - 1)) & 1) and ((bitmask >> (b - 1)) & 1):
                return False
        
        return True

    def reciprocal_sum_float(bitmask):
        """Compute float reciprocal sum for fast comparison."""
        total = 0.0
        for x in range(1, N + 1):
            if (bitmask >> (x - 1)) & 1:
                total += 1.0 / x
        return total

    def greedy_fill(bitmask, current_sum):
        """
        Greedily add valid elements to bitmask.
        Prioritizes smaller x (higher reciprocal).
        Returns updated bitmask and sum.
        """
        improved = True
        while improved:
            improved = False
            for x in range(1, N + 1):
                if not ((bitmask >> (x - 1)) & 1):
                    if is_valid_addition(bitmask, x):
                        bitmask |= (1 << (x - 1))
                        current_sum += 1.0 / x
                        improved = True
                        # Restart loop to ensure we pick up any new possibilities
                        # Actually, since we iterate 1..N, and adding x only 
                        # restricts future additions, we don't need to restart 
                        # for correctness of "maximal" set, but for "greedy best",
                        # picking smallest valid first is key.
                        # Since we break and restart, we ensure the absolute 
                        # smallest valid x is always picked first.
                        break
        return bitmask, current_sum

    def local_search_optimize(bitmask, current_sum, time_limit):
        """
        Local search using 1-removal + greedy-fill moves.
        This allows escaping local optima where a single swap is not enough,
        but removing one element unlocks multiple better additions.
        """
        improved_global = True
        while improved_global:
            improved_global = False
            if (time.time() - start_time) > time_limit:
                break
            
            # Try removing each element x from the current set
            # We iterate from largest to smallest x, as removing large x is "cheap"
            # and might unlock small, valuable additions.
            # However, removing small x is expensive. We only do it if it leads to net gain.
            
            # To save time, we can limit the search or just iterate all.
            # N=256 is small.
            for x in range(N, 0, -1):
                if (time.time() - start_time) > time_limit:
                    break
                
                if not ((bitmask >> (x - 1)) & 1):
                    continue
                
                # Simulate removing x
                temp_bitmask = bitmask & ~(1 << (x - 1))
                temp_sum = current_sum - (1.0 / x)
                
                # Greedily fill temp_bitmask
                # We want to see if the net sum improves
                # If temp_sum is already much lower, we might prune?
                # Max possible gain from adding all remaining elements is sum(1/i for i not in temp)
                # But let's just run greedy fill. It's fast enough for N=256.
                
                # Optimization: Only try to add elements y < x? 
                # No, removing x might unlock y > x too, but y > x has lower value.
                # However, we just want to maximize sum.
                # Let's run full greedy fill on temp_bitmask.
                
                new_bitmask, new_sum = greedy_fill(temp_bitmask, temp_sum)
                
                if new_sum > current_sum + 1e-12:
                    # Accept the move
                    bitmask = new_bitmask
                    current_sum = new_sum
                    improved_global = True
                    # Restart the outer while loop to re-evaluate all removals
                    # because the set has changed significantly
                    break
        
        return bitmask, current_sum

    # Phase 1: Generate initial solutions
    best_bitmask = 0
    best_sum = 0.0
    
    orders = [
        list(range(1, N + 1)),  # Ascending
        list(range(N, 0, -1)),  # Descending
    ]
    
    # Add some random orders
    for _ in range(10):
        order = list(range(1, N + 1))
        random.shuffle(order)
        orders.append(order)
    
    for order in orders:
        bm = 0
        s = 0.0
        for x in order:
            if is_valid_addition(bm, x):
                bm |= (1 << (x - 1))
                s += 1.0 / x
        # Fill up
        bm, s = greedy_fill(bm, s)
        
        if s > best_sum:
            best_sum = s
            best_bitmask = bm
    
    # Phase 2: Local Search on best initial
    best_bitmask, best_sum = local_search_optimize(best_bitmask, best_sum, DEADLINE - 5)
    
    # Phase 3: Multiple restarts with perturbation
    # We use the remaining time for restarts
    while (time.time() - start_time) < DEADLINE - 2:
        # Perturb current best
        current = best_bitmask
        elems = [x for x in range(1, N + 1) if (current >> (x - 1)) & 1]
        
        if not elems:
            break
        
        # Remove 1-5 random elements
        num_remove = random.randint(1, min(5, len(elems)))
        to_remove = random.sample(elems, num_remove)
        for r in to_remove:
            current &= ~(1 << (r - 1))
        
        # Re-optimize
        current_sum = reciprocal_sum_float(current)
        current, current_sum = greedy_fill(current, current_sum)
        current, current_sum = local_search_optimize(current, current_sum, DEADLINE - 2)
        
        if current_sum > best_sum + 1e-12:
            best_sum = current_sum
            best_bitmask = current
    
    # Final polish
    best_bitmask, best_sum = local_search_optimize(best_bitmask, best_sum, DEADLINE - 1)
    
    # Convert bitmask to list
    result = [x for x in range(1, N + 1) if (best_bitmask >> (x - 1)) & 1]
    
    return result
# EVOLVE-BLOCK-END
