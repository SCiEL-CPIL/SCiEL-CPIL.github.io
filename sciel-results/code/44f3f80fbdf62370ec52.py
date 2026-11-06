"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 55
    
    TARGET = 200
    ALL = list(range(1, 101))
    
    def greedy(order):
        """
        Greedy construction based on a given order of elements.
        Returns a valid subset.
        """
        S = []
        # reachable[s] is True if sum s can be formed by a subset of S
        reachable = [False] * (TARGET + 1)
        reachable[0] = True
        
        for x in order:
            # Check if adding x creates a sum of TARGET
            # We need to check if any existing sum s satisfies s + x == TARGET
            # i.e., s == TARGET - x
            # If so, we cannot add x.
            
            # Also, we need to update reachable sums.
            # We only care about sums < TARGET.
            
            if reachable[TARGET - x]:
                continue
            
            # If we can add x, update reachable sums
            # We iterate downwards to avoid using x multiple times (though 0/1 knapsack style)
            # Actually, since we are just checking existence, and we process x once,
            # we can just mark new sums.
            
            new_reachable = reachable[:]
            for s in range(TARGET - x, -1, -1):
                if reachable[s]:
                    ns = s + x
                    if ns < TARGET:
                        new_reachable[ns] = True
            
            S.append(x)
            reachable = new_reachable
            
        return S

    # Generate multiple orders
    orders = []
    
    # 1. Descending (Seed strategy)
    orders.append(list(range(100, 0, -1)))
    
    # 2. Ascending
    orders.append(list(range(1, 101)))
    
    # 3. Random shuffles
    # Use a fixed seed for reproducibility in evaluation if needed, or random
    # The task says "Python standard library... available".
    # We should ensure we use enough time.
    num_random_orders = 50
    for _ in range(num_random_orders):
        o = ALL[:]
        random.shuffle(o)
        orders.append(o)
        
    # Also try some heuristic orders
    # Sort by "value" but with noise?
    # Or sort by "rarity" of sum contribution?
    
    best_set = []
    
    for o in orders:
        if time.time() > deadline - 5:
            break
        S = greedy(o)
        if len(S) > len(best_set):
            best_set = S
        # Optional: If we found a very large set, we might stop early?
        # Max possible is 100.
        if len(best_set) == 100:
            break

    # If no set found (should not happen), return a small valid set
    if not best_set:
        # {100} is valid
        return [100]
        
    return best_set

# EVOLVE-BLOCK-END
