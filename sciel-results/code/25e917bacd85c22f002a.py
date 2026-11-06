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

    best_set = []
    best_len = 0
    
    # Adaptive ordering strategy
    # Maintain a "priority" for each element.
    # Elements that were skipped in previous runs get higher priority to be tried earlier.
    priorities = {x: 1.0 for x in ALL}
    
    # Initial static orders
    static_orders = [
        list(range(100, 0, -1)),
        list(range(1, 101))
    ]
    
    for o in static_orders:
        if time.time() > deadline - 2:
            break
        S = greedy(o)
        if len(S) > best_len:
            best_len = len(S)
            best_set = S
        if best_len == 100:
            return best_set

    # Adaptive random search
    # We run as many iterations as possible until deadline
    while time.time() < deadline - 2:
        # Generate order based on priorities
        # Sort by priority descending, then randomize ties
        # To ensure diversity, we can also mix in pure random sometimes
        if random.random() < 0.1:
            order = ALL[:]
            random.shuffle(order)
        else:
            # Sort by priority, add small random noise to break ties and maintain exploration
            order = sorted(ALL, key=lambda x: (priorities[x] + random.uniform(0, 0.1), x), reverse=True)
        
        S = greedy(order)
        
        if len(S) > best_len:
            best_len = len(S)
            best_set = S
            if best_len == 100:
                return best_set
        
        # Update priorities:
        # Elements in S: slightly decrease priority (they were easy/early)
        # Elements NOT in S: increase priority (they were skipped, try earlier next time)
        for x in ALL:
            if x in S:
                priorities[x] *= 0.9
            else:
                priorities[x] *= 1.1
                
    if not best_set:
        return [100]
        
    return best_set

# EVOLVE-BLOCK-END
