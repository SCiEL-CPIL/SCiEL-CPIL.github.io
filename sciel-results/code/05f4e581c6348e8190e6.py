import time
import random

# EVOLVE-BLOCK-START

def construct():
    N = 512
    start_time = time.time()
    time_limit = 85.0  # Leave 5s margin for evaluator overhead

    # Precompute conflicts for each x
    # conflicts[x] is a list of tuples (u, v, w) such that {x, u, v, w} forms a 4-AP
    conflicts = [[] for _ in range(N + 1)]
    
    # Generate all 4-APs (a, a+d, a+2d, a+3d)
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            
            # Add to conflicts for each element
            conflicts[a].append((b, c, e))
            conflicts[b].append((a, c, e))
            conflicts[c].append((a, b, e))
            conflicts[e].append((a, b, c))

    def can_add(x, S):
        """Check if x can be added to S without forming a 4-AP."""
        for u, v, w in conflicts[x]:
            if u in S and v in S and w in S:
                return False
        return True

    # Greedy Initialization
    S = set()
    # Use a random order for initialization to potentially find a better local optimum
    # We will run a few random greedy passes and keep the best one as the starting point
    best_init_S = set()
    best_init_size = 0
    
    rng = random.Random(42) # Fixed seed for reproducibility of the init phase
    
    # Try a few random greedy initializations
    for _ in range(5):
        S_temp = set()
        order = list(range(1, N + 1))
        rng.shuffle(order)
        for x in order:
            if can_add(x, S_temp):
                S_temp.add(x)
        if len(S_temp) > best_init_size:
            best_init_size = len(S_temp)
            best_init_S = set(S_temp)
            
    S = best_init_S
    best_S = set(S)
    best_size = len(S)
    
    # Local Search
    # We will use a "kick" strategy.
    # 1. Try to add all possible elements.
    # 2. If stuck, remove k random elements and try again.
    
    # To make "try to add all" efficient, we iterate over a shuffled list of all N elements
    # and check if they are in S. If not, try to add.
    
    # We will loop until time runs out.
    # We track best_S.
    
    # Use a new random generator for the search phase to avoid correlation with init
    search_rng = random.Random()
    
    kick_size = 1
    no_improve_count = 0
    max_no_improve = 50
    
    # We'll keep a list of all elements to shuffle
    all_elements = list(range(1, N + 1))
    
    while time.time() - start_time < time_limit:
        # Phase 1: Greedy completion
        # Shuffle to avoid bias
        search_rng.shuffle(all_elements)
        
        added_any = False
        for x in all_elements:
            if x in S:
                continue
            if can_add(x, S):
                S.add(x)
                added_any = True
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
            no_improve_count = 0
            # Reset kick size on improvement
            kick_size = 1
        else:
            no_improve_count += 1
        
        if not added_any:
            # Stuck in local optimum
            # Perform a kick
            if len(S) <= kick_size:
                # Can't kick if set is too small
                break
            
            # Remove kick_size random elements from S
            elements_in_S = list(S)
            search_rng.shuffle(elements_in_S)
            to_remove = elements_in_S[:kick_size]
            for y in to_remove:
                S.remove(y)
            
            # Increase kick size if we've been stuck for a while
            if no_improve_count > max_no_improve:
                kick_size = min(kick_size + 1, 15)
                no_improve_count = 0
                max_no_improve = int(max_no_improve * 1.5)
                
    return sorted(best_S)

# EVOLVE-BLOCK-END
