import random
import time

def construct():
    N = 256
    # Precompute conflicts
    # conflict[i][j] is True if (i+1, j+1) is a bad pair
    # We want to avoid bad pairs.
    # Let's store neighbors in the CONFLICT graph (bad pairs).
    # If i and j are in conflict, we can't have both.
    # This is an Independent Set problem on the Conflict Graph.
    
    conflict = [0] * N
    for i in range(N):
        a = i + 1
        for j in range(i + 1, N):
            b = j + 1
            if (a * b) % (a + b) == 0:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
                
    def popcount(x):
        return bin(x).count('1')
        
    def greedy_independent_set(order):
        """
        order: list of indices to consider
        Returns a bitset of the independent set found.
        """
        selected = 0
        for i in order:
            # Check if i conflicts with any selected node
            if (conflict[i] & selected) == 0:
                selected |= (1 << i)
        return selected

    def local_search(initial_set, time_limit=50):
        """
        Tries to improve the set by adding vertices or swapping.
        Returns the best set found.
        """
        current = initial_set
        best = initial_set
        best_size = popcount(current)
        start_time = time.time()
        
        # Vertices not in current set
        all_mask = (1 << N) - 1
        excluded = all_mask & ~current
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            
            # 1. Try to add any single vertex that has no conflicts
            # We can do this efficiently by checking conflicts.
            # Let's iterate through excluded vertices.
            # To make it faster, we can prioritize vertices with fewer conflicts with current set.
            # But simple iteration is okay for N=256.
            
            addable = 0
            e = excluded
            while e:
                v_bit = e & -e
                idx = v_bit.bit_length() - 1
                if (conflict[idx] & current) == 0:
                    addable |= v_bit
                e &= e - 1
                
            if addable:
                # Add all addable vertices?
                # Adding one might block another.
                # Greedy: Add them one by one, updating 'current' and 'excluded'.
                # Let's just add the first one found, or all if they don't conflict with each other.
                # Since they have no conflict with 'current', do they conflict with each other?
                # If v1 and v2 are both addable, they might conflict with each other.
                # So we should add them greedily.
                new_current = current
                new_excluded = excluded
                a = addable
                while a:
                    v_bit = a & -a
                    idx = v_bit.bit_length() - 1
                    # Check if it conflicts with already added in this step
                    if (conflict[idx] & new_current) == 0:
                        new_current |= v_bit
                        new_excluded &= ~v_bit
                    a &= a - 1
                
                current = new_current
                excluded = new_excluded
                if popcount(current) > best_size:
                    best = current
                    best_size = popcount(current)
                improved = True
                continue # Loop again to check for more additions
            
            # 2. If no single vertex can be added, try swaps.
            # Remove one vertex u from current, then try to add many vertices.
            # This is expensive. Let's try a simpler heuristic:
            # Pick a vertex u in current with high "blocking" power?
            # Or just random restarts of the greedy phase.
            
            # Given the time limit, and that we already did a greedy pass,
            # if we are stuck, we break and rely on multiple restarts.
            # But let's try one simple move:
            # Pick a random vertex u in current.
            # Remove it.
            # Try to greedily add vertices that were previously excluded.
            # If size increases, keep it.
            
            # To avoid infinite loops or slow progress, we limit this phase.
            # If we don't improve in a few swaps, we stop.
            
            # Actually, for N=256, the greedy set is likely quite good.
            # The main gain comes from better initial ordering.
            
            break # Break local search, rely on restarts
            
        return best

    # Strategy: Multiple random restarts with different orderings
    # 1. Natural order
    # 2. Reverse order
    # 3. Random orders
    # 4. Order by "conflict degree" (ascending or descending)
    
    degrees = [popcount(conflict[i]) for i in range(N)]
    
    orders = []
    orders.append(list(range(N)))
    orders.append(list(range(N-1, -1, -1)))
    orders.append(sorted(range(N), key=lambda x: degrees[x]))
    orders.append(sorted(range(N), key=lambda x: -degrees[x]))
    
    # Add some random orders
    for _ in range(10):
        o = list(range(N))
        random.shuffle(o)
        orders.append(o)
        
    best_global = 0
    best_global_size = 0
    
    start_total = time.time()
    time_limit = 55 # Leave 5s for overhead
    
    for order in orders:
        if time.time() - start_total > time_limit:
            break
        s = greedy_independent_set(order)
        s = local_search(s, time_limit=2) # Short local search per restart
        
        size = popcount(s)
        if size > best_global_size:
            best_global_size = size
            best_global = s
            
    # Final polish on the best set found
    # Try a longer local search on the best set
    best_final = local_search(best_global, time_limit=5)
    
    result = []
    for i in range(N):
        if (best_final >> i) & 1:
            result.append(i + 1)
            
    return result
