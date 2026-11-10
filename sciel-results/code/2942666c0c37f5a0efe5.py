import time
from math import gcd
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 512
    nodes = list(range(1, n + 1))
    
    # 1. Build adjacency as bitmasks
    # adj[i] contains bit j if lcm(i, j) <= 512
    adj = [0] * (n + 1)
    
    # Optimization: Compute lcm(i, j) <= n
    # We can precompute this more efficiently or just loop. 
    # 512*512/2 ~ 130k operations, which is very fast in Python if optimized.
    
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            g = gcd(i, j)
            l = (i // g) * j
            if l <= n:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
        if time.time() > deadline:
            return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
            
    def popcount(x):
        return bin(x).count('1')
        
    def verify_solution(lst):
        if not lst:
            return False
        # Check pairwise LCM
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                a, b = lst[i], lst[j]
                g = gcd(a, b)
                if (a // g) * b > n:
                    return False
        return True
        
    def greedy_clique(order):
        """
        Greedily build a clique by iterating through `order` and adding
        a node if it is compatible with all nodes already in the clique.
        """
        clique = []
        clique_mask = 0
        for v in order:
            # Check if v is compatible with all in clique
            # This is equivalent to checking if (adj[v] & clique_mask) == clique_mask
            if (adj[v] & clique_mask) == clique_mask:
                clique.append(v)
                clique_mask |= (1 << v)
        return clique

    def local_search_swap(clique):
        """
        Try to improve the clique by swapping one node out and adding one or more nodes in.
        Returns a new clique if improved, else the original.
        """
        current_size = len(clique)
        best_clique = list(clique)
        best_size = current_size
        
        clique_mask = 0
        for v in clique:
            clique_mask |= (1 << v)
            
        # Try removing each node
        for i in range(len(clique)):
            v_remove = clique[i]
            temp_mask = clique_mask & ~(1 << v_remove)
            
            # Try to add nodes that were previously incompatible or just not chosen
            # We iterate through all nodes not in the clique
            added = []
            added_mask = 0
            
            # To speed up, we can sort candidates by degree or just iterate
            # We want to add as many as possible that are compatible with temp_mask
            # and compatible with each other.
            # Since we only remove 1, we can try to add 2 or more to improve size.
            
            # Simple heuristic: try to add nodes in descending order of degree in the remaining graph
            # But for simplicity and speed, let's just try to add any node that fits.
            # We do a greedy fill on the remaining space.
            
            candidates = []
            for v in range(1, n + 1):
                if (clique_mask & (1 << v)):
                    continue # Already in clique (including v_remove, which we excluded from temp_mask but is in clique_mask)
                # Check compatibility with temp_mask
                if (adj[v] & temp_mask) == temp_mask:
                    candidates.append(v)
                    
            # Sort candidates by degree (compatibility count) descending to pick "best" ones first
            # Degree in the context of the current clique + other candidates
            # For simplicity, use global degree or degree in temp_mask
            candidates.sort(key=lambda x: -popcount(adj[x] & temp_mask))
            
            for v in candidates:
                if (adj[v] & (temp_mask | added_mask)) == (temp_mask | added_mask):
                    added.append(v)
                    added_mask |= (1 << v)
                    # If we've already added enough to beat best_size, we can stop
                    # New size = (current_size - 1) + len(added)
                    if (current_size - 1 + len(added)) <= best_size:
                        # We might still find a better combination later, but if we just want improvement:
                        pass
                        # Actually, we should keep adding to maximize size for this specific removal
                        # But if we've already exceeded best_size, we can update best and continue to see if we can do even better?
                        # No, if we find a valid clique of size > best_size, we update. 
                        # We should finish the greedy fill to get the maximal clique for this removal.
            
            new_size = current_size - 1 + len(added)
            if new_size > best_size:
                best_size = new_size
                # Reconstruct clique
                new_clique = [v for v in clique if v != v_remove]
                new_clique.extend(added)
                best_clique = new_clique
                clique_mask = temp_mask | added_mask # Update mask for subsequent checks? No, we reset per iteration.
                # Note: updating clique_mask here is tricky because we're iterating over the original clique.
                # It's safer to just compute the mask from best_clique if we need it, but we don't need it for the next i iteration 
                # because we start from the original clique each time.
                
        return best_clique

    # 2. Generate multiple orderings for Greedy
    active_nodes = list(range(1, n + 1))
    
    # Heuristic 1: By value (ascending)
    order1 = list(active_nodes)
    
    # Heuristic 2: By value (descending)
    order2 = list(reversed(active_nodes))
    
    # Heuristic 3: By degree (number of compatible nodes) descending
    # Precompute degrees
    degrees = [popcount(adj[i]) for i in range(n + 1)]
    order3 = sorted(active_nodes, key=lambda x: -degrees[x])
    
    # Heuristic 4: By odd part, then value
    def odd_part(x):
        while x % 2 == 0:
            x //= 2
        return x
    order4 = sorted(active_nodes, key=lambda x: (odd_part(x), x))
    
    # Heuristic 5: By "core-ness" - prefer powers of 2?
    # Powers of 2 are compatible with everything that divides them? No.
    # Powers of 2 are compatible with any number if the number is a power of 2? No.
    # lcm(2^a, 2^b) = 2^max(a,b). So all powers of 2 are mutually compatible.
    # Also 2^a is compatible with any number whose odd part is 1? No.
    # 2^a is compatible with x if lcm(2^a, x) <= 512.
    # If x = 2^k * m (m odd), lcm = 2^max(a,k) * m.
    # So if we pick a large power of 2, we restrict the odd parts of other numbers.
    # If we pick small powers of 2, we allow larger odd parts?
    # Actually, the set of all powers of 2 is a clique of size 10.
    # The set of all divisors of 512 is a clique of size 10.
    
    # Let's run greedy on all orderings
    best_clique = []
    best_size = 0
    
    for order in [order1, order2, order3, order4]:
        c = greedy_clique(order)
        if len(c) > best_size:
            best_size = len(c)
            best_clique = c
            
    # 3. Local Search (Swap)
    # Run local search on the best clique found
    improved_clique = local_search_swap(best_clique)
    if len(improved_clique) > best_size:
        best_clique = improved_clique
        best_size = len(best_clique)
        
    # 4. Randomized Restart Greedy
    # If we have time, try random perturbations
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        for _ in range(100): # Try 100 random restarts
            if time.time() > deadline - 2.0:
                break
            
            # Shuffle the order slightly or use a random subset?
            # Better: Randomly select a starting node, then greedily add neighbors
            # Or: Shuffle the global order and run greedy
            
            shuffled_order = list(active_nodes)
            random.shuffle(shuffled_order)
            
            # Bias towards high degree nodes?
            # Just run greedy on shuffled
            c = greedy_clique(shuffled_order)
            
            if len(c) > best_size:
                best_size = len(c)
                best_clique = c
                
            # Also try local search on this new clique if it's promising
            if len(c) >= best_size - 2: # Only if it's close
                c_improved = local_search_swap(c)
                if len(c_improved) > best_size:
                    best_size = len(c_improved)
                    best_clique = c_improved

    # Final verification
    if not verify_solution(best_clique):
        # Fallback to a known valid set: powers of 2
        return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        
    return best_clique
