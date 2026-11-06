import math
import random
import time

def construct():
    n = 256
    max_sum = 2 * n
    squares = set()
    i = 1
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1

    # Build conflict graph
    # conflicts[u] is a set of v such that u+v is a square
    conflicts = [set() for _ in range(n + 1)]
    for a in range(1, n + 1):
        if 2 * a in squares:
            # Diagonal conflict with itself? 
            # The problem says "a+b is never a perfect square for a<=b in the set, INCLUDING a=b".
            # So if 2a is a square, a cannot be in the set at all?
            # Yes, if a is in the set, then pair (a,a) has sum 2a. If 2a is square, it's invalid.
            # So such 'a' are forbidden.
            pass 
        
    # Identify forbidden nodes (diagonal conflict)
    forbidden = set()
    for a in range(1, n + 1):
        if 2 * a in squares:
            forbidden.add(a)
            
    valid_nodes = [a for a in range(1, n + 1) if a not in forbidden]
    
    # Build conflicts among valid nodes
    for a in valid_nodes:
        for b in valid_nodes:
            if a < b and (a + b) in squares:
                conflicts[a].add(b)
                conflicts[b].add(a)
                
    # Simulated Annealing
    # State: set of active nodes
    # We start with a greedy solution to have a good initial state
    
    def greedy_init():
        s = set()
        # Sort by degree? Or random?
        # Let's use random order for diversity, or min-degree for quality.
        # Min-degree greedy is usually strong.
        remaining = set(valid_nodes)
        while remaining:
            min_deg = float('inf')
            chosen = None
            for v in remaining:
                deg = len(conflicts[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    chosen = v
            s.add(chosen)
            remaining.remove(chosen)
            for neighbor in conflicts[chosen]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
        return s

    # We will run SA starting from greedy_init
    # Penalty M should be large enough to prefer conflict-free states
    # But we allow conflicts to escape.
    # If we have a conflict, we can't count the size as "valid".
    # Let's define Energy = -|S| + M * (number of conflicts in S)
    # We want to minimize Energy.
    # A valid set has Energy = -|S|.
    # An invalid set with k conflicts has Energy = -|S| + M*k.
    # If M is large, we prefer valid sets.
    
    M = 1000  # Large penalty
    
    def get_conflict_count(s):
        # Count pairs (a,b) in s with a<=b and a+b in squares
        # Note: diagonal conflicts are already removed by 'forbidden', so a!=b for conflicts in valid_nodes
        # Wait, if a is in valid_nodes, 2a is NOT in squares. So diagonal is safe.
        # So we only check a < b.
        count = 0
        s_list = list(s)
        for i in range(len(s_list)):
            a = s_list[i]
            for b in s_list[i+1:]:
                if (a + b) in squares:
                    count += 1
        return count

    # Optimization: Precompute neighbors
    # To calculate delta energy quickly:
    # Adding v:
    #   New conflicts = len(conflicts[v] & s)
    #   Delta E = M * new_conflicts - 1
    # Removing v:
    #   Removed conflicts = len(conflicts[v] & s)
    #   Delta E = -M * removed_conflicts + 1
    
    start_time = time.time()
    timeout = 50 # Leave 10s buffer
    
    best_set = greedy_init()
    best_size = len(best_set)
    
    random.seed(42)
    
    # Multiple restarts
    for restart in range(10):
        current_set = set(best_set)
        # Or start from random greedy?
        # Let's start from greedy for speed.
        
        T = 50.0
        T_min = 0.01
        cooling = 0.95
        
        # Pre-calculate current conflict count?
        # For SA, we usually track the number of conflicts or just compute delta.
        # Since we start valid, conflicts = 0.
        # If we add a node with k neighbors in S, conflicts become k.
        # We need to track the set of conflicts? Or just the count?
        # If we add v, conflicts increase by k.
        # If we remove v, conflicts decrease by k.
        # This works if we only add/remove one node at a time.
        
        current_conflicts = 0 # Should be 0 if starting from valid
        
        while T > T_min:
            if time.time() - start_time > timeout:
                break
            
            # Pick a random move
            # 50% chance add, 50% chance remove?
            # Or pick a random node from all valid_nodes.
            v = random.choice(valid_nodes)
            
            if v in current_set:
                # Remove v
                k = len(conflicts[v] & current_set)
                delta_E = -M * k + 1
                if delta_E < 0 or random.random() < math.exp(delta_E / T):
                    current_set.remove(v)
                    current_conflicts -= k
            else:
                # Add v
                k = len(conflicts[v] & current_set)
                delta_E = M * k - 1
                if delta_E < 0 or random.random() < math.exp(delta_E / T):
                    current_set.add(v)
                    current_conflicts += k
            
            # Check if current state is valid and better
            if current_conflicts == 0:
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
            
            T *= cooling
            
        # After cooling, if we have conflicts, try to fix them greedily?
        # Or just rely on the best valid state found.
        
    # Final verification
    result = sorted(best_set)
    for i, a in enumerate(result):
        for b in result[i:]:
            s_val = a + b
            root = math.isqrt(s_val)
            if root * root == s_val:
                raise ValueError(f"Conflict: {a} + {b} = {s_val}")
                
    return result
