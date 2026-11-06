"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    # 255 = 3 * 5 * 17
    # Candidate numbers must share a factor with 255 (to have gcd>1 with 255)
    # AND must pairwise share a factor with each other
    
    # Build candidate pool: numbers in [1,255] that share a factor with 255
    # i.e., divisible by 3, 5, or 17
    candidates = [x for x in range(1, 256) if x % 3 == 0 or x % 5 == 0 or x % 17 == 0]
    # 255 must be included
    if 255 not in candidates:
        candidates.append(255)
    
    n = len(candidates)
    idx = {v: i for i, v in enumerate(candidates)}
    
    # Build adjacency: edge between i,j if gcd(candidates[i], candidates[j]) > 1
    # Use bitmasks for efficiency
    adj = [0] * n  # adj[i] is a bitmask of neighbors of i
    
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # We need a clique that includes vertex for 255
    v255 = idx[255]
    
    # Greedy coloring upper bound + branch and bound for max clique
    # Since n can be up to ~170, we use a heuristic approach
    
    # Strategy: greedy construction + local improvement (simulated annealing style)
    
    def greedy_clique():
        """Greedy: start with 255, add vertices that connect to all current members."""
        clique = {v255}
        # Sort candidates by degree (number of neighbors in full graph) descending
        degrees = [bin(adj[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda x: -degrees[x])
        
        for v in order:
            if v in clique:
                continue
            # Check if v connects to all in clique
            compatible = True
            for u in clique:
                if not (adj[u] & (1 << v)):
                    compatible = False
                    break
            if compatible:
                clique.add(v)
        return clique
    
    def can_add(clique, v):
        """Check if v can be added to clique (connects to all members)."""
        for u in clique:
            if not (adj[u] & (1 << v)):
                return False
        return True
    
    def remove_from_clique(clique, v):
        """Try to remove v and add something else to improve."""
        pass
    
    # Start with greedy
    best_clique = greedy_clique()
    best_size = len(best_clique)
    
    # Local search: try swapping out one element for another
    # Represent clique as a set
    clique = set(best_clique)
    
    # Precompute for each vertex, which other vertices it's NOT connected to
    # non_neighbors[i] = bitmask of vertices not connected to i (excluding self)
    non_neighbors = [0] * n
    for i in range(n):
        # All vertices except i
        all_mask = (1 << n) - 1
        non_neighbors[i] = all_mask & ~(adj[i] | (1 << i))
    
    # Simulated annealing / local search
    random.seed(42)
    current_size = len(clique)
    best_size = current_size
    best_clique = set(clique)
    
    temp = 10.0
    cooling = 0.9995
    
    while time.time() < deadline:
        # Try a random move: remove one random element, add one random compatible element
        if current_size == 0:
            break
        
        # Randomly pick an element to remove
        clique_list = list(clique)
        remove_v = random.choice(clique_list)
        
        # Find candidates to add: vertices not in clique that connect to all remaining
        remaining = clique - {remove_v}
        
        # Find all vertices that connect to all in remaining
        addable = []
        for v in range(n):
            if v in remaining:
                continue
            if v == remove_v:
                continue
            # Check v connects to all in remaining
            ok = True
            for u in remaining:
                if not (adj[u] & (1 << v)):
                    ok = False
                    break
            if ok:
                addable.append(v)
        
        if not addable:
            # Can't add anything, just skip
            continue
        
        add_v = random.choice(addable)
        
        # Perform the swap
        clique.remove(remove_v)
        clique.add(add_v)
        new_size = len(clique)
        
        # Accept or reject based on simulated annealing
        delta = new_size - current_size
        if delta >= 0 or random.random() < math.exp(delta / max(temp, 0.01)):
            current_size = new_size
            if current_size > best_size:
                best_size = current_size
                best_clique = set(clique)
        else:
            # Reject: undo
            clique.remove(add_v)
            clique.add(remove_v)
        
        temp *= cooling
        if temp < 0.01:
            temp = 0.01
    
    # Final greedy polish: try to add any remaining compatible vertices
    for v in range(n):
        if v not in best_clique:
            if can_add(best_clique, v):
                best_clique.add(v)
    
    # Convert back to list of values
    result = sorted([candidates[v] for v in best_clique])
    
    # Ensure 255 is in the result
    if 255 not in result:
        result.append(255)
    
    return result
# EVOLVE-BLOCK-END
