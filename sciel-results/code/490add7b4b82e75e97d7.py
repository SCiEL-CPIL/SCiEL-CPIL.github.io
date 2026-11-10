import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    N = 512
    
    # Precompute lcm for all pairs
    # lcm[i][j] = lcm(i+1, j+1) using 0-indexed
    # We'll use a 2D list for fast access
    
    # Actually, let's use a different approach:
    # For each number x in [1, 512], compute which numbers y have lcm(x,y) <= 512
    # compat[x] is a set (or bitset) of y such that lcm(x,y) <= 512
    
    # Use Python integers as bitsets for efficiency
    # bit i (0-indexed) represents number i+1
    
    # Precompute compatibility bitsets
    compat = [0] * (N + 1)  # 1-indexed, compat[x] is bitset of compatible y's
    for x in range(1, N + 1):
        mask = 0
        for y in range(1, N + 1):
            g = gcd(x, y)
            lcm_val = x // g * y
            if lcm_val <= N:
                mask |= (1 << (y - 1))
        compat[x] = mask
    
    # The problem is: find a maximum subset S of {1,...,512} such that for all a,b in S, lcm(a,b) <= 512
    # This is equivalent to: for all a,b in S, a is in compat[b] (and b is in compat[a])
    # In graph terms: vertices 1..512, edge between a and b if lcm(a,b) > 512
    # We want a maximum independent set in this conflict graph
    # Equivalently: maximum clique in the compatibility graph (edge if lcm <= 512)
    
    # Let's use a branch-and-bound approach for maximum clique in the compatibility graph
    # The compatibility graph has an edge between a and b if lcm(a,b) <= 512
    
    # For maximum clique with bitsets:
    # We maintain a set of candidates (as a bitset) that are all compatible with the current clique
    # and try to add vertices from this candidate set
    
    # Maximum clique algorithm (Bron-Kerbosch variant with coloring for bound)
    
    # First, let's get a good initial lower bound using a greedy heuristic
    
    def greedy_clique(candidates_mask):
        """Find a large clique greedily from candidates_mask"""
        clique = []
        remaining = candidates_mask
        
        while remaining and time.time() < deadline:
            # Pick the vertex with the most connections within remaining
            best_v = -1
            best_count = -1
            m = remaining
            pos = 0
            while m:
                if m & 1:
                    v = pos + 1
                    # Count how many in remaining are compatible with v
                    count = bin(compat[v] & remaining).count('1')
                    if count > best_count:
                        best_count = count
                        best_v = v
                m >>= 1
                pos += 1
            
            if best_v == -1:
                break
            
            clique.append(best_v)
            # Update remaining: keep only vertices compatible with best_v
            remaining = remaining & compat[best_v]
            remaining &= ~(1 << (best_v - 1))  # remove best_v itself
        
        return clique
    
    # Get initial solution
    all_mask = (1 << N) - 1
    initial_clique = greedy_clique(all_mask)
    best_clique = set(initial_clique)
    best_size = len(best_clique)
    
    # Now do local search / branch and bound to improve
    
    # Branch and bound for maximum clique
    # We'll use a recursive approach with pruning
    
    # For efficiency, let's use a different strategy:
    # Start from the greedy solution and try to improve it
    
    # Strategy: try to add one more element by removing some existing elements
    # Or: use a more systematic search
    
    # Let's try a simulated annealing / local search approach
    # that can escape local optima
    
    import random
    
    current_set = set(best_clique)
    current_size = best_size
    
    # Precompute: for each element, its compatibility with the current set
    # We'll do iterative improvement
    
    def get_compatible_additions(s):
        """Get all elements that can be added to set s"""
        s_mask = 0
        for x in s:
            s_mask |= (1 << (x - 1))
        
        additions = []
        for x in range(1, N + 1):
            if x not in s:
                if (compat[x] & s_mask) == s_mask:
                    additions.append(x)
        return additions
    
    def get_conflicts(s):
        """For each element in s, get which other elements in s it conflicts with... wait, 
           in our formulation, if s is a valid clique, there are no conflicts within s.
           We need: for each element x in s, which elements y NOT in s would become 
           incompatible if we remove x? That's not quite right either.
           
           Let me think differently: 
           For local search, we can try:
           1. Add one element (if possible)
           2. Remove one element and add one or more
           3. Swap: remove one, add one
        """
        pass
    
    # Local search: try to improve the current set
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try to add any compatible element
        s_mask = 0
        for x in current_set:
            s_mask |= (1 << (x - 1))
        
        for x in range(1, N + 1):
            if x not in current_set:
                if (compat[x] & s_mask) == s_mask:
                    current_set.add(x)
                    current_size += 1
                    improved = True
                    s_mask |= (1 << (x - 1))
        
        if improved:
            continue
        
        # Try 1-out, multiple-in moves
        for remove_x in list(current_set):
            if time.time() > deadline:
                break
            
            trial_set = current_set - {remove_x}
            trial_mask = 0
            for y in trial_set:
                trial_mask |= (1 << (y - 1))
            
            # Try to add as many compatible elements as possible
            added_count = 0
            for x in range(1, N + 1):
                if x not in trial_set and x != remove_x:
                    if (compat[x] & trial_mask) == trial_mask:
                        trial_set.add(x)
                        trial_mask |= (1 << (x - 1))
                        added_count += 1
            
            if len(trial_set) > current_size:
                current_set = trial_set
                current_size = len(current_set)
                improved = True
                break
    
    # Try random restarts with different greedy orderings
    for restart in range(200):
        if time.time() > deadline:
            break
        
        # Shuffle the vertex order for greedy
        vertices = list(range(1, N + 1))
        random.shuffle(vertices)
        
        clique = []
        remaining = all_mask
        
        for v in vertices:
            if time.time() > deadline:
                break
            if remaining & (1 << (v - 1)):
                # Check if v is compatible with all in clique
                clique_mask = 0
                for u in clique:
                    clique_mask |= (1 << (u - 1))
                if (compat[v] & clique_mask) == clique_mask:
                    clique.append(v)
                    remaining = remaining & compat[v]
                    remaining &= ~(1 << (v - 1))
        
        if len(clique) > current_size:
            current_set = set(clique)
            current_size = len(current_set)
            
            # Try to improve this new solution
            s_mask = 0
            for x in current_set:
                s_mask |= (1 << (x - 1))
            
            for x in range(1, N + 1):
                if x not in current_set:
                    if (compat[x] & s_mask) == s_mask:
                        current_set.add(x)
                        current_size += 1
                        s_mask |= (1 << (x - 1))
    
    # Final local search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        s_mask = 0
        for x in current_set:
            s_mask |= (1 << (x - 1))
        
        # Try to add
        for x in range(1, N + 1):
            if x not in current_set:
                if (compat[x] & s_mask) == s_mask:
                    current_set.add(x)
                    current_size += 1
                    improved = True
                    s_mask |= (1 << (x - 1))
        
        if improved:
            continue
        
        # Try 1-out, multiple-in
        for remove_x in list(current_set):
            if time.time() > deadline:
                break
            
            trial_set = current_set - {remove_x}
            trial_mask = 0
            for y in trial_set:
                trial_mask |= (1 << (y - 1))
            
            added_count = 0
            for x in range(1, N + 1):
                if x not in trial_set and x != remove_x:
                    if (compat[x] & trial_mask) == trial_mask:
                        trial_set.add(x)
                        trial_mask |= (1 << (x - 1))
                        added_count += 1
            
            if len(trial_set) > current_size:
                current_set = trial_set
                current_size = len(current_set)
                improved = True
                break
    
    return list(current_set)
