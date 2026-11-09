import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill

    n = 16
    k = 4
    max_target = 250

    # Generate all 4-subsets as bitmasks (0-indexed internally, 1-indexed in output)
    all_sets = []
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append(mask)

    total = len(all_sets)

    # Precompute conflict graph: two sets conflict if they can form a sunflower with some third set
    # A sunflower with core C means: |A∩B| = |A∩C| = |B∩C| = |C| and A∩B∩C = C
    # For two sets A, B to conflict, there exists some C such that A∩B∩C = A∩B and |A∩C|=|A∩B| and |B∩C|=|A∩B|
    # This means C must contain A∩B, and |A∩C|=|A∩B| means C∩(A\B)=∅, similarly C∩(B\A)=∅
    # So C must be a superset of A∩B and disjoint from (A\B)∪(B\A)
    
    # Build conflict adjacency list
    conflict_adj = [set() for _ in range(total)]
    
    # For each pair (i, j), check if they can form a sunflower with some third set
    # This is expensive O(n^3) but we can optimize
    # Two sets A, B conflict if there exists C such that:
    #   C ⊇ A∩B, C ∩ (A\B) = ∅, C ∩ (B\A) = ∅, and |C|=4
    # Let I = A∩B, D1 = A\B, D2 = B\A
    # C must satisfy: I ⊆ C, C ∩ D1 = ∅, C ∩ D2 = ∅, |C|=4
    # So C must choose (4-|I|) elements from [n]\I\D1\D2
    # Such a C exists iff |[n]\I\D1\D2| >= 4-|I|
    
    # Precompute for faster checking
    # Actually, let's just check if the condition holds for each pair
    # But this is O(n^2) pairs, and for each we need to check existence of C
    
    # Better approach: for each pair (i,j), compute the number of valid C's
    # If > 0, they conflict
    
    # Let's precompute masks and use bit operations
    # For pair (i, j):
    #   I = all_sets[i] & all_sets[j]
    #   D1 = all_sets[i] & ~all_sets[j]
    #   D2 = all_sets[j] & ~all_sets[i]
    #   Available = ~(I | D1 | D2) & ((1 << n) - 1)
    #   need = 4 - popcount(I)
    #   If popcount(Available) >= need, then conflict exists
    
    full_mask = (1 << n) - 1
    
    # Precompute popcount for all possible masks? Or use int.bit_count()
    # Python 3.10+ has int.bit_count()
    
    for i in range(total):
        mi = all_sets[i]
        for j in range(i + 1, total):
            mj = all_sets[j]
            I = mi & mj
            I_count = I.bit_count()
            D1 = mi & ~mj
            D2 = mj & ~mi
            Available = ~(I | D1 | D2) & full_mask
            need = 4 - I_count
            if Available.bit_count() >= need:
                conflict_adj[i].add(j)
                conflict_adj[j].add(i)
    
    # Now use greedy independent set with randomized ordering
    best_family = []
    best_size = 0
    
    def greedy_independent_set(order, current_best_size=0):
        """
        Greedily select an independent set from the given order.
        A set can be added if it has no conflicts with already selected sets.
        """
        selected = set()
        F = []
        
        for pos, idx in enumerate(order):
            if time.time() > deadline:
                break
            
            # Check if idx conflicts with any selected set
            can_add = True
            for s in selected:
                if s in conflict_adj[idx]:
                    can_add = False
                    break
            
            if can_add:
                selected.add(idx)
                F.append(idx)
                
                if len(F) >= max_target:
                    break
                # Early termination heuristic
                if len(F) + (len(order) - pos - 1) <= current_best_size:
                    break
        
        return F
    
    def local_improvement(F, max_iter=100):
        """
        Try to improve the family by swapping one set for another.
        F is a list of set indices.
        """
        F_set = set(F)
        improved = True
        iterations = 0
        
        while improved and iterations < max_iter and time.time() < deadline - 5.0:
            improved = False
            iterations += 1
            
            # Try removing each set and adding a non-conflicting one
            for i in range(len(F)):
                if time.time() > deadline - 5.0:
                    break
                removed = F[i]
                F_set.remove(removed)
                
                # Find a set not in F_set that doesn't conflict with remaining F_set
                best_add = -1
                for candidate in range(total):
                    if candidate in F_set:
                        continue
                    # Check if candidate conflicts with any in F_set
                    can_add = True
                    for s in F_set:
                        if s in conflict_adj[candidate]:
                            can_add = False
                            break
                    if can_add:
                        # Prefer candidates that conflict with fewer existing sets (more "room")
                        best_add = candidate
                        break
                
                if best_add != -1:
                    F_set.add(best_add)
                    F[i] = best_add
                    improved = True
                    break  # Restart after one swap
        
        return F
    
    # 1. Deterministic pass (natural order)
    if time.time() < deadline:
        order = list(range(total))
        F = greedy_independent_set(order, best_size)
        if len(F) > best_size:
            best_family = F
            best_size = len(F)
    
    # 2. Random restarts with different seeds/orders
    rng = random.Random(12345)
    
    # Precompute degree for weighted random ordering
    degrees = [len(conflict_adj[i]) for i in range(total)]
    
    while time.time() < deadline - 3.0:  # Leave 3s for cleanup/return
        order = list(range(total))
        rng.shuffle(order)
        
        F = greedy_independent_set(order, best_size)
        
        if len(F) > best_size:
            best_family = F
            best_size = len(F)
            if best_size >= max_target:
                break
        
        # Local improvement on the best family found so far
        if time.time() < deadline - 5.0 and best_size > 0:
            improved_F = local_improvement(best_family[:], max_iter=50)
            if len(improved_F) > best_size:
                best_family = improved_F
                best_size = len(best_family)
                if best_size >= max_target:
                    break

    # Convert best_family masks to sorted 1-indexed lists
    result = []
    for idx in best_family:
        mask = all_sets[idx]
        elems = []
        for i in range(n):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))

    return result
