import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin before 60s
    
    N = 256
    
    # Precompute perfect squares up to 2*N = 512
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1
    
    # Build adjacency/conflict sets
    adj = [set() for _ in range(N)]
    
    # Identify vertices with self-loops (2*v is a square)
    forbidden = set()
    
    for i in range(N):
        v = i + 1
        # Check self-loop
        if 2 * v in squares:
            forbidden.add(i)
        
        # Find neighbors
        for sq in squares:
            u = sq - v
            if 1 <= u <= N:
                j = u - 1
                adj[i].add(j)
    
    # Candidates are vertices without self-loops
    candidates = [i for i in range(N) if i not in forbidden]
    
    # Precompute degrees for greedy ordering
    degrees = [len(adj[i]) for i in range(N)]
    
    # Modular residue classes: squares mod 8 are {0,1,4}
    # If a+b is a square, then a+b mod 8 ∈ {0,1,4}
    # This gives structure we can exploit for seeding
    def residue_class(v):
        """Return the residue class of v+1 mod 8"""
        return (v + 1) % 8
    
    # Group candidates by residue class
    residue_groups = {}
    for v in candidates:
        rc = residue_class(v)
        if rc not in residue_groups:
            residue_groups[rc] = []
        residue_groups[rc].append(v)
    
    best_size = 0
    best_set = frozenset()
    
    def greedy_solve(order=None):
        """
        Greedy construction. If order is None, use a randomized degree-based order.
        """
        if order is None:
            order = sorted(candidates, key=lambda v: (degrees[v], random.random()))
        
        current_set = set()
        
        for v in order:
            if not (adj[v] & current_set):
                current_set.add(v)
                
        return current_set

    def local_improve(s):
        """Try to add vertices that don't conflict."""
        s = set(s)
        improved = True
        while improved:
            improved = False
            for v in candidates:
                if v in s:
                    continue
                if not (adj[v] & s):
                    s.add(v)
                    improved = True
        return s

    def local_search_12(s, max_iterations=200):
        """
        1-2 exchange local search:
        - Try to add vertices (add-only)
        - If stuck, try removing one vertex and adding up to two others
        """
        s = set(s)
        s = local_improve(s)
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iterations and (time.time() - start_time < DEADLINE - 1):
            improved = False
            iter_count += 1
            
            # Try 1-2 exchanges: remove v from s, then try to add up to 2 new vertices
            vertices_to_try = list(s)
            random.shuffle(vertices_to_try)
            
            for v in vertices_to_try:
                s.remove(v)
                
                # Try to add vertices that were previously blocked by v
                # First, try to add any vertex not in s with no conflicts
                s_new = set(s)
                
                # Collect candidates that could be added
                addable = []
                for u in candidates:
                    if u in s_new:
                        continue
                    if not (adj[u] & s_new):
                        addable.append(u)
                
                if len(addable) >= 2:
                    # Try pairs of addable vertices
                    random.shuffle(addable)
                    for i in range(min(len(addable), 20)):
                        for j in range(i+1, min(len(addable), 20)):
                            u1, u2 = addable[i], addable[j]
                            if u1 == u2:
                                continue
                            # Check if u1 and u2 conflict with each other
                            if u2 in adj[u1]:
                                continue
                            s_test = set(s_new)
                            s_test.add(u1)
                            s_test.add(u2)
                            if len(s_test) > len(s) + 1:
                                s = s_test
                                improved = True
                                break
                        if improved:
                            break
                
                if improved:
                    break
                else:
                    # Restore v
                    s.add(v)
                    # Try to fill up again after restoring
                    s = local_improve(s)
            
            # Also try simple additions in case we missed some
            if not improved:
                s = local_improve(s)
                if len(s) > best_size:
                    best_size = len(s)
                    best_set = frozenset(s)
        
        return s

    # Modular-residue-based seeding strategy
    def modular_seed():
        """
        Seed based on modular residue classes.
        Squares mod 8 are {0, 1, 4}.
        If a+b is a square, a+b mod 8 ∈ {0,1,4}.
        So if a mod 8 = r, then b mod 8 must satisfy (r+b) mod 8 ∈ {0,1,4}.
        
        We can try to pick vertices from residue classes that are "safe" together.
        """
        # For each residue r, which residues s can pair with r such that r+s mod 8 is a square mod 8?
        # Square residues mod 8: {0, 1, 4}
        safe_pairs = {}
        for r in range(8):
            safe_pairs[r] = []
            for s in range(8):
                if (r + s) % 8 in (0, 1, 4):
                    safe_pairs[r].append(s)
        
        # Try different modular seeding strategies
        # Strategy: pick vertices from residue classes that don't conflict with each other
        # i.e., if we pick from class r, we should avoid class s where (r+s)%8 is a square mod 8
        
        # Actually, for an independent set, we want NO pair to have sum a square.
        # So if we pick from class r, we should NOT pick from class s if (r+s)%8 ∈ {0,1,4}
        # This is a necessary condition, not sufficient.
        
        # Let's just use this to create diverse initial orderings
        orderings = []
        
        # Strategy 1: Group by residue, process groups in different orders
        for perm_seed in range(5):
            random.seed(perm_seed * 12345 + 67890)
            group_order = list(residue_groups.keys())
            random.shuffle(group_order)
            
            order = []
            for rc in group_order:
                group = residue_groups[rc][:]
                random.shuffle(group)
                order.extend(group)
            
            # Fill in any missing candidates
            for v in candidates:
                if v not in order:
                    order.append(v)
            
            orderings.append(order)
        
        random.seed()  # Reset random seed
        return orderings

    # Iterated Local Search / Random Restart Greedy
    modular_orderings = modular_seed()
    mo_idx = 0
    
    while time.time() - start_time < DEADLINE:
        # Choose strategy
        r = random.random()
        if r < 0.3:
            # Strategy 1: Random order
            order = candidates[:]
            random.shuffle(order)
        elif r < 0.6:
            # Strategy 2: Degree-based order with random tie-breaking
            order = sorted(candidates, key=lambda v: (degrees[v], random.random()))
        else:
            # Strategy 3: Modular-residue-based ordering
            if mo_idx < len(modular_orderings):
                order = modular_orderings[mo_idx]
                mo_idx += 1
            else:
                # Regenerate modular orderings
                modular_orderings = modular_seed()
                mo_idx = 0
                order = modular_orderings[mo_idx]
                mo_idx += 1
        
        s = greedy_solve(order)
        s = local_improve(s)
        
        # Do 1-2 exchange local search if we have time
        if time.time() - start_time < DEADLINE - 3:
            s = local_search_12(s)
        
        if len(s) > best_size:
            best_size = len(s)
            best_set = frozenset(s)
            
    # Convert 0-based indices back to 1-based values
    return sorted([v + 1 for v in best_set])

# EVOLVE-BLOCK-END
