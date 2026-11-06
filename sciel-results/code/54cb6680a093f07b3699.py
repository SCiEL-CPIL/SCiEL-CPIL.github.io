import time
from math import gcd

def construct():
    start_time = time.time()
    limit = 512
    deadline = start_time + 80  # Safety margin

    # 1. Precompute compatibility graph
    # compat[i] is a bitmask where bit j is 1 if lcm(i, j) <= limit
    compat = [0] * (limit + 1)
    for i in range(1, limit + 1):
        mask = 0
        for j in range(1, limit + 1):
            g = gcd(i, j)
            if (i // g) * j <= limit:
                mask |= (1 << j)
        compat[i] = mask

    # 2. Initial Solution: Max divisors of any number <= limit
    best_size = 0
    best_clique = []
    
    # Check all n, find max divisors
    max_divs = 0
    best_n = 1
    for n in range(1, limit + 1):
        d = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                d += 1
                if i != n // i:
                    d += 1
        if d > max_divs:
            max_divs = d
            best_n = n
            
    # Get divisors of best_n
    divs = [i for i in range(1, limit + 1) if best_n % i == 0]
    best_size = len(divs)
    best_clique = divs
    
    # 3. Branch and Bound for Max Clique in Compatibility Graph
    # State: candidates (bitmask), current_clique (list)
    # We use a global best
    
    def expand(candidates, current_clique):
        nonlocal best_size, best_clique
        
        # Pruning 1: Loose bound
        if len(current_clique) + candidates.bit_count() <= best_size:
            return
            
        # Pruning 2: If no candidates, update best
        if candidates == 0:
            if len(current_clique) > best_size:
                best_size = len(current_clique)
                best_clique = current_clique.copy()
            return

        # Optional: Greedy Coloring Bound
        # If candidates.bit_count() is large, coloring might help.
        # For now, let's rely on the loose bound + good initial solution.
        # If this is too slow, we will add coloring.
        
        # Branching
        # Pick a vertex v from candidates
        # Heuristic: Pick vertex with highest degree in the candidate subgraph?
        # Or just the lowest set bit?
        # Let's pick the lowest set bit for simplicity, or highest degree.
        
        # To improve, let's pick a vertex that is in many cliques?
        # Simple: Pick the first available vertex.
        # Better: Pick vertex with max degree in candidates.
        
        # Finding max degree vertex in bitmask subgraph is O(N).
        # v = argmax_{u in candidates} popcount(compat[u] & candidates)
        
        max_deg = -1
        v = -1
        c = candidates
        while c:
            # Get lowest set bit index
            lsb = c & -c
            idx = lsb.bit_length() - 1
            deg = (compat[idx] & candidates).bit_count()
            if deg > max_deg:
                max_deg = deg
                v = idx
            c ^= lsb
            
        if v == -1:
            return

        bit_v = 1 << v

        # Branch 1: Include v
        # New candidates must be compatible with v
        new_candidates = candidates & compat[v]
        # Remove v from new_candidates? 
        # The clique is formed by v + clique from new_candidates.
        # Note: v is not in new_candidates if we don't add it back?
        # Usually, we remove v from the candidate set before branching?
        # Standard:
        # Branch 1: Include v. New candidates = (candidates \ {v}) & compat[v].
        # Branch 2: Exclude v. New candidates = candidates \ {v}.
        
        candidates_without_v = candidates & ~bit_v
        
        # Include v
        expand(new_candidates & ~bit_v, current_clique + [v]) # Wait, new_candidates already excludes v? 
        # compat[v] might contain v? lcm(v,v)=v<=512, so yes.
        # So new_candidates = candidates & compat[v].
        # We should remove v from the recursion to avoid self-loop issues? 
        # The clique is a set. 
        # If we include v, the next vertices must be in candidates \ {v} and compatible with v.
        
        # Exclude v
        expand(candidates_without_v, current_clique)

    # Start B&B
    # Initial candidates: all vertices
    all_candidates = (1 << (limit + 1)) - 2 # bits 1..512
    # We already have a good best_size.
    # To speed up, we can restrict candidates to those compatible with the initial clique?
    # No, we want to find a *better* clique.
    # We can just run B&B on the whole graph.
    
    # To prevent stack overflow or infinite loops, we rely on pruning.
    # With N=512 and good pruning, this should be fine.
    
    # One optimization: If best_size is already 24, and we know 24 is likely optimal, 
    # we can just return. But let's try to find 25+.
    
    # Run B&B
    # To avoid recursion depth issues, we can use a stack or increase limit.
    # Python default recursion limit is 1000. Depth is at most 512. So it's fine.
    
    expand(all_candidates, [])
    
    return sorted(best_clique)
