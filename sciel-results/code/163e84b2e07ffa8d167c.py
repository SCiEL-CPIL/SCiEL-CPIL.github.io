"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 285  # safety margin before 330s
    
    def make_paley(q):
        residues = {(x * x) % q for x in range(1, q)}
        adj = [0] * q
        for i in range(q):
            for j in range(i + 1, q):
                if (j - i) % q in residues:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj
    
    def get_4cliques(adj, n):
        """Return list of 4-cliques as bitmasks (each is a bitmask of 4 vertices)."""
        cliques = []
        for a in range(n - 3):
            for b in range(a + 1, n - 2):
                if not (adj[a] & (1 << b)):
                    continue
                for c in range(b + 1, n - 1):
                    if not (adj[a] & (1 << c)) or not (adj[b] & (1 << c)):
                        continue
                    for d in range(c + 1, n):
                        if not (adj[a] & (1 << d)) or not (adj[b] & (1 << d)) or not (adj[c] & (1 << d)):
                            continue
                        cliques.append((1 << a) | (1 << b) | (1 << c) | (1 << d))
        return cliques
    
    def get_4indeps(adj, n):
        """Return list of 4-independent-sets as bitmasks."""
        indeps = []
        for a in range(n - 3):
            for b in range(a + 1, n - 2):
                if adj[a] & (1 << b):
                    continue
                for c in range(b + 1, n - 1):
                    if (adj[a] & (1 << c)) or (adj[b] & (1 << c)):
                        continue
                    for d in range(c + 1, n):
                        if (adj[a] & (1 << d)) or (adj[b] & (1 << d)) or (adj[c] & (1 << d)):
                            continue
                        indeps.append((1 << a) | (1 << b) | (1 << c) | (1 << d))
        return indeps
    
    def is_55_free(adj, n, cliques4, indeps4):
        """Check if graph has no K5 or I5. Uses precomputed 4-cliques and 4-indeps."""
        # K5 exists if any 4-clique has a common neighbor outside itself
        for mask in cliques4:
            # Find common neighbors of all 4 vertices in the clique
            a = mask.bit_length() - 1
            b = (mask & ~(1 << a)).bit_length() - 1
            c = (mask & ~(1 << a) & ~(1 << b)).bit_length() - 1
            d = (mask & ~(1 << a) & ~(1 << b) & ~(1 << c)).bit_length() - 1
            common = adj[a] & adj[b] & adj[c] & adj[d]
            if common & ~mask:
                return False
        # I5 exists if any 4-independent-set has a common non-neighbor outside itself
        full_mask = (1 << n) - 1
        for mask in indeps4:
            a = mask.bit_length() - 1
            b = (mask & ~(1 << a)).bit_length() - 1
            c = (mask & ~(1 << a) & ~(1 << b)).bit_length() - 1
            d = (mask & ~(1 << a) & ~(1 << b) & ~(1 << c)).bit_length() - 1
            non_common = (~adj[a]) & (~adj[b]) & (~adj[c]) & (~adj[d]) & full_mask
            if non_common & ~mask:
                return False
        return True
    
    def check_candidate(adj, n, s_mask, cliques4, indeps4):
        """Check if adding a vertex connected to s_mask creates K5 or I5.
        K5 forms iff there's a 4-clique entirely in s_mask.
        I5 forms iff there's a 4-independent-set entirely in ~s_mask.
        """
        full_mask = (1 << n) - 1
        comp_mask = full_mask & ~s_mask
        # Check if any 4-clique is subset of s_mask
        for mask in cliques4:
            if mask & ~s_mask == 0:  # mask is subset of s_mask
                return False
        # Check if any 4-independent-set is subset of comp_mask
        for mask in indeps4:
            if mask & ~comp_mask == 0:  # mask is subset of comp_mask
                return False
        return True
    
    def try_add_vertex(adj, n, cliques4, indeps4, deadline):
        """Try to add a vertex. Returns (new_adj, new_n, new_cliques4, new_indeps4) or None."""
        full_mask = (1 << n) - 1
        
        # Strategy 1: Vertex duplication - copy neighborhood of existing vertex
        for v in range(n):
            if time.time() > deadline:
                break
            s_mask = adj[v]  # neighborhood of v
            if check_candidate(adj, n, s_mask, cliques4, indeps4):
                new_adj = adj + [0]
                new_adj[n] = s_mask
                for u in range(n):
                    if s_mask & (1 << u):
                        new_adj[u] |= (1 << n)
                new_cliques = get_4cliques(new_adj, n + 1)
                new_indeps = get_4indeps(new_adj, n + 1)
                return new_adj, n + 1, new_cliques, new_indeps
        
        # Strategy 2: Try subsets of varying sizes
        # Prefer medium-sized subsets
        for size_frac in [0.3, 0.5, 0.7, 0.4, 0.6, 0.5, 0.5, 0.3, 0.7, 0.5]:
            if time.time() > deadline:
                break
            target_size = int(n * size_frac)
            target_size = max(1, min(n - 1, target_size))
            
            for attempt in range(2000):
                if time.time() > deadline:
                    break
                # Random subset of target size
                s = random.sample(range(n), target_size)
                s_mask = 0
                for v in s:
                    s_mask |= (1 << v)
                
                if check_candidate(adj, n, s_mask, cliques4, indeps4):
                    new_adj = adj + [0]
                    new_adj[n] = s_mask
                    for u in range(n):
                        if s_mask & (1 << u):
                            new_adj[u] |= (1 << n)
                    new_cliques = get_4cliques(new_adj, n + 1)
                    new_indeps = get_4indeps(new_adj, n + 1)
                    return new_adj, n + 1, new_cliques, new_indeps
        
        # Strategy 3: Greedy - start with a small set and grow
        if time.time() < deadline - 5:
            # Start with a random vertex's neighborhood, then try to add more
            base_v = random.randint(0, n - 1)
            s_mask = adj[base_v]
            
            # Try to add vertices one at a time
            remaining = [v for v in range(n) if not (s_mask & (1 << v))]
            random.shuffle(remaining)
            
            for v in remaining:
                if time.time() > deadline:
                    break
                new_s_mask = s_mask | (1 << v)
                if check_candidate(adj, n, new_s_mask, cliques4, indeps4):
                    s_mask = new_s_mask
            
            if check_candidate(adj, n, s_mask, cliques4, indeps4):
                new_adj = adj + [0]
                new_adj[n] = s_mask
                for u in range(n):
                    if s_mask & (1 << u):
                        new_adj[u] |= (1 << n)
                new_cliques = get_4cliques(new_adj, n + 1)
                new_indeps = get_4indeps(new_adj, n + 1)
                return new_adj, n + 1, new_cliques, new_indeps
        
        return None
    
    # Start with Paley 37
    adj = make_paley(37)
    n = 37
    best_adj = adj
    best_n = n
    
    cliques4 = get_4cliques(adj, n)
    indeps4 = get_4indeps(adj, n)
    
    if not is_55_free(adj, n, cliques4, indeps4):
        # Should not happen, but just in case
        return n, [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i] & (1 << j)]
    
    # Try to add vertices
    for iteration in range(50):
        if time.time() > deadline:
            break
        result = try_add_vertex(adj, n, cliques4, indeps4, deadline)
        if result is None:
            break
        adj, n, cliques4, indeps4 = result
        if n > best_n:
            best_adj = adj
            best_n = n
    
    # Try random restarts with perturbations
    random.seed(12345)
    for restart in range(20):
        if time.time() > deadline:
            break
        perturbed = list(make_paley(37))
        pn = 37
        # Remove a few random edges
        for _ in range(random.randint(1, 5)):
            a = random.randint(0, pn - 1)
            b = random.randint(0, pn - 1)
            if a != b and (perturbed[a] & (1 << b)):
                perturbed[a] &= ~(1 << b)
                perturbed[b] &= ~(1 << a)
        
        p_cliques = get_4cliques(perturbed, pn)
        p_indeps = get_4indeps(perturbed, pn)
        
        if not is_55_free(perturbed, pn, p_cliques, p_indeps):
            continue
        
        for iteration in range(30):
            if time.time() > deadline:
                break
            result = try_add_vertex(perturbed, pn, p_cliques, p_indeps, deadline)
            if result is None:
                break
            perturbed, pn, p_cliques, p_indeps = result
            if pn > best_n:
                best_adj = perturbed
                best_n = pn
    
    # Convert to edge list
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i] & (1 << j):
                edges.append((i, j))
    
    return best_n, edges

# EVOLVE-BLOCK-END
