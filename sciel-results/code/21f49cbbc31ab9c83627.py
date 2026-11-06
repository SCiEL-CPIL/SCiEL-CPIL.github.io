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
    
    def find_k5_or_i5(adj, n):
        """Return True if graph has K5 or independent 5-set."""
        # Check K5: find 4-cliques, then check if any vertex connects to all 4
        # For each 4-tuple, check if it's a clique
        for a in range(n - 4):
            for b in range(a + 1, n - 3):
                if not (adj[a] & (1 << b)):
                    continue
                for c in range(b + 1, n - 2):
                    if not (adj[a] & (1 << c)) or not (adj[b] & (1 << c)):
                        continue
                    for d in range(c + 1, n - 1):
                        if not (adj[a] & (1 << d)) or not (adj[b] & (1 << d)) or not (adj[c] & (1 << d)):
                            continue
                        # Found 4-clique {a,b,c,d}, check if any e connects to all
                        common = adj[a] & adj[b] & adj[c] & adj[d]
                        # Remove a,b,c,d themselves
                        mask = ~(1 << a) & ~(1 << b) & ~(1 << c) & ~(1 << d)
                        if common & mask:
                            return True
        # Check independent 5-set: find 4-independent-sets, then check if any vertex is non-adjacent to all 4
        for a in range(n - 4):
            for b in range(a + 1, n - 3):
                if adj[a] & (1 << b):
                    continue
                for c in range(b + 1, n - 2):
                    if (adj[a] & (1 << c)) or (adj[b] & (1 << c)):
                        continue
                    for d in range(c + 1, n - 1):
                        if (adj[a] & (1 << d)) or (adj[b] & (1 << d)) or (adj[c] & (1 << d)):
                            continue
                        # Found 4-independent-set {a,b,c,d}, check if any e is non-adjacent to all
                        non_common = (~adj[a]) & (~adj[b]) & (~adj[c]) & (~adj[d])
                        mask = ~(1 << a) & ~(1 << b) & ~(1 << c) & ~(1 << d)
                        if non_common & mask:
                            return True
        return False
    
    def is_55_free(adj, n):
        return not find_k5_or_i5(adj, n)
    
    def get_4cliques(adj, n):
        """Return list of 4-cliques as tuples."""
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
                        cliques.append((a, b, c, d))
        return cliques
    
    def get_4indeps(adj, n):
        """Return list of 4-independent-sets as tuples."""
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
                        indeps.append((a, b, c, d))
        return indeps
    
    def try_add_vertex(adj, n, cliques4, indeps4, deadline):
        """Try to add a vertex. Returns (new_adj, new_n, new_cliques4, new_indeps4) or None."""
        if not cliques4 and not indeps4:
            # Any subset works
            s = random.sample(range(n), n // 2)
            new_adj = adj + [0]
            for v in s:
                new_adj[n] |= (1 << v)
                new_adj[v] |= (1 << n)
            # Recompute cliques and indeps for new graph
            new_cliques = get_4cliques(new_adj, n + 1)
            new_indeps = get_4indeps(new_adj, n + 1)
            return new_adj, n + 1, new_cliques, new_indeps
        
        # Try random subsets
        for attempt in range(10000):
            if time.time() > deadline:
                break
            size = random.randint(max(1, n // 4), min(n - 1, 3 * n // 4))
            s = random.sample(range(n), size)
            s_set = set(s)
            
            valid = True
            for quad in cliques4:
                if all(v in s_set for v in quad):
                    valid = False
                    break
            if valid:
                for quad in indeps4:
                    if all(v in s_set for v in quad):
                        valid = False
                        break
            
            if valid:
                new_adj = adj + [0]
                for v in s:
                    new_adj[n] |= (1 << v)
                    new_adj[v] |= (1 << n)
                
                # Verify the new graph is 55-free
                if is_55_free(new_adj, n + 1):
                    new_cliques = get_4cliques(new_adj, n + 1)
                    new_indeps = get_4indeps(new_adj, n + 1)
                    return new_adj, n + 1, new_cliques, new_indeps
        return None
    
    # Start with Paley 37
    adj = make_paley(37)
    n = 37
    best_adj = adj
    best_n = n
    
    if not is_55_free(adj, n):
        # Should not happen, but just in case
        return n, [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i] & (1 << j)]
    
    cliques4 = get_4cliques(adj, n)
    indeps4 = get_4indeps(adj, n)
    
    # Try to add vertices
    for iteration in range(30):
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
    for restart in range(10):
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
        
        if not is_55_free(perturbed, pn):
            continue
        
        p_cliques = get_4cliques(perturbed, pn)
        p_indeps = get_4indeps(perturbed, pn)
        
        for iteration in range(20):
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
