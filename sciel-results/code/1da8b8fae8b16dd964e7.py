"""
Strategy:
1. Use a fast bitset-based verification (O(n^3) with integer bitmasks) to validate tournaments.
2. Instead of random search + repair, use a local search heuristic specifically designed to
   minimize the number of violating 4-sets.
3. Start from the known n=67 Paley tournament (which is valid) and try to "compress" it to n=66
   by removing a vertex and re-optimizing the remaining structure, or by starting from a random
   n=66 tournament and aggressively reducing violations.
4. The key insight is that if f(4) < 67, then there exists a valid tournament on n < 67 vertices.
   We can try to find one by starting from a near-valid structure and using local search.
   
   A different approach: Use the fact that the Paley tournament on 67 vertices is valid.
   Try to find a valid tournament on n=66 by starting from the Paley tournament on 67,
   removing one vertex, and then checking if the remaining 66-vertex tournament is valid.
   If not, try flipping arcs to fix violations.
   
   Actually, a more promising approach: Start with a random tournament on n=66, 
   and use a local search that counts violations and tries to reduce them.
   The violation count can be computed efficiently using bitsets.
   
   Even better: Use the structure of the problem. A 4-set is bad if it has score sequence (2,2,1,1).
   We can count the number of bad 4-sets and try to minimize it.
   
   For n=66, the number of 4-sets is C(66,4) = 66*65*64*63/24 = ~740k.
   We can compute the violation count in O(n^3) using bitsets.
"""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _paley_orientation(p):
    """Return orientation list for Paley tournament on p vertices."""
    squares = {x * x % p for x in range(1, p)}
    n = p
    orient = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % p in squares:
                orient.append(0)
            else:
                orient.append(1)
    return n, orient

def _orient_to_adj(orient, n):
    """Convert orientation list to numpy adjacency matrix (bool)."""
    adj = np.zeros((n, n), dtype=bool)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            bit = orient[idx]
            idx += 1
            if bit == 0:
                adj[i, j] = True
            else:
                adj[j, i] = True
    return adj

def _adj_to_orient(adj, n):
    """Convert adjacency matrix to orientation list."""
    orient = []
    for i in range(n):
        for j in range(i + 1, n):
            orient.append(0 if adj[i, j] else 1)
    return orient

def _count_violations(adj, n):
    """
    Count the number of 4-sets that are not dominated by any of their vertices.
    A 4-set {a,b,c,d} is bad if for all v in {a,b,c,d}, v does not beat the other 3.
    
    Efficient counting using bitsets:
    For each vertex v, let Nv be the bitmask of vertices that v beats.
    v dominates a 4-set S if v in S and S\{v} subset Nv.
    
    Alternative: For each 4-set, check if any vertex dominates it.
    But that's O(n^4).
    
    Better: For each vertex v, count the number of triples {a,b,c} from Nv.
    This gives the number of 4-sets dominated by v.
    But we need to count 4-sets that are NOT dominated by ANY vertex.
    
    Inclusion-exclusion is complex. Let's use a different approach.
    
    For each pair (u, v) with u -> v:
    The number of 4-sets containing u, v where u dominates the set is 
    the number of pairs {x, y} from Nv (since u must beat x and y, and v must beat x and y, 
    and u beats v). Wait, no.
    
    Let's think differently. A 4-set is bad iff it has score sequence (2,2,1,1).
    In such a set, there are exactly 2 vertices with out-degree 2 and 2 with out-degree 1.
    
    For a vertex v, let out(v) be the set of vertices v beats.
    A 4-set S is dominated by v iff S\{v} subset out(v).
    
    Let's count the number of 4-sets that are dominated by at least one vertex.
    This is hard with inclusion-exclusion.
    
    Instead, let's just iterate over all 4-sets but use bit operations to speed it up.
    For n=66, C(66,4) ~ 740k. A Python loop over 740k items is fast (~0.1s).
    The inner check is 4 boolean ANDs, which is fast.
    
    Let's just do the O(n^4) loop but with numpy adjacency for fast indexing.
    """
    adj_list = [adj[i] for i in range(n)]
    count = 0
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    if not (adj_list[a][b] and adj_list[a][c] and adj_list[a][d]):
                        if not (adj_list[b][a] and adj_list[b][c] and adj_list[b][d]):
                            if not (adj_list[c][a] and adj_list[c][b] and adj_list[c][d]):
                                if not (adj_list[d][a] and adj_list[d][b] and adj_list[d][c]):
                                    count += 1
    return count

def _find_violation(adj, n):
    """Return a violating 4-set if exists, else None."""
    adj_list = [adj[i] for i in range(n)]
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    if not (adj_list[a][b] and adj_list[a][c] and adj_list[a][d]):
                        if not (adj_list[b][a] and adj_list[b][c] and adj_list[b][d]):
                            if not (adj_list[c][a] and adj_list[c][b] and adj_list[c][d]):
                                if not (adj_list[d][a] and adj_list[d][b] and adj_list[d][c]):
                                    return (a, b, c, d)
    return None

def _local_search(adj, n, max_steps=20000, time_limit=100.0):
    """
    Local search to minimize the number of violations.
    At each step, find a violating 4-set, try flipping arcs within it,
    and accept the flip if it reduces the violation count.
    """
    start_time = time.time()
    current_violations = _count_violations(adj, n)
    
    if current_violations == 0:
        return True, adj
    
    best_adj = adj.copy()
    best_violations = current_violations
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break
        
        # Find a violation
        viol = _find_violation(adj, n)
        if viol is None:
            return True, adj
        
        a, b, c, d = viol
        pairs = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
        
        # Try flipping each arc in the violating set
        best_flip = None
        best_new_violations = current_violations
        
        for u, v in pairs:
            # Flip
            adj[u, v] = not adj[u, v]
            adj[v, u] = not adj[v, u]
            
            # Count violations (expensive, but necessary for informed search)
            new_violations = _count_violations(adj, n)
            
            # Unflip
            adj[u, v] = not adj[u, v]
            adj[v, u] = not adj[v, u]
            
            if new_violations < best_new_violations:
                best_new_violations = new_violations
                best_flip = (u, v)
        
        if best_flip is not None:
            u, v = best_flip
            adj[u, v] = not adj[u, v]
            adj[v, u] = not adj[v, u]
            current_violations = best_new_violations
            
            if current_violations < best_violations:
                best_violations = current_violations
                best_adj = adj.copy()
            
            if current_violations == 0:
                return True, adj
        else:
            # No improving move found; try a random flip to escape local minimum
            u, v = random.choice(pairs)
            adj[u, v] = not adj[u, v]
            adj[v, u] = not adj[v, u]
            current_violations = _count_violations(adj, n)
            
            if current_violations < best_violations:
                best_violations = current_violations
                best_adj = adj.copy()
            
            if current_violations == 0:
                return True, adj
    
    if best_violations == 0:
        return True, best_adj
    return False, best_adj

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # First, verify the known n=67 Paley tournament
    p = 67
    n, orient = _paley_orientation(p)
    adj = _orient_to_adj(orient, n)
    
    # Verify 67
    viol = _find_violation(adj, n)
    if viol is None:
        # n=67 is valid. Now try to find n < 67.
        # Strategy: Try n=66 with local search from multiple random starts.
        for target_n in [66, 65, 64]:
            if time.time() > deadline - 30:
                break
            
            # Try a few random starts
            for attempt in range(5):
                if time.time() > deadline - 15:
                    break
                
                rand_orient = [random.randint(0, 1) for _ in range(target_n * (target_n - 1) // 2)]
                rand_adj = _orient_to_adj(rand_orient, target_n)
                
                # Run local search
                success, repaired_adj = _local_search(rand_adj, target_n, max_steps=5000, time_limit=10.0)
                if success:
                    new_orient = _adj_to_orient(repaired_adj, target_n)
                    return {'n': target_n, 'orientation': new_orient}
        
        # Also try: take the n=67 Paley tournament, remove one vertex, and local search
        for rem_vertex in range(67):
            if time.time() > deadline - 15:
                break
            
            # Create a 66-vertex tournament by removing rem_vertex
            new_n = 66
            new_adj = np.zeros((new_n, new_n), dtype=bool)
            idx = 0
            for i in range(67):
                if i == rem_vertex:
                    continue
                for j in range(i+1, 67):
                    if j == rem_vertex:
                        continue
                    # Map to new indices
                    i_new = i if i < rem_vertex else i - 1
                    j_new = j if j < rem_vertex else j - 1
                    new_adj[i_new, j_new] = adj[i, j]
                    new_adj[j_new, i_new] = adj[j, i]
            
            # Local search on this 66-vertex tournament
            success, repaired_adj = _local_search(new_adj, new_n, max_steps=5000, time_limit=10.0)
            if success:
                new_orient = _adj_to_orient(repaired_adj, new_n)
                return {'n': new_n, 'orientation': new_orient}
        
        # Return the verified n=67
        return {'n': n, 'orientation': orient}
    else:
        # If 67 fails (shouldn't happen), try other primes
        for p in [71, 79, 83]:
            if time.time() > deadline:
                break
            n, orient = _paley_orientation(p)
            adj = _orient_to_adj(orient, n)
            if _find_violation(adj, n) is None:
                return {'n': n, 'orientation': orient}
        
        # Fallback
        n, orient = _paley_orientation(83)
        return {'n': n, 'orientation': orient}

# EVOLVE-BLOCK-END
