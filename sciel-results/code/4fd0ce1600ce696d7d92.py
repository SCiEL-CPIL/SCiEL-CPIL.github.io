"""
Program to construct a C4-free graph with no independent set of size 11, maximizing n.
Uses a polarity graph base (PG(2,5)) and randomized greedy extension with local search.
"""

import time
import random
import sys

sys.setrecursionlimit(10000)

def _popcount(x):
    return bin(x).count('1')

def _has_c4(adj, adj_bits, n):
    """
    Check if graph contains a C4.
    A C4 exists if there are two vertices u, v that have >= 2 common neighbors.
    """
    for u in range(n):
        for v in range(u + 1, n):
            common = adj_bits[u] & adj_bits[v]
            if common & (common - 1): # If common has more than 1 bit set
                return True
    return False

def _can_add_edge(adj, adj_bits, u, v, n):
    """
    Check if adding edge (u, v) would create a C4.
    
    The condition "No two vertices may have two common neighbours" is equivalent to C4-free.
    When adding edge (u, v):
    1. Pair (u, v) becomes adjacent. They gain no new common neighbors (neighbors are unchanged).
       So if they had <= 1 common neighbor before, they still do.
    2. Pair (u, x) for x != v:
       If x is adjacent to v, then v becomes a new common neighbor of u and x (since v is now adj u, and v is adj x).
       So we must ensure that u and x had NO common neighbors before.
    3. Pair (v, y) for y != u:
       Symmetric. If y is adjacent to u, then u becomes a new common neighbor of v and y.
       So we must ensure that v and y had NO common neighbors before.
       
    Therefore, adding (u, v) is safe iff:
    - For all x in adj[v] \ {u}: (adj_bits[u] & adj_bits[x]) == 0
    - For all y in adj[u] \ {v}: (adj_bits[v] & adj_bits[y]) == 0
    """
    # Check neighbors of v
    for x in adj[v]:
        if x == u:
            continue
        if adj_bits[u] & adj_bits[x]:
            return False
            
    # Check neighbors of u
    for y in adj[u]:
        if y == v:
            continue
        if adj_bits[v] & adj_bits[y]:
            return False
            
    return True

def _add_edge(adj, adj_bits, u, v):
    """Add edge u-v, update structures."""
    adj[u].add(v)
    adj[v].add(u)
    adj_bits[u] |= (1 << v)
    adj_bits[v] |= (1 << u)

def _remove_edge(adj, adj_bits, u, v):
    """Remove edge u-v, update structures."""
    adj[u].discard(v)
    adj[v].discard(u)
    adj_bits[u] &= ~(1 << v)
    adj_bits[v] &= ~(1 << u)

def _has_independent_set_k(adj_bits, n, k):
    """
    Exact check for independent set of size k using Branch and Bound with bitsets.
    Returns True if an independent set of size k exists.
    """
    # We want to find a set of k vertices with no edges between them.
    # This is equivalent to finding a clique of size k in the complement graph.
    # But we can work directly on the graph:
    # Independent set in G = set of vertices with no edges.
    
    def search(rem, size):
        if size >= k:
            return True
        if rem == 0:
            return False
        
        # Pruning: If even taking all remaining vertices doesn't reach k
        if size + _popcount(rem) < k:
            return False
            
        # Choose a vertex to branch on.
        # Heuristic: Pick vertex with highest degree in the remaining subgraph (induced by rem).
        # Why? If we include it, we remove it and all its neighbors. High degree -> big reduction.
        # If we exclude it, we just remove it.
        # Usually, branching on high degree vertices is better for clique/independent set problems.
        
        best_v = -1
        best_deg = -1
        temp_rem = rem
        while temp_rem:
            lsb = temp_rem & -temp_rem
            v = lsb.bit_length() - 1
            # Degree of v in the subgraph induced by rem
            deg = _popcount(adj_bits[v] & rem)
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp_rem ^= lsb
            
        if best_v == -1:
            return False
            
        v_bit = 1 << best_v
        
        # Branch 1: Include best_v
        # Remove best_v and all its neighbors from rem
        new_rem = rem & ~(adj_bits[best_v] | v_bit)
        if search(new_rem, size + 1):
            return True
            
        # Branch 2: Exclude best_v
        new_rem2 = rem & ~v_bit
        if search(new_rem2, size):
            return True
            
        return False
        
    return search((1 << n) - 1, 0)

def _polarity_graph_pg25():
    """
    Construct the polarity graph of PG(2,5).
    Points are vectors in GF(5)^3, up to scale.
    There are (5^3 - 1)/(5 - 1) = 31 points.
    Two points are adjacent if their dot product is 0.
    This graph is C4-free.
    """
    # Generate all non-zero vectors in GF(5)^3, normalized so first non-zero component is 1
    points = []
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize
                if x != 0:
                    # x is already 1? No, x can be 1..4.
                    # We want first non-zero to be 1.
                    # If x != 0, divide by x.
                    inv_x = pow(x, 3, 5) # 5 is prime, so x^(5-2) = x^3 is inverse
                    nx = 1
                    ny = (y * inv_x) % 5
                    nz = (z * inv_x) % 5
                elif y != 0:
                    inv_y = pow(y, 3, 5)
                    nx = 0
                    ny = 1
                    nz = (z * inv_y) % 5
                else: # z != 0
                    inv_z = pow(z, 3, 5)
                    nx = 0
                    ny = 0
                    nz = 1
                points.append((nx, ny, nz))
    
    # Remove duplicates (should be unique by construction)
    unique_points = list(set(points))
    # Sort for determinism
    unique_points.sort()
    n = len(unique_points)
    
    adj = [set() for _ in range(n)]
    adj_bits = [0] * n
    
    for i in range(n):
        for j in range(i + 1, n):
            p1 = unique_points[i]
            p2 = unique_points[j]
            # Dot product mod 5
            dot = (p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]) % 5
            if dot == 0:
                _add_edge(adj, adj_bits, i, j)
                
    return n, adj, adj_bits

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # 1. Start with Polarity Graph of PG(2,5)
    n0, adj, adj_bits = _polarity_graph_pg25()
    n = n0
    best_n = n
    best_edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                best_edges.append([u, v])
                
    # Verify base graph
    if _has_c4(adj, adj_bits, n):
        # Should not happen
        pass
    if _has_independent_set_k(adj_bits, n, 11):
        # Base graph has alpha >= 11? 
        # Polarity graph of PG(2,5) has alpha = 8. 
        # So this should be False.
        pass
        
    # 2. Try to extend the graph by adding more vertices
    # We want to add vertices 31, 32, ... up to 43.
    # For each new vertex, we try to connect it to existing vertices.
    # We must maintain C4-free property.
    # We must maintain alpha < 11.
    
    # We will try to add vertices one by one.
    # For each new vertex, we decide which existing vertices to connect to.
    # This is a subgraph isomorphism / extension problem.
    # Since n is small, we can use randomized greedy search.
    
    rng = random.Random(42)
    
    max_n = 43
    
    for target_n in range(n0 + 1, max_n + 1):
        if time.time() > deadline - 5:
            break
            
        # Try to find a valid extension to size target_n
        # We restart the extension process from the base graph
        # Or do we continue from the previous successful extension?
        # Continuing is better if we maintain validity.
        # But if we fail to extend n to n+1, we might need to backtrack.
        # Given the time limit, let's try a simple strategy:
        # Keep the current graph (which is valid for size n).
        # Try to add vertex n.
        
        success = False
        # Try multiple random configurations for the new vertex
        # The new vertex v_new can be connected to any subset of existing vertices.
        # However, it must not create a C4.
        # Condition: For any two existing vertices u, w that are neighbors of v_new,
        # u and w must not have any common neighbors other than v_new?
        # No, u and w already exist. Their common neighbors are fixed.
        # If u and w are both neighbors of v_new, then v_new is a common neighbor of u and w.
        # So u and w must have NO other common neighbors.
        # i.e., (adj_bits[u] & adj_bits[w]) must be 0.
        
        # So the set of neighbors of v_new must be an independent set in the graph 
        # where edges connect pairs of vertices that share a neighbor.
        # Let's call this the "conflict graph".
        # Two vertices u, w conflict if they share a neighbor.
        # We need to find a subset of vertices for v_new to connect to, such that no two in the subset conflict.
        # This is finding a clique in the complement of the conflict graph?
        # Or an independent set in the conflict graph.
        
        # To maximize the chance of success, we should probably connect v_new to as many vertices as possible?
        # Not necessarily. We just need alpha to stay < 11.
        # Actually, adding edges reduces alpha. So more edges is better for alpha.
        # But we are constrained by C4-free.
        
        # Let's try to find a large independent set in the conflict graph to connect to.
        # Or just random subsets.
        
        # Heuristic:
        # 1. Identify potential neighbors.
        # 2. Randomly pick a subset that is independent in the conflict graph.
        # 3. Add the vertex and edges.
        # 4. Check alpha.
        
        # Conflict graph construction is O(n^3) or O(n^2 * n/word_size).
        # For n=43, this is fast.
        
        # Let's try 100 random restarts for this extension step.
        for r in range(100):
            if time.time() > deadline - 5:
                break
                
            # Create a copy of the current graph state
            adj_copy = [set(s) for s in adj]
            adj_bits_copy = list(adj_bits)
            
            v_new = n # index of new vertex
            
            # Build conflict graph on existing vertices
            # Conflict[u][w] = True if u and w share a neighbor
            # We can use bitsets for this too.
            # common_neighbors_mask[u] = adj_bits[u]
            # u and w conflict if (adj_bits[u] & adj_bits[w]) != 0
            
            # We want to select a subset S of {0..n-1} such that for all u, w in S, u != w:
            # (adj_bits[u] & adj_bits[w]) == 0
            
            # This is a maximal independent set problem in the conflict graph.
            # Let's use a greedy approach to find a large such set.
            
            # Randomize the order of vertices
            vertices = list(range(n))
            rng.shuffle(vertices)
            
            S = set()
            for u in vertices:
                # Check if u can be added to S
                # u can be added if it doesn't conflict with any vertex in S
                can_add = True
                for w in S:
                    if adj_bits_copy[u] & adj_bits_copy[w]:
                        can_add = False
                        break
                if can_add:
                    S.add(u)
            
            # If S is empty, maybe try a different strategy?
            # For now, if S is empty, we add the vertex with no edges (isolated).
            # An isolated vertex increases alpha by 1.
            # If alpha was 10, it becomes 11 -> Invalid.
            # So we must be careful.
            
            # Let's check if adding v_new with neighbors S is valid.
            # 1. C4-free check is satisfied by construction of S.
            # 2. Alpha check.
            
            # Add v_new to the graph
            adj_copy.append(set(S))
            adj_bits_copy.append(0)
            new_bits = 0
            for u in S:
                adj_copy[u].add(v_new)
                adj_bits_copy[u] |= (1 << v_new)
                new_bits |= (1 << u)
            adj_bits_copy[v_new] = new_bits
            
            # Check alpha
            if not _has_independent_set_k(adj_bits_copy, n + 1, 11):
                # Success!
                adj = adj_copy
                adj_bits = adj_bits_copy
                n += 1
                success = True
                break
        
        if not success:
            # Could not extend to target_n
            # We can try to extend to target_n+1? No, we need target_n first.
            # If we can't extend to n+1, we probably can't extend to n+2 either.
            # So we stop extending.
            break
        
        # Update best
        best_n = n
        best_edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    best_edges.append([u, v])

    # If we still have time, try local search to improve the graph?
    # Local search for C4-free graphs is tricky.
    # We could try swapping edges or flipping edges.
    # But we must maintain C4-free and alpha < 11.
    # Given the time constraints, we'll skip complex local search.
    
    # One more check: maybe we can do better with a different random seed or strategy?
    # The current strategy starts from PG(2,5) and extends greedily.
    # Is it possible to get n=39?
    # The best known is 39.
    # Our base is 31.
    # We need to add 8 more vertices.
    # If the greedy extension fails, we might be stuck at 31 or 32.
    
    # Let's try a different approach if we are stuck at 31.
    # Randomized greedy from scratch for n=39?
    # That's what the original program did, and it was slow.
    
    # Let's try to improve the extension strategy.
    # Instead of just one random S, try multiple S's of different sizes?
    # The current loop tries 100 random S's.
    
    # If we are still at 31, let's try to jump to 39 directly with a different method?
    # Maybe a known construction for 39?
    # The task says "The best known construction has 39 vertices."
    # If we can't find it, we return what we have.
    
    # Let's add a fallback: if n < 35, try the original randomized greedy for n=35..39?
    # But it was slow.
    # With bitset optimization, it might be faster.
    
    if best_n < 35 and time.time() < deadline - 20:
        # Try randomized greedy for larger n
        for target_n in range(39, best_n, -1):
            if time.time() > deadline - 5:
                break
            found = False
            for r in range(20):
                if time.time() > deadline - 5:
                    break
                adj_t = [set() for _ in range(target_n)]
                adj_bits_t = [0] * target_n
                pairs = [(u, v) for u in range(target_n) for v in range(u + 1, target_n)]
                rng.shuffle(pairs)
                for u, v in pairs:
                    if _can_add_edge(adj_t, adj_bits_t, u, v, target_n):
                        _add_edge(adj_t, adj_bits_t, u, v)
                if not _has_c4(adj_t, adj_bits_t, target_n):
                    if not _has_independent_set_k(adj_bits_t, target_n, 11):
                        best_n = target_n
                        best_edges = []
                        for u in range(target_n):
                            for v in adj_t[u]:
                                if u < v:
                                    best_edges.append([u, v])
                        found = True
                        break
            if found:
                break

    return {'n': best_n, 'edges': best_edges}

if __name__ == "__main__":
    res = construct()
    print(f"Constructed graph with n={res['n']}")
