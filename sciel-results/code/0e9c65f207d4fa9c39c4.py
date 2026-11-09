"""
Program to construct a C4-free graph with no independent set of size 11, maximizing n.
Uses randomized greedy construction and local search with bitset optimization.
"""

import time
import random
import sys

# Set recursion limit if needed, though we'll use iterative approaches where possible
sys.setrecursionlimit(10000)

def _popcount(x):
    return bin(x).count('1')

def _has_independent_set_fast(adj_bits, n, k):
    """
    Fast check for independent set of size k using bitsets and Branch and Bound.
    """
    def search(rem, size):
        if size >= k:
            return True
        if rem == 0:
            return False
        
        # Pruning: If even taking all remaining vertices doesn't reach k
        if size + _popcount(rem) < k:
            return False
            
        # Choose a vertex to branch on.
        # Heuristic: Pick vertex with highest degree in the remaining subgraph.
        # This maximizes the reduction in search space for the "Include" branch.
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

def _create_adjacency(n):
    """Create adjacency lists and bitsets."""
    adj = [set() for _ in range(n)]
    # Bitsets for fast intersection
    adj_bits = [0] * n
    return adj, adj_bits

def _add_edge(adj, adj_bits, u, v):
    """Add edge u-v, update structures."""
    adj[u].add(v)
    adj[v].add(u)
    adj_bits[u] |= (1 << v)
    adj_bits[v] |= (1 << u)

def _has_c4(adj, adj_bits, n):
    """
    Check if graph contains a C4.
    A C4 exists if there are two vertices u, v (non-adjacent) that have >= 2 common neighbors.
    We check this during construction or verification.
    """
    # This is O(n^2 * (n/word_size)) which is fast for n <= 43
    for u in range(n):
        for v in range(u + 1, n):
            if (adj_bits[u] >> v) & 1:
                continue # Adjacent vertices don't form opposite corners of C4 directly in this check logic
            common = adj_bits[u] & adj_bits[v]
            if common & (common - 1): # If common has more than 1 bit set
                return True
    return False

def _count_common_neighbors(adj_bits, u, v):
    """Count common neighbors of u and v."""
    common = adj_bits[u] & adj_bits[v]
    return bin(common).count('1')

def _can_add_edge(adj, adj_bits, u, v, n):
    """
    Check if adding edge (u, v) would create a C4.
    Adding (u, v) creates a C4 if there exists a path u-w-z-v of length 3 that is not using u-v.
    Actually, simpler: u and v already share some neighbors. If they share >= 1 neighbor w, 
    adding edge (u, v) creates a triangle, not a C4 immediately. 
    Wait, the definition of C4-free is: no two vertices have 2 or more common neighbors.
    If we add edge (u, v), it doesn't change the common neighbors of other pairs? 
    No, it changes the adjacency. 
    The condition "No two vertices may have two common neighbours" is the C4-free condition.
    
    If we add edge (u, v):
    1. u and v become adjacent.
    2. For any other pair (x, y), their common neighbors are unchanged UNLESS x=u or y=v etc.
    3. Specifically, we must ensure that after adding (u, v), no pair has >= 2 common neighbors.
    
    Let's check which pairs might violate the condition after adding (u, v):
    - Pair (u, v): They are now adjacent. The condition applies to non-adjacent vertices? 
      The task says: "No two vertices may have two common neighbours (no C4)".
      In graph theory, a C4 is a cycle of length 4. A cycle u-w-z-v-u requires u and v to be non-adjacent? 
      No, u and v are opposite in the cycle u-w-z-v. So u and v are NOT adjacent in the cycle.
      If u and v ARE adjacent, they can't be opposite in a C4. 
      However, if u and v are adjacent, can they have 2 common neighbors? 
      If u and v are adjacent and share neighbors w, z, then u-w-z-v-u is a C4? 
      Yes, if w != z. 
      So, even for adjacent vertices, having 2 common neighbors creates a C4 (u-w-z-v-u).
      Wait, the cycle is u-w-z-v-u. The edges are (u,w), (w,z), (z,v), (v,u).
      So yes, any pair of vertices (adjacent or not) must have at most 1 common neighbor.
      
    Therefore, adding edge (u, v) is safe if and only if:
    1. u and v do not already have 2 common neighbors. (If they have 2, a C4 already exists or will be reinforced? 
       Actually, if they have 2 common neighbors, the graph ALREADY has a C4 (u-w-z-v-u is not a cycle if u-v is not edge, 
       but u-w-z-v is a path. If u-v is not edge, u-w-z-v-u is not a cycle. 
       Wait. C4 is a cycle. 
       If u, v are non-adjacent and have 2 common neighbors w, z, then u-w-z-v-u is a C4? 
       Edges: (u,w), (w,z), (z,v), (v,u)? No, (v,u) is not an edge.
       The cycle would be u-w-z-v... where is the 4th edge? 
       The cycle is u-w-z-v-u. This requires edge (v,u). 
       So if u and v are NON-adjacent, and have 2 common neighbors, do they form a C4? 
       No. The cycle is u-w-z-v. That's a path of length 3. 
       To close the cycle, we need an edge between u and v. 
       So, if u and v are NON-adjacent and have 2 common neighbors, we do NOT necessarily have a C4.
       Example: Star graph K1,3. Center c, leaves l1, l2, l3. 
       l1 and l2 have 1 common neighbor (c). 
       l1 and l3 have 1 common neighbor (c).
       No C4.
       
       Let's re-read the standard definition. 
       A graph is C4-free if it contains no cycle of length 4.
       A cycle of length 4 is a sequence of distinct vertices v1, v2, v3, v4 such that 
       (v1,v2), (v2,v3), (v3,v4), (v4,v1) are edges.
       
       Condition: "No two vertices may have two common neighbours".
       If u and v have common neighbors w and z.
       If u and v are adjacent: Cycle u-w-z-v-u. Edges (u,w), (w,z), (z,v), (v,u). This is a C4.
       If u and v are NOT adjacent: The path u-w-z-v exists. To form a C4, we need a 4th edge. 
       Is there a theorem? 
       "A graph is C4-free if and only if every pair of vertices has at most one common neighbor."
       Let's check.
       If every pair has at most 1 common neighbor:
       Suppose there is a C4: v1-v2-v3-v4-v1.
       v1 and v3 have common neighbors v2 and v4.
       So v1 and v3 have 2 common neighbors.
       This contradicts the condition.
       So YES, the condition "No two vertices may have two common neighbours" is equivalent to C4-free.
       
       Therefore, when adding edge (u, v):
       We must ensure that NO pair of vertices has 2 or more common neighbors.
       
       Which pairs are affected by adding (u, v)?
       Only pairs that include u or v.
       
       1. Pair (u, v):
          Before adding, they had k common neighbors.
          After adding, they are adjacent.
          They still have the same k common neighbors.
          If k >= 2, then (u, v) already violated the condition? 
          No, if they are non-adjacent and have k=2 common neighbors, does that form a C4?
          Let's trace: u-w1-z-v-u? No, u-v is not edge.
          The cycle would be u-w1-v-w2-u?
          If u and v have common neighbors w1 and w2.
          Edges: (u,w1), (w1,v), (v,w2), (w2,u).
          This is a cycle u-w1-v-w2-u.
          Yes! This is a C4.
          So, if u and v (non-adjacent) have 2 common neighbors, a C4 ALREADY EXISTS.
          
          So, if the graph is currently C4-free, then u and v (non-adjacent) have at most 1 common neighbor.
          When we add edge (u, v), they become adjacent.
          Do they still have at most 1 common neighbor? Yes, the common neighbors didn't change.
          So pair (u, v) is safe.
       
       2. Pair (u, x) where x != v:
          Before: Common neighbors of u and x.
          After: v is now a neighbor of u. Is v a neighbor of x?
          If v is a neighbor of x, then v becomes a new common neighbor of u and x.
          So we must check: If v is in adj[x], then u and x gain v as a common neighbor.
          If u and x ALREADY had 1 common neighbor, adding v makes it 2 -> C4 created.
          If u and x ALREADY had 0 common neighbors, adding v makes it 1 -> OK.
          If u and x ALREADY had >= 1 common neighbor, we need to count them.
          
          So, for each neighbor x of v (x != u), we must check if u and x have any common neighbors other than v.
          Actually, simpler: 
          Count common neighbors of u and x BEFORE adding (u,v).
          Let c = common_count(u, x).
          If x is adjacent to v, then after adding (u,v), v becomes a common neighbor of u and x (since v is adj u and v is adj x).
          So new common count = c + 1.
          We need new common count <= 1.
          So we need c <= 0.
          So if x is adjacent to v, u and x must have NO common neighbors.
          
       3. Pair (v, y) where y != u:
          Symmetric to above.
          For each neighbor y of u (y != v), if y is adjacent to v, then u and v... wait.
          Pair (v, y). v is adj to y? 
          We are adding (u, v).
          v's neighbors are unchanged. u's neighbors gain v.
          So common neighbors of v and y change only if y is adjacent to u.
          If y is adjacent to u, then u becomes a new common neighbor of v and y.
          So for each neighbor y of u (y != v), if y is adjacent to v, then v and y gain u as a common neighbor.
          We need common_count(v, y) + 1 <= 1 => common_count(v, y) == 0.
          
       4. Pair (x, y) where x != u, v and y != u, v:
          Unchanged.
       
       So, adding edge (u, v) is safe if and only if:
       - For every x in adj[v] \ {u}: common_count(u, x) == 0
       - For every y in adj[u] \ {v}: common_count(v, y) == 0
       
       Note: common_count(u, x) counts neighbors of u that are also neighbors of x.
       
    """
    # Check neighbors of v
    for x in adj[v]:
        if x == u:
            continue
        # Check if u and x have any common neighbors
        # common neighbors of u and x are (adj_bits[u] & adj_bits[x])
        if adj_bits[u] & adj_bits[x]:
            return False
            
    # Check neighbors of u
    for y in adj[u]:
        if y == v:
            continue
        # Check if v and y have any common neighbors
        # common neighbors of v and y are (adj_bits[v] & adj_bits[y])
        if adj_bits[v] & adj_bits[y]:
            return False
            
    return True

def _independent_set_greedy(adj, n, limit=11):
    """
    Try to find an independent set of size 'limit'.
    Returns the set if found, else None.
    Uses a greedy approach with backtracking if needed, but for n<=43, a simple 
    branch and bound is fast enough.
    """
    # Greedy first: pick vertex with min degree
    remaining = set(range(n))
    indep = set()
    while remaining:
        # Pick vertex with fewest neighbors in remaining
        best_v = min(remaining, key=lambda v: len(adj[v] & remaining))
        indep.add(best_v)
        remaining -= adj[best_v]
        remaining.discard(best_v)
        if len(indep) >= limit:
            return indep
            
    if len(indep) >= limit:
        return indep
        
    return None

def _has_independent_set_k(adj, n, k):
    """
    Exact check for independent set of size k using Branch and Bound.
    """
    # Pre-sort vertices by degree (heuristic)
    # We'll use a simple recursive search
    best_found = [False]
    
    def search(remaining, chosen):
        if len(chosen) == k:
            best_found[0] = True
            return True
        if not remaining:
            return False
        if len(chosen) + len(remaining) < k:
            return False
            
        # Pick a vertex to branch on
        # Heuristic: pick vertex with highest degree in remaining subgraph? 
        # Or just first one?
        # Let's pick the one with highest degree to maximize pruning
        v = max(remaining, key=lambda x: len(adj[x] & remaining))
        
        # Option 1: Include v
        new_remaining = remaining - adj[v]
        new_remaining.discard(v)
        chosen.add(v)
        if search(new_remaining, chosen):
            return True
        chosen.discard(v)
        
        # Option 2: Exclude v
        remaining2 = set(remaining)
        remaining2.discard(v)
        if search(remaining2, chosen):
            return True
            
        return False

    return search(set(range(n)), set())

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    best_n = 30
    best_edges = []
    
    # Initial fallback: 10 disjoint triangles (n=30)
    # Triangles are C4-free. Alpha = 10 (one from each triangle).
    # We need Alpha < 11. So n=30 is valid.
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        best_edges += [[a, b], [a, c], [b, c]]
        
    rng = random.Random(42)
    
    # Try to find larger graphs
    # We iterate n from 40 down to 31
    for n in range(40, 30, -1):
        if time.time() > deadline - 10:
            break
            
        # Try multiple random restarts
        max_restarts = 20
        found = False
        
        for r in range(max_restarts):
            if time.time() > deadline - 10:
                break
                
            adj = [set() for _ in range(n)]
            adj_bits = [0] * n
            
            # Phase 1: Greedy random edge addition
            # We want to add edges that are "safe" (don't create C4)
            # To maximize density, we should prioritize edges that connect low-degree vertices?
            # Or just random?
            # Let's shuffle vertex pairs and try to add them.
            
            # To make it faster, we can try to build from a base structure.
            # But random is more general.
            
            # Optimization: Pre-generate list of pairs
            pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
            rng.shuffle(pairs)
            
            edges_added = 0
            for u, v in pairs:
                if time.time() > deadline - 10:
                    break
                    
                if _can_add_edge(adj, adj_bits, u, v, n):
                    _add_edge(adj, adj_bits, u, v)
                    edges_added += 1
                    # Heuristic: if we've added enough edges, check alpha
                    # But checking alpha is expensive.
                    # Let's just finish adding edges first, then check.
                    # However, if we add too many, we might get stuck with high alpha?
                    # No, more edges -> lower alpha.
                    # So adding as many as possible is good.
            
            # Check C4 just in case (should be free by construction)
            if _has_c4(adj, adj_bits, n):
                continue
                
            # Check Independence Number
            # Use fast bitset check for k=11
            has_indep = _has_independent_set_fast(adj_bits, n, 11)
            
            if not has_indep:
                # Valid graph found!
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append([u, v])
                if n > best_n:
                    best_n = n
                    best_edges = edges
                found = True
                break # Move to next n (smaller)
            
            # If has_indep, we failed to find a valid graph for this n with this restart.
            # Try next restart.
        
        if found:
            # We found a valid graph for this n.
            # Since we iterate n from 43 down to 31, the first n for which we find a valid graph
            # is the maximum n we can achieve with this strategy.
            # We should return immediately.
            return {'n': n, 'edges': edges}

    # Re-do the loop properly: 43 down to 31
    # If we find a valid graph for n, we can return it immediately 
    # UNLESS we want to see if n+1 is possible? 
    # But the loop goes DOWN. 
    # So if we find n=40, we haven't tried 41 yet? 
    # Wait, `range(40, 30, -1)` starts at 40. 
    # We should start at 43.
    
    # Let's clear best and try again with correct range
    best_n = 30
    best_edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        best_edges += [[a, b], [a, c], [b, c]]

    for n in range(43, 30, -1):
        if time.time() > deadline - 5:
            break
            
        # Try to find a valid graph of size n
        valid_found = False
        max_restarts = 30 # Increase restarts for smaller n as they are easier? 
        # Actually, larger n are harder to find valid. 
        # So we should give larger n MORE time/restarts.
        
        # Dynamic restarts:
        restarts = 10 if n > 38 else 30
        
        for r in range(restarts):
            if time.time() > deadline - 5:
                break
                
            adj = [set() for _ in range(n)]
            adj_bits = [0] * n
            
            pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
            rng.shuffle(pairs)
            
            # Greedy fill
            for u, v in pairs:
                if time.time() > deadline - 5:
                    break
                if _can_add_edge(adj, adj_bits, u, v, n):
                    _add_edge(adj, adj_bits, u, v)
            
            # Verify C4 (should be clean)
            if _has_c4(adj, adj_bits, n):
                continue
            
            # Check Alpha
            if not _has_independent_set_k(adj, n, 11):
                # Success
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append([u, v])
                best_n = n
                best_edges = edges
                valid_found = True
                break # Found one for this n. Since n is max in this iteration, we can return?
                # Yes, because we are going down from 43. 
                # If we find a valid graph for 43, it's the best possible (up to limit).
                # If we find one for 42, it's the best so far.
                # We should return immediately to save time for other potential improvements?
                # No, the task is to MAXIMIZE n. 
                # If we find a valid graph for n, any n' < n is worse.
                # So we can return immediately.
                return {'n': n, 'edges': edges}
        
        # If we didn't find a valid graph for n, continue to n-1
    
    return {'n': best_n, 'edges': best_edges}

if __name__ == "__main__":
    # Simple test
    res = construct()
    print(f"Constructed graph with n={res['n']}")
