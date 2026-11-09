"""Construct a C4-free graph on n vertices with minimum degree >= n-39.

Strategy:
  1. Start from a complete graph on n vertices.
  2. Greedily remove edges that create C4s, prioritizing removal of edges
     that are in the most C4s, to minimize total edges removed.
  3. Verify the result meets the minimum-degree constraint.
  4. If not, try a local search / edge-addition repair.
  5. Try multiple n values from high to low; return the best valid graph.

The key insight: a C4-free graph on n vertices with min degree >= n-39
means each vertex is missing at most 39 edges. The complement has max degree <= 39.
We want to find a C4-free graph as dense as possible.

Alternative approach: build the complement. The complement H has max degree <= 39
and must be such that G = complement(H) is C4-free. G is C4-free iff H contains
no induced structure... actually it's easier to work with G directly.

Let me use a greedy edge-addition approach:
- Start with empty graph on n vertices.
- Repeatedly add the edge (u,v) that maximizes some heuristic, as long as
  adding it doesn't create a C4.
- An edge (u,v) creates a C4 iff there exist a,b such that edges (u,a),(a,b),(b,v)
  and (v,a),(a,b),(b,u) exist... no, C4 is u-a-v-b-u or u-a-b-v-u etc.
  
  More precisely: adding (u,v) creates a C4 iff there exist distinct a,b != u,v
  such that (u,a), (a,b), (b,v) all exist AND (u,b), (a,v) also... no.
  
  A C4 containing edge (u,v) would be u - a - v - b - u where a,b are distinct
  and not u,v. So we need: (u,a) in E, (a,v) in E, (v,b) in E, (b,u) in E.
  Equivalently: there exist a,b with (u,a),(a,v) in E and (u,b),(b,v) in E.
  I.e., u and v have at least two common neighbors.
  
  So: adding (u,v) creates a C4 iff |N(u) ∩ N(v)| >= 2.

This is the key check. Let me implement greedy edge addition:
- For n vertices, try to add all possible edges while maintaining C4-freedom.
- Use a priority: add edges that give the most "room" or use a random order with restarts.

Actually, for maximizing minimum degree, we want the graph to be as regular as possible.
A good approach:
- Use a randomized greedy: shuffle edges, add if C4-free.
- Try multiple random orderings, keep the best.

For n=46, we need min degree >= 7 (46-39=7). For n=47, min degree >= 8.

Let me try a more structured approach:
- Start with a cycle or some base structure, then add edges greedily.
- Or: use the fact that polarity graphs / finite geometry give C4-free graphs.

Actually, the simplest strong approach: randomized greedy edge addition with many restarts.
For each restart:
  - Create a random permutation of all C(n,2) edges.
  - For each edge in order, add it if it doesn't create a C4.
  - At the end, check min degree.
  - Keep the best result.

The C4 check: maintain adjacency sets. When adding (u,v), check if |adj[u] ∩ adj[v]| >= 2.
This is O(min(deg(u), deg(v))) per check.

For n=47, there are 1081 edges. Each check is fast. Many restarts should be feasible.

Let me also try a "targeted" approach: first ensure all vertices reach the minimum degree,
then try to add more edges.
"""

import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    best_result = {'n': 39, 'edges': []}
    best_score = 39  # n is the score (maximize n)
    
    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        if time.time() > deadline - 10:
            break
        
        min_deg_needed = n - 39
        
        # Try multiple random restarts
        found = False
        num_restarts = 200  # will be limited by time
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            # Random greedy edge addition
            all_edges = list(combinations(range(n), 2))
            random.shuffle(all_edges)
            
            adj = [set() for _ in range(n)]
            edge_set = set()
            
            for (u, v) in all_edges:
                # Check if adding (u,v) creates a C4
                # C4 exists iff u and v have >= 2 common neighbors
                common = adj[u] & adj[v]
                if len(common) < 2:
                    adj[u].add(v)
                    adj[v].add(u)
                    edge_set.add((u, v))
            
            # Check minimum degree
            min_deg = min(len(adj[i]) for i in range(n))
            
            if min_deg >= min_deg_needed:
                edges = [[u, v] for (u, v) in edge_set]
                result = {'n': n, 'edges': edges}
                
                # Verify it's truly C4-free (safety check for small n)
                if _verify_c4_free(n, edges):
                    if n > best_score:
                        best_score = n
                        best_result = result
                    found = True
                    break  # found valid graph for this n, move to next n
        
        if found:
            break  # found the largest n, return it
    
    return best_result


def _verify_c4_free(n, edges):
    """Verify the graph has no C4. O(E * n) worst case, fine for n<=47."""
    adj = [set() for _ in range(n)]
    for (u, v) in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # For each pair of adjacent vertices, check they don't have 2 common neighbors
    # A C4 is u-a-v-b-u, so for edge (u,v), if they share 2+ neighbors, C4 exists
    for (u, v) in edges:
        common = adj[u] & adj[v]
        if len(common) >= 2:
            return False
    
    return True


# Quick self-test
if __name__ == "__main__":
    result = construct()
    print(f"n={result['n']}, edges={len(result['edges'])}")
    if result['n'] > 39:
        print(f"min_degree={min(len([e for e in result['edges'] if i in e]) for i in range(result['n']))}")
        print(f"C4-free: {_verify_c4_free(result['n'], result['edges'])}")
