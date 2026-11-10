"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

N = 20

def construct():
    """
    Construct a K3,3-free graph on 20 vertices maximizing edges.
    
    Strategy:
    1. Start with the complete graph K_20 (190 edges).
    2. Use bitmask representation for fast common-neighbor computation.
    3. Greedily remove edges that are part of K3,3 subgraphs.
    4. Try to add back removed edges if they don't create K3,3.
    5. Use deadline guard to ensure completion within time limit.
    """
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s limit
    
    # Bitmask representation: adj[i] is a bitmask of neighbors of vertex i
    # For n=20, we can use Python integers as bitmasks
    adj = [0] * N
    for i in range(N):
        for j in range(N):
            if i != j:
                adj[i] |= (1 << j)
    
    def count_edges():
        total = 0
        for i in range(N):
            total += bin(adj[i]).count('1')
        return total // 2
    
    def is_k33_free():
        """Check if graph is K3,3-free using bitmask operations."""
        # A K3,3 exists if there are two disjoint sets {a,b,c} and {d,e,f}
        # such that all 9 edges between them exist.
        # Equivalently: three vertices have at least 3 common neighbors that are
        # also connected to each other... no wait.
        # K3,3 means: exists partition {u1,u2,u3} | {v1,v2,v3} such that
        # ui-vj edge exists for all i,j.
        # So: exists 3 vertices u1,u2,u3 with common neighbor set S = N(u1)&N(u2)&N(u3)
        # where |S| >= 3, and those 3 neighbors in S are NOT connected to each other? 
        # No! In K3,3, the neighbors on the other side are NOT connected to each other.
        # Actually, K3,3 just requires the 9 cross edges. The vertices within each part
        # don't need to be non-adjacent - that would be K6.
        # 
        # Wait, let me re-read the task: "no three vertices have three common neighbours"
        # This means: for any 3 vertices {a,b,c}, |N(a) ∩ N(b) ∩ N(c)| <= 2
        # 
        # Actually that's the constraint! Let me verify: if 3 vertices have 3 common 
        # neighbors, say d,e,f, then we have a-d, a-e, a-f, b-d, b-e, b-f, c-d, c-e, c-f
        # which is exactly K3,3 with parts {a,b,c} and {d,e,f}.
        # 
        # So the constraint is: for all triples (a,b,c), |N(a)∩N(b)∩N(c)| <= 2
        
        for a, b, c in combinations(range(N), 3):
            common = adj[a] & adj[b] & adj[c]
            if bin(common).count('1') >= 3:
                return False
        return True
    
    def find_violation():
        """Find a triple with >= 3 common neighbors. Returns (a,b,c) or None."""
        for a, b, c in combinations(range(N), 3):
            common = adj[a] & adj[b] & adj[c]
            if bin(common).count('1') >= 3:
                return (a, b, c)
        return None
    
    def edge_in_k33(u, v):
        """
        Check if edge (u,v) is in any K3,3.
        An edge (u,v) is in a K3,3 if there exist a,b (neighbors of u, not v) and
        c,d (neighbors of v, not u) such that... 
        Actually simpler: edge (u,v) is in K3,3 iff there exist 2 other vertices 
        that are common neighbors of some triple involving u and v...
        
        Let me think differently. Edge (u,v) participates in a K3,3 if:
        There exist a,b,c where u is in {a,b,c} and v is in the common neighbor set,
        or v is in {a,b,c} and u is in the common neighbor set,
        or u,v are both in {a,b,c} (impossible since they're distinct) or both in neighbor set.
        
        Case 1: u is one of the 3 "left" vertices. Then we need 2 more left vertices a,b
        such that v is a common neighbor of {u,a,b} and there's a third common neighbor w.
        So: |N(u) ∩ N(a) ∩ N(b) ∩ {v,w,...}| >= 3 where v is one of them.
        
        Simpler approach: just check if removing (u,v) makes the graph K3,3-free 
        when it wasn't before. But that's expensive.
        
        For greedy removal, let's just find any violation and remove an edge from it.
        """
        pass
    
    # Phase 1: Greedily remove edges to make graph K3,3-free
    removed_edges = []
    
    while time.time() < DEADLINE:
        viol = find_violation()
        if viol is None:
            break
        
        a, b, c = viol
        common = adj[a] & adj[b] & adj[c]
        
        # Find an edge to remove. We want to remove an edge that breaks this violation.
        # Removing edge (a, x) where x is a common neighbor of a,b,c will reduce
        # |N(a)∩N(b)∩N(c)| by 1.
        # We should remove the edge that appears in the most violations.
        
        # Get the common neighbors
        common_neighbors = [i for i in range(N) if (common >> i) & 1]
        
        # Try each common neighbor and see which edge removal is best
        best_edge = None
        best_score = -1
        
        for x in common_neighbors:
            # Try removing edge (a, x)
            # Count how many violations this edge participates in (approximate)
            # For simplicity, just pick the first one or use a heuristic
            score = 0
            # Check how many other pairs (b',c') with x as common neighbor
            # This is expensive, so let's just use a simple heuristic
            # Remove the edge (a,x) if x is not adjacent to b or c (less "structural")
            if not ((adj[b] >> x) & 1):
                score += 1
            if not ((adj[c] >> x) & 1):
                score += 1
            
            if score > best_score:
                best_score = score
                best_edge = (a, x)
        
        if best_edge is None:
            best_edge = (a, common_neighbors[0])
        
        u, v = best_edge
        # Remove edge
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        removed_edges.append((u, v))
        
        if time.time() > DEADLINE - 10:
            break
    
    # Phase 2: Try to add back removed edges
    if time.time() < DEADLINE - 5:
        for u, v in removed_edges:
            if time.time() > DEADLINE - 3:
                break
            # Try adding edge (u,v)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if not is_k33_free():
                # Revert
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
    
    # Phase 3: Try adding all non-edges
    if time.time() < DEADLINE - 2:
        for i in range(N):
            if time.time() > DEADLINE - 1:
                break
            for j in range(i+1, N):
                if time.time() > DEADLINE - 1:
                    break
                if not ((adj[i] >> j) & 1):
                    # Try adding edge (i,j)
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    
                    if not is_k33_free():
                        adj[i] &= ~(1 << j)
                        adj[j] &= ~(1 << i)
    
    # Convert to edge list
    edges = []
    for i in range(N):
        for j in range(i+1, N):
            if (adj[i] >> j) & 1:
                edges.append([i, j])
    
    return edges
# EVOLVE-BLOCK-END
