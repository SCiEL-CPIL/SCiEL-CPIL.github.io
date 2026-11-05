import time
import random
from collections import defaultdict

def construct():
    """Construct a dense K2,3-free graph on 20 vertices, maximizing edges."""
    start_time = time.time()
    deadline = start_time + 145  # 150s wall-clock, 5s safety margin
    n = 20
    
    # We'll represent the graph using integer bitmasks for adjacency
    # neighbors[i] is an integer where bit j is set if (i,j) is an edge
    neighbors = [0] * n
    
    def edge_count():
        total = 0
        for i in range(n):
            total += neighbors[i].bit_count()
        return total // 2
    
    def is_valid_addition(u, v):
        """Check if adding edge (u,v) would create a K2,3 violation."""
        # A K2,3 exists if there are two vertices with at least 3 common neighbors
        # After adding (u,v), we need to check:
        # 1. For any vertex w, if w is neighbor of both u and v (after addition),
        #    then u and v would have at least that many common neighbors
        # 2. More generally, adding (u,v) affects common neighbor counts for pairs
        #    involving u or v
        
        # Common neighbors of u and v after adding edge:
        # These are vertices w such that w is adjacent to both u and v
        # Note: u is adjacent to v (we're adding it), so we count w != u, w != v
        common_uv = neighbors[u] & neighbors[v]
        # Remove u and v themselves from consideration (they can't be their own common neighbors in the K2,3 sense)
        common_uv &= ~(1 << u)
        common_uv &= ~(1 << v)
        if common_uv.bit_count() >= 3:
            return False
        
        # Now check all other pairs (a,b) where at least one of a or b is u or v
        # and the other is some w, to see if adding (u,v) creates a new K2,3
        for w in range(n):
            if w == u or w == v:
                continue
            
            # Check pair (u, w): after adding (u,v), does u and w have >= 3 common neighbors?
            # Common neighbors of u and w: vertices x adjacent to both u and w
            # Before: common_uw = neighbors[u] & neighbors[w]
            # After adding (u,v): if v is adjacent to w, then v becomes a new common neighbor of u and w
            # Also, if u is adjacent to w... wait, we need to think carefully.
            
            # Common neighbors of (u,w) after adding edge (u,v):
            # A vertex x is a common neighbor of u and w if x is adjacent to both u and w.
            # The edge (u,v) being added means:
            # - v is now adjacent to u. If v is also adjacent to w, then v is a common neighbor of u and w.
            # - u is now adjacent to v. This doesn't directly affect common neighbors of (u,w) unless...
            #   Actually, common neighbors of (u,w) are vertices x where x~u and x~w.
            #   Adding edge (u,v) changes the adjacency of u and v.
            #   So v becomes adjacent to u. If v~w, then v is now a common neighbor of u and w.
            #   Also, does this affect any other vertex being a common neighbor? No, because
            #   we're only adding one edge, so only the adjacency involving u and v changes.
            
            # So: common neighbors of (u,w) after = (neighbors[u] & neighbors[w]) plus potentially v
            # But neighbors[u] already includes v after we add it? No, we're checking BEFORE adding.
            # Let me reframe: current common neighbors of (u,w) = neighbors[u] & neighbors[w]
            # After adding (u,v): u gains neighbor v, v gains neighbor u.
            # New common neighbors of (u,w) = (neighbors[u] | (1<<v)) & (neighbors[w] | (1<<u if w==v else 0))
            # This is getting complex. Let me use a simpler approach.
            
            # Simpler: after adding (u,v), for pair (u,w):
            # common neighbors = vertices x such that x is adjacent to u AND x is adjacent to w
            # x is adjacent to u: either x was already adjacent to u, or x == v (new edge)
            # x is adjacent to w: x was already adjacent to w (adding (u,v) only affects u and v's adjacencies)
            #   unless x == u or x == v, but x can't be u (not its own neighbor) 
            #   Actually x can be v: is v adjacent to w? That's neighbors[w] has bit v set.
            
            # Let me just compute directly:
            # After adding (u,v), neighbor set of u = neighbors[u] | (1<<v)
            # After adding (u,v), neighbor set of w = neighbors[w] (unchanged, unless w==v, but w!=v here)
            # Common neighbors of (u,w) = (neighbors[u] | (1<<v)) & neighbors[w]
            # But we should exclude u and w themselves? No, common neighbors are other vertices.
            # Actually in K2,3, the two centers are u and w, and the three leaves are common neighbors.
            # The leaves must be distinct from u and w.
            
            common_uw_after = (neighbors[u] | (1 << v)) & neighbors[w]
            common_uw_after &= ~(1 << u)
            common_uw_after &= ~(1 << w)
            if common_uw_after.bit_count() >= 3:
                return False
            
            # Check pair (v, w): after adding (u,v), does v and w have >= 3 common neighbors?
            # After adding (u,v), neighbor set of v = neighbors[v] | (1<<u)
            # Neighbor set of w = neighbors[w] (unchanged)
            # Common neighbors of (v,w) = (neighbors[v] | (1<<u)) & neighbors[w]
            common_vw_after = (neighbors[v] | (1 << u)) & neighbors[w]
            common_vw_after &= ~(1 << v)
            common_vw_after &= ~(1 << w)
            if common_vw_after.bit_count() >= 3:
                return False
        
        return True
    
    def try_add_edge(u, v):
        """Add edge (u,v) if valid. Returns True if added."""
        if u >= v:
            u, v = v, u
        if (neighbors[u] >> v) & 1:
            return False  # Already an edge
        if is_valid_addition(u, v):
            neighbors[u] |= (1 << v)
            neighbors[v] |= (1 << u)
            return True
        return False
    
    def greedy_complete():
        """Try to add all possible edges greedily."""
        added = True
        while added:
            added = False
            for u in range(n):
                for v in range(u + 1, n):
                    if not ((neighbors[u] >> v) & 1):
                        if try_add_edge(u, v):
                            added = True
                            if time.time() > deadline:
                                return
    
    # Start with empty graph and greedily add edges
    greedy_complete()
    best_edges = edge_count()
    
    # Now try local search: remove some edges and try to re-add
    if time.time() < deadline:
        for restart in range(20):
            if time.time() > deadline:
                break
            random.seed(42 + restart)
            
            # Start from current best or a random perturbation
            neighbors_copy = neighbors[:]
            for i in range(n):
                neighbors_copy[i] = neighbors[i]
            
            # Remove a few random edges to explore
            current_edges = edge_count()
            edge_list = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (neighbors_copy[i] >> j) & 1:
                        edge_list.append((i, j))
            
            # Remove 3-8 random edges
            num_remove = random.randint(3, 8)
            random.shuffle(edge_list)
            for u, v in edge_list[:num_remove]:
                neighbors_copy[u] &= ~(1 << v)
                neighbors_copy[v] &= ~(1 << u)
            
            # Save original neighbors for restoration if we don't improve
            original_neighbors = neighbors[:]
            neighbors = neighbors_copy[:]
            
            # Greedily try to add edges back
            greedy_complete()
            
            new_edges = edge_count()
            if new_edges > best_edges:
                best_edges = new_edges
                neighbors_copy = neighbors[:]
                for i in range(n):
                    neighbors_copy[i] = neighbors[i]
            else:
                # Restore
                neighbors = original_neighbors[:]
            
            # Try more aggressive: start fresh with random initialization
            if restart % 5 == 4 and time.time() < deadline - 10:
                # Random dense start
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < 0.5:
                            neighbors[i] |= (1 << j)
                            neighbors[j] |= (1 << i)
                
                # Now try to fix violations by removing edges
                # Find all K2,3 violations and remove edges to fix them
                improved = True
                while improved and time.time() < deadline - 5:
                    improved = False
                    for u in range(n):
                        for v in range(u + 1, n):
                            common = neighbors[u] & neighbors[v]
                            common &= ~(1 << u)
                            common &= ~(1 << v)
                            if common.bit_count() >= 3:
                                # Remove a random edge incident to a common neighbor
                                common_list = [w for w in range(n) if w != u and w != v and (common >> w) & 1]
                                if common_list:
                                    w = random.choice(common_list)
                                    # Remove edge (w, u) or (w, v)
                                    if random.random() < 0.5:
                                        neighbors[w] &= ~(1 << u)
                                        neighbors[u] &= ~(1 << w)
                                    else:
                                        neighbors[w] &= ~(1 << v)
                                        neighbors[v] &= ~(1 << w)
                                    improved = True
                
                # Greedily add more
                greedy_complete()
                new_edges = edge_count()
                if new_edges > best_edges:
                    best_edges = new_edges
                    for i in range(n):
                        neighbors_copy[i] = neighbors[i]
    
    # Final greedy pass to ensure maximum
    if time.time() < deadline - 2:
        # Work with the best configuration
        best_neighbors = neighbors_copy[:]
        for i in range(n):
            neighbors[i] = best_neighbors[i]
        greedy_complete()
    
    # Convert to edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (neighbors[i] >> j) & 1:
                edges.append([i, j])
    
    return edges
