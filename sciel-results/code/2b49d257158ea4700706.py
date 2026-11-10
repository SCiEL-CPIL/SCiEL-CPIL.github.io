import time
import random
import sys
from itertools import combinations

def construct():
    def instance(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def greedy_max_clique_partition(adj, n, deadline):
        """
        Greedy partition by repeatedly extracting a large clique.
        Uses Bron-Kerbosch with pivot to find maximal cliques,
        then greedily picks the largest one each iteration.
        """
        graph = adj[:]
        cliques = []
        
        def max_clique(R, P, X):
            """
            Bron-Kerbosch with pivot to find the largest clique.
            R: current clique (bitmask)
            P: candidates (bitmask)
            X: excluded (bitmask)
            Returns the largest clique found.
            """
            best = R
            
            if P == 0 and X == 0:
                return R
            
            pivot_mask = P | X
            if pivot_mask == 0:
                candidates = P
            else:
                best_pivot = -1
                max_deg = -1
                pm = pivot_mask
                while pm:
                    lsb = pm & -pm
                    u = lsb.bit_length() - 1
                    deg = (graph[u] & P).bit_count()
                    if deg > max_deg:
                        max_deg = deg
                        best_pivot = u
                    pm ^= lsb
                
                if best_pivot != -1:
                    candidates = P & ~graph[best_pivot]
                else:
                    candidates = P
            
            c = candidates
            while c:
                lsb = c & -c
                v = lsb.bit_length() - 1
                
                new_R = R | lsb
                new_P = P & graph[v]
                new_X = X & graph[v]
                
                res = max_clique(new_R, new_P, new_X)
                if res.bit_count() > best.bit_count():
                    best = res
                
                P ^= lsb
                X |= lsb
                c ^= lsb
                
                if time.time() > deadline:
                    return best
            
            return best

        while True:
            has_edges = False
            for u in range(n):
                if graph[u]:
                    has_edges = True
                    break
            
            if not has_edges:
                break
            
            best_clique_mask = max_clique(0, (1 << n) - 1, 0)
            
            if best_clique_mask.bit_count() < 2:
                for u in range(n):
                    if graph[u]:
                        lsb = graph[u] & -graph[u]
                        v = lsb.bit_length() - 1
                        best_clique_mask = (1 << u) | (1 << v)
                        break
            
            verts = []
            mask = best_clique_mask
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                verts.append(v)
                mask ^= lsb
            
            if len(verts) >= 2:
                cliques.append(verts)
                for i in range(len(verts)):
                    u = verts[i]
                    for j in range(i + 1, len(verts)):
                        v = verts[j]
                        graph[u] &= ~(1 << v)
                        graph[v] &= ~(1 << u)
            
            if time.time() > deadline:
                break
        
        for u in range(n):
            mask = graph[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                if v > u:
                    cliques.append([u, v])
                    graph[u] &= ~(1 << v)
                    graph[v] &= ~(1 << u)
                mask ^= lsb
        
        return cliques

    def greedy_sequential_clique_partition(adj, n, deadline):
        """
        Sequential greedy: process edges one by one, assign each to the
        smallest-indexed existing clique that can accommodate it (maintaining
        the clique property), or create a new clique if none works.
        This is equivalent to greedy coloring of the line graph.
        """
        graph = adj[:]
        cliques = []  # list of vertex sets (as lists)
        # For each clique, maintain the set of vertices as a bitmask for fast intersection
        clique_masks = []
        
        # Collect all edges
        edges = []
        for u in range(n):
            mask = graph[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                if v > u:
                    edges.append((u, v))
                mask ^= lsb
        
        # Process edges in order of decreasing "popularity" (sum of degrees)
        # to pack more edges into larger cliques
        edge_order = sorted(range(len(edges)), key=lambda i: -(graph[edges[i][0]].bit_count() + graph[edges[i][1]].bit_count()))
        
        # Track which edges are covered
        covered = [False] * len(edges)
        
        for idx in edge_order:
            if covered[idx]:
                continue
            u, v = edges[idx]
            
            # Try to fit this edge into an existing clique
            best_clique_idx = -1
            best_clique_size = 0
            
            for ci, cmask in enumerate(clique_masks):
                # Check if u and v can be added to this clique
                # All vertices in the clique must be adjacent to both u and v
                # And u must be adjacent to v (already true since it's an edge)
                new_mask = cmask | (1 << u) | (1 << v)
                
                # Check if all pairs in new_mask form a clique
                valid = True
                # u must be adjacent to all in cmask
                if not (cmask & ~graph[u]) == 0:
                    valid = False
                # v must be adjacent to all in cmask
                elif not (cmask & ~graph[v]) == 0:
                    valid = False
                # u and v are adjacent (guaranteed)
                
                if valid:
                    size = new_mask.bit_count()
                    if size > best_clique_size:
                        best_clique_size = size
                        best_clique_idx = ci
            
            if best_clique_idx >= 0:
                # Add u and v to this clique
                cmask = clique_masks[best_clique_idx]
                clique_masks[best_clique_idx] = cmask | (1 << u) | (1 << v)
                # Update the clique list
                verts = cliques[best_clique_idx]
                if u not in verts:
                    verts.append(u)
                if v not in verts:
                    verts.append(v)
            else:
                # Create a new clique
                cliques.append([u, v])
                clique_masks.append((1 << u) | (1 << v))
            
            covered[idx] = True
            
            # Also mark any other edges that are now "covered" by being within this clique
            # Actually, we need to be careful: an edge is covered if both endpoints are in the same clique
            # Let's re-check: we should only mark edges that are fully contained in some clique
            # This is complex, so let's just track which edges we've explicitly assigned
            # and at the end, verify all edges are covered
            
            if time.time() > deadline:
                break
        
        # Verify all edges are covered; if not, add remaining as 2-cliques
        # Build a set of all edges in cliques
        covered_edges = set()
        for clique in cliques:
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u, v = clique[i], clique[j]
                    if u > v:
                        u, v = v, u
                    covered_edges.add((u, v))
        
        # Check for uncovered edges
        for u in range(n):
            mask = graph[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                if v > u:
                    if (u, v) not in covered_edges:
                        cliques.append([u, v])
                        covered_edges.add((u, v))
                mask ^= lsb
        
        return cliques

    start_time = time.time()
    deadline = start_time + 140
    
    adj_1 = instance(32, 0.6, 601)
    adj_2 = instance(64, 0.55, 602)
    
    # Run both approaches and pick the better one
    cliques_1_max = greedy_max_clique_partition(adj_1, 32, deadline)
    cliques_2_max = greedy_max_clique_partition(adj_2, 64, deadline)
    
    total_max = len(cliques_1_max) + len(cliques_2_max)
    
    if time.time() < deadline - 5:
        cliques_1_seq = greedy_sequential_clique_partition(adj_1, 32, deadline)
        cliques_2_seq = greedy_sequential_clique_partition(adj_2, 64, deadline)
        
        total_seq = len(cliques_1_seq) + len(cliques_2_seq)
        
        if total_seq < total_max:
            return [cliques_1_seq, cliques_2_seq]
    
    return [cliques_1_max, cliques_2_max]

if __name__ == "__main__":
    res = construct()
    print(f"Instance 1: {len(res[0])} cliques")
    print(f"Instance 2: {len(res[1])} cliques")
    print(f"Total: {len(res[0]) + len(res[1])}")
