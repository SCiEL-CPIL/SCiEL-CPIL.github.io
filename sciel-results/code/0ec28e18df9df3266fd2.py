"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []

    for (n, p, seed) in specs:
        if time.time() > deadline - 5:
            # Fallback: just return remaining edges as individual pieces
            edges = instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue

        edges = instance(n, p, seed)
        edge_set = set(edges)
        
        # Build adjacency
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        remaining = set(edge_set)
        pieces = []
        
        # Greedy cycle extraction: repeatedly find a cycle in the remaining graph
        # and remove its edges. This minimizes the number of cycle pieces.
        # After no more cycles can be found, remaining edges are single-edge pieces.
        
        max_iters = 10000
        iter_count = 0
        
        while remaining and iter_count < max_iters:
            iter_count += 1
            if time.time() > deadline - 2:
                break
            
            # Find a cycle using DFS
            # We look for any cycle in the remaining subgraph
            visited = {}  # node -> (parent, state) where state is 0=visiting, 1=done
            found_cycle = None
            parent = {}
            
            def dfs(u):
                visited[u] = 0
                for v in sorted(adj[u]):
                    if (u, v) not in remaining and (v, u) not in remaining:
                        continue
                    if v not in visited:
                        parent[v] = u
                        result = dfs(v)
                        if result is not None:
                            return result
                    elif visited[v] == 0:
                        # Found a back edge u->v, extract cycle
                        cycle = [u, v]
                        cur = u
                        while cur != v:
                            cur = parent[cur]
                            cycle.append(cur)
                        return cycle
                visited[u] = 1
                return None
            
            for start in range(n):
                if start not in visited:
                    found_cycle = dfs(start)
                    if found_cycle is not None:
                        break
            
            if found_cycle is None:
                break
            
            # Normalize the cycle: it's a list of vertices [v0, v1, ..., vk]
            # where consecutive vertices are edges, and (vk, v0) is the closing edge
            # Actually from the DFS above, the cycle is [u, v, ..., v] where the last element is v
            # Let's re-extract more carefully
            
            # The cycle found: u -> v (back edge), then path from u back to v via parents
            # cycle list: [u, v, parent[u], parent[parent[u]], ..., v]
            # So the vertices in order are: cycle[0], cycle[1], ..., cycle[-1] where cycle[0]==cycle[-1]
            # Wait, let me re-examine. The cycle list built is:
            # cycle = [u, v], then we go cur=u, append parent[u], until cur==v
            # So cycle = [u, v, parent[u], parent[parent[u]], ..., v]
            # The actual cycle vertices (without duplication) are: u, v, parent[u], ..., parent^k[u] where parent^k[u] = v
            # But we want the cycle in order. The edges are:
            # (u, v) [the back edge], (v, parent[v])? No...
            
            # Let me redo this more carefully.
            # In DFS, when we find back edge u->v (v is ancestor of u):
            # The cycle is: v -> ... -> u -> v
            # Path from v to u: v, parent[v] is wrong. parent[u] is the node before u on the tree path.
            # Tree path from root to u: root, ..., v, ..., u
            # So the cycle vertices in order: v, (child of v on path to u), ..., u, v
            # 
            # My code: cycle = [u, v], then cur=u, append parent[cur] until cur==v
            # So cycle = [u, v, parent[u], parent[parent[u]], ..., v]
            # The cycle in proper order (starting from v): v, ..., parent[u], u, v
            # Reversed from what I have minus the first element... 
            # 
            # Let's just use: the vertices in the cycle are cycle[0:-1] which is [u, v, parent[u], ..., parent^k[u]]
            # where parent^k[u] is the child of v on the path to u... this is getting confusing.
            # 
            # Let me just verify the edges exist and use the cycle as a piece.
            # The piece is given as a list of vertices where consecutive pairs are edges and the last-first is an edge.
            
            # From the extraction:
            # cycle_list = [u, v, p1, p2, ..., v] where p1=parent[u], p2=parent[p1], etc.
            # Edges in the cycle: (u,v), (v, p1)? No. (u, p1) is an edge (tree edge), (p1, p2) is an edge, etc.
            # And (last before final v, v) is an edge.
            # 
            # Actually the edges are:
            # Back edge: (u, v)
            # Tree edges: (u, parent[u]), (parent[u], parent[parent[u]]), ..., (parent^k[u], v)
            # So the cycle in order: u, v, then going up: but v is an ancestor.
            # The cycle is: u -> v (back edge), then v -> child_on_path -> ... -> u (tree edges reversed)
            # 
            # The vertices in order around the cycle: [u, v, child_of_v_on_path, ..., parent[u], u]
            # But my list is [u, v, parent[u], parent[parent[u]], ..., v]
            # 
            # Let me just take the unique vertices and verify. For the piece, I need an ordering
            # where consecutive vertices (and last-to-first) are edges in the remaining set.
            
            # Simplest: take the cycle vertices as they appear, dedupe the last (which equals some earlier one)
            # The cycle list ends with v (the ancestor). The first element is u.
            # The actual cycle vertices in order: v, ..., u (going down the tree from v to u), then back to v.
            # My list: [u, v, parent[u], parent[parent[u]], ..., v]
            # Reversed: [v, ..., parent[parent[u]], parent[u], v, u]
            # Drop the last v (duplicate of first): [v, ..., parent[parent[u]], parent[u], v, u] -> no
            # 
            # OK let me just build the cycle properly.
            
            # Re-extract:
            # The back edge is (u, v) where v is an ancestor of u.
            # Path from v down to u in the tree: v = w0, w1, w2, ..., wk = u where wi+1 is child of wi
            # In my parent array, parent[wi+1] = wi
            # So from u, following parents: u, parent[u], parent[parent[u]], ..., v
            # This gives us the path from u up to v: u, p1, p2, ..., v
            # The cycle in order: v, ..., p2, p1, u, v (going down then back up via back edge)
            # Or equivalently: u, p1, p2, ..., v, u (going up then back down via back edge)
            # 
            # My cycle list is [u, v, p1, p2, ..., v] where p1=parent[u], etc.
            # The cycle vertices in order (u to v via tree, then v to u via back edge):
            # u, p1, p2, ..., v, u
            # That's: cycle_list[0], cycle_list[2], cycle_list[3], ..., cycle_list[-1], cycle_list[0]
            # i.e., skip the second element (v) and append it at the end... no.
            # 
            # cycle_list = [u, v, p1, p2, ..., v]
            # Path u->v via tree: u, p1, p2, ..., v = cycle_list[0], cycle_list[2], cycle_list[3], ..., cycle_list[-1]
            # Then back edge v->u closes it.
            # So the cycle piece is: [cycle_list[0], cycle_list[2], cycle_list[3], ..., cycle_list[-1]]
            # Let's call this: cycle_verts = [cycle_list[0]] + cycle_list[2:]
            
            cycle_verts = [found_cycle[0]] + found_cycle[2:]
            # cycle_verts should be [u, p1, p2, ..., v] where the last is v (the ancestor)
            # Edges: (u,p1), (p1,p2), ..., (v, u) [closing]
            # Verify all edges exist in remaining
            valid = True
            for i in range(len(cycle_verts)):
                a = cycle_verts[i]
                b = cycle_verts[(i + 1) % len(cycle_verts)]
                e = tuple(sorted((a, b)))
                if e not in remaining:
                    valid = False
                    break
            
            if not valid or len(cycle_verts) < 3:
                break
            
            pieces.append(cycle_verts)
            for i in range(len(cycle_verts)):
                a = cycle_verts[i]
                b = cycle_verts[(i + 1) % len(cycle_verts)]
                remaining.discard(tuple(sorted((a, b))))
        
        # Remaining edges become individual pieces
        for e in remaining:
            pieces.append(list(e))
        
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
