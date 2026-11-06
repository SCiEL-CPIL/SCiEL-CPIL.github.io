import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 110.0  # Leave 10s margin for 120s limit
    
    # 1. Graph Generation (Exact match to verifier)
    def get_instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        for _ in range(4 * n * d):
            u = rng.randrange(n)
            v = rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    instances = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        edges = get_instance(*spec)
        n = spec[0]
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Precompute F(e): vertices that cannot be in the same induced matching as e
        edge_F = {}
        for e in edges:
            u, v = e
            F = set()
            F.add(u)
            F.add(v)
            F.update(adj[u])
            F.update(adj[v])
            edge_F[e] = F
            
        instances.append({
            'n': n,
            'edges': edges,
            'edge_F': edge_F,
            'adj': adj
        })
        
    results = []
    
    def solve_instance(inst):
        n = inst['n']
        edges = inst['edges']
        edge_F = inst['edge_F']
        adj = inst['adj']
        
        # Greedy initialization: sort edges by degree sum descending
        edge_order = sorted(edges, key=lambda e: len(adj[e[0]]) + len(adj[e[1]]), reverse=True)
        
        colour_of_edge = {}
        class_verts = []  # class_verts[c] = set of vertices in colour class c
        class_edges = []  # class_edges[c] = list of edges in colour class c
        
        for e in edge_order:
            Fe = edge_F[e]
            found = False
            for c in range(len(class_verts)):
                if class_verts[c].isdisjoint(Fe):
                    colour_of_edge[e] = c
                    class_verts[c].update(e)
                    class_edges[c].append(e)
                    found = True
                    break
            if not found:
                c = len(class_verts)
                colour_of_edge[e] = c
                class_verts.append(set(e))
                class_edges.append([e])
        
        def get_num_colours():
            return len(class_edges)
        
        def compact():
            """Remove empty classes and reindex."""
            new_class_verts = []
            new_class_edges = []
            old_to_new = {}
            for c in range(len(class_verts)):
                if class_edges[c]:
                    old_to_new[c] = len(new_class_verts)
                    new_class_verts.append(class_verts[c])
                    new_class_edges.append(class_edges[c])
            class_verts.clear()
            class_edges.clear()
            class_verts.extend(new_class_verts)
            class_edges.extend(new_class_edges)
            for e in colour_of_edge:
                old_c = colour_of_edge[e]
                if old_c in old_to_new:
                    colour_of_edge[e] = old_to_new[old_c]
        
        def try_relocate(e, from_c):
            """Try to move edge e from class from_c to a lower-index class."""
            Fe = edge_F[e]
            # Try classes in order from 0 to from_c-1
            for c2 in range(from_c):
                if class_verts[c2].isdisjoint(Fe):
                    # Move e from from_c to c2
                    class_edges[from_c].remove(e)
                    class_verts[from_c].discard(e[0])
                    class_verts[from_c].discard(e[1])
                    class_edges[c2].append(e)
                    class_verts[c2].add(e[0])
                    class_verts[c2].add(e[1])
                    colour_of_edge[e] = c2
                    return True
            return False
        
        def try_swap(e1, c1, e2, c2):
            """Try to swap e1 (in c1) with e2 (in c2)."""
            Fe1 = edge_F[e1]
            Fe2 = edge_F[e2]
            
            # Check if e1 can go to c2 (without e2)
            verts_c2_without_e2 = class_verts[c2] - set(e2)
            if not verts_c2_without_e2.isdisjoint(Fe1):
                return False
            
            # Check if e2 can go to c1 (without e1)
            verts_c1_without_e1 = class_verts[c1] - set(e1)
            if not verts_c1_without_e1.isdisjoint(Fe2):
                return False
            
            # Perform swap
            class_edges[c1].remove(e1)
            class_verts[c1].discard(e1[0])
            class_verts[c1].discard(e1[1])
            class_edges[c2].remove(e2)
            class_verts[c2].discard(e2[0])
            class_verts[c2].discard(e2[1])
            
            class_edges[c1].append(e2)
            class_verts[c1].add(e2[0])
            class_verts[c1].add(e2[1])
            class_edges[c2].append(e1)
            class_verts[c2].add(e1[0])
            class_verts[c2].add(e1[1])
            
            colour_of_edge[e1] = c2
            colour_of_edge[e2] = c1
            return True
        
        # Local search loop
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Step 1: Try to relocate edges from high-index classes to lower-index classes
            # Process classes from highest to lowest
            for c1 in range(len(class_edges) - 1, -1, -1):
                if not class_edges[c1]:
                    continue
                
                # Try to relocate each edge in class c1
                edges_to_try = list(class_edges[c1])
                for e in edges_to_try:
                    if colour_of_edge.get(e) != c1:
                        continue  # Edge was already moved
                    if try_relocate(e, c1):
                        improved = True
                        break  # Restart the outer loop to re-evaluate state
                
                if improved:
                    break
            
            # Step 2: If no relocations, try swaps to enable future relocations
            if not improved and time.time() < deadline - 10:
                # Try swaps between pairs of classes
                # Focus on swaps that might empty a high-index class
                for c1 in range(len(class_edges) - 1, 0, -1):
                    if not class_edges[c1]:
                        continue
                    
                    # If class c1 has only 1 edge, try to swap it out
                    if len(class_edges[c1]) == 1:
                        e1 = class_edges[c1][0]
                        Fe1 = edge_F[e1]
                        
                        # Try to swap with an edge in a lower class
                        for c2 in range(c1):
                            if not class_edges[c2]:
                                continue
                            
                            for e2 in class_edges[c2]:
                                Fe2 = edge_F[e2]
                                
                                # Check if swap is valid
                                verts_c1_without_e1 = class_verts[c1] - set(e1)
                                verts_c2_without_e2 = class_verts[c2] - set(e2)
                                
                                if verts_c2_without_e2.isdisjoint(Fe1) and \
                                   verts_c1_without_e1.isdisjoint(Fe2):
                                    # Perform swap
                                    if try_swap(e1, c1, e2, c2):
                                        improved = True
                                        break
                            
                            if improved:
                                break
                    
                    if improved:
                        break
            
            # Compact after any improvement
            if improved:
                compact()
        
        # Final compaction
        compact()
        
        # Calculate result
        if not class_edges:
            return []
        
        # Map back to original edge order
        res = [0] * len(edges)
        for i, e in enumerate(edges):
            res[i] = colour_of_edge[e]
        return res
    
    for inst in instances:
        results.append(solve_instance(inst))
        
    return results
