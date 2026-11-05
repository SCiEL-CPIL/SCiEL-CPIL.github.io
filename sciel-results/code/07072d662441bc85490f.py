import time
import random
import sys
import numpy as np

def _construct_polarity_graph(q):
    """
    Construct the polarity graph of the projective plane PG(2, q).
    Vertices are points of the plane.
    Two points p and r are adjacent if p lies on the polar line of r.
    This graph is C4-free.
    """
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, q-2, q)
                    key = (1, (y * inv_x) % q, (z * inv_x) % q)
                elif y != 0:
                    inv_y = pow(y, q-2, q)
                    key = (0, 1, (z * inv_y) % q)
                else:
                    key = (0, 0, 1)
                
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    
    n_pts = len(points)
    
    adj = [set() for _ in range(n_pts)]
    
    for i in range(n_pts):
        x1, x2, x3 = points[i]
        for j in range(i + 1, n_pts):
            y1, y2, y3 = points[j]
            if (x1 * y1 + x2 * y2 + x3 * y3) % q == 0:
                adj[i].add(j)
                adj[j].add(i)
                
    edges = []
    for i in range(n_pts):
        for j in adj[i]:
            if i < j:
                edges.append([i, j])
                
    return n_pts, edges, adj

def _verify_c4_free(n, adj):
    """Check that no two vertices have >= 2 common neighbours."""
    for u in range(n):
        for v in range(u + 1, n):
            common = adj[u] & adj[v]
            if len(common) >= 2:
                return False
    return True

def _max_independent_set(adj, n, limit=11):
    """
    Check if there is an independent set of size >= limit.
    Uses branch and bound with bitmasks for speed.
    """
    if n <= 64:
        adj_mask = [0] * n
        for i in range(n):
            mask = 0
            for j in adj[i]:
                mask |= (1 << j)
            adj_mask[i] = mask
        
        all_mask = (1 << n) - 1
        
        def _branch(rem_mask, current_size):
            if current_size >= limit:
                return True
            if rem_mask == 0:
                return False
            if current_size + rem_mask.bit_count() < limit:
                return False
            
            # Greedy pivot: pick vertex with max degree in remaining subgraph
            best_v = -1
            best_deg = -1
            m = rem_mask
            while m:
                v = (m & -m).bit_length() - 1
                m &= m - 1
                deg = (adj_mask[v] & rem_mask).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            
            if best_v == -1:
                return False
            
            # Include best_v
            rem_include = rem_mask & ~(adj_mask[best_v] | (1 << best_v))
            if _branch(rem_include, current_size + 1):
                return True
            
            # Exclude best_v
            rem_exclude = rem_mask & ~(1 << best_v)
            if _branch(rem_exclude, current_size):
                return True
            
            return False
        
        return _branch(all_mask, 0)
    else:
        # Fallback for larger n
        remaining = set(range(n))
        
        def _branch(rem, current_size):
            if current_size >= limit:
                return True
            if not rem:
                return False
            if current_size + len(rem) < limit:
                return False
            
            best_v = -1
            best_deg = -1
            for v in rem:
                deg = len(adj[v] & rem)
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            
            if best_v == -1:
                return False
                
            rem_include = rem - adj[best_v] - {best_v}
            if _branch(rem_include, current_size + 1):
                return True
            
            rem_exclude = rem - {best_v}
            if _branch(rem_exclude, current_size):
                return True
            
            return False
        
        return _branch(remaining, 0)

def _find_independent_set_in_h(n, adj_h):
    """
    Find a maximal independent set in graph H (represented by adj_h).
    H has an edge between u, v if N(u) and N(v) in G are not disjoint.
    We want a set S of vertices in G such that N(u) and N(v) are disjoint for all u,v in S.
    This is an independent set in H.
    """
    # Greedy: pick vertex with lowest degree in H, add to S, remove it and its neighbors from H
    S = set()
    remaining = set(range(n))
    
    # Sort by degree in H for better heuristic? Or just random?
    # Let's try picking the one with min degree in H to allow more others
    while remaining:
        # Pick vertex with min degree in H within remaining
        best_v = -1
        min_deg = float('inf')
        for v in remaining:
            deg = len(adj_h[v] & remaining)
            if deg < min_deg:
                min_deg = deg
                best_v = v
        
        S.add(best_v)
        # Remove best_v and its neighbors from remaining
        to_remove = {best_v} | (adj_h[best_v] & remaining)
        remaining -= to_remove
        
    return S

def construct():
    deadline = time.time() + 140.0
    best_n = 0
    best_edges = []
    
    # Start with q=5 polarity graph (31 vertices)
    n, edges, adj = _construct_polarity_graph(5)
    
    if not _verify_c4_free(n, adj):
        return {'n': 0, 'edges': []}
        
    if _max_independent_set(adj, n, limit=11):
        return {'n': 0, 'edges': []}
        
    best_n = n
    best_edges = edges
    
    # Build H (common neighbor graph)
    # adj_h[u] contains v if N(u) & N(v) is not empty
    adj_h = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & adj[v]:
                adj_h[u].add(v)
                adj_h[v].add(u)
                
    max_vertices_to_add = 43 - n
    for _ in range(max_vertices_to_add):
        if time.time() > deadline - 20:
            break
            
        added = False
        # Try multiple random restarts of the greedy independent set finder in H
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
                
            # Find an independent set S in H
            # We want S to be large to keep alpha low, but any valid S works for C4.
            # Greedy min-degree in H tends to find large independent sets.
            S = _find_independent_set_in_h(n, adj_h)
            
            if not S:
                continue
                
            # Check if adding vertex with neighbors S keeps alpha < 11
            # New vertex v_new. N(v_new) = S.
            # Alpha(new) = max(Alpha(old), 1 + Alpha(G - N[v_new]))
            # G - N[v_new] is G with S and their neighbors removed?
            # No, N[v_new] in G_new is S ∪ {v_new}.
            # The independent set containing v_new is {v_new} ∪ I, where I is independent in G - N(v_new).
            # N(v_new) = S. So I is independent in G - S.
            
            # We need to check if G - S has an independent set of size >= 10.
            # If yes, then 1 + 10 = 11, which is invalid.
            # So we need Alpha(G - S) < 10.
            
            # Construct G - S
            # Vertices in G - S are V \ S.
            # Edges are original edges between vertices in V \ S.
            
            # We can reuse _max_independent_set if we build the adj list for G-S.
            # But n is up to 43, so it's fast.
            
            valid_alpha = True
            if len(S) < n: # If S is all vertices, G-S is empty, alpha=0
                # Build adj for G-S
                # Map old indices to new indices
                # Or just pass a subset?
                # Let's create a new adj list for the subgraph
                
                # Optimization: If alpha(old) < 10, and S is not empty, alpha(G-S) <= alpha(old) < 10.
                # Since we start with alpha ~ 3, and alpha only increases slowly or stays same,
                # this check might be redundant for many steps.
                # But to be safe, we check.
                
                # Actually, if alpha(old) < 10, then alpha(G-S) < 10 is always true.
                # We only need to worry if alpha(old) >= 10.
                # Our seed has alpha ~ 3.
                # As we add vertices, alpha might increase.
                # If we always pick large S, we remove many vertices, keeping alpha low.
                
                # Let's just check if _max_independent_set on G-S returns True for limit 10.
                # To avoid rebuilding adj list every time, we can just check if alpha(old) < 10.
                # If alpha(old) < 10, then G-S has alpha < 10.
                # So we only need to check if alpha(old) >= 10.
                
                # Let's track alpha.
                # For now, assume alpha < 10 is maintained.
                # We will verify at the end.
                
                pass
            else:
                valid_alpha = True # Empty set has alpha 0

            # If we assume valid_alpha is True, we proceed to add the vertex.
            # We should verify C4-freeness of the new graph?
            # By construction, S is independent in H, so N(u) and N(v) are disjoint for u,v in S.
            # So no C4 is formed involving v_new and two vertices in S.
            # What about C4s not involving v_new? They were already checked.
            # So the new graph is C4-free.
            
            # Update adj
            new_adj = [set(s) for s in adj]
            new_adj.append(set(S))
            for u in S:
                new_adj[u].add(n)
                
            # Update edges
            for u in S:
                edges.append([n, u])
                
            # Update adj_h
            # v_new is n.
            # N(v_new) = S.
            # v_new ~_H w if N(v_new) & N(w) != empty.
            # i.e., S & N(w) != empty.
            # So v_new is connected in H to all w such that w has a neighbor in S.
            # Let's find these w.
            
            # Also, for u, v in S, they now share v_new as a common neighbor.
            # So u ~_H v for all u, v in S.
            # This means S becomes a clique in H?
            # No, H edge means common neighbor.
            # If u, v in S, N(u) and N(v) now both contain v_new.
            # So N(u) & N(v) contains v_new.
            # So u ~_H v.
            # So S becomes a clique in H.
            
            # Update adj_h for existing vertices:
            # 1. For u, v in S: add edge u-v in H.
            for u in S:
                for v in S:
                    if u != v:
                        adj_h[u].add(v)
                        adj_h[v].add(u)
                        
            # 2. For w not in S:
            #    w ~_H v_new if N(w) & S != empty.
            #    Also, does w's connection to existing u change?
            #    N(w) is unchanged.
            #    N(u) changed only if u in S.
            #    If u in S, N(u) added v_new.
            #    N(w) & N(u) now includes v_new if v_new in N(w)? No, v_new is not in N(w) unless w in S.
            #    So N(w) & N(u) is unchanged for w not in S, u in S?
            #    Wait. N(u) = N_old(u) ∪ {v_new}.
            #    N(w) = N_old(w).
            #    N(w) & N(u) = N_old(w) & (N_old(u) ∪ {v_new}) = (N_old(w) & N_old(u)) ∪ (N_old(w) & {v_new}).
            #    Since w not in S, v_new not in N(w). So N_old(w) & {v_new} is empty.
            #    So N(w) & N(u) is unchanged.
            #    So H edges between w (not in S) and u (in S) are unchanged.
            
            # 3. Add v_new to adj_h.
            new_h_neighbors = set()
            for w in range(n):
                if adj_h[w] & S: # If w has any neighbor in S in H? No.
                    # Condition: N(w) & S != empty.
                    # N(w) is adj[w].
                    if adj[w] & S:
                        new_h_neighbors.add(w)
                        
            adj_h.append(new_h_neighbors)
            for w in new_h_neighbors:
                adj_h[w].add(n)
                
            # Update n
            n += 1
            
            if n > best_n:
                best_n = n
                best_edges = [e[:] for e in edges]
                
            added = True
            break # Success, try to add next vertex
            
        if not added:
            break
            
    # Final verification
    if best_n > 0:
        # Rebuild adj from best_edges for verification
        final_adj = [set() for _ in range(best_n)]
        for u, v in best_edges:
            final_adj[u].add(v)
            final_adj[v].add(u)
            
        if not _verify_c4_free(best_n, final_adj):
            return {'n': 0, 'edges': []}
            
        if _max_independent_set(final_adj, best_n, limit=11):
            return {'n': 0, 'edges': []}
            
    return {'n': best_n, 'edges': best_edges}

