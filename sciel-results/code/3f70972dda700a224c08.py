"""Packing edge-disjoint cycles on one vertex set."""
# EVOLVE-BLOCK-START
import time
import sys

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

DEADLINE = 85  # Safety margin before 90s kill

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    n = P['n']
    edges_list = P['edges']
    m = len(edges_list)
    
    # Build edge index mapping
    edge_index = {}
    for i, (u, v) in enumerate(edges_list):
        if u > v:
            u, v = v, u
        edge_index[(u, v)] = i
    
    # Adjacency with edge indices
    adj = [set() for _ in range(n)]
    adj_edges = [[] for _ in range(n)]  # list of (neighbor, edge_idx)
    for idx, (u, v) in enumerate(edges_list):
        adj[u].add(v)
        adj[v].add(u)
        adj_edges[u].append((v, idx))
        adj_edges[v].append((u, idx))
        
    # Sort adjacency lists for deterministic and potentially faster search
    # We can sort by the edge index or neighbor
    for v in range(n):
        adj_edges[v].sort()

    # Theoretical max cycles for 6-regular graph is 3
    max_possible_cycles = 3
    
    # We will try to find 3 edge-disjoint Hamiltonian cycles
    # If we can't find 3, we return fewer (but we expect to find 3)
    
    def find_hamiltonian_cycles(k_target):
        """Find up to k_target edge-disjoint Hamiltonian cycles."""
        used_edges = [False] * m
        cycles = []
        
        # Pre-compute edge masks for faster checking? 
        # Python ints are fast enough for 70 bits.
        
        def dfs_cycle(start, path, visited_mask, used_edges_ref, depth, remaining_deg):
            """
            Find a Hamiltonian cycle starting at 'start'.
            remaining_deg[v] is the number of unused edges incident to v.
            """
            if time.time() > deadline:
                return None
            
            if depth == n:
                # Check if we can close: last vertex connected to start
                last = path[-1]
                key = (min(last, start), max(last, start))
                if key in edge_index:
                    eidx = edge_index[key]
                    if not used_edges_ref[eidx]:
                        return list(path)
                return None
            
            last = path[-1]
            
            # Heuristic: Choose next vertex. 
            # We are forced to continue from 'last'.
            # We should try neighbors that have lower remaining degree first (MRV heuristic on the neighbor's options).
            # Actually, since we are at 'last', we just iterate its neighbors.
            # But we can order them by the remaining degree of the neighbor.
            
            candidates = []
            for neighbor, eidx in adj_edges[last]:
                if visited_mask & (1 << neighbor):
                    continue
                if used_edges_ref[eidx]:
                    continue
                # Check if neighbor has enough remaining degree to complete the cycle
                # It needs to connect to all other unvisited vertices eventually.
                # A simple check: remaining_deg[neighbor] must be at least 2 (one in, one out) 
                # unless it's the last one, but we haven't reached the last one yet.
                if remaining_deg[neighbor] < 2:
                    continue
                candidates.append((remaining_deg[neighbor], neighbor, eidx))
            
            # Sort by remaining degree (lowest first)
            candidates.sort()
            
            for rem_deg, neighbor, eidx in candidates:
                visited_mask_new = visited_mask | (1 << neighbor)
                path.append(neighbor)
                used_edges_ref[eidx] = True
                
                # Update remaining degrees
                remaining_deg[last] -= 1
                remaining_deg[neighbor] -= 1
                
                # Check if any vertex now has degree < 2 and is unvisited?
                # If an unvisited vertex has degree < 2, it can't be part of the cycle.
                # But we already checked the neighbor. Other vertices might be affected?
                # No, only 'last' and 'neighbor' changed.
                
                # Forward checking: 
                # If we have visited n-1 vertices, the last unvisited vertex must be connected to 'start' and 'neighbor'.
                if depth + 1 == n - 1:
                    # Find the last unvisited vertex
                    unvisited = []
                    for v in range(n):
                        if not (visited_mask_new & (1 << v)):
                            unvisited.append(v)
                    if len(unvisited) == 1:
                        last_unvisited = unvisited[0]
                        # Check if last_unvisited is connected to 'start' and 'neighbor'
                        key1 = (min(last_unvisited, start), max(last_unvisited, start))
                        key2 = (min(last_unvisited, neighbor), max(last_unvisited, neighbor))
                        
                        e1 = edge_index.get(key1, -1)
                        e2 = edge_index.get(key2, -1)
                        
                        if e1 == -1 or e2 == -1 or used_edges_ref[e1] or used_edges_ref[e2]:
                            # Cannot close
                            pass # Backtrack
                        else:
                            # Valid, proceed (or we could just let the next step handle it)
                            pass
                    
                result = dfs_cycle(start, path, visited_mask_new, used_edges_ref, depth + 1, remaining_deg)
                
                # Undo
                remaining_deg[last] += 1
                remaining_deg[neighbor] += 1
                used_edges_ref[eidx] = False
                path.pop()
                
                if result is not None:
                    return result
            
            return None
        
        # Try to find cycles one by one
        for c in range(k_target):
            if time.time() > deadline - 5:
                break
            
            # Reset remaining degrees
            remaining_deg = [0] * n
            for v in range(n):
                for neighbor, eidx in adj_edges[v]:
                    if not used_edges[eidx]:
                        remaining_deg[v] += 1
            
            # Start vertex 0 is fixed.
            # We need to ensure vertex 0 has degree >= 2.
            if remaining_deg[0] < 2:
                break
                
            found = dfs_cycle(0, [0], 1, used_edges, 1, remaining_deg)
            
            if found is None:
                break
            
            # Mark edges of this cycle as used
            for i in range(n):
                u = found[i]
                v = found[(i + 1) % n]
                key = (min(u, v), max(u, v))
                eidx = edge_index[key]
                used_edges[eidx] = True
            
            cycles.append(found)
        
        return cycles

    best_cycles = find_hamiltonian_cycles(max_possible_cycles)
    
    if not best_cycles:
        # Fallback: Try to find any cycle if the strict Hamiltonian search failed due to time or dead ends
        # But for this graph, 3 Hamiltonian cycles exist.
        # If we fail, we return an empty list or a small cycle?
        # The contract says "nonempty list".
        # Let's try a simple triangle as a last resort to satisfy "nonempty" if we crash.
        # But we should be able to find 3.
        
        # Simple triangle fallback
        for i, (u, v) in enumerate(edges_list):
            if time.time() > deadline:
                break
            for w in adj[u] & adj[v]:
                if w != u and w != v:
                    return [[u, v, w]]
                    
    # Convert to list of lists of ints
    result = []
    for cycle in best_cycles:
        result.append([int(v) for v in cycle])
        
    return result

# EVOLVE-BLOCK-END
