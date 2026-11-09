import time
import sys

def construct():
    # Seed data
    P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}
    n = P['n']
    edges = P['edges']
    
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    # Build adjacency bitmask
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Precompute valid_part[mask]: True if mask is a clique or independent set
    size = 1 << n
    valid_part = [False] * size
    valid_part[0] = True  # Empty set is valid (but we won't use it as a part)
    
    # Efficiently compute valid_part using incremental approach
    # A set S is a clique iff for all v in S, adj[v] & (S \ {v}) == S \ {v}
    # A set S is independent iff for all v in S, adj[v] & (S \ {v}) == 0
    
    # Precompute is_clique and is_independent for all masks
    is_clique = [False] * size
    is_independent = [False] * size
    is_clique[0] = True
    is_independent[0] = True
    
    for mask in range(1, size):
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # For clique: v must be adjacent to all in rest, and rest must be a clique
        is_clique[mask] = is_clique[rest] and ((adj[v] & rest) == rest)
        
        # For independent: v must not be adjacent to any in rest, and rest must be independent
        is_independent[mask] = is_independent[rest] and ((adj[v] & rest) == 0)
        
        valid_part[mask] = is_clique[mask] or is_independent[mask]
    
    # Check deadline
    if time.time() > deadline:
        # Fallback: each vertex its own part
        return [[i] for i in range(n)]
    
    # Backtracking with branch-and-bound to find minimum cochromatic partition
    # State: assign vertices 0..n-1 to parts (cliques or independent sets)
    # Each part has a type: 'clique' or 'independent'
    # Symmetry breaking: first vertex goes to part 0
    
    best_solution = None
    best_num_parts = n + 1
    
    # parts[i] = list of vertices in part i
    # part_type[i] = 'clique' or 'independent'
    parts = []
    part_type = []
    
    def can_add_to_part(v, part_idx):
        """Check if vertex v can be added to part part_idx"""
        ptype = part_type[part_idx]
        part_verts = parts[part_idx]
        if not part_verts:
            return True
        if ptype == 'clique':
            # v must be adjacent to all vertices in the part
            for u in part_verts:
                if not (adj[v] & (1 << u)):
                    return False
            return True
        else:  # independent
            # v must not be adjacent to any vertex in the part
            for u in part_verts:
                if adj[v] & (1 << u):
                    return False
            return True
    
    def backtrack(v_idx, num_parts_used):
        nonlocal best_solution, best_num_parts
        
        if time.time() > deadline:
            return False
        
        # Pruning: if we already have a solution with <= num_parts_used parts, prune
        if num_parts_used >= best_num_parts:
            return False
        
        if v_idx == n:
            # Found a complete solution
            if num_parts_used < best_num_parts:
                best_num_parts = num_parts_used
                best_solution = [list(p) for p in parts]
            return True
        
        # Try adding vertex v_idx to existing parts
        for p_idx in range(num_parts_used):
            if can_add_to_part(v_idx, p_idx):
                parts[p_idx].append(v_idx)
                if backtrack(v_idx + 1, num_parts_used):
                    return True
                parts[p_idx].pop()
        
        # Try creating a new part (only if we haven't exceeded best)
        if num_parts_used < best_num_parts:
            # Symmetry breaking: if this is the first vertex, it must go to part 0
            # For subsequent new parts, we only create them if all existing parts were tried
            # and none worked (which is implicit in the loop above)
            
            # Create a new clique part
            parts.append([v_idx])
            part_type.append('clique')
            if backtrack(v_idx + 1, num_parts_used + 1):
                return True
            parts.pop()
            part_type.pop()
            
            # Create a new independent part
            parts.append([v_idx])
            part_type.append('independent')
            if backtrack(v_idx + 1, num_parts_used + 1):
                return True
            parts.pop()
            part_type.pop()
        
        return False
    
    # Start backtracking
    backtrack(0, 0)
    
    # If backtracking found a solution, use it
    if best_solution is not None:
        best_parts = best_solution
    else:
        # Fallback: greedy approach
        es = {tuple(e) for e in edges} | {tuple(reversed(e)) for e in edges}
        parts = []
        for v in range(n):
            for part in parts:
                trial = part + [v]
                hits = [(u, w) in es for i, u in enumerate(trial) for w in trial[i+1:]]
                if all(hits) or not any(hits):
                    part.append(v)
                    break
            else:
                parts.append([v])
        best_parts = parts
    
    # Verify the solution
    def verify(parts):
        seen = set()
        for part in parts:
            if not part:
                return False
            for v in part:
                if v in seen:
                    return False
                seen.add(v)
            # Check if part is clique or independent
            is_cli = True
            is_ind = True
            for i in range(len(part)):
                for j in range(i + 1, len(part)):
                    u, w = part[i], part[j]
                    if adj[u] & (1 << w):
                        is_ind = False
                    else:
                        is_cli = False
                    if not is_cli and not is_ind:
                        return False
        return len(seen) == n
    
    if not verify(best_parts):
        # Final fallback: each vertex its own part
        best_parts = [[i] for i in range(n)]
    
    return best_parts
