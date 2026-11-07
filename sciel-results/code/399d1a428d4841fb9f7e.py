import time
import math
from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmask
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    full_mask = (1 << n) - 1
    
    # Precompute is_clique and is_ind for all masks
    is_clique = [False] * (1 << n)
    is_ind = [False] * (1 << n)
    is_clique[0] = True
    is_ind[0] = True
    
    for mask in range(1, 1 << n):
        # Get lowest set bit
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # Check clique
        if is_clique[rest]:
            # v must be connected to all in rest
            if (adj[v] & rest) == rest:
                is_clique[mask] = True
            else:
                is_clique[mask] = False
        else:
            is_clique[mask] = False
            
        # Check independent
        if is_ind[rest]:
            # v must NOT be connected to any in rest
            if (adj[v] & rest) == 0:
                is_ind[mask] = True
            else:
                is_ind[mask] = False
        else:
            is_ind[mask] = False

    valid_masks = []
    for mask in range(1, 1 << n):
        if is_clique[mask] or is_ind[mask]:
            valid_masks.append(mask)
            
    # Remove masks that are subsets of other valid masks (dominated)
    # A mask S is dominated if there exists S' superset of S that is also valid
    # Since we want to minimize number of parts, larger sets are preferred
    # But we can't just remove subsets because a subset might be needed if the superset conflicts
    # Actually for cochromatic partition, if S is a clique and S' is a clique superset, S is dominated
    # Same for independent sets
    # But if S is clique and S' is independent, they are different types
    # Let's keep all valid masks for now and let the solver handle it
    
    # For MILP: binary variables x_{v,i} = 1 if vertex v is assigned to set i
    # But this is complex. Let's use a simpler approach:
    # Binary variables y_i = 1 if set i is used
    # x_{v,i} = 1 if vertex v is in set i
    # Constraints:
    # 1. Each vertex is in exactly one set: sum_i x_{v,i} = 1 for all v
    # 2. x_{v,i} <= y_i for all v, i
    # 3. If set i is used, it must be a valid set: this is encoded by only having valid sets as options
    #    But we need to ensure that if y_i = 1, then the set of vertices assigned to i is exactly set i
    #    This requires: x_{v,i} = 1 for all v in set_i if y_i = 1
    #    And x_{v,i} = 0 for all v not in set_i
    #    So: x_{v,i} = y_i for v in set_i
    #         x_{v,i} = 0 for v not in set_i
    #    This means x_{v,i} is determined by y_i if v is in set_i
    
    # Actually, let's use a different formulation:
    # Let the valid sets be indexed by i = 0, ..., m-1
    # y_i = 1 if set i is selected
    # Constraint: for each vertex v, sum_{i: v in set_i} y_i >= 1 (vertex is covered)
    # But this allows overlapping sets, which is not allowed
    # We need: for each vertex v, sum_{i: v in set_i} y_i = 1 (vertex is in exactly one set)
    # This works! Because if y_i = 1 and y_j = 1 and v is in both set_i and set_j,
    # then the sum would be 2, violating the constraint
    # So this formulation ensures each vertex is in exactly one selected set
    
    # Objective: minimize sum_i y_i
    
    m = len(valid_masks)
    if m == 0:
        # Should not happen
        return [[i] for i in range(n)]
    
    # Build constraint matrix A for: A @ y = b where b = 1 for each vertex
    # A[v, i] = 1 if vertex v is in valid_masks[i], else 0
    A = np.zeros((n, m), dtype=int)
    for i, mask in enumerate(valid_masks):
        mm = mask
        while mm:
            lsb = mm & -mm
            v = lsb.bit_length() - 1
            A[v, i] = 1
            mm ^= lsb
    
    # Objective: minimize sum y_i
    c = np.ones(m, dtype=float)
    
    # Constraints: A @ y = 1 (each vertex covered exactly once)
    # y_i in {0, 1}
    
    # Use MILP
    try:
        res = milp(
            c=c,
            constraints=LinearConstraint(A, 1, 1),
            integrality=np.ones(m, dtype=int),
            bounds=Bounds(0, 1)
        )
        
        if res.status == 0 and res.x is not None:
            # Extract solution
            selected = np.where(res.x > 0.5)[0]
            if len(selected) > 0:
                result = []
                for i in selected:
                    mask = valid_masks[i]
                    verts = []
                    mm = mask
                    while mm:
                        lsb = mm & -mm
                        idx = lsb.bit_length() - 1
                        verts.append(idx)
                        mm ^= lsb
                    result.append(verts)
                
                # Verify: all vertices covered exactly once
                all_verts = []
                for part in result:
                    all_verts.extend(part)
                if len(all_verts) == n and len(set(all_verts)) == n:
                    # Verify each part is a clique or independent set
                    for part in result:
                        part_set = set(part)
                        is_clq = True
                        is_ind = True
                        for u in part:
                            for v in part:
                                if u != v:
                                    if (adj[u] & (1 << v)) == 0:
                                        is_clq = False
                                    else:
                                        is_ind = False
                        if not (is_clq or is_ind):
                            break
                    else:
                        return result
                
                # If verification failed, fall through to backtracking
            else:
                # No sets selected, shouldn't happen
                pass
        else:
            # MILP failed, fall through to backtracking
            pass
    except Exception:
        # Fall through to backtracking
        pass
    
    # Fallback: original backtracking approach
    # Sort valid masks by size descending to try larger sets first
    valid_masks.sort(key=lambda x: x.bit_count(), reverse=True)
    
    # Precompute for each vertex v, the list of valid masks containing v
    valid_masks_by_vertex = [[] for _ in range(n)]
    for S in valid_masks:
        m2 = S
        while m2:
            lsb = m2 & -m2
            v = lsb.bit_length() - 1
            valid_masks_by_vertex[v].append(S)
            m2 ^= lsb

    def solve(rem, parts_left):
        if rem == 0:
            return True
        if parts_left == 0:
            return False
        
        # Lower bound
        if parts_left == 1:
            return is_clique[rem] or is_ind[rem]
            
        # Pick a vertex v in rem
        v_bit = rem & -rem
        v = v_bit.bit_length() - 1
        
        # Candidates: S in valid_masks_by_vertex[v] such that S & rem == S
        candidates = []
        for S in valid_masks_by_vertex[v]:
            if S & rem == S:
                candidates.append(S)
                
        if not candidates:
            return False
            
        # Sort by size descending
        candidates.sort(key=lambda x: x.bit_count(), reverse=True)
        
        # Additional pruning:
        # If the largest candidate size is S_max, we need at least ceil(popcount(rem)/S_max) parts.
        max_s = candidates[0].bit_count()
        if parts_left < math.ceil(rem.bit_count() / max_s):
            return False
            
        for S in candidates:
            if solve(rem ^ S, parts_left - 1):
                return True
        return False

    # Find minimum k
    best_k = None
    for k in range(1, n + 1):
        if solve(full_mask, k):
            best_k = k
            break
            
    if best_k is None:
        best_k = n # Should not happen
        
    # Reconstruct partition
    def find_partition(rem, parts_left, current_path):
        if rem == 0:
            return current_path
        
        if parts_left == 0:
            return None
            
        if parts_left == 1:
            if is_clique[rem] or is_ind[rem]:
                return current_path + [rem]
            return None

        v_bit = rem & -rem
        v = v_bit.bit_length() - 1
        
        candidates = []
        for S in valid_masks_by_vertex[v]:
            if S & rem == S:
                candidates.append(S)
                
        if not candidates:
            return None
            
        candidates.sort(key=lambda x: x.bit_count(), reverse=True)
        
        max_s = candidates[0].bit_count()
        if parts_left < math.ceil(rem.bit_count() / max_s):
            return None
            
        for S in candidates:
            res = find_partition(rem ^ S, parts_left - 1, current_path + [S])
            if res:
                return res
        return None

    partition_masks = find_partition(full_mask, best_k, [])
    
    # Convert masks to list of vertex lists
    result = []
    for mask in partition_masks:
        verts = []
        m2 = mask
        while m2:
            lsb = m2 & -m2
            idx = lsb.bit_length() - 1
            verts.append(idx)
            m2 ^= lsb
        result.append(verts)
        
    return result

if __name__ == "__main__":
    construct()
