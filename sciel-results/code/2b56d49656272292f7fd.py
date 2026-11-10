import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    best_n = None
    best_edges = None
    best_m = float('inf')
    
    def is_2_colorable(n, edges):
        clauses = []
        for e in edges:
            mask = 0
            for v in e:
                mask |= (1 << v)
            clauses.append((1, mask))
            clauses.append((0, mask))
            
        def solve(pos_mask, neg_mask, unassigned_mask, clauses):
            # Unit Propagation
            changed = True
            while changed:
                changed = False
                new_clauses = []
                for (type, mask) in clauses:
                    # Check satisfied
                    if type == 1:
                        if mask & pos_mask:
                            continue
                    else:
                        if mask & neg_mask:
                            continue
                    
                    unassigned_bits = mask & unassigned_mask
                    if unassigned_bits == 0:
                        return False
                    elif unassigned_bits & (unassigned_bits - 1) == 0:
                        v = unassigned_bits.bit_length() - 1
                        if type == 1:
                            pos_mask |= (1 << v)
                        else:
                            neg_mask |= (1 << v)
                        unassigned_mask &= ~(1 << v)
                        changed = True
                        break
                    else:
                        new_clauses.append((type, mask))
                        
                if changed:
                    clauses = new_clauses
                else:
                    clauses = new_clauses
                    break
            
            if unassigned_mask == 0:
                return True
            
            # Choose variable
            # Pick from first clause
            v = (clauses[0][1] & unassigned_mask).bit_length() - 1
            
            # Try 1
            if solve(pos_mask | (1 << v), neg_mask, unassigned_mask & ~(1 << v), clauses):
                return True
            
            # Try 0
            if solve(pos_mask, neg_mask | (1 << v), unassigned_mask & ~(1 << v), clauses):
                return True
            
            return False
        
        return solve(0, 0, (1 << n) - 1, clauses)

    # Search
    for n in range(10, 17):
        if time.time() > deadline:
            break
        for m in range(40, 151):
            if time.time() > deadline:
                break
            for _ in range(5):
                if time.time() > deadline:
                    break
                edges = []
                seen = set()
                while len(edges) < m:
                    e = tuple(sorted(random.sample(range(n), 5)))
                    if e not in seen:
                        seen.add(e)
                        edges.append(list(e))
                
                if not is_2_colorable(n, edges):
                    # Minimize
                    min_edges = edges[:]
                    for i in range(len(min_edges) - 1, -1, -1):
                        if time.time() > deadline:
                            break
                        trial = min_edges[:i] + min_edges[i+1:]
                        if not is_2_colorable(n, trial):
                            min_edges = trial
                            
                    if len(min_edges) < best_m:
                        best_m = len(min_edges)
                        best_n = n
                        best_edges = min_edges

    if best_edges is None:
        from itertools import combinations
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
        
    return {'n': best_n, 'edges': best_edges}
