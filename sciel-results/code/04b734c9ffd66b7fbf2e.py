import time
import random
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _is_valid_biclique(left, right, adj, remaining_edges):
    """Check that every cross pair is in remaining_edges (and hence in adj)."""
    left_set = set(left)
    right_set = set(right)
    if not left or not right:
        return False
    if left_set & right_set:
        return False
    for u in left:
        for v in right:
            if v not in adj[u]:
                return False
            # Check it's in remaining edges
            if (u, v) not in remaining_edges and (v, u) not in remaining_edges:
                return False
    return True


def _find_max_biclique(start_vertex, remaining_edges, adj, n):
    """Find a large biclique containing start_vertex on the left side."""
    # Try to grow a biclique with start_vertex on left
    best_left = {start_vertex}
    best_right = set()
    
    # Start with all neighbors of start_vertex as potential right side
    candidates_right = set(adj[start_vertex])
    candidates_right &= {v for u, v in remaining_edges} | {u for u, v in remaining_edges}
    
    # Greedily build: start with right = neighbors of start that have edges in remaining
    right = set()
    for v in sorted(candidates_right):
        # Check if v can be in right: all current left vertices must connect to v
        can_add = True
        for u in best_left:
            if (u, v) not in remaining_edges and (v, u) not in remaining_edges:
                can_add = False
                break
        if can_add:
            right.add(v)
    
    # Now try to add more left vertices
    changed = True
    while changed:
        changed = False
        for u in range(n):
            if u in best_left or u in right:
                continue
            can_add = True
            for v in right:
                if (u, v) not in remaining_edges and (v, u) not in remaining_edges:
                    can_add = False
                    break
            if can_add and right:
                best_left.add(u)
                changed = True
    
    if not right:
        return None, None
    
    return list(best_left), list(right)


def _greedy_partition(remaining_edges, adj, n, deadline):
    """Greedy: repeatedly extract largest biclique."""
    bicliques = []
    rem = set()
    for u, v in remaining_edges:
        rem.add((u, v))
        rem.add((v, u))
    
    while rem:
        if time.time() > deadline:
            break
        
        # Find the edge to start from
        edge = next(iter(rem))
        u, v = edge
        
        # Try multiple candidates: u as left, v as left, and some high-degree vertices
        candidates = [u, v]
        # Add a few random vertices from remaining edges
        all_verts = set()
        for a, b in rem:
            all_verts.add(a)
            all_verts.add(b)
        random.shuffle(all_verts)
        for w in all_verts[:5]:
            if w not in candidates:
                candidates.append(w)
        
        best_biclique = None
        best_size = -1
        
        for cand in candidates:
            left, right = _find_max_biclique(cand, rem, adj, n)
            if left and right:
                size = len(left) * len(right)
                if size > best_size:
                    best_size = size
                    best_biclique = (left, right)
        
        if best_biclique is None:
            # Fallback: single edge
            best_biclique = ([u], [v])
        
        left, right = best_biclique
        bicliques.append((left, right))
        
        # Remove covered edges
        left_set = set(left)
        right_set = set(right)
        to_remove = set()
        for a, b in rem:
            if (a in left_set and b in right_set) or (a in right_set and b in left_set):
                to_remove.add((a, b))
                to_remove.add((b, a))
        rem -= to_remove
    
    return bicliques


def _try_merge(bicliques, adj, n):
    """Try to merge pairs of bicliques if their union is still a valid biclique."""
    merged = True
    while merged:
        merged = False
        new_bicliques = []
        used = [False] * len(bicliques)
        
        for i in range(len(bicliques)):
            if used[i]:
                continue
            left_i, right_i = bicliques[i]
            left_set_i = set(left_i)
            right_set_i = set(right_i)
            
            for j in range(i + 1, len(bicliques)):
                if used[j]:
                    continue
                left_j, right_j = bicliques[j]
                
                # Try all 4 ways to merge:
                # (left_i ∪ left_j, right_i ∩ right_j) - if intersection nonempty
                # (left_i ∩ left_j, right_i ∪ right_j) - if intersection nonempty
                # (left_i ∪ left_j, right_i ∪ right_j) - check validity
                # (left_i, right_i ∪ right_j) if left_i disjoint from right_j and valid
                # (left_i ∪ left_j, right_i) if right_i disjoint from left_j and valid
                
                candidates = []
                
                # Option 1: union both sides
                l_union = left_set_i | set(left_j)
                r_union = right_set_i | set(right_j)
                if l_union and r_union and not (l_union & r_union):
                    candidates.append((list(l_union), list(r_union)))
                
                # Option 2: left union, right intersection
                r_inter = right_set_i & set(right_j)
                if l_union and r_inter and not (l_union & r_inter):
                    candidates.append((list(l_union), list(r_inter)))
                
                # Option 3: left intersection, right union
                l_inter = left_set_i & set(left_j)
                if l_inter and r_union and not (l_inter & r_union):
                    candidates.append((list(l_inter), list(r_union)))
                
                # Option 4: left_i, right union (if left_i disjoint from right_j)
                if left_set_i and r_union and not (left_set_i & r_union):
                    # Check left_i disjoint from right_j
                    if not (left_set_i & set(right_j)):
                        candidates.append((list(left_set_i), list(r_union)))
                
                # Option 5: left union, right_j (if right_j disjoint from left_i)
                if l_union and set(right_j) and not (l_union & set(right_j)):
                    if not (set(left_i) & set(right_j)):
                        candidates.append((list(l_union), list(right_j)))
                
                for cl, cr in candidates:
                    # Check validity
                    valid = True
                    cl_set = set(cl)
                    cr_set = set(cr)
                    if not cl or not cr or cl_set & cr_set:
                        continue
                    for a in cl:
                        for b in cr:
                            if b not in adj[a]:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        new_bicliques.append((cl, cr))
                        used[i] = True
                        used[j] = True
                        merged = True
                        break
                if merged:
                    break
            if merged:
                break
        
        if not merged:
            for i in range(len(bicliques)):
                if not used[i]:
                    new_bicliques.append(bicliques[i])
        
        bicliques = new_bicliques
    
    return bicliques


def _local_search(bicliques, adj, n, deadline):
    """Try to reduce biclique count by moving edges or restructuring."""
    # Simple local search: try to combine bicliques more aggressively
    # For now, just do the merge pass again
    bicliques = _try_merge(bicliques, adj, n)
    return bicliques


def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    n = P['n']
    adj = _build_adj()
    
    # Build remaining edges set
    remaining_edges = set()
    for u, v in P['edges']:
        remaining_edges.add((u, v))
        remaining_edges.add((v, u))
    
    # Run greedy partition with multiple restarts
    best_bicliques = None
    best_count = float('inf')
    
    # First run: deterministic greedy
    bicliques = _greedy_partition(remaining_edges, adj, n, deadline)
    bicliques = _try_merge(bicliques, adj, n)
    count = len(bicliques)
    if count < best_count:
        best_count = count
        best_bicliques = bicliques
    
    # Try a few random restarts if time permits
    random.seed(42)
    for restart in range(3):
        if time.time() > deadline - 10.0:
            break
        
        # Shuffle the order in which we process edges
        shuffled_edges = list(P['edges'])
        random.shuffle(shuffled_edges)
        
        # Build a shuffled remaining edges
        rem_shuffled = set()
        for u, v in shuffled_edges:
            rem_shuffled.add((u, v))
            rem_shuffled.add((v, u))
        
        bicliques2 = _greedy_partition(rem_shuffled, adj, n, deadline)
        bicliques2 = _try_merge(bicliques2, adj, n)
        count2 = len(bicliques2)
        if count2 < best_count:
            best_count = count2
            best_bicliques = bicliques2
    
    # Final local search / merge
    if time.time() < deadline - 5.0:
        best_bicliques = _try_merge(best_bicliques, adj, n)
    
    # Convert to required format: list of [left, right] with Python ints and lists
    result = []
    for left, right in best_bicliques:
        left_ints = [int(v) for v in left]
        right_ints = [int(v) for v in right]
        result.append([left_ints, right_ints])
    
    return result
