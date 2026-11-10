import itertools
import random
import time
import sys

sys.setrecursionlimit(10000)

def _is_non_2_colorable_fast(n, edges):
    if not edges:
        return False

    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)

    full_mask = (1 << n) - 1

    vertex_degree = [0] * n
    incident_edges = [[] for _ in range(n)]
    for i, mask in enumerate(edge_masks):
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            vertex_degree[idx] += 1
            incident_edges[idx].append(i)
            m ^= lsb

    def propagate(color1, unassigned):
        changed = True
        while changed:
            changed = False
            for mask in edge_masks:
                assigned_in_edge = mask & ~unassigned
                unassigned_in_edge = mask & unassigned
                
                if unassigned_in_edge == 0:
                    if (mask & ~color1 & full_mask) == 0:
                        return None, None, True
                    if (mask & color1) == 0:
                        return None, None, True
                    continue

                if unassigned_in_edge and (unassigned_in_edge & (unassigned_in_edge - 1)) == 0:
                    other_4 = mask ^ unassigned_in_edge
                    if (other_4 & ~color1 & full_mask) == 0 and (other_4 & color1) == other_4:
                        unassigned ^= unassigned_in_edge
                        changed = True
                    elif (other_4 & color1) == 0:
                        color1 |= unassigned_in_edge
                        unassigned ^= unassigned_in_edge
                        changed = True

        return color1, unassigned, False

    def backtrack(color1, unassigned):
        color1, unassigned, conflict = propagate(color1, unassigned)
        if conflict:
            return False

        if unassigned == 0:
            for mask in edge_masks:
                if (mask & ~color1 & full_mask) == 0:
                    return False
                if (mask & color1) == 0:
                    return False
            return True

        best_v_bit = 0
        best_score = -1
        
        m = unassigned
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            score = 0
            for e_idx in incident_edges[idx]:
                mask = edge_masks[e_idx]
                if mask & lsb:
                    un_in_edge = mask & unassigned
                    if un_in_edge == lsb:
                        other_4 = mask ^ lsb
                        if (other_4 & ~color1 & full_mask) == 0:
                            score += 1
                        elif (other_4 & color1) == 0:
                            score += 1
            if score > best_score or (score == best_score and vertex_degree[idx] > (vertex_degree[best_v_bit.bit_length() - 1] if best_v_bit else 0)):
                best_score = score
                best_v_bit = lsb
            m ^= lsb
        
        v_bit = best_v_bit

        if backtrack(color1 | v_bit, unassigned ^ v_bit):
            return True

        if backtrack(color1, unassigned ^ v_bit):
            return True

        return False

    initial_color1 = 0
    initial_unassigned = full_mask ^ 1

    initial_color1, initial_unassigned, conflict = propagate(initial_color1, initial_unassigned)
    if conflict:
        return True

    if initial_unassigned == 0:
        for mask in edge_masks:
            if (mask & ~initial_color1 & full_mask) == 0:
                return False
            if (mask & initial_color1) == 0:
                return False
        return False

    if backtrack(initial_color1, initial_unassigned):
        return False

    return True

def _greedy_remove(n, edges, deadline, rng):
    current_edges = edges[:]
    removed_any = True
    while removed_any and (time.time() < deadline):
        removed_any = False
        indices = list(range(len(current_edges)))
        rng.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            if i >= len(current_edges):
                continue
            e = current_edges[i]
            new_edges = current_edges[:i] + current_edges[i + 1:]
            if _is_non_2_colorable_fast(n, new_edges):
                current_edges = new_edges
                removed_any = True

    return current_edges

def construct():
    start_time = time.time()
    deadline = start_time + 145

    best_edges = None
    best_len = float('inf')
    best_n = 9

    # Focus on n=9
    n = 9
    all_edges = [list(e) for e in itertools.combinations(range(n), 5)]
    
    # More restarts
    num_restarts = 50
    
    for seed in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        rng = random.Random(seed * 100 + n)
        reduced = _greedy_remove(n, all_edges, deadline, rng)

        if len(reduced) < best_len:
            best_len = len(reduced)
            best_edges = reduced[:]
            best_n = n

        if best_len <= 55:
            break

    if best_edges is None:
        n = 9
        best_edges = [list(e) for e in itertools.combinations(range(n), 5)]
        best_n = n

    if not _is_non_2_colorable_fast(best_n, best_edges):
        n = 9
        best_edges = [list(e) for e in itertools.combinations(range(n), 5)]
        best_n = n

    return {'n': best_n, 'edges': best_edges}
