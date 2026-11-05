"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import heapq
from collections import defaultdict

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin before 120s kill

    d = 7
    n = 2 ** d

    # Precompute all edges as (u, v, bit) where u < v
    all_edges = []
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges.append((x, y, i))

    num_edges = len(all_edges)
    edge_index = {}
    for idx, (x, y, bit) in enumerate(all_edges):
        edge_index[(x, y)] = idx

    # Precompute for each edge the list of (other_edge1, other_edge2, other_edge3)
    # such that {edge, oe1, oe2, oe3} forms a C4.
    # We'll store this as: for edge e, list of (e1, e2, e3) where {e, e1, e2, e3} is a C4
    # We also need: for each pair of edges (e1, e2), which edges e form a C4 with them?
    # That's harder. Instead, for each edge e, store the list of C4s it's in.
    
    edge_cycles = [[] for _ in range(num_edges)]
    for idx, (x, y, bit_i) in enumerate(all_edges):
        for j in range(d):
            if j == bit_i:
                continue
            c = x ^ (1 << j)
            e = y ^ (1 << j)
            # The four edges of the C4:
            # (x, y, bit_i), (c, e, bit_i), (x, c, j), (y, e, j)
            e1_key = (min(x, y), max(x, y))  # already the edge
            e2_key = (min(c, e), max(c, e))
            e3_key = (min(x, c), max(x, c))
            e4_key = (min(y, e), max(y, e))
            
            e2_idx = edge_index.get(e2_key)
            e3_idx = edge_index.get(e3_key)
            e4_idx = edge_index.get(e4_key)
            
            if e2_idx is not None and e3_idx is not None and e4_idx is not None:
                edge_cycles[idx].append((e2_idx, e3_idx, e4_idx))

    def is_valid_add(chosen, edge_idx):
        """Check if adding edge_idx to chosen creates a C4."""
        for (e1, e2, e3) in edge_cycles[edge_idx]:
            if e1 in chosen and e2 in chosen and e3 in chosen:
                return False
        return True

    def greedy_mcs(seed):
        """
        Maximum Cardinality Search heuristic:
        Maintain a count of how many selected edges each unselected edge is "conflicting with".
        At each step, pick the edge with the minimum conflict count (least constrained).
        Add it, then update conflict counts of edges that share a C4 with it.
        """
        rng = random.Random(seed)
        
        # conflict_count[i] = number of selected edges j such that i and j share a C4
        # Actually, we want: for each unselected edge i, how many C4s would be "almost complete"
        # i.e., how many C4s involving i have all other 3 edges selected.
        # That's what is_valid_add checks. But we want a cheaper proxy.
        
        # Let's use: degree[i] = number of C4s that edge i is in that have >=1 other edge selected
        # No, simpler: count[i] = number of selected edges that share at least one C4 with i
        # Actually, the key constraint is: edge i cannot be added if ANY of its C4s has all 3 other edges selected.
        # So let's track for each C4, how many of its 4 edges are selected.
        # When a C4 has 3 selected, the 4th is blocked.
        
        # Alternative approach: track for each edge i, the number of "blocked C4s" it's in.
        # A C4 is "blocked for i" if the other 3 edges are all selected.
        # is_valid_add is true iff blocked_count[i] == 0.
        
        # Let's precompute all C4s as tuples of 4 edge indices.
        all_c4s = []
        seen_c4s = set()
        for idx in range(num_edges):
            x, y, bit_i = all_edges[idx]
            for j in range(d):
                if j == bit_i:
                    continue
                c = x ^ (1 << j)
                e = y ^ (1 << j)
                e2_key = (min(c, e), max(c, e))
                e3_key = (min(x, c), max(x, c))
                e4_key = (min(y, e), max(y, e))
                
                e2_idx = edge_index.get(e2_key)
                e3_idx = edge_index.get(e3_key)
                e4_idx = edge_index.get(e4_key)
                
                if e2_idx is not None and e3_idx is not None and e4_idx is not None:
                    c4 = tuple(sorted([idx, e2_idx, e3_idx, e4_idx]))
                    if c4 not in seen_c4s:
                        seen_c4s.add(c4)
                        all_c4s.append(c4)
        
        # For each edge, list of C4s it belongs to
        edge_to_c4s = defaultdict(list)
        for c4_idx, c4 in enumerate(all_c4s):
            for e in c4:
                edge_to_c4s[e].append(c4_idx)
        
        # c4_selected_count[c4_idx] = number of selected edges in this C4
        c4_selected_count = [0] * len(all_c4s)
        
        chosen = set()
        out = []
        
        # blocked_count[i] = number of C4s containing i where the other 3 are all selected
        # i.e., c4_selected_count[c4] == 3 and i is in that c4 and i is not selected
        blocked_count = [0] * num_edges
        
        unselected = set(range(num_edges))
        
        # Use a priority queue: (blocked_count, random_tiebreaker, edge_idx)
        # We want to pick the edge with minimum blocked_count
        heap = []
        for i in range(num_edges):
            heap.append((0, rng.random(), i))
        heapq.heapify(heap)
        
        in_heap = [True] * num_edges
        
        while unselected:
            if time.time() - start_time > time_limit - 5.0:
                break
            
            # Pop the edge with minimum blocked_count
            found = False
            while heap:
                bc, _, idx = heapq.heappop(heap)
                if not in_heap[idx] or idx not in unselected:
                    continue
                if bc != blocked_count[idx]:
                    # Stale entry, skip
                    continue
                # This is the best candidate
                found = True
                break
            
            if not found:
                # Rebuild heap
                heap = []
                for i in unselected:
                    heap.append((blocked_count[i], rng.random(), i))
                heapq.heapify(heap)
                if not heap:
                    break
                bc, _, idx = heapq.heappop(heap)
                in_heap[idx] = False
                if idx not in unselected:
                    continue
            
            in_heap[idx] = False
            unselected.discard(idx)
            
            # Check if actually valid (blocked_count should be 0, but verify)
            if blocked_count[idx] > 0:
                # Shouldn't happen if we maintain correctly, but safety
                continue
            
            chosen.add(idx)
            out.append([all_edges[idx][0], all_edges[idx][1]])
            
            # Update blocked_count for all edges in C4s that idx belongs to
            for c4_idx in edge_to_c4s[idx]:
                c4 = all_c4s[c4_idx]
                c4_selected_count[c4_idx] += 1
                new_count = c4_selected_count[c4_idx]
                
                if new_count == 3:
                    # One edge in this C4 is now blocked
                    for e in c4:
                        if e not in chosen:
                            blocked_count[e] += 1
                            if e in unselected:
                                heapq.heappush(heap, (blocked_count[e], rng.random(), e))
        
        return out, chosen

    def local_search_mcs(out, chosen, rng, time_remaining):
        """Local search on top of MCS result."""
        no_improve_count = 0
        best_size = len(out)
        
        while time.time() - start_time < time_limit - 3.0:
            # Try to add random edges
            added = False
            for _ in range(10):
                candidate = rng.choice(all_edges)
                cx, cy, cbit = candidate
                cidx = edge_index.get((min(cx, cy), max(cx, cy)))
                if cidx is not None and cidx not in chosen and is_valid_add(chosen, cidx):
                    chosen.add(cidx)
                    out.append([cx, cy])
                    added = True
                    break
            
            if added:
                no_improve_count = 0
                if len(out) > best_size:
                    best_size = len(out)
                continue
            
            # Try 1-remove, 2-add
            if out:
                idx = rng.randint(0, len(out) - 1)
                ex, ey = out[idx]
                edge_key = (ex, ey)
                eidx = edge_index.get((min(ex, ey), max(ex, ey)))
                
                if eidx is not None:
                    chosen.discard(eidx)
                    out.pop(idx)
                    original_len = len(out)
                    
                    added_count = 0
                    for _ in range(2):
                        found = False
                        for _ in range(30):
                            candidate = rng.choice(all_edges)
                            cx, cy, cbit = candidate
                            cidx = edge_index.get((min(cx, cy), max(cx, cy)))
                            if cidx is not None and cidx not in chosen and is_valid_add(chosen, cidx):
                                chosen.add(cidx)
                                out.append([cx, cy])
                                added_count += 1
                                found = True
                                break
                        if not found:
                            break
                    
                    if len(out) <= original_len:
                        # Revert
                        for _ in range(added_count):
                            last_edge = out.pop()
                            lidx = edge_index.get((min(last_edge[0], last_edge[1]), max(last_edge[0], last_edge[1])))
                            if lidx is not None:
                                chosen.discard(lidx)
                        chosen.add(eidx)
                        out.append([ex, ey])
                        no_improve_count += 1
                    else:
                        no_improve_count = 0
                        if len(out) > best_size:
                            best_size = len(out)
            
            # Kick
            if no_improve_count > 20:
                remove_count = min(5, len(out))
                for _ in range(remove_count):
                    idx = rng.randint(0, len(out) - 1)
                    removed = out.pop(idx)
                    ridx = edge_index.get((min(removed[0], removed[1]), max(removed[0], removed[1])))
                    if ridx is not None:
                        chosen.discard(ridx)
                
                # Re-add systematically
                for edge in all_edges:
                    x, y, bit_i = edge
                    eidx = edge_index.get((min(x, y), max(x, y)))
                    if eidx is not None and eidx not in chosen and is_valid_add(chosen, eidx):
                        chosen.add(eidx)
                        out.append([x, y])
                
                no_improve_count = 0
                if len(out) > best_size:
                    best_size = len(out)
        
        return out, chosen

    # Run multiple MCS restarts
    best_result = None
    best_count = 0

    num_restarts = 30
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit - 5.0:
            break

        rng = random.Random(restart * 7919 + 13)
        out, chosen = greedy_mcs(restart)

        if len(out) > best_count:
            best_count = len(out)
            best_result = [list(e) for e in out]

        # Local search improvement
        if time.time() - start_time < time_limit - 5.0:
            out, chosen = local_search_mcs(out, chosen, rng, time_limit - time.time() - start_time)
            if len(out) > best_count:
                best_count = len(out)
                best_result = [list(e) for e in out]

    # Final polish: try to add any remaining edges
    if best_result is not None:
        final_chosen = set()
        for e in best_result:
            final_chosen.add(edge_index.get((min(e[0], e[1]), max(e[0], e[1]))))
        
        for edge in all_edges:
            x, y, bit_i = edge
            eidx = edge_index.get((min(x, y), max(x, y)))
            if eidx is not None and eidx not in final_chosen and is_valid_add(final_chosen, eidx):
                best_result.append([x, y])
                final_chosen.add(eidx)

    if best_result is None:
        # Fallback to deterministic greedy
        chosen = set()
        best_result = []
        for edge in all_edges:
            x, y, bit_i = edge
            eidx = edge_index.get((min(x, y), max(x, y)))
            if eidx is not None and eidx not in chosen and is_valid_add(chosen, eidx):
                chosen.add(eidx)
                best_result.append([x, y])

    return best_result
# EVOLVE-BLOCK-END
