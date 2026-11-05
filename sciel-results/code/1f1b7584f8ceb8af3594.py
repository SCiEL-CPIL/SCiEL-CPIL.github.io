import random
import time
import math

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill

    d = 7
    n = 2 ** d

    # Precompute all edges as (u, v, bit) where u < v
    all_edges = []
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges.append((x, y, i))
    
    # Precompute for each edge the list of (other_edge1, other_edge2, other_edge3)
    # such that {edge, oe1, oe2, oe3} forms a C4.
    edge_cycles = {}  # (x,y) -> list of tuples of 3 other edges
    for (x, y, bit_i) in all_edges:
        cycles = []
        for j in range(d):
            if j == bit_i:
                continue
            c = x ^ (1 << j)
            e = y ^ (1 << j)
            oe1 = (min(x ^ (1 << j), y ^ (1 << j)), max(x ^ (1 << j), y ^ (1 << j)))
            oe2 = (min(x, c), max(x, c))
            oe3 = (min(y, e), max(y, e))
            cycles.append((oe1, oe2, oe3))
        edge_cycles[(x, y)] = cycles

    def is_valid_add(chosen, edge):
        """Check if adding edge to chosen creates a C4."""
        x, y, bit_i = edge
        for (oe1, oe2, oe3) in edge_cycles[(x, y)]:
            if oe1 in chosen and oe2 in chosen and oe3 in chosen:
                return False
        return True

    def greedy_randomized(seed):
        """Run greedy with randomized edge order."""
        rng = random.Random(seed)
        edges = list(all_edges)
        rng.shuffle(edges)

        chosen = set()
        out = []

        for edge in edges:
            x, y, bit_i = edge
            if is_valid_add(chosen, edge):
                chosen.add((x, y))
                out.append([x, y])

        return out, chosen

    def exhaustive_add(chosen, out):
        """Try to add any valid edge repeatedly until no more can be added."""
        improved = True
        while improved:
            improved = False
            for edge in all_edges:
                x, y, bit_i = edge
                if (x, y) not in chosen and is_valid_add(chosen, edge):
                    chosen.add((x, y))
                    out.append([x, y])
                    improved = True
        return out, chosen

    def local_search(out, chosen, rng, start_time, time_limit):
        """
        Local search with 1-1 swaps and additions.
        Moves:
        1. Try to add a random edge. If valid, accept.
        2. 1-1 swap: remove edge A, add edge B (B != A). Accept if net gain >= 0.
        3. 1-2 swap: remove edge A, try to add 2 edges. Accept if net gain > 0.
        """
        current_score = len(chosen)
        best_score = current_score
        best_out = [list(e) for e in out]
        
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.9995
        
        while time.time() - start_time < time_limit - 2.0:
            # Try to add a random edge
            added = False
            for _ in range(15):
                candidate = rng.choice(all_edges)
                cx, cy, cbit = candidate
                if (cx, cy) not in chosen:
                    if is_valid_add(chosen, candidate):
                        chosen.add((cx, cy))
                        out.append([cx, cy])
                        current_score += 1
                        added = True
                        if current_score > best_score:
                            best_score = current_score
                            best_out = [list(e) for e in out]
                        break
            
            if not added:
                # Stuck. Try local moves.
                if out:
                    # 1-1 swap: remove one edge, try to add a different one
                    swap_done = False
                    for _ in range(30):
                        idx = rng.randint(0, len(out) - 1)
                        ex, ey = out[idx]
                        edge_key = (ex, ey)
                        
                        # Try to find an edge that becomes valid after removing edge_key
                        # We need an edge B such that:
                        # - B is not in chosen (or B == edge_key, but we're removing it)
                        # - After removing edge_key, B is valid
                        # - B is not edge_key
                        # Efficient: check a few random candidates
                        for _ in range(5):
                            candidate = rng.choice(all_edges)
                            cx, cy, cbit = candidate
                            cand_key = (cx, cy)
                            if cand_key == edge_key:
                                continue
                            if cand_key in chosen:
                                continue
                            # Check if candidate is valid after removing edge_key
                            # We can simulate: temporarily remove edge_key
                            chosen.discard(edge_key)
                            valid = is_valid_add(chosen, candidate)
                            if valid:
                                # Accept the swap: we removed edge_key and add candidate
                                # Net change: 0 (removed 1, added 1)
                                # But this might enable future additions
                                chosen.add(cand_key)
                                out[idx] = [cx, cy]
                                swap_done = True
                                break
                            else:
                                # Restore
                                chosen.add(edge_key)
                        
                        if swap_done:
                            break
                    
                    if not swap_done:
                        # 1-2 swap: remove one edge, try to add 2 edges
                        idx = rng.randint(0, len(out) - 1)
                        ex, ey = out[idx]
                        edge_key = (ex, ey)
                        
                        # Remove
                        chosen.discard(edge_key)
                        out.pop(idx)
                        current_score -= 1
                        
                        # Try to add 2 edges
                        added_count = 0
                        added_edges = []
                        for _ in range(30):
                            candidate = rng.choice(all_edges)
                            cx, cy, cbit = candidate
                            if (cx, cy) not in chosen:
                                if is_valid_add(chosen, candidate):
                                    chosen.add((cx, cy))
                                    out.append([cx, cy])
                                    added_edges.append((cx, cy))
                                    added_count += 1
                                    current_score += 1
                                    if current_score > best_score:
                                        best_score = current_score
                                        best_out = [list(e) for e in out]
                                    if added_count >= 2:
                                        break
                        
                        # If we didn't recover, we are at a lower score.
                        # This is allowed in SA, but we track best.
            
            # Cool down
            T *= cooling_rate
            if T < T_min:
                T = T_min

        # Final exhaustive add pass
        best_out, _ = exhaustive_add(set(e for e in best_out), best_out)
        
        return best_out, len(best_out)

    # Run multiple randomized greedy restarts
    best_result = None
    best_count = 0

    num_restarts = 80
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit - 15.0:
            break

        rng = random.Random(restart * 12345 + 67890)
        out, chosen = greedy_randomized(restart)

        # Exhaustive add after greedy
        out, chosen = exhaustive_add(chosen, out)

        if len(out) > best_count:
            best_count = len(out)
            best_result = [list(e) for e in out]

        # Local search improvement
        if time.time() - start_time < time_limit - 10.0:
            out, score = local_search(out, chosen, rng, start_time, time_limit - 5.0)
            if score > best_count:
                best_count = score
                best_result = out

    if best_result is None:
        # Fallback to deterministic greedy
        chosen = set()
        best_result = []
        for edge in all_edges:
            x, y, bit_i = edge
            if is_valid_add(chosen, edge):
                chosen.add((x, y))
                best_result.append([x, y])

    return best_result
