import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    d = 7
    n = 1 << d
    total_edges = d * (1 << (d - 1))  # 448
    
    # Build all edges
    all_edges = []
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges.append((x, y))
    
    # Build all 4-cycles
    four_cycles = []
    for v in range(n):
        for i in range(d):
            for j in range(i + 1, d):
                e1 = v ^ (1 << i)
                e2 = v ^ (1 << j)
                e3 = v ^ (1 << i) ^ (1 << j)
                edges = [
                    (min(v, e1), max(v, e1)),
                    (min(v, e2), max(v, e2)),
                    (min(e1, e3), max(e1, e3)),
                    (min(e2, e3), max(e2, e3)),
                ]
                four_cycles.append(edges)
    
    # Map edge index
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    
    # For each edge, list of cycle indices it belongs to
    edge_cycles = [[] for _ in range(len(all_edges))]
    for ci, cycle in enumerate(four_cycles):
        for e in cycle:
            edge_cycles[edge_to_idx[e]].append(ci)
    
    # Greedy with random ordering
    def greedy_random(seed):
        rng = random.Random(seed)
        order = list(range(len(all_edges)))
        rng.shuffle(order)
        chosen = [False] * len(all_edges)
        # cycle_count[c] = number of edges in cycle c that are chosen
        cycle_count = [0] * len(four_cycles)
        
        for ei in order:
            e = all_edges[ei]
            # Check if adding this edge completes any cycle
            ok = True
            for ci in edge_cycles[ei]:
                if cycle_count[ci] == 3:  # other 3 edges already chosen
                    ok = False
                    break
            if ok:
                chosen[ei] = True
                for ci in edge_cycles[ei]:
                    cycle_count[ci] += 1
        return chosen, cycle_count
    
    # Try multiple greedy constructions
    best_chosen = None
    best_count = 0
    for s in range(30):
        if time.time() > deadline - 40:
            break
        chosen, _ = greedy_random(s)
        cnt = sum(chosen)
        if cnt > best_count:
            best_count = cnt
            best_chosen = chosen[:]
    
    if best_chosen is None:
        best_chosen, _ = greedy_random(0)
        best_count = sum(best_chosen)
    
    # Local search: try to add edges, then 1-flip and 2-flip
    def local_search(chosen, max_time=100.0):
        n_edges = len(all_edges)
        n_cycles = len(four_cycles)
        cycle_count = [0] * n_cycles
        for ei in range(n_edges):
            if chosen[ei]:
                for ci in edge_cycles[ei]:
                    cycle_count[ci] += 1
        
        # First, try to add any edge that doesn't complete a cycle
        changed = True
        while changed:
            changed = False
            for ei in range(n_edges):
                if not chosen[ei]:
                    ok = True
                    for ci in edge_cycles[ei]:
                        if cycle_count[ci] == 3:
                            ok = False
                            break
                    if ok:
                        chosen[ei] = True
                        for ci in edge_cycles[ei]:
                            cycle_count[ci] += 1
                        changed = True
        
        # Now try 2-flip moves
        end_time = time.time() + min(max_time, deadline - time.time() - 10)
        rng = random.Random(42)
        
        for _ in range(10000):
            if time.time() > end_time:
                break
            
            # Pick two random chosen edges to remove
            chosen_indices = [i for i in range(n_edges) if chosen[i]]
            if len(chosen_indices) < 2:
                break
            
            e1, e2 = rng.sample(chosen_indices, 2)
            
            # Remove them
            for ei in (e1, e2):
                chosen[ei] = False
                for ci in edge_cycles[ei]:
                    cycle_count[ci] -= 1
            
            # Try to add two random unchosen edges
            unchosen = [i for i in range(n_edges) if not chosen[i]]
            if len(unchosen) < 2:
                # Put them back
                for ei in (e1, e2):
                    chosen[ei] = True
                    for ci in edge_cycles[ei]:
                        cycle_count[ci] += 1
                continue
            
            improved = False
            for _ in range(3):
                a, b = rng.sample(unchosen, 2)
                # Check if adding a and b is valid
                # Temporarily mark them
                valid = True
                # Check cycles affected by a
                for ci in edge_cycles[a]:
                    if cycle_count[ci] + (1 if chosen[b] else 0) + (1 if ci in edge_cycles[b] else 0) >= 4:
                        # Actually need to check: after adding a and b, is any cycle complete?
                        pass
                # Simpler: just check all cycles that contain a or b
                affected = set(edge_cycles[a] + edge_cycles[b])
                for ci in affected:
                    cnt = cycle_count[ci]
                    # Add contribution from a if not already counted
                    if all_edges[a] in four_cycles[ci]:
                        cnt += 1
                    if all_edges[b] in four_cycles[ci]:
                        cnt += 1
                    if cnt >= 4:
                        valid = False
                        break
                
                if valid:
                    chosen[a] = True
                    chosen[b] = True
                    for ci in edge_cycles[a]:
                        cycle_count[ci] += 1
                    for ci in edge_cycles[b]:
                        cycle_count[ci] += 1
                    improved = True
                    break
            
            if not improved:
                # Put e1, e2 back
                for ei in (e1, e2):
                    chosen[ei] = True
                    for ci in edge_cycles[ei]:
                        cycle_count[ci] += 1
        
        return chosen
    
    best_chosen = local_search(best_chosen[:], max_time=80.0)
    best_count = sum(best_chosen)
    
    # Try a few more random restarts with local search
    for s in range(30, 50):
        if time.time() > deadline - 30:
            break
        chosen, _ = greedy_random(s)
        improved = local_search(chosen, max_time=30.0)
        cnt = sum(improved)
        if cnt > best_count:
            best_count = cnt
            best_chosen = improved
    
    # Final: try to add any remaining edges
    cycle_count = [0] * len(four_cycles)
    for ei in range(len(all_edges)):
        if best_chosen[ei]:
            for ci in edge_cycles[ei]:
                cycle_count[ci] += 1
    for ei in range(len(all_edges)):
        if not best_chosen[ei]:
            ok = True
            for ci in edge_cycles[ei]:
                if cycle_count[ci] == 3:
                    ok = False
                    break
            if ok:
                best_chosen[ei] = True
                for ci in edge_cycles[ei]:
                    cycle_count[ci] += 1
    
    result = [[all_edges[ei][0], all_edges[ei][1]] for ei in range(len(all_edges)) if best_chosen[ei]]
    return result
