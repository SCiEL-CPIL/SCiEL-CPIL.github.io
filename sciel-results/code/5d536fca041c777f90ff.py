import time
import random

N = 40

def _validate(edges, cycles):
    n = N
    edge_set = set()
    for u, v in edges:
        if not (0 <= u < n and 0 <= v < n) or u == v:
            return False
        if u > v:
            u, v = v, u
        if (u, v) in edge_set:
            return False
        edge_set.add((u, v))

    if len(cycles) != n - 2:
        return False

    for i, cycle in enumerate(cycles):
        L = i + 3
        if len(cycle) != L:
            return False
        if len(set(cycle)) != L:
            return False
        for j in range(L):
            u = cycle[j]
            v = cycle[(j + 1) % L]
            if u > v:
                u, v = v, u
            if (u, v) not in edge_set:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    # Target: Find S subset of {1..39} such that S-S covers 1..38
    # Minimize |S|
    
    # Initial greedy S
    S = set()
    covered = set()
    # Add 1 and 39 to ensure L=40 cycle
    S.add(1)
    S.add(39)
    covered.add(38)
    
    # Greedy add
    while len(covered) < 38:
        missing = [d for d in range(1, 39) if d not in covered]
        if not missing:
            break
        # Pick x that covers most missing
        best_x = -1
        best_gain = -1
        for x in range(1, 40):
            if x in S:
                continue
            gain = 0
            for y in S:
                d = abs(x - y)
                if d in missing:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_x = x
        if best_x == -1:
            break
        S.add(best_x)
        for y in S:
            d = abs(best_x - y)
            if d <= 38:
                covered.add(d)
                
    best_S = set(S)
    best_size = len(best_S)
    
    # Local Search to reduce size
    # We try to remove elements or swap.
    # Since we want to minimize size, we try to remove first.
    # If we can't remove, we try to swap to enable removal later?
    # Actually, just try to remove random elements if coverage is maintained.
    # And try to add elements to allow removal of others?
    # Simpler: Randomized restarts of the greedy or just local search on the set.
    
    def check_coverage(S_set):
        cov = set()
        S_list = sorted(S_set)
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                d = S_list[j] - S_list[i]
                if d <= 38:
                    cov.add(d)
        return cov == set(range(1, 39))
        
    current_S = set(best_S)
    
    while time.time() < deadline - 5:
        # Try to remove
        removed = False
        candidates = list(current_S)
        random.shuffle(candidates)
        for x in candidates:
            new_S = current_S - {x}
            if check_coverage(new_S):
                current_S = new_S
                if len(current_S) < best_size:
                    best_size = len(current_S)
                    best_S = set(current_S)
                removed = True
                break
        if removed:
            continue
            
        # If no removal, try a random swap to potentially open up removals
        # Or just restart with a slightly different greedy?
        # Let's try a random swap
        if len(current_S) > 1:
            x = random.choice(list(current_S))
            y = random.randint(1, 39)
            if y not in current_S:
                new_S = (current_S - {x}) | {y}
                # Only accept if it doesn't increase size (it doesn't) 
                # and maybe has better potential? 
                # For simplicity, accept if it covers all (it might not)
                # But we want to maintain validity.
                # If it doesn't cover, we can't use it as a base for removal.
                # So we only swap if it maintains coverage?
                # That's a very strict constraint.
                # Maybe just restart the greedy with a random seed?
                pass
                
    # Use best_S found
    S_final = best_S
    
    # Build Edges
    path_edges = [(i, i + 1) for i in range(1, N - 1)]
    spoke_edges = [(0, s) for s in S_final]
    edges = path_edges + spoke_edges
    
    # Build Cycles
    cycles = []
    S_list = sorted(S_final)
    S_set = set(S_final)
    
    for L in range(3, N + 1):
        if L == N:
            # 0, 1, ..., 39
            cycles.append(list(range(N)))
        else:
            diff = L - 2
            # Find u in S such that u+diff in S
            found = False
            for u in S_list:
                v = u + diff
                if v in S_set:
                    cycle_verts = [0] + list(range(u, v + 1))
                    cycles.append(cycle_verts)
                    found = True
                    break
            if not found:
                # Fallback: This should not happen if S covers all diffs
                # But if it does, we need a valid cycle.
                # We can't easily construct one without searching.
                # If this happens, we return a valid but larger graph (59 edges)
                # to ensure validity.
                ham_edges = [(i, (i + 1) % N) for i in range(N)]
                chord_edges = [(0, d) for d in range(2, 21)]
                fallback_edges = ham_edges + chord_edges
                fallback_cycles = []
                for Ll in range(3, N + 1):
                    if Ll == N:
                        fallback_cycles.append(list(range(N)))
                    elif Ll <= 21:
                        fallback_cycles.append(list(range(Ll)))
                    else:
                        end_vertex = N + 1 - Ll
                        fallback_cycles.append([0] + list(range(N - 1, end_vertex - 1, -1)))
                return {'edges': fallback_edges, 'cycles': fallback_cycles}

    if not _validate(edges, cycles):
        # Fallback to 59-edge solution
        ham_edges = [(i, (i + 1) % N) for i in range(N)]
        chord_edges = [(0, d) for d in range(2, 21)]
        fallback_edges = ham_edges + chord_edges
        fallback_cycles = []
        for Ll in range(3, N + 1):
            if Ll == N:
                fallback_cycles.append(list(range(N)))
            elif Ll <= 21:
                fallback_cycles.append(list(range(Ll)))
            else:
                end_vertex = N + 1 - Ll
                fallback_cycles.append([0] + list(range(N - 1, end_vertex - 1, -1)))
        return {'edges': fallback_edges, 'cycles': fallback_cycles}

    return {'edges': edges, 'cycles': cycles}
