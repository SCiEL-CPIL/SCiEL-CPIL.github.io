import time
import random

_START = time.time()
_DEADLINE = 145

def _elapsed():
    return time.time() - _START

def _get_triangles(adj, n):
    """Return list of triangles (tuples of 3 vertices) in the graph."""
    triangles = []
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] & (1 << j)):
                continue
            common = adj[i] & adj[j] & ~((1 << (j + 1)) - 1)
            c = common
            while c:
                lsb = c & -c
                k = lsb.bit_length() - 1
                triangles.append((i, j, k))
                c ^= lsb
    return triangles

def _get_ind_5s(adj, n):
    """Return list of independent sets of size 5 in the graph."""
    ind_5s = []
    def backtrack(candidates, current_set):
        if len(current_set) == 5:
            ind_5s.append(tuple(current_set))
            return
        if candidates.bit_count() < 5 - len(current_set):
            return
        
        # Heuristic: pick vertex with max non-neighbors
        best_v = -1
        best_count = -1
        c = candidates
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            new_cand = candidates & ~adj[v] & ~(1 << v)
            cnt = new_cand.bit_count()
            if cnt > best_count:
                best_count = cnt
                best_v = v
            c ^= lsb
        
        if best_v == -1:
            return
            
        v = best_v
        backtrack(candidates & ~adj[v] & ~(1 << v), current_set + [v])
        backtrack(candidates & ~(1 << v), current_set)

    backtrack((1 << n) - 1, [])
    return ind_5s

def _try_extend(adj, n, time_limit):
    """
    Try to extend the graph on n-1 vertices to n vertices.
    """
    start = time.time()
    n_old = n - 1
    
    # Precompute constraints
    triangles = _get_triangles(adj, n_old)
    ind_5s = _get_ind_5s(adj, n_old)
    
    if not triangles and not ind_5s:
        # No constraints, just add isolated or complete vertex?
        # But we need to maintain K4-free and Ind6-free.
        # If no triangles, any S is K3-free.
        # If no ind 5s, any S is fine for Ind6.
        # So we can just pick a random S.
        pass

    # Local search on the adjacency vector of the new vertex
    # State: bitmask of neighbors of new vertex
    best_score = float('inf')
    best_state = None
    
    # Initialize with a few random states
    states = []
    for _ in range(10):
        state = 0
        for i in range(n_old):
            if random.random() < 0.5:
                state |= (1 << i)
        states.append(state)
        
    # Also try states based on heuristics
    # e.g., connect to non-neighbors of a random vertex
    
    for state in states:
        # Evaluate
        # Count triangles fully in state
        tri_violations = 0
        for t in triangles:
            if (state & (1 << t[0])) and (state & (1 << t[1])) and (state & (1 << t[2])):
                tri_violations += 1
                
        # Count ind 5s fully in complement
        comp_state = ((1 << n_old) - 1) & ~state
        ind_violations = 0
        for i in ind_5s:
            if (comp_state & (1 << i[0])) and (comp_state & (1 << i[1])) and (comp_state & (1 << i[2])) and (comp_state & (1 << i[3])) and (comp_state & (1 << i[4])):
                ind_violations += 1
                
        score = tri_violations + ind_violations
        if score < best_score:
            best_score = score
            best_state = state
            if score == 0:
                break

    if best_score > 0:
        # Perform local search
        current_state = best_state
        for step in range(1000):
            if time.time() - start > time_limit:
                break
            if best_score == 0:
                break
                
            # Flip a random bit
            bit = random.randint(0, n_old - 1)
            new_state = current_state ^ (1 << bit)
            
            # Evaluate new state
            tri_violations = 0
            for t in triangles:
                if (new_state & (1 << t[0])) and (new_state & (1 << t[1])) and (new_state & (1 << t[2])):
                    tri_violations += 1
            comp_state = ((1 << n_old) - 1) & ~new_state
            ind_violations = 0
            for i in ind_5s:
                if (comp_state & (1 << i[0])) and (comp_state & (1 << i[1])) and (comp_state & (1 << i[2])) and (comp_state & (1 << i[3])) and (comp_state & (1 << i[4])):
                    ind_violations += 1
                    
            new_score = tri_violations + ind_violations
            
            if new_score < best_score:
                current_state = new_state
                best_score = new_score
                best_state = new_state
            elif new_score == best_score:
                # Accept with some probability to escape local minima
                if random.random() < 0.5:
                    current_state = new_state
            # Note: We don't update best_state if we accept a worse move, 
            # but we might want to track the best seen so far separately.
            # For simplicity, let's just keep the best found so far.
            
            # Actually, standard local search keeps the current state, 
            # and we should check if current_state is better than best_state.
            # But here we are doing a hill climbing on the score.
            # Let's restart if stuck?
            
        if best_score > 0:
            return None

    # Construct new adjacency
    new_adj = adj[:] + [0]
    for i in range(n_old):
        if best_state & (1 << i):
            new_adj[n-1] |= (1 << i)
            new_adj[i] |= (1 << (n-1))
            
    return new_adj

def _paley_graph(p):
    if p % 4 != 1:
        return None
    squares = {x * x % p for x in range(1, p)}
    adj = [0] * p
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def construct():
    adj = _paley_graph(17)
    if adj is None:
        return {'n': 0, 'edges': []}
    
    current_n = 17
    best_adj = adj
    
    for target_n in range(18, 36):
        if _elapsed() > _DEADLINE - 10:
            break
            
        time_slice = 2.0
        if target_n > 25:
            time_slice = 1.0
        if target_n > 30:
            time_slice = 0.5
            
        if _elapsed() + time_slice > _DEADLINE - 5:
            break
            
        new_adj = _try_extend(best_adj, target_n, time_slice)
        if new_adj is not None:
            best_adj = new_adj
            current_n = target_n
    
    edges = []
    for i in range(current_n):
        for j in range(i + 1, current_n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
                
    return {'n': current_n, 'edges': edges}
