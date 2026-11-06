"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    d = 7
    n_vertices = 1 << d
    all_edges = []
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges.append((x, y))
    
    # Precompute for each edge, the 3 other edges that would complete a 4-cycle
    # An edge (u, v) with u < v. Let bit = u ^ v.
    # The 4-cycle is completed by 3 other edges.
    # If u and v differ in bit i, the other two vertices are u^(1<<j) and v^(1<<j) for j != i.
    # The 3 edges are:
    # 1. (u, u^(1<<j))
    # 2. (v, v^(1<<j))
    # 3. (u^(1<<j), v^(1<<j))
    # Note: We need to check if these 3 edges are present.
    # We can map edge -> index for fast lookup.
    edge_index = {e: i for i, e in enumerate(all_edges)}
    
    # conflicts[i] is a list of lists of 3 edge indices
    # If edge i is added, and all 3 edges in conflicts[i][k] are present, a cycle is formed.
    # Actually, it's better to store for each edge, the set of "triads" that it participates in.
    # A triad is a set of 3 edges. If all 4 edges (triad + current) are present, it's a C4.
    # Let's store: for each edge index, a list of tuples of 3 other edge indices.
    conflict_trads = [[] for _ in range(len(all_edges))]
    
    for idx, (u, v) in enumerate(all_edges):
        bit = u ^ v
        for j in range(d):
            if (1 << j) == bit:
                continue
            # The 3 other edges
            u2 = u ^ (1 << j)
            v2 = v ^ (1 << j)
            
            e1 = (min(u, u2), max(u, u2))
            e2 = (min(v, v2), max(v, v2))
            e3 = (min(u2, v2), max(u2, v2))
            
            i1 = edge_index.get(e1)
            i2 = edge_index.get(e2)
            i3 = edge_index.get(e3)
            
            if i1 is not None and i2 is not None and i3 is not None:
                conflict_trads[idx].append((i1, i2, i3))
    
    # State: boolean array of length len(all_edges)
    # 1 if edge is present, 0 otherwise
    # We want to maximize sum(state) such that no C4 is fully present.
    
    def count_cycles(state):
        count = 0
        for i, s in enumerate(state):
            if s:
                for triad in conflict_trads[i]:
                    if state[triad[0]] and state[triad[1]] and state[triad[2]]:
                        count += 1
        return count
    
    def is_valid(state):
        return count_cycles(state) == 0

    # Initialize: Random dense set
    # Start with probability 0.5 or 0.8?
    # Greedy gets ~300. 448 edges total.
    # Let's try starting with 0.7 probability.
    state = [1 if random.random() < 0.7 else 0 for _ in range(len(all_edges))]
    
    # Remove cycles to make it valid initially
    # Simple loop: while any cycle, remove a random edge from a random cycle
    # This might be slow if done naively.
    # Better: just run the SA from a random start, but the objective function penalizes cycles.
    # Objective: Value = num_edges - penalty * num_cycles
    # We want to reach a state with 0 cycles.
    
    # Let's use a specific penalty that forces validity.
    # If we have k edges and c cycles, score = k - 1000*c.
    # Maximize score.
    
    best_state = None
    best_score = -1
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    alpha = 0.999
    
    current_score = len([x for x in state if x]) - 1000 * count_cycles(state)
    
    # To speed up, we can keep a list of "active" edges and check only those?
    # Or just update the score incrementally.
    
    # Let's try a simpler Hill Climbing with restarts first, or SA with incremental updates.
    # Incremental update for adding edge i:
    # Gain: 1 (edge count)
    # Cost: 1000 * (number of new cycles formed)
    # New cycles formed: for each triad in conflict_trads[i], if all 3 are present, it forms a cycle.
    
    # Incremental update for removing edge i:
    # Gain: -1 (edge count)
    # Cost: -1000 * (number of cycles destroyed)
    # Cycles destroyed: for each triad in conflict_trads[i], if all 3 are present, it destroys a cycle.
    # Note: A cycle is destroyed if the edge being removed was part of it AND the other 3 were present.
    
    def get_score_delta_add(i, state):
        if state[i]: return 0
        gain = 1
        cycles_formed = 0
        for triad in conflict_trads[i]:
            if state[triad[0]] and state[triad[1]] and state[triad[2]]:
                cycles_formed += 1
        return gain - 1000 * cycles_formed
        
    def get_score_delta_remove(i, state):
        if not state[i]: return 0
        gain = -1
        cycles_destroyed = 0
        for triad in conflict_trads[i]:
            if state[triad[0]] and state[triad[1]] and state[triad[2]]:
                cycles_destroyed += 1
        return gain + 1000 * cycles_destroyed
        
    # Initial state cleanup: remove edges to break cycles
    # Just remove random edges until valid
    while count_cycles(state) > 0 and time.time() < deadline - 5:
        # Find a cycle
        found = False
        for i, s in enumerate(state):
            if s:
                for triad in conflict_trads[i]:
                    if state[triad[0]] and state[triad[1]] and state[triad[2]]:
                        # Remove one of these 4 edges
                        candidates = [i, triad[0], triad[1], triad[2]]
                        victim = random.choice(candidates)
                        state[victim] = 0
                        found = True
                        break
                if found: break
        if not found: break # Should not happen if count_cycles > 0
        
    current_score = len([x for x in state if x])
    
    if is_valid(state):
        best_state = state[:]
        best_score = current_score
    
    # Main Loop
    steps = 0
    while time.time() < deadline:
        steps += 1
        # Pick random edge
        i = random.randint(0, len(all_edges) - 1)
        
        if state[i]:
            delta = get_score_delta_remove(i, state)
        else:
            delta = get_score_delta_add(i, state)
            
        new_score = current_score + delta
        
        if new_score >= current_score:
            state[i] = 1 - state[i]
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_state = state[:]
        else:
            # Accept with probability exp(delta / T)
            # delta is negative
            if random.random() < pow(2.71828, delta / T):
                state[i] = 1 - state[i]
                current_score = new_score
        
        T *= alpha
        if T < T_min:
            T = T_min
            
        # Occasionally restart if stuck?
        # Or just let it cool down.
        
        # Check validity of best_state periodically? 
        # best_state is always valid if we only update when improving score 
        # and initial state was valid.
        # Wait, if we accept a bad move (negative delta), we might enter an invalid state.
        # The score function penalizes cycles, so invalid states have lower scores.
        # But we want the *best valid* state.
        # If current state is invalid, its score is low.
        # We should track the best VALID state.
        
        # Optimization: Only update best_state if current state is valid?
        # Checking validity is O(N).
        # But we can check if count_cycles == 0.
        # We don't track count_cycles incrementally easily in the SA loop without extra work.
        # However, if we use a large penalty, the "score" is a proxy for validity + size.
        # A state with 0 cycles and 300 edges has score 300.
        # A state with 1 cycle and 305 edges has score 305 - 1000 = -695.
        # So the maximum score will always correspond to a valid state (0 cycles) with max edges,
        # provided the penalty is large enough (> max possible edges).
        # Max edges 448. Penalty 1000 is safe.
        
        # So best_score tracks the best valid state automatically?
        # Yes, because any invalid state has a score < 448 - 1000 = -552, 
        # while any valid state has score >= 0.
        # So the global maximum of the score function is a valid state.
        
    # Convert best_state to list of edges
    result = []
    for i, s in enumerate(best_state):
        if s:
            result.append(list(all_edges[i]))
            
    return result
# EVOLVE-BLOCK-END
