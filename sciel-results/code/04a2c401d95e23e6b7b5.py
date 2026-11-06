import time
import random
import math

def construct():
    """
    Strategy:
    1. Formulate the problem as a global optimization over all potential edges.
    2. State: A dictionary mapping each potential edge (u,v) to a state:
       - 0: Absent
       - 1..20: Present with colour (1-based for internal logic, map to 0-19 later)
       Actually, let's use 0 for absent, 1-20 for present with colour c in 0-19.
       So colour c is stored as c+1.
    
    3. Objective: Minimize Penalty = W_C4 * (Number of non-rainbow 4-cycles) - W_E * (Number of edges).
       We want to maximize edges, so we minimize -edges.
       Let's define Energy = C4_violations * 100 - edge_count.
       We want to minimize Energy.
    
    4. Algorithm: WalkSAT / Local Search
       - Start with a random valid or near-valid state.
       - Iteratively pick a variable (edge) to flip.
       - Bias towards variables involved in violated clauses (non-rainbow C4s) or missing edges.
       - Flip the variable to the value that minimizes local energy change.
       - If no improvement, accept random flips to escape local minima.
    
    5. Specifics for n=20:
       - 190 potential edges.
       - Number of 4-cycles in K20 is C(20,4) * 3 = 4845 * 3 = 14535.
       - Checking all C4s on every step is O(14535 * 4) ~ 60k ops. 
       - With 150s budget, we can do ~1000-5000 steps if optimized.
       - Optimization: Only check C4s affected by the changed edge.
       - An edge (u,v) is in a C4 if there is a path of length 3 between u and v.
       - The C4s involving edge (u,v) are of the form u-v-x-y-u.
       - This requires x adjacent to v and y adjacent to u, and x-y edge present.
       - Or u-v-x-y-u means edges (u,v), (v,x), (x,y), (y,u).
       - So for a fixed edge (u,v), the C4s containing it are determined by pairs (x,y) such that
         (v,x) exists, (x,y) exists, (y,u) exists.
       - We can precompute or compute on the fly the list of C4s for each edge.
       - Since the graph changes, the list of C4s changes.
       - However, we can maintain a count of violations per edge? No, violations are per C4.
       - Let's maintain a global count of violations and a set of violated C4s.
       - When an edge changes, we only need to re-evaluate C4s that contain that edge.
       - How many C4s contain a specific edge in K20?
         For edge (u,v), a C4 is u-v-x-y-u.
         x must be a neighbor of v, y must be a neighbor of u, and (x,y) must be an edge.
         Also x != u, y != v, x != y.
         In K20, max ~ 18*18 = 324 C4s per edge.
         This is small enough to check on the fly.
    """
    
    n = 20
    deadline = time.time() + 140  # Safety margin
    
    # List of all potential edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    num_edges = len(all_edges)
    
    # Precompute for each edge (u,v), the list of "witness" pairs (x,y) that form a C4 with it.
    # A C4 containing (u,v) is u-v-x-y-u.
    # Edges: (u,v), (v,x), (x,y), (y,u).
    # So for a fixed (u,v), we need to find all x,y such that:
    # 1. x != u, x != v
    # 2. y != u, y != v
    # 3. x != y
    # 4. Edge (v,x) exists
    # 5. Edge (x,y) exists
    # 6. Edge (y,u) exists
    # Note: The C4 is defined by the set of 4 vertices {u,v,x,y}.
    # There are 3 C4s on a set of 4 vertices.
    # To avoid double counting, we can just count the number of 4-cycles in the standard way.
    # But for local search, we need to know which C4s are violated.
    
    # Let's represent the state as an array `state` of length num_edges.
    # state[i] = 0 if edge i is absent.
    # state[i] = c+1 if edge i is present with colour c (0-19).
    
    # Helper to get edge index
    def get_eidx(u, v):
        if u > v: u, v = v, u
        return edge_to_idx[(u, v)]
    
    # Helper to check if edge exists
    def has_edge(u, v, state):
        idx = get_eidx(u, v)
        return state[idx] != 0
    
    # Helper to get colour of edge (0 if absent)
    def get_colour(u, v, state):
        idx = get_eidx(u, v)
        return state[idx] - 1 # Return -1 if absent? No, 0 is absent. 
        # Let's say state[idx] is 0..19 for colours, and 20 for absent?
        # Easier: state[idx] in 0..19 means present with colour state[idx].
        # state[idx] == 20 means absent.
        pass

    # Revised State Representation:
    # state[i] in 0..19: Edge present, colour = state[i]
    # state[i] == 20: Edge absent
    
    def get_colour(u, v, state):
        idx = get_eidx(u, v)
        if state[idx] == 20: return None
        return state[idx]
        
    def is_present(u, v, state):
        return state[get_eidx(u, v)] != 20

    # Count violations (non-rainbow C4s)
    # A C4 is a set of 4 vertices {a,b,c,d} and a cycle order.
    # Standard C4 counting: sum_{u<v} C(common_neighbors(u,v), 2)
    # But we need to check if the colours are distinct.
    # For each pair (u,v) with common neighbors x,y, the C4 is u-x-v-y-u.
    # The edges are (u,x), (x,v), (v,y), (y,u).
    # We need colours of these 4 edges to be distinct.
    
    def count_violations(state):
        violations = 0
        # For each pair of vertices (u, v) that could be opposite in a C4?
        # No, in the formula sum_{u<v} C(common(u,v), 2), u and v are opposite.
        # The C4 is u - x - v - y - u.
        # Edges: (u,x), (x,v), (v,y), (y,u).
        for u in range(n):
            for v in range(u + 1, n):
                # Find common neighbors
                common = []
                for w in range(n):
                    if w == u or w == v: continue
                    if is_present(u, w, state) and is_present(v, w, state):
                        common.append(w)
                
                k = len(common)
                if k < 2: continue
                
                # For each pair of common neighbors (x, y)
                for i in range(k):
                    for j in range(i + 1, k):
                        x = common[i]
                        y = common[j]
                        
                        # Check if edges (x,y) exists? 
                        # Wait, the C4 is u-x-v-y-u.
                        # The edges are (u,x), (x,v), (v,y), (y,u).
                        # We already know (u,x), (x,v), (v,y), (y,u) exist because x,y are common neighbors of u,v.
                        # Do we need (x,y) to exist? NO.
                        # The cycle is u-x-v-y-u.
                        # Edges: (u,x), (x,v), (v,y), (y,u).
                        # These are exactly the edges implied by x,y being common neighbors of u,v.
                        # So we just need to check the colours of these 4 edges.
                        
                        c1 = get_colour(u, x, state)
                        c2 = get_colour(x, v, state)
                        c3 = get_colour(v, y, state)
                        c4 = get_colour(y, u, state)
                        
                        if c1 is None or c2 is None or c3 is None or c4 is None:
                            continue # Should not happen if is_present check was correct
                        
                        # Check if rainbow
                        if len({c1, c2, c3, c4}) < 4:
                            violations += 1
                            
        return violations

    def count_edges(state):
        return sum(1 for s in state if s != 20)

    def get_energy(state):
        # Maximize edges, minimize violations
        # Energy = Violations * BIG_PENALTY - Edges
        # We want to minimize this.
        v = count_violations(state)
        e = count_edges(state)
        return v * 1000 - e

    # Local Search
    # State: list of ints
    # Move: Change state[i] to a new value.
    
    def random_state(density=0.5):
        state = []
        for _ in range(num_edges):
            if random.random() < density:
                state.append(random.randint(0, 19))
            else:
                state.append(20)
        return state

    def step(state, T):
        # Pick a random edge index
        i = random.randint(0, num_edges - 1)
        u, v = all_edges[i]
        
        old_val = state[i]
        
        # Determine new value
        # Try to find a value that minimizes energy change locally?
        # Or just pick a random value?
        # WalkSAT: Pick a variable involved in a violated clause.
        # Here, if we change edge (u,v), it affects C4s containing (u,v).
        # Which C4s contain (u,v)?
        # C4s where (u,v) is one of the 4 edges.
        # In the u-x-v-y-u structure:
        # (u,v) could be (u,x) if x=v? No.
        # (u,v) could be (u,x) where x is a neighbor.
        # Let's identify C4s affected by changing (u,v).
        # A C4 is u-x-v-y-u.
        # (u,v) is an edge in the C4 if:
        # 1. (u,v) == (u,x) => x=v. Then C4 is u-v-v-y-u? No, simple graph.
        #    Actually, the vertices are distinct.
        #    If (u,v) is an edge in the C4, then u and v are adjacent in the cycle.
        #    So the cycle is u-v-x-y-u or u-v-y-x-u.
        #    Case 1: u-v-x-y-u. Edges: (u,v), (v,x), (x,y), (y,u).
        #    Case 2: u-v-y-x-u. Edges: (u,v), (v,y), (y,x), (x,u).
        #    
        #    So for a fixed edge (u,v), the affected C4s are those of the form:
        #    u-v-x-y-u where (v,x), (x,y), (y,u) exist.
        #    And u-v-y-x-u where (v,y), (y,x), (x,u) exist.
        #    These are essentially the same set of C4s, just traversed differently.
        #    The set of 4 vertices is {u,v,x,y}.
        #    For a fixed pair {u,v}, the C4s containing edge (u,v) are determined by pairs (x,y)
        #    such that (v,x), (x,y), (y,u) exist.
        
        # To be efficient, we can just calculate the global violation count change?
        # No, we need to know which C4s are violated to bias the search.
        # Given the small n=20, we can just recalculate the global violation count
        # for the few candidate new values for state[i].
        
        # Candidate values: 0..19 (present) and 20 (absent).
        # Total 21 candidates.
        # For each candidate, we need to calculate the change in violations.
        # Instead of full recalculation, we can calculate violations for C4s involving (u,v).
        
        # Let's identify all C4s involving (u,v).
        # A C4 involving (u,v) has vertices {u, v, x, y}.
        # Edges in C4: (u,v), (v,x), (x,y), (y,u) OR (u,v), (v,y), (y,x), (x,u).
        # Since (u,v) is fixed, we need x,y such that the other 3 edges exist.
        
        affected_c4s = []
        # Find all x, y
        for x in range(n):
            if x == u or x == v: continue
            if not is_present(v, x, state): continue
            for y in range(n):
                if y == u or y == v or y == x: continue
                if not is_present(x, y, state): continue
                if not is_present(y, u, state): continue
                # We have a C4 u-v-x-y-u
                # Store the tuple of 4 edges involved in colour check
                # Edges: (u,v), (v,x), (x,y), (y,u)
                # Note: (u,v) is the one being changed.
                # The other 3 are (v,x), (x,y), (y,u).
                affected_c4s.append(((u,v), (v,x), (x,y), (y,u)))
                
        # Note: We might double count C4s if we iterate x,y symmetrically?
        # u-v-x-y-u is the same set of vertices as u-v-y-x-u?
        # No, the cycle direction matters for the edge set?
        # Edges in u-v-x-y-u: {(u,v), (v,x), (x,y), (y,u)}.
        # Edges in u-v-y-x-u: {(u,v), (v,y), (y,x), (x,u)}.
        # These are different sets of edges unless x=y (impossible) or specific symmetries.
        # Actually, for a fixed set of 4 vertices {u,v,x,y}, there are 3 C4s.
        # 1. u-v-x-y-u (Edges: uv, vx, xy, yu)
        # 2. u-v-y-x-u (Edges: uv, vy, yx, xu)
        # 3. u-x-v-y-u (Edges: ux, xv, vy, yu) -- This one does NOT contain edge (u,v).
        # So for edge (u,v), only C4s 1 and 2 are relevant?
        # Wait, C4 is a subgraph. The subgraph is defined by the 4 edges.
        # If we change (u,v), we affect any C4 that uses edge (u,v).
        # The C4s using (u,v) are exactly those where u and v are adjacent in the cycle.
        # In the set {u,v,x,y}, u and v are adjacent in C4 1 and C4 2.
        # In C4 3, u is adjacent to x and y, v is adjacent to x and y. u and v are opposite.
        # So yes, only C4s where u,v are adjacent.
        
        # To avoid duplicates in affected_c4s, note that for a fixed set {u,v,x,y},
        # the pair (x,y) in my loop corresponds to specific C4s.
        # My loop finds x such that (v,x) exists, and y such that (x,y) and (y,u) exist.
        # This finds C4 u-v-x-y-u.
        # It will also find C4 u-v-y-x-u when x and y are swapped in the logic?
        # If I swap x and y in the C4 u-v-x-y-u, I get u-v-y-x-u.
        # Does my loop find it?
        # For C4 u-v-y-x-u, the edges are (u,v), (v,y), (y,x), (x,u).
        # In my loop:
        # x' = y. Check (v,y) exists. Yes.
        # y' = x. Check (y', x')? i.e., (x,y) exists. Yes.
        # Check (y', u)? i.e., (x,u) exists. Yes.
        # So yes, it finds both.
        # Are they distinct C4s?
        # C4 1 edges: uv, vx, xy, yu.
        # C4 2 edges: uv, vy, yx, xu.
        # They are distinct unless the graph has symmetries that make the edge sets identical,
        # which would require vx=vy (x=y, impossible) or similar.
        # So we might have duplicate entries in affected_c4s if the same set of edges is found?
        # No, the tuple of edges is different.
        # However, for the purpose of counting violations, we just need to check if the 4 edges have distinct colours.
        
        # Let's calculate the change in violations for each candidate value.
        # Current violations in affected C4s:
        curr_viol = 0
        for c4_edges in affected_c4s:
            # c4_edges is ((u,v), (v,x), (x,y), (y,u))
            # Get colours
            cols = [get_colour(e[0], e[1], state) for e in c4_edges]
            if None in cols: continue # Should not happen for existing C4 structure
            if len(set(cols)) < 4:
                curr_viol += 1
                
        best_new_val = old_val
        best_delta = 0 # We want to minimize energy. Delta = NewEnergy - OldEnergy.
        # Energy = 1000 * Violations - Edges.
        # Change in Violations: dV. Change in Edges: dE.
        # dEnergy = 1000 * dV - dE.
        
        for new_val in range(21): # 0..19 colours, 20 absent
            if new_val == old_val: continue
            
            # Calculate new violations in affected C4s
            new_viol = 0
            for c4_edges in affected_c4s:
                # Update colour of (u,v) in the check
                # c4_edges[0] is (u,v)
                cols = [get_colour(e[0], e[1], state) for e in c4_edges]
                # Replace colour of (u,v)
                # (u,v) is at index 0 in c4_edges
                if new_val == 20:
                    cols[0] = None # Absent
                else:
                    cols[0] = new_val
                    
                if None in cols: 
                    # If an edge becomes absent, the C4 is no longer a C4 in the graph?
                    # Wait. The definition of "4-cycle is rainbow" applies to subgraphs that ARE 4-cycles.
                    # If an edge is removed, the subgraph is no longer a 4-cycle.
                    # So it should NOT be counted as a violation.
                    # It should be counted as 0 violations for that C4.
                    continue 
                
                if len(set(cols)) < 4:
                    new_viol += 1
                    
            dV = new_viol - curr_viol
            
            # Change in Edges
            if old_val == 20 and new_val != 20:
                dE = 1
            elif old_val != 20 and new_val == 20:
                dE = -1
            else:
                dE = 0
                
            dEnergy = 1000 * dV - dE
            
            if dEnergy < best_delta:
                best_delta = dEnergy
                best_new_val = new_val
            elif dEnergy == best_delta and random.random() < 0.5:
                best_new_val = new_val

        # WalkSAT noise: With some probability, pick a random value
        if random.random() < 0.1:
            best_new_val = random.randint(0, 20)
            
        state[i] = best_new_val
        return state

    # Main Loop
    best_state = None
    best_energy = float('inf')
    
    # Initial state: Try to start with a dense random graph
    # Or start with empty and add?
    # Let's start with a random graph with density 0.8
    state = random_state(0.8)
    current_energy = get_energy(state)
    
    if current_energy < best_energy:
        best_energy = current_energy
        best_state = state.copy()
        
    T = 50.0
    cooling = 0.999
    min_T = 1.0
    
    start_time = time.time()
    steps = 0
    
    while time.time() - start_time < 130:
        old_energy = current_energy
        state = step(state, T)
        new_energy = get_energy(state)
        
        delta = new_energy - old_energy
        
        # Accept if better or with probability exp(-delta/T)
        if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_state = state.copy()
        else:
            # Revert
            # We didn't store the old state, so we need to revert the specific change
            # This is inefficient. Better to store state before step.
            # Let's modify step to return state and allow revert, or just keep a copy.
            # For simplicity and speed, let's keep a copy of the state before each step?
            # No, that's too much memory copying.
            # Let's refactor step to modify in place and return the change made.
            pass # This logic is flawed in the current draft. 
            # Let's restart the loop structure with proper state management.
            
        T *= cooling
        if T < min_T: T = min_T
        steps += 1
        
        if steps % 1000 == 0:
            # Check time
            if time.time() - start_time > 125:
                break

    # The above loop had a revert issue. Let's implement a cleaner version.
    
    def local_search():
        state = random_state(0.8)
        current_energy = get_energy(state)
        best_state = state.copy()
        best_energy = current_energy
        
        T = 50.0
        cooling = 0.999
        min_T = 1.0
        
        start_time = time.time()
        
        while time.time() - start_time < 125:
            old_state = state.copy() # Expensive but n=190 is small
            old_energy = current_energy
            
            # Perform step
            i = random.randint(0, num_edges - 1)
            u, v = all_edges[i]
            old_val = state[i]
            
            # Find affected C4s
            affected_c4s = []
            for x in range(n):
                if x == u or x == v: continue
                if not is_present(v, x, old_state): continue
                for y in range(n):
                    if y == u or y == v or y == x: continue
                    if not is_present(x, y, old_state): continue
                    if not is_present(y, u, old_state): continue
                    affected_c4s.append(((u,v), (v,x), (x,y), (y,u)))
                    
            curr_viol = 0
            for c4_edges in affected_c4s:
                cols = [get_colour(e[0], e[1], old_state) for e in c4_edges]
                if None in cols: continue
                if len(set(cols)) < 4:
                    curr_viol += 1
                    
            best_new_val = old_val
            best_delta = 0
            
            for new_val in range(21):
                if new_val == old_val: continue
                new_viol = 0
                for c4_edges in affected_c4s:
                    cols = [get_colour(e[0], e[1], old_state) for e in c4_edges]
                    if new_val == 20:
                        cols[0] = None
                    else:
                        cols[0] = new_val
                    if None in cols: continue
                    if len(set(cols)) < 4:
                        new_viol += 1
                        
                dV = new_viol - curr_viol
                if old_val == 20 and new_val != 20: dE = 1
                elif old_val != 20 and new_val == 20: dE = -1
                else: dE = 0
                
                dEnergy = 1000 * dV - dE
                if dEnergy < best_delta:
                    best_delta = dEnergy
                    best_new_val = new_val
                elif dEnergy == best_delta and random.random() < 0.5:
                    best_new_val = new_val
                    
            if random.random() < 0.1:
                best_new_val = random.randint(0, 20)
                
            state[i] = best_new_val
            new_energy = get_energy(state)
            delta = new_energy - old_energy
            
            if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_state = state.copy()
            else:
                state = old_state # Revert
                
            T *= cooling
            if T < min_T: T = min_T
            
        return best_state, best_energy

    # Run multiple restarts
    final_best_state = None
    final_best_energy = float('inf')
    
    restarts = 10
    for r in range(restarts):
        if time.time() > deadline: break
        st, en = local_search()
        if en < final_best_energy:
            final_best_energy = en
            final_best_state = st
            # If we found a perfect solution (0 violations), we can stop early?
            # But we want max edges. 0 violations is the constraint.
            # If en == -edge_count, and we can't add more edges without violations, we stop.
            # Check if 0 violations:
            v = count_violations(st)
            if v == 0:
                # Check if we can add any more edges?
                # The local search should have optimized this.
                # We can break if we are confident.
                pass

    if final_best_state is None:
        # Fallback: return empty graph
        return []
        
    # Extract edges
    coloured_edges = []
    for i, (u, v) in enumerate(all_edges):
        c = final_best_state[i]
        if c != 20:
            coloured_edges.append([u, v, c])
            
    return coloured_edges

# Quick sanity check for the logic
if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
