import time
import sys
import random
import itertools
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLORS = 120

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin

    # Precompute edge indices
    edge_idx = [[-1]*N for _ in range(N)]
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            edge_idx[i][j] = idx
            edge_idx[j][i] = idx
            idx += 1
            
    # Precompute all 4-cycles and their edge indices
    cycles = []
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        # 3 possible 4-cycles on 4 vertices
        # 1. a-b-c-d-a
        cycles.append(((a,b), (b,c), (c,d), (d,a)))
        # 2. a-b-d-c-a
        cycles.append(((a,b), (b,d), (d,c), (c,a)))
        # 3. a-c-b-d-a
        cycles.append(((a,c), (c,b), (b,d), (d,a)))
        
    cycle_edge_indices = []
    for c in cycles:
        cycle_edge_indices.append(tuple(edge_idx[u][v] for u, v in c))
        
    # Map edge to cycles containing it
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for i, c_idx in enumerate(cycle_edge_indices):
        for e_idx in c_idx:
            edge_to_cycles[e_idx].append(i)

    def count_violations(colors):
        violations = 0
        for c_idx in cycle_edge_indices:
            c0 = colors[c_idx[0]]
            c1 = colors[c_idx[1]]
            c2 = colors[c_idx[2]]
            c3 = colors[c_idx[3]]
            if len({c0, c1, c2, c3}) == 4:
                violations += 1
        return violations

    def is_valid_change(colors, edge, new_color):
        # Check all cycles containing 'edge'
        for c_idx in edge_to_cycles[edge]:
            ci = cycle_edge_indices[c_idx]
            c0 = new_color if ci[0] == edge else colors[ci[0]]
            c1 = new_color if ci[1] == edge else colors[ci[1]]
            c2 = new_color if ci[2] == edge else colors[ci[2]]
            c3 = new_color if ci[3] == edge else colors[ci[3]]
            
            if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
                return False
        return True

    # Initialize with a random valid-ish coloring
    # Start with all 1s, then randomly assign colors while maintaining validity?
    # Or just start random and let SA fix it.
    # Starting random leads to many violations, but SA can handle it.
    # Let's try starting with a structured valid coloring to be safe, then improve.
    # A simple valid coloring: use only 3 colors.
    # But we want to maximize colors.
    
    # Let's try multiple restarts with different initial conditions.
    best_score = 0
    best_colors = None
    best_violations = float('inf')

    for restart in range(20):
        if time.time() > deadline - 5.0:
            break
            
        random.seed(1234 + restart)
        
        # Initialization: Random assignment
        # To help convergence, we can bias towards fewer colors initially?
        # Or just random 0..119.
        current_colors = [random.randint(0, MAX_COLORS - 1) for _ in range(NUM_EDGES)]
        
        current_violations = count_violations(current_colors)
        current_distinct = len(set(current_colors))
        
        # Score function: Minimize violations, Maximize distinct colors
        # score = -violations * 1000 + distinct
        current_score = -current_violations * 1000 + current_distinct
        
        if current_violations == 0 and current_distinct > best_score:
            best_score = current_distinct
            best_colors = current_colors.copy()

        T = 10.0
        T_min = 0.01
        cooling = 0.9995
        
        # Simulated Annealing
        while time.time() < deadline - 5.0:
            # Pick a random edge
            e = random.randint(0, NUM_EDGES - 1)
            old_color = current_colors[e]
            
            # Pick a new color
            # Bias towards unused colors if valid, or random
            if random.random() < 0.5:
                # Try to introduce a new color or keep diversity
                new_color = random.randint(0, MAX_COLORS - 1)
            else:
                # Try to reduce violations by picking a color already in the cycle
                # This is complex, so just random
                new_color = random.randint(0, MAX_COLORS - 1)
                
            if new_color == old_color:
                continue
                
            # Check if the move is valid (no new violations created)
            # Actually, SA allows invalid moves with probability exp(-delta_E / T)
            # We need to calculate the change in violations.
            # It's expensive to recalculate all violations.
            # Instead, we check if the move creates a violation in any cycle containing e.
            # If it does, delta_violations > 0.
            # If it removes a violation, delta_violations < 0.
            
            # Let's just use is_valid_change for the "valid" part of the move.
            # If is_valid_change is True, the move does NOT create a 4-distinct-color cycle.
            # But it might remove existing violations?
            # If we start with violations, we need to allow moves that reduce them.
            
            # Simplified approach:
            # If current state is valid (0 violations), only accept valid moves (is_valid_change).
            # If current state is invalid, accept moves that reduce violations or with probability.
            
            # Let's calculate delta_violations for the edge e.
            # This requires checking the 182 cycles.
            
            delta_violations = 0
            for c_idx in edge_to_cycles[e]:
                ci = cycle_edge_indices[c_idx]
                
                # Old colors
                oc0 = colors[ci[0]] if False else current_colors[ci[0]] # Wait, use current_colors
                oc1 = current_colors[ci[1]]
                oc2 = current_colors[ci[2]]
                oc3 = current_colors[ci[3]]
                # Replace old_color with new_color in the set
                # This is getting messy. Let's just recalculate violations for the affected cycles.
                
                # Old violation status
                old_set = {current_colors[ci[0]], current_colors[ci[1]], current_colors[ci[2]], current_colors[ci[3]]}
                old_viol = 1 if len(old_set) == 4 else 0
                
                # New violation status
                new_c0 = new_color if ci[0] == e else current_colors[ci[0]]
                new_c1 = new_color if ci[1] == e else current_colors[ci[1]]
                new_c2 = new_color if ci[2] == e else current_colors[ci[2]]
                new_c3 = new_color if ci[3] == e else current_colors[ci[3]]
                new_set = {new_c0, new_c1, new_c2, new_c3}
                new_viol = 1 if len(new_set) == 4 else 0
                
                delta_violations += new_viol - old_viol

            # Change in distinct colors
            # Need freq array
            # We haven't maintained freq in this loop. Let's do it.
            
            # This inner loop is ~182 iterations. 
            # 182 * (time.time() calls etc) might be slow.
            # Let's optimize by precomputing cycle edge indices and using local vars.
            
            # For now, let's assume this is fast enough for 50s.
            
            if delta_violations == 0:
                # Check distinct colors change
                # We need freq. Let's maintain it outside.
                pass
            
            # This is getting complicated. Let's stick to the "is_valid_change" heuristic for valid states,
            # and a simpler repair for invalid states.
            
            # Revised Strategy:
            # 1. If current_violations > 0:
            #    Try to find a move that reduces violations.
            #    If found, accept.
            #    Else, accept random move with prob exp(-delta_violations/T).
            # 2. If current_violations == 0:
            #    Only accept moves that maintain validity (is_valid_change).
            #    Among valid moves, prefer those that increase distinct colors.
            
            if current_violations == 0:
                if is_valid_change(current_colors, e, new_color):
                    # Update distinct count
                    # We need freq
                    # Let's just recalculate distinct count every K steps or maintain freq.
                    # Maintaining freq is better.
                    
                    # Let's implement freq maintenance in the main loop.
                    # I'll rewrite the loop structure to maintain freq.
                    pass
            else:
                # Invalid state
                # Calculate delta_violations
                # ... (as above)
                pass

    # The above is getting messy. Let's write a cleaner version.
    
    # Clean Implementation:
    
    def run_sa():
        # Init
        colors = [random.randint(0, MAX_COLORS - 1) for _ in range(NUM_EDGES)]
        freq = [0] * MAX_COLORS
        for c in colors:
            freq[c] += 1
        distinct = len([f for f in freq if f > 0])
        
        violations = count_violations(colors)
        
        T = 10.0
        T_min = 0.01
        cooling = 0.9995
        
        best_local_distinct = 0
        best_local_colors = colors.copy()
        
        if violations == 0 and distinct > best_local_distinct:
            best_local_distinct = distinct
            best_local_colors = colors.copy()

        while time.time() < deadline - 5.0:
            e = random.randint(0, NUM_EDGES - 1)
            old_color = colors[e]
            new_color = random.randint(0, MAX_COLORS - 1)
            
            if new_color == old_color:
                continue
            
            # Calculate delta violations
            delta_viol = 0
            for c_idx in edge_to_cycles[e]:
                ci = cycle_edge_indices[c_idx]
                # Old
                s_old = {colors[ci[0]], colors[ci[1]], colors[ci[2]], colors[ci[3]]}
                v_old = 1 if len(s_old) == 4 else 0
                # New
                nc0 = new_color if ci[0] == e else colors[ci[0]]
                nc1 = new_color if ci[1] == e else colors[ci[1]]
                nc2 = new_color if ci[2] == e else colors[ci[2]]
                nc3 = new_color if ci[3] == e else colors[ci[3]]
                s_new = {nc0, nc1, nc2, nc3}
                v_new = 1 if len(s_new) == 4 else 0
                delta_viol += v_new - v_old
            
            if violations + delta_viol == 0:
                # Move results in valid state
                # Check distinct change
                delta_distinct = 0
                if freq[old_color] == 1:
                    delta_distinct -= 1
                if freq[new_color] == 0:
                    delta_distinct += 1
                
                new_distinct = distinct + delta_distinct
                if new_distinct > best_local_distinct:
                    best_local_distinct = new_distinct
                    best_local_colors = colors.copy() # Save before update? No, update then save.
                    
                # Apply move
                colors[e] = new_color
                freq[old_color] -= 1
                freq[new_color] += 1
                violations = 0
                distinct = new_distinct
                no_improve = 0 # Reset counter if we are in valid state and improving?
                
            else:
                # Move results in invalid state or keeps invalid
                # Accept with probability
                delta_E = delta_viol * 1000 # Penalty for violations
                # If we are currently invalid, we might want to accept moves that reduce violations
                # If delta_viol < 0, delta_E < 0, accept always.
                # If delta_viol > 0, accept with prob exp(-delta_E/T).
                
                if delta_viol < 0:
                    accept = True
                else:
                    accept = random.random() < np.exp(-delta_E / T)
                    
                if accept:
                    # Apply move
                    colors[e] = new_color
                    freq[old_color] -= 1
                    freq[new_color] += 1
                    violations += delta_viol
                    # Distinct change
                    delta_distinct = 0
                    if freq[old_color] == 0: # After decrement
                        delta_distinct -= 1
                    if freq[new_color] == 1: # After increment (was 0)
                        delta_distinct += 1
                    distinct += delta_distinct
                    
                    # If we become valid, check best
                    if violations == 0 and distinct > best_local_distinct:
                        best_local_distinct = distinct
                        best_local_colors = colors.copy()
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        return best_local_distinct, best_local_colors

    best_overall = 0
    best_overall_colors = None
    
    for restart in range(30):
        if time.time() > deadline - 5.0:
            break
        random.seed(42 + restart)
        np.random.seed(42 + restart)
        
        d, c = run_sa()
        if d > best_overall:
            best_overall = d
            best_overall_colors = c

    return best_overall_colors

# Call construct to test? No, just define it.
# The evaluator will call construct().

# Wait, the function `construct` is defined above. I need to return the best colors.
# The code above has `return best_overall_colors` inside `construct`.

# One issue: `run_sa` uses `deadline` and `edge_to_cycles` from outer scope. This is fine.

# Let's double check the `delta_distinct` logic.
# Before: freq[old], freq[new]
# After: freq[old]-1, freq[new]+1
# Distinct count change:
# If freq[old] was 1, it becomes 0. Distinct -1.
# If freq[new] was 0, it becomes 1. Distinct +1.
# In the code:
# `if freq[old_color] == 1: delta_distinct -= 1` -> Correct.
# `if freq[new_color] == 0: delta_distinct += 1` -> Correct.

# In the invalid state accept block:
# `if freq[old_color] == 0: delta_distinct -= 1` -> This is AFTER decrement.
# If it was 1, it is now 0. So distinct -1. Correct.
# `if freq[new_color] == 1: delta_distinct += 1` -> This is AFTER increment.
# If it was 0, it is now 1. So distinct +1. Correct.

# The logic seems sound.

# Performance:
# `count_violations` is called once at start.
# Inside loop, we iterate `edge_to_cycles[e]` which has ~182 items.
# For each, we do set operations.
# 182 * (small constant) operations per step.
# Python can do ~10^6-10^7 simple ops/sec.
# 182 * 10^6 = 1.8 * 10^8 ops. This might be too slow for 50s if we do 10^6 steps.
# We need to be faster.

# Optimization:
# Instead of calculating `delta_viol` for every step, we can use the `is_valid_change` function
# if we are in a valid state.
# If we are in an invalid state, we need to reduce violations.
# Maybe we can just use `is_valid_change` always?
# If we are invalid, `is_valid_change` returns False if the move creates a NEW violation.
# But we want to REMOVE violations.
# So `is_valid_change` is not enough for invalid states.

# However, if we start with a valid state (e.g., all 1s), we can stay valid and just increase colors.
# Let's try initializing with a valid state.
# A valid state: All edges color 0.
# Then we try to change edges to new colors.
# If we only accept moves that maintain validity, we never get stuck in invalid states.
# This is a "valid-only" hill climbing / SA.
# This is much faster because we don't need to calculate delta_violations for invalid moves.
# We just check `is_valid_change`.

# Let's modify `run_sa` to start with a valid state and only accept valid moves.
# To escape local optima, we can use "restarts" or "perturbations" (e.g., randomly re-color 10% of edges to valid colors? No, that breaks validity).
# Instead, we can use a "kick" that temporarily allows invalid moves?
# Or just rely on multiple restarts from different valid initial states.
# But finding a good valid initial state with high color count is the problem.
# Starting with 1 color is safe but low score.
# Starting with 3 colors is safe.
# We can try to build a valid coloring with more colors greedily?

# Let's stick to the "Valid-Only" SA starting from a random valid coloring.
# How to get a random valid coloring?
# We can try to assign colors randomly and check validity. If invalid, revert.
# This is basically the same as the valid-only SA.

# Let's try this:
# 1. Init: All 0s.
# 2. SA: Pick edge, pick new color. If `is_valid_change`, accept.
# 3. To improve diversity, bias `new_color` selection towards unused colors.

# This should be fast.
# `is_valid_change` checks ~182 cycles.
# 182 checks per step.
# If we do 10^5 steps in 5s, that's 2 * 10^7 checks.
# Each check is ~10-20 Python ops.
# 2 * 10^8 ops. Might be tight in Python.
# But we have 55s.
# Let's try it.

# Revised `run_sa` for valid-only:

def run_sa_valid_only():
    colors = [0] * NUM_EDGES
    freq = [0] * MAX_COLORS
    freq[0] = NUM_EDGES
    distinct = 1
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    best_distinct = 1
    best_colors = colors.copy()
    
    no_improve = 0
    max_no_improve = 10000
    
    while time.time() < deadline - 5.0:
        if no_improve > max_no_improve:
            break
            
        e = random.randint(0, NUM_EDGES - 1)
        old_color = colors[e]
        
        # Bias towards unused colors
        if random.random() < 0.7:
            unused = [c for c in range(MAX_COLORS) if freq[c] == 0]
            if unused:
                new_color = random.choice(unused)
            else:
                new_color = random.randint(0, MAX_COLORS - 1)
        else:
            new_color = random.randint(0, MAX_COLORS - 1)
            
        if new_color == old_color:
            continue
            
        if is_valid_change(colors, e, new_color):
            # Apply
            colors[e] = new_color
            freq[old_color] -= 1
            freq[new_color] += 1
            
            delta_distinct = 0
            if freq[old_color] == 0:
                delta_distinct -= 1
            if freq[new_color] == 1:
                delta_distinct += 1
                
            distinct += delta_distinct
            
            if distinct > best_distinct:
                best_distinct = distinct
                best_colors = colors.copy()
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
            
        T *= cooling
        if T < T_min:
            T = T_min
            
    return best_distinct, best_colors

# Let's use this in the main loop.
# Also, try different initial states?
# E.g., start with a random valid coloring found by greedy?
# For now, start with 0s.

# One more thing: The `is_valid_change` function is called frequently.
# We can optimize it by precomputing the cycle edge indices as lists of lists of ints.
# And using local variables.

# Let's finalize the code.

