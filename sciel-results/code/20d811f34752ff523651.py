import numpy as np
import time
import random
import sys

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

# Precompute edge indices
edge_idx = np.zeros((N, N), dtype=np.int32)
idx = 0
for i in range(N):
    for j in range(i + 1, N):
        edge_idx[i, j] = idx
        edge_idx[j, i] = idx
        idx += 1

# Precompute all 4-cycles
# A 4-cycle is defined by 4 distinct vertices {a, b, c, d}.
# The three possible cycles on these vertices are:
# 1. a-b-c-d-a (edges: ab, bc, cd, da)
# 2. a-b-d-c-a (edges: ab, bd, dc, ca)
# 3. a-c-b-d-a (edges: ac, cb, bd, da)
cycles = []
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            for d in range(c + 1, N):
                # Cycle 1: a-b-c-d
                cycles.append((edge_idx[a, b], edge_idx[b, c], edge_idx[c, d], edge_idx[d, a]))
                # Cycle 2: a-b-d-c
                cycles.append((edge_idx[a, b], edge_idx[b, d], edge_idx[d, c], edge_idx[c, a]))
                # Cycle 3: a-c-b-d
                cycles.append((edge_idx[a, c], edge_idx[c, b], edge_idx[b, d], edge_idx[d, a]))

NUM_CYCLES = len(cycles)

# For each edge, precompute which cycles it belongs to
# edge_to_cycles[edge_id] = list of cycle indices
edge_to_cycles = [[] for _ in range(NUM_EDGES)]
for i, c in enumerate(cycles):
    for e in c:
        edge_to_cycles[e].append(i)

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s/90s limit
    
    best_distinct = 0
    best_colors = None
    
    # Fallback valid solution: color(i,j) = i
    # This uses 15 colors.
    fallback = np.zeros(NUM_EDGES, dtype=np.int32)
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            fallback[idx] = i
            idx += 1
    
    # CSP Backtracking with Forward Checking
    # State:
    # colors: array of 120 ints, -1 for unassigned
    # domains: list of sets (or bitmasks) for each edge, indicating allowed colors
    # We use bitmasks for speed. 120 colors fit in a Python int (arbitrary precision).
    # Full domain mask: (1 << 120) - 1
    
    full_mask = (1 << 120) - 1
    
    def solve_with_restart(max_steps):
        nonlocal best_distinct, best_colors
        nonlocal start_time, time_limit
        
        # Randomized order of edges to break symmetry and allow different search paths
        edge_order = list(range(NUM_EDGES))
        random.shuffle(edge_order)
        
        # colors[e] = -1 if unassigned, else color value
        colors = np.full(NUM_EDGES, -1, dtype=np.int32)
        # domains[e] = bitmask of allowed colors
        domains = [full_mask] * NUM_EDGES
        
        # To maximize distinct colors, we can't just pick any valid color.
        # We will use a heuristic: prefer colors that are "new" (not used yet) 
        # but this is hard to do in pure CSP without lookahead.
        # Alternative: Just find a valid solution, then try to increase distinctness?
        # No, CSP can be guided. 
        # Let's try a simple strategy: 
        # 1. Find ANY valid solution quickly.
        # 2. Then, use a local search phase to increase distinct colors?
        # The prompt asks for a DIFFERENT strategy. 
        # Pure CSP with random value ordering might find a valid solution with random distinct count.
        # To maximize distinctness, we can bias the value selection.
        
        # Heuristic for value selection:
        # When assigning color to edge e, pick a color from its domain.
        # To maximize distinctness, we prefer colors that appear less frequently in the current assignment.
        # Or simply: iterate through colors 0..119, and if valid, pick one randomly from the valid ones?
        # If we pick randomly, we might use few colors.
        # Let's try: Pick a random color from the domain. 
        # But to bias towards high distinctness, we can track used colors.
        
        used_colors = np.zeros(120, dtype=np.int32)
        
        # Backtracking stack
        # We implement iterative backtracking to avoid recursion limits and allow time checks.
        
        # However, implementing full backtracking with forward checking in Python 
        # with bitmask domains might be slow if we do it naively.
        
        # Let's try a simpler CSP: 
        # Assign edges in a fixed (shuffled) order.
        # For each edge, try colors from its domain.
        # If conflict, backtrack.
        
        # To make it fast, we limit the depth or use a greedy approach with restarts.
        # "Greedy CSP" (no backtracking, just restart on failure) is often faster.
        # Given N=16, the constraint graph is dense. Backtracking might be deep.
        # Let's use "Randomized Greedy CSP":
        # 1. Shuffle edge order.
        # 2. For each edge, pick a random valid color from its domain.
        # 3. Update domains of neighbors (forward checking).
        # 4. If domain becomes empty, fail and restart.
        
        # To maximize distinct colors, we can modify the value selection:
        # Instead of uniform random from domain, we can weight by 1/frequency?
        # Or just uniform random is fine for finding *a* valid solution, 
        # and we run many restarts to find one with high distinctness.
        
        # Actually, a better strategy for MAX distinct:
        # We want to use as many different colors as possible.
        # If we assign colors randomly, we'll likely use ~sqrt(120) * something colors?
        # No, if we have 120 edges and pick from 120 colors, we might use many.
        # But constraints force repetition.
        
        # Let's implement the Greedy CSP.
        
        def try_greedy():
            nonlocal colors, domains, used_colors
            nonlocal best_distinct, best_colors
            
            colors = np.full(NUM_EDGES, -1, dtype=np.int32)
            domains = [full_mask] * NUM_EDGES
            used_colors = np.zeros(120, dtype=np.int32)
            
            # To bias towards high distinctness, we can shuffle the color order 
            # and pick the first available? No, that's deterministic.
            # Let's pick randomly from the domain.
            
            for e in edge_order:
                # Get allowed colors
                mask = domains[e]
                if mask == 0:
                    return False # Dead end
                
                # Convert mask to list of colors? 
                # For 120 bits, iterating is fast enough if we do it smartly.
                # We want to pick a color. 
                # To encourage distinctness, we can prefer colors with low count.
                # But calculating counts for all 120 colors every step is O(120*120) = 14400 ops.
                # 120 steps * 14400 = 1.7M ops. This is fine.
                
                # Get list of available colors
                # Optimization: Precompute bit positions?
                # Just iterate 0..119
                available = []
                # We can bias: 
                # If we just pick random, we might miss the chance to use a new color.
                # Let's pick a random color from the available set.
                
                # To speed up, we can just pick a random integer 0..119 and check if in mask?
                # No, that's inefficient if domain is small.
                
                # Let's extract the set bits.
                # Since 120 is small, a simple loop is okay.
                for c in range(120):
                    if (mask >> c) & 1:
                        available.append(c)
                        
                if not available:
                    return False
                
                # Choose a color
                # Strategy: With probability 0.5, pick a color not yet used (if available).
                # Otherwise, pick random.
                # This biases towards diversity.
                
                new_colors = [c for c in available if used_colors[c] == 0]
                if new_colors and random.random() < 0.7:
                    chosen = random.choice(new_colors)
                else:
                    chosen = random.choice(available)
                
                # Assign
                colors[e] = chosen
                used_colors[chosen] += 1
                
                # Forward Checking:
                # For each cycle involving e, check if the constraint is violated or domain reduced.
                # A cycle (e1, e2, e3, e4) is violated if:
                # - 3 edges are assigned distinct colors c1, c2, c3.
                # - The 4th edge e4 is assigned c4 != c1, c2, c3.
                # In Forward Checking, if e is the last one to be assigned in a cycle?
                # No, we check if the assignment of e *removes* options from neighbors.
                
                # Constraint: No 4 distinct colors.
                # This means for any 4 edges in a cycle, they cannot have 4 distinct values.
                # Equivalently, at least two edges must share a color.
                
                # When we assign color C to edge e:
                # Look at all cycles containing e.
                # Let the other 3 edges be e1, e2, e3.
                # If e1, e2, e3 are all assigned distinct colors C1, C2, C3:
                #   Then e must NOT be C1, C2, or C3? 
                #   Wait. If e1=C1, e2=C2, e3=C3 (distinct), and we assign e=C.
                #   If C is distinct from C1, C2, C3, then we have 4 distinct colors -> VIOLATION.
                #   So, if e1, e2, e3 are assigned distinct colors, e is FORBIDDEN from taking any color distinct from them?
                #   No, e MUST take a color that is one of C1, C2, C3.
                #   So, if e1, e2, e3 are assigned distinct colors, the domain of e is restricted to {C1, C2, C3}.
                
                # If e1, e2, e3 have some duplicates, the constraint is looser.
                # E.g., e1=C1, e2=C1, e3=C2.
                # If e=C3 (distinct from C1, C2), colors are C1, C1, C2, C3 -> 3 distinct. OK.
                # If e=C1, colors C1, C1, C2, C1 -> 2 distinct. OK.
                # So in this case, e can be ANY color?
                # Yes, because even if e is a new color, the set is {C1, C2, C3} which has size 3.
                
                # General Rule for Forward Checking:
                # For a cycle (e, e1, e2, e3):
                # If e1, e2, e3 are all assigned:
                #   Let S be the set of colors of e1, e2, e3.
                #   If len(S) == 3:
                #       Then e MUST take a color in S.
                #       So, domain of e = domain of e intersect S.
                #   If len(S) < 3:
                #       e can take any color (constraint is satisfied regardless of e's color, 
                #       because max distinct colors in the cycle will be at most 3).
                #       Wait, if e takes a color not in S, the set becomes S U {C_e}. Size is len(S)+1 <= 3+1=4?
                #       If len(S)=2, and C_e is new, size is 3. OK.
                #       If len(S)=1, and C_e is new, size is 2. OK.
                #       So yes, if len(S) < 3, no restriction on e.
                
                # If some of e1, e2, e3 are UNASSIGNED:
                # We cannot prune e's domain based on future assignments?
                # Forward checking usually prunes the domain of *unassigned* neighbors based on the *assigned* variable.
                # Here, e is assigned. We need to prune the domains of e1, e2, e3 if they are unassigned.
                
                # Let's rephrase:
                # We just assigned color C to e.
                # For each cycle (e, e1, e2, e3):
                #   Check the status of e1, e2, e3.
                #   If e1, e2, e3 are all assigned:
                #       We already checked validity when we assigned them? 
                #       No, we check consistency NOW.
                #       If the 4 colors (C, C1, C2, C3) have 4 distinct values -> FAIL.
                #   If some are unassigned:
                #       We need to update their domains.
                #       Case: e1 unassigned, e2, e3 assigned.
                #       If e2 and e3 have distinct colors C2, C3:
                #           Then e1 MUST be C, C2, or C3?
                #           Let's check: Cycle colors: C, C1, C2, C3.
                #           If C1 is new (not C, C2, C3), then we have 4 distinct -> VIOLATION.
                #           So e1 is FORBIDDEN from any color not in {C, C2, C3}.
                #           So domain(e1) = domain(e1) & {C, C2, C3}.
                #       If e2 and e3 have the same color C2:
                #           Cycle colors: C, C1, C2, C2.
                #           If C1 is new, set is {C, C2, C1}. Size 3. OK.
                #           So no restriction on e1?
                #           Wait, if C1 is distinct from C and C2, size is 3.
                #           If C1 is C, size 2.
                #           If C1 is C2, size 2.
                #           So yes, no restriction if the other two assigned edges have the same color.
                
                # General Logic for updating domain of unassigned edge X in cycle (e, X, Y, Z) given e=C:
                # Let Y, Z be the other two edges.
                # If Y and Z are assigned:
                #   If color(Y) != color(Z):
                #       X must be in {C, color(Y), color(Z)}
                #   Else:
                #       No restriction (X can be anything)
                # If Y or Z is unassigned:
                #   No restriction on X from this cycle yet (will be handled when Y/Z are assigned)
                
                # Implementation:
                failed = False
                for cycle_id in edge_to_cycles[e]:
                    c_tuple = cycles[cycle_id]
                    # Find the other 3 edges
                    others = [x for x in c_tuple if x != e]
                    e1, e2, e3 = others
                    
                    # Check if all 3 are assigned
                    if colors[e1] != -1 and colors[e2] != -1 and colors[e3] != -1:
                        c1, c2, c3 = colors[e1], colors[e2], colors[e3]
                        if len({chosen, c1, c2, c3}) == 4:
                            return False # Violation detected
                    
                    # Update domains of unassigned edges
                    # Case 1: e1 unassigned, e2, e3 assigned
                    if colors[e1] == -1 and colors[e2] != -1 and colors[e3] != -1:
                        c2, c3 = colors[e2], colors[e3]
                        if c2 != c3:
                            allowed = (1 << chosen) | (1 << c2) | (1 << c3)
                            domains[e1] &= allowed
                            if domains[e1] == 0:
                                return False
                    # Case 2: e2 unassigned, e1, e3 assigned
                    if colors[e2] == -1 and colors[e1] != -1 and colors[e3] != -1:
                        c1, c3 = colors[e1], colors[e3]
                        if c1 != c3:
                            allowed = (1 << chosen) | (1 << c1) | (1 << c3)
                            domains[e2] &= allowed
                            if domains[e2] == 0:
                                return False
                    # Case 3: e3 unassigned, e1, e2 assigned
                    if colors[e3] == -1 and colors[e1] != -1 and colors[e2] != -1:
                        c1, c2 = colors[e1], colors[e2]
                        if c1 != c2:
                            allowed = (1 << chosen) | (1 << c1) | (1 << c2)
                            domains[e3] &= allowed
                            if domains[e3] == 0:
                                return False
                            
                if time.time() - start_time > time_limit:
                    return False
                    
            # If we reach here, all edges assigned and no violations found
            distinct = len(set(colors))
            if distinct > best_distinct:
                best_distinct = distinct
                best_colors = colors.copy()
            return True
        
        # Run multiple restarts
        restarts = 0
        while time.time() - start_time < time_limit:
            restarts += 1
            # Use a different random seed for edge order each restart
            random.seed(restarts * 12345)
            edge_order = list(range(NUM_EDGES))
            random.shuffle(edge_order)
            
            # Try greedy
            if try_greedy():
                # We found a valid solution. 
                # To find better distinctness, we can continue restarting.
                # But greedy with random value choice might be stuck at ~20-30 colors.
                # We can add a local search phase here?
                # No, keep it pure CSP.
                pass
            
            # If we find a solution with very high distinctness, we might stop early?
            # Max possible is 120. 
            # If best_distinct > 60, it's very good.
            if best_distinct > 80:
                break

    solve_with_restart(1000)
    
    if best_colors is None:
        return fallback.tolist()
    
    return best_colors.tolist()

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Result length: {len(res)}")
    print(f"Distinct colors: {len(set(res))}")
