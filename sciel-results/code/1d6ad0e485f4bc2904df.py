import time
import random
import itertools
import math
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    def get_fallback(n):
        # A valid coloring for small n is trivial (e.g., all 0s for n<=5, since C6 needs 6 vertices)
        # For n=5, no C6 exists.
        # For n=6, K6 has C6s. We need to be careful.
        # Actually, for n=5, any coloring works.
        colors = [0] * (n * (n - 1) // 2)
        return {'n': n, 'colors': colors}

    best_result = get_fallback(5)

    def edge_index(i, j, n):
        if i > j:
            i, j = j, i
        return i * (n - 2) - i * (i - 1) // 2 + j - 1

    def build_cycles(n):
        cycles = []
        verts = list(range(n))
        for combo in itertools.combinations(verts, 6):
            v0 = combo[0]
            others = combo[1:]
            for perm in itertools.permutations(others):
                if perm[0] > perm[-1]:
                    continue
                path = [v0] + list(perm)
                cycle_edges = []
                for k in range(6):
                    u = path[k]
                    v = path[(k + 1) % 6]
                    cycle_edges.append(edge_index(u, v, n))
                cycles.append(cycle_edges)
        return cycles

    def solve_walksat(n, time_budget, seed_offset=0):
        if time.time() > deadline - 5:
            return None

        m = n * (n - 1) // 2
        random.seed(42 + seed_offset)

        cycles = build_cycles(n)
        num_cycles = len(cycles)
        
        if num_cycles == 0:
            colors = [0] * m
            return {'n': n, 'colors': colors}

        # Map edge to list of cycle indices
        edge_to_cycles = [[] for _ in range(m)]
        for ci, ce in enumerate(cycles):
            for e in ce:
                edge_to_cycles[e].append(ci)

        # Initial random coloring
        colors = [random.randint(0, 3) for _ in range(m)]
        
        # is_mono[cycle_index] = True if cycle is monochromatic
        is_mono = [False] * num_cycles
        bad_count = 0
        
        # Compute initial state
        for ci, ce in enumerate(cycles):
            c0 = colors[ce[0]]
            if all(colors[e] == c0 for e in ce):
                is_mono[ci] = True
                bad_count += 1

        if bad_count == 0:
            return {'n': n, 'colors': colors}

        start_iter_time = time.time()
        max_iter = 1000000 # High cap, time is the real limit
        
        for i in range(max_iter):
            if time.time() > start_iter_time + time_budget:
                break
            if time.time() > deadline - 5:
                break
            
            if bad_count == 0:
                return {'n': n, 'colors': colors}

            # Pick a random bad cycle
            # To avoid O(N) scan for a bad cycle, we can maintain a list of bad cycle indices
            # But removing from a list is O(N).
            # Alternative: Pick a random edge, if it's in a bad cycle, pick one.
            # Better: Maintain a list `bad_cycles_list`.
            # Since we don't have it, let's just pick a random edge and check if it belongs to any bad cycle.
            # If yes, pick one. If no, repeat.
            # This might be inefficient if bad_count is small.
            
            # Let's maintain a list of bad cycle indices for efficiency.
            # We'll update it in the flip step.
            # For now, let's just pick a random edge and hope it's in a bad cycle?
            # No, that's bad.
            
            # Let's keep a list `bad_indices`.
            # We need to initialize it.
            # Re-initializing logic:
            # We'll just scan for a bad cycle if the list is not maintained, but that's slow.
            # Let's implement the list maintenance.
            
            pass # Placeholder, we need to restructure the loop to maintain bad_indices

        # Restructure:
        bad_indices = [i for i in range(num_cycles) if is_mono[i]]
        
        for i in range(max_iter):
            if time.time() > start_iter_time + time_budget:
                break
            if time.time() > deadline - 5:
                break
            
            if not bad_indices:
                return {'n': n, 'colors': colors}

            # Pick random bad cycle
            ci = bad_indices[random.randint(0, len(bad_indices) - 1)]
            ce = cycles[ci]
            
            # Pick random edge in this cycle
            e = ce[random.randint(0, 5)]
            
            c_old = colors[e]
            # Pick new color
            c_new = random.randint(0, 3)
            while c_new == c_old:
                c_new = random.randint(0, 3)
            
            # Apply flip
            colors[e] = c_new
            
            # Update is_mono and bad_indices for all cycles containing e
            # We need to update bad_indices carefully.
            # We will create a new set of bad indices? No, that's O(N).
            # We will update the list in place.
            
            # Cycles containing e
            affected_cycles = edge_to_cycles[e]
            
            # We will mark which cycles changed status
            # and update bad_indices.
            
            # To avoid O(N) update of bad_indices, we can use a "dirty" flag or just rebuild?
            # Rebuilding bad_indices is O(N_cycles) which is ~700k. Too slow per step.
            
            # Instead, we can maintain bad_indices as a set?
            # Set operations are O(1) average.
            
            # Let's switch bad_indices to a set.
            # But we need to pick randomly from a set.
            # We can convert to list for picking, but that's O(K) where K is bad_count.
            # If bad_count is small, this is fine.
            
            # Let's stick to a list, but update it by:
            # 1. Identifying cycles that were bad and are no longer bad.
            # 2. Identifying cycles that were not bad and are now bad.
            # 3. Removing/Adding from the list.
            # Removal from list is O(N).
            
            # Given the constraints, let's just pick a random edge from the whole graph,
            # and if it's in a bad cycle, use that.
            # Probability of edge being in a bad cycle:
            # Each bad cycle has 6 edges. Total bad edges = 6 * bad_count.
            # Total edges = m.
            # Prob = 6 * bad_count / m.
            # For n=17, m=136. bad_count ~ 100. Prob ~ 600/136 > 1.
            # So almost every edge is in a bad cycle.
            # This strategy works well when bad_count is high.
            # When bad_count is low, we might need many tries.
            
            # Let's use the "Pick random edge" strategy.
            
            # Re-do the loop logic:
            # We already flipped e. We need to update the state.
            # But we picked e from a specific bad cycle ci.
            # We must update ci's status.
            
            # Let's revert the "pick random edge" idea and stick to "pick from bad_indices".
            # To make removal from bad_indices fast, we can use a dictionary or just accept O(N) if N is small?
            # N_cycles is 700k. O(N) is too slow.
            
            # Solution: Use a set for bad_indices.
            # To pick randomly: convert to list.
            # If bad_count is 100, converting a set of 100 to list is fast.
            # As bad_count drops, it's faster.
            
            # Let's restart the solve function with this design.

        return None

    # Corrected solve function
    def solve_walksat_v2(n, time_budget, seed_offset=0):
        if time.time() > deadline - 5:
            return None

        m = n * (n - 1) // 2
        random.seed(42 + seed_offset)

        cycles = build_cycles(n)
        num_cycles = len(cycles)
        
        if num_cycles == 0:
            colors = [0] * m
            return {'n': n, 'colors': colors}

        edge_to_cycles = [[] for _ in range(m)]
        for ci, ce in enumerate(cycles):
            for e in ce:
                edge_to_cycles[e].append(ci)

        colors = [random.randint(0, 3) for _ in range(m)]
        
        is_mono = [False] * num_cycles
        bad_set = set()
        
        for ci, ce in enumerate(cycles):
            c0 = colors[ce[0]]
            if all(colors[e] == c0 for e in ce):
                is_mono[ci] = True
                bad_set.add(ci)

        if not bad_set:
            return {'n': n, 'colors': colors}

        start_iter_time = time.time()
        
        for i in range(1000000):
            if time.time() > start_iter_time + time_budget:
                break
            if time.time() > deadline - 5:
                break
            
            if not bad_set:
                return {'n': n, 'colors': colors}

            # Pick random bad cycle
            # Convert set to list for random access
            bad_list = list(bad_set)
            ci = bad_list[random.randint(0, len(bad_list) - 1)]
            ce = cycles[ci]
            
            # Pick random edge in cycle
            e = ce[random.randint(0, 5)]
            
            c_old = colors[e]
            c_new = random.randint(0, 3)
            while c_new == c_old:
                c_new = random.randint(0, 3)
            
            colors[e] = c_new
            
            # Update affected cycles
            affected = edge_to_cycles[e]
            
            for aci in affected:
                ace = cycles[aci]
                
                # Check if aci was bad
                was_bad = is_mono[aci]
                
                # Check if it is bad now
                # It is bad if all edges have the same color.
                # Since e changed, we only need to check if the other 5 edges match the new color of e (c_new).
                # If they match c_new, it is now bad.
                # If they don't match c_new, it is not bad.
                # Note: It cannot be bad with c_old anymore because e is c_new.
                
                is_bad_now = True
                for ed in ace:
                    if ed == e:
                        continue
                    if colors[ed] != c_new:
                        is_bad_now = False
                        break
                
                if was_bad and not is_bad_now:
                    is_mono[aci] = False
                    bad_set.discard(aci)
                elif not was_bad and is_bad_now:
                    is_mono[aci] = True
                    bad_set.add(aci)
                # If was_bad and is_bad_now? Impossible, e changed color.
                # If not was_bad and not is_bad_now? No change.

        return None

    # Targets
    targets = [18, 17, 16, 15, 14, 13, 12]
    
    # Time allocation
    time_alloc = {18: 20, 17: 40, 16: 30, 15: 20, 14: 10, 13: 5, 12: 5}

    for n in targets:
        if time.time() > deadline - 5:
            break
            
        tb = time_alloc.get(n, 10)
        
        # Try multiple restarts if time permits
        # For n=18, maybe 1 restart is enough if it's hard.
        # For n=17, it should be easy.
        num_restarts = 2 if n >= 17 else 1
        restart_budget = tb / num_restarts
        
        for r in range(num_restarts):
            if time.time() > deadline - 5:
                break
            result = solve_walksat_v2(n, restart_budget, seed_offset=r * 1000 + n)
            if result:
                return result

    # Fallback
    return best_result

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
