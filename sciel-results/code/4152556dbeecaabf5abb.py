import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 148.0  # Safety margin before 150s limit
    
    def time_check():
        return time.time() > deadline

    # We aim to find a valid 4-coloring of K_n with no monochromatic C6.
    # Start with n=17 (known to exist). If found, try n=18.
    
    # Helper: Check if a specific set of edges forms a C6 in a given color
    # This is slow, so we use delta updates in the main loop.
    
    # Pre-compute all 6-cycles in K_n
    # A 6-cycle is a set of 6 edges.
    # Number of 6-cycles in K_n: n! / (12 * (n-6)!)
    # For n=17: 17! / (12 * 11!) = (17*16*15*14*13*12) / 12 = 17*16*15*14*13 = 742,560
    # This is manageable.
    
    def generate_6_cycles(n):
        cycles = []
        # We can represent a cycle by its 6 edges.
        # To avoid duplicates, we can fix vertex 0 in the cycle? No, cycles can be anywhere.
        # It's easier to iterate over combinations of 6 vertices and then permutations?
        # That's too slow.
        # Better: Iterate over all 6-tuples (v0..v5) with v0 < v1 < ... ? No, cycles are not ordered that way simply.
        
        # Let's use a smarter generation.
        # A C6 is a sequence of 6 distinct vertices v0..v5 such that (v0,v1), (v1,v2), ..., (v5,v0) are edges.
        # We can generate them by iterating over all 6-combinations of vertices, and for each combination,
        # check the 60 distinct cycles (Hamiltonian cycles on K6).
        
        from itertools import combinations, permutations
        
        # For each set of 6 vertices, there are (6-1)! / 2 = 60 distinct cycles.
        # Total combinations C(17, 6) = 12376.
        # 12376 * 60 = 742,560 cycles.
        
        # We will store each cycle as a tuple of 6 edge indices.
        # We need a mapping from (i, j) to edge index.
        edge_index = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_index[(i, j)] = idx
                edge_index[(j, i)] = idx
                idx += 1
        
        num_edges = n * (n - 1) // 2
        
        # Generate all cycles
        for verts in combinations(range(n), 6):
            # Generate all Hamiltonian cycles on these 6 vertices
            # Fix v0 to be the smallest vertex in the set to reduce symmetry?
            # Actually, we just need the set of edges.
            # We can iterate over permutations of the other 5 vertices.
            # (v0, v1, v2, v3, v4, v5, v0)
            # To avoid duplicates, we can fix v0 and v1?
            # The number of distinct cycles on 6 labeled nodes is 60.
            # We can just generate all 5! = 120 permutations of the last 5, and divide by 2 (direction).
            # Or just generate all and use a set of frozensets of edge indices?
            # That might be memory heavy but 742k cycles * 6 edges = 4.4M ints. OK.
            
            v0 = verts[0]
            others = verts[1:]
            for perm in permutations(others):
                # Cycle: v0 -> perm[0] -> perm[1] -> perm[2] -> perm[3] -> perm[4] -> v0
                # To avoid duplicates (reverse direction), we can enforce perm[0] < perm[4]?
                # No, that's tricky.
                # Let's just generate all 120 and use a set of frozensets of edges.
                # Wait, 12376 * 120 = 1.4M checks. Fast enough.
                
                edges = []
                valid = True
                prev = v0
                for v in perm:
                    u, w = sorted((prev, v))
                    edges.append(edge_index[(u, w)])
                    prev = v
                # Close the cycle
                u, w = sorted((prev, v0))
                edges.append(edge_index[(u, w)])
                
                # Check if this is a simple cycle (distinct vertices are guaranteed by perms)
                # The edges are distinct?
                # In a simple graph, yes.
                
                # To avoid duplicates, we can normalize the edge set.
                # But we need to know the structure for delta updates.
                # Actually, for delta updates, we just need to know which cycles contain a given edge.
                # So let's build: edge_idx -> list of cycle_idx
                
                # Let's store the cycle as a tuple of edge indices.
                # We will add it to a global list if it's not a duplicate.
                # Using a set of frozensets is the easiest way to deduplicate.
                # But we need to map back to the tuple for the list.
                
                # Optimization: 
                # Instead of storing all cycles explicitly and deduplicating,
                # we can just generate them and count how many times each edge appears?
                # No, we need to check if a specific cycle is monochromatic.
                
                # Let's stick to:
                # 1. Generate all cycles.
                # 2. Store them in a list `all_cycles`.
                # 3. Build `edge_to_cycles` mapping.
                
                # To save memory and time, we can skip the deduplication step if we are careful,
                # but duplicates will just mean we check the same cycle multiple times.
                # 1.4M iterations is fast.
                
                all_cycles.append(tuple(edges))
                
        # Wait, the list `all_cycles` is local to the function. I need to define it outside or pass it.
        # Let's refactor.
        
        return all_cycles, edge_index, num_edges

    # Refactored: Generate cycles and edge mapping
    def prepare_graph(n):
        from itertools import combinations, permutations
        
        edge_index = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_index[(i, j)] = idx
                edge_index[(j, i)] = idx
                idx += 1
        
        num_edges = n * (n - 1) // 2
        all_cycles = []
        
        for verts in combinations(range(n), 6):
            v0 = verts[0]
            others = verts[1:]
            seen_in_this_vert_set = set()
            for perm in permutations(others):
                edges = []
                prev = v0
                for v in perm:
                    u, w = sorted((prev, v))
                    edges.append(edge_index[(u, w)])
                    prev = v
                u, w = sorted((prev, v0))
                edges.append(edge_index[(u, w)])
                
                # Normalize for deduplication
                # A cycle is the same if the set of edges is the same.
                key = frozenset(edges)
                if key not in seen_in_this_vert_set:
                    seen_in_this_vert_set.add(key)
                    all_cycles.append(tuple(edges))
                    
        # Build edge to cycles map
        edge_to_cycles = [[] for _ in range(num_edges)]
        for c_idx, cycle in enumerate(all_cycles):
            for e_idx in cycle:
                edge_to_cycles[e_idx].append(c_idx)
                
        return all_cycles, edge_to_cycles, edge_index, num_edges

    # We will try to find a coloring for n=17 first.
    # If successful, we might try n=18 with remaining time, but n=17 is the goal to beat n=9 or match seed.
    # The seed returns n=17. So we must at least return n=17.
    # The task says "Maximize n". Seed is 1.0. If we find 18, we get > 1.0.
    
    # Let's start with n=17.
    n = 17
    all_cycles, edge_to_cycles, edge_index, num_edges = prepare_graph(n)
    
    # Initial random coloring
    colors = [random.randint(0, 3) for _ in range(num_edges)]
    
    # Count monochromatic cycles
    # For each cycle, check if all 6 edges have the same color.
    mono_count = 0
    cycle_mono_status = [0] * len(all_cycles) # 1 if mono, 0 else
    
    for c_idx, cycle in enumerate(all_cycles):
        c0 = colors[cycle[0]]
        is_mono = True
        for e_idx in cycle[1:]:
            if colors[e_idx] != c0:
                is_mono = False
                break
        if is_mono:
            mono_count += 1
            cycle_mono_status[c_idx] = 1
        else:
            cycle_mono_status[c_idx] = 0
            
    # Simulated Annealing
    # We want to minimize mono_count.
    # Temperature T starts high, decreases.
    # Move: pick a random edge, change its color to a random other color.
    # Delta:
    #   Let old_color = colors[e]
    #   Let new_color = c_new
    #   For each cycle in edge_to_cycles[e]:
    #       If cycle was mono (with old_color):
    #           It becomes non-mono if new_color != old_color (which it is).
    #           So delta -= 1
    #       If cycle was non-mono:
    #           Check if it becomes mono with new_color.
    #           If all other 5 edges have color new_color, then it becomes mono.
    #           So delta += 1
    
    # We need to be careful with the "becomes mono" check.
    # It requires checking 5 other edges.
    # Total edges per cycle = 6.
    # For each move, we check up to ~ (num_cycles_per_edge) cycles.
    # How many cycles share an edge?
    # In K17, an edge (i,j) is in how many C6s?
    # Fix edge (0,1). We need 4 more vertices from 15.
    # The cycle must go 0-1-...-0.
    # The path from 1 to 0 of length 5 (4 intermediate vertices) must be in the remaining 15 vertices.
    # Number of such paths?
    # It's roughly proportional to C(15, 4) * (something).
    # C(15,4) = 1365.
    # So ~1000-2000 cycles per edge.
    # 1000 checks per move.
    # 150s / 1000 = 150 moves? No, Python is slower.
    # 1000 iterations in Python is ~0.01s? Maybe 0.05s.
    # So 150s / 0.05s = 3000 moves.
    # That's very few.
    
    # We need to speed up the delta calculation.
    # Optimization:
    # Precompute for each edge and each color, the number of "potential mono cycles" that would be formed if this edge were that color?
    # No, the other 5 edges must match.
    
    # Alternative:
    # Keep a count for each cycle: how many edges of each color does it have?
    # `cycle_color_counts[c_idx][color]`
    # If we change edge e in cycle c from c_old to c_new:
    #   cycle_color_counts[c][c_old] -= 1
    #   cycle_color_counts[c][c_new] += 1
    #   If cycle_color_counts[c][c_new] == 6:
    #       delta += 1 (became mono)
    #   If cycle_color_counts[c][c_old] == 5 (before decrement) i.e. 4 after?
    #       Wait.
    #   Before change:
    #       If count[c_old] == 6, it was mono. After change, count[c_old] = 5. Not mono. Delta -= 1.
    #       If count[c_old] < 6, it was not mono.
    #   After change:
    #       If count[c_new] == 6, it is mono. Delta += 1.
    #       Else not mono.
    
    # This is O(1) per cycle!
    # We just need to update the counts.
    
    # Let's re-initialize with counts.
    
    def init_counts(colors, all_cycles):
        counts = [[0]*4 for _ in range(len(all_cycles))]
        for c_idx, cycle in enumerate(all_cycles):
            for e_idx in cycle:
                counts[c_idx][colors[e_idx]] += 1
        return counts

    counts = init_counts(colors, all_cycles)
    
    # Recalculate mono_count from counts
    mono_count = sum(1 for c in counts if any(x == 6 for x in c))
    
    # SA Loop
    T = 1.0
    cooling = 0.999
    
    # We want to find mono_count == 0.
    # If we reach 0, we can stop and verify.
    
    best_mono = mono_count
    best_colors = colors[:]
    
    while not time_check():
        if mono_count == 0:
            # Found a valid coloring!
            break
            
        # Pick random edge
        e_idx = random.randint(0, num_edges - 1)
        old_color = colors[e_idx]
        
        # Pick new color different from old
        new_color = random.choice([c for c in range(4) if c != old_color])
        
        # Calculate delta
        delta = 0
        for c_idx in edge_to_cycles[e_idx]:
            c_counts = counts[c_idx]
            
            # Old state
            was_mono = (c_counts[old_color] == 6)
            
            # Apply change
            c_counts[old_color] -= 1
            c_counts[new_color] += 1
            
            # New state
            is_mono = (c_counts[new_color] == 6)
            
            if was_mono and not is_mono:
                delta -= 1
            elif not was_mono and is_mono:
                delta += 1
            # If was_mono and is_mono: impossible since old != new
            # If not was_mono and not is_mono: delta 0
            
        # Acceptance
        if delta < 0 or random.random() < pow(2.71828, -delta / T):
            # Accept
            colors[e_idx] = new_color
            mono_count += delta
            if mono_count < best_mono:
                best_mono = mono_count
                best_colors = colors[:]
        else:
            # Reject: revert counts
            for c_idx in edge_to_cycles[e_idx]:
                c_counts = counts[c_idx]
                c_counts[old_color] += 1
                c_counts[new_color] -= 1
                
        T *= cooling
        
    # After SA, if we have a valid coloring (mono_count == 0), return it.
    # If not, we might have a good candidate.
    # But we MUST return a valid coloring.
    # If best_mono > 0, we have to fall back to a known valid coloring or a smaller n.
    
    # Let's verify the best_colors found.
    # And also, if we didn't find 0, we should try to find a known n=17 coloring using the cyclic method from the seed,
    # but faster? Or just use the seed's method if it works?
    # The seed's method worked (score 1.0).
    # But my SA might find n=18!
    
    # If mono_count == 0 for n=17, we have a valid n=17.
    # Can we try n=18?
    # If we have time left.
    
    if mono_count == 0:
        # Convert best_colors to the required format
        result_colors = best_colors[:]
        return {'n': 17, 'colors': result_colors}
        
    # If SA failed to find 0 for n=17, we need a fallback.
    # The seed program finds n=17 via cyclic search.
    # Let's implement a faster cyclic search or just use the seed's logic if it's fast enough?
    # The seed's logic iterates 4^8 = 65536 cases.
    # Each case checks 4 colors for C6.
    # has_c6_in_diffs is O(D^5) roughly?
    # D is small (4 or 8 diffs).
    # It should be fast.
    
    # Let's try the cyclic search for n=17 as a fallback.
    
    pairs = [(1, 16), (2, 15), (3, 14), (4, 13), (5, 12), (6, 11), (7, 10), (8, 9)]
    
    def has_c6_in_diffs(D):
        D_set = set(D)
        # Check for C6 in Cayley graph
        # v0=0
        # v1 in D
        # v2 in v1+D
        # ...
        # v5 in v4+D
        # -v5 in D
        
        # Optimization: Precompute sums?
        # For small D, direct check is fine.
        
        for v1 in D_set:
            for v2 in [(v1 + d) % 17 for d in D_set]:
                if v2 == 0 or v2 == v1: continue
                for v3 in [(v2 + d) % 17 for d in D_set]:
                    if v3 in (0, v1, v2): continue
                    for v4 in [(v3 + d) % 17 for d in D_set]:
                        if v4 in (0, v1, v2, v3): continue
                        for v5 in [(v4 + d) % 17 for d in D_set]:
                            if v5 in (0, v1, v2, v3, v4): continue
                            if (-v5) % 17 in D_set:
                                return True
        return False

    found_n17 = None
    for assignment in range(4**8):
        if time_check(): break
        
        colors_for_pairs = [0] * 8
        temp = assignment
        for i in range(8):
            colors_for_pairs[i] = temp % 4
            temp //= 4
        
        diff_sets = [set() for _ in range(4)]
        for i, c in enumerate(colors_for_pairs):
            p = pairs[i]
            diff_sets[c].add(p[0])
            diff_sets[c].add(p[1])
        
        valid = True
        for c in range(4):
            if has_c6_in_diffs(diff_sets[c]):
                valid = False
                break
        
        if valid:
            color_map = {}
            for c in range(4):
                for d in diff_sets[c]:
                    color_map[d] = c
            
            n = 17
            colors_list = []
            for i in range(n):
                for j in range(i + 1, n):
                    diff = (j - i) % 17
                    colors_list.append(color_map[diff])
            
            # Verify
            def verify_no_c6(colors_list, n):
                for c in range(4):
                    adj = [[False] * n for _ in range(n)]
                    idx = 0
                    for i in range(n):
                        for j in range(i + 1, n):
                            if colors_list[idx] == c:
                                adj[i][j] = adj[j][i] = True
                            idx += 1
                    # Check for C6
                    for a in range(n):
                        for b in range(n):
                            if not adj[a][b]: continue
                            for c_v in range(n):
                                if c_v == a or c_v == b or not adj[b][c_v]: continue
                                for d_v in range(n):
                                    if d_v in (a, b, c_v) or not adj[c_v][d_v]: continue
                                    for e_v in range(n):
                                        if e_v in (a, b, c_v, d_v) or not adj[d_v][e_v]: continue
                                        if adj[e_v][a] and e_v != b:
                                            return False
                return True
            
            if verify_no_c6(colors_list, n):
                found_n17 = {'n': n, 'colors': colors_list}
                break

    if found_n17:
        return found_n17

    # If all else fails, return n=9 (trivial)
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
