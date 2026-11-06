import random
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n = 20
    num_colors = 20
    
    # Precompute cycles
    all_cycles = []
    edge_to_cycles = defaultdict(list)
    
    for u, v, w, x in itertools.combinations(range(n), 4):
        cycles = [
            [(u, v), (v, w), (w, x), (x, u)],
            [(u, v), (v, x), (x, w), (w, u)],
            [(u, w), (w, v), (v, x), (x, u)],
        ]
        for cycle in cycles:
            norm_cycle = tuple(sorted((min(e), max(e)) for e in cycle))
            if len(set(norm_cycle)) == 4:
                idx = len(all_cycles)
                all_cycles.append(norm_cycle)
                for e in norm_cycle:
                    edge_to_cycles[e].append(idx)
    
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def can_add_edge(e, c, edge_colors):
        for idx in edge_to_cycles[e]:
            cycle = all_cycles[idx]
            other_colors = []
            valid = True
            for oe in cycle:
                if oe == e:
                    continue
                if oe not in edge_colors:
                    valid = False
                    break
                other_colors.append(edge_colors[oe])
            if not valid:
                continue
            if c in other_colors:
                return False
            if len(set(other_colors)) < 3:
                return False
        return True

    def find_best_color(e, edge_colors):
        """Find a valid color, preferring those that minimize future conflicts."""
        valid_colors = []
        # Count usage of colors in incident cycles
        color_freq = defaultdict(int)
        for idx in edge_to_cycles[e]:
            cycle = all_cycles[idx]
            for oe in cycle:
                if oe != e and oe in edge_colors:
                    color_freq[edge_colors[oe]] += 1
        
        for c in range(num_colors):
            if can_add_edge(e, c, edge_colors):
                valid_colors.append((color_freq[c], c))
        
        if not valid_colors:
            return None
        
        # Sort by frequency (ascending) so we prefer rare colors
        valid_colors.sort()
        # Pick randomly from the top few to maintain diversity
        top_k = min(3, len(valid_colors))
        _, c = random.choice(valid_colors[:top_k])
        return c

    # Phase 1: Greedy Random Initialization
    edge_colors = {}
    random.shuffle(all_edges)
    for e in all_edges:
        c = find_best_color(e, edge_colors)
        if c is not None:
            edge_colors[e] = c

    # Phase 2: Local Search
    best_count = len(edge_colors)
    best_state = dict(edge_colors)
    missing_set = set(e for e in all_edges if e not in edge_colors)
    
    def save_if_better():
        nonlocal best_count, best_state
        if len(edge_colors) > best_count:
            best_count = len(edge_colors)
            best_state = dict(edge_colors)

    # Move types
    while time.time() < deadline:
        # 1. Add
        if missing_set:
            # Sample a few missing edges to find a good candidate
            sample_size = min(len(missing_set), 20)
            sample = random.sample(list(missing_set), sample_size)
            best_add = None
            best_score = -1
            for e in sample:
                c = find_best_color(e, edge_colors)
                if c is not None:
                    # Score: how many colors are available? (proxy for flexibility)
                    # Or just take the first one that works?
                    # Let's just take the first one to save time, or pick random
                    best_add = (e, c)
                    break # Optimization: don't search all if one works
                    
            if best_add:
                e, c = best_add
                edge_colors[e] = c
                missing_set.discard(e)
                save_if_better()
                continue

        # 2. Recolor
        if edge_colors:
            e_recolor = random.choice(list(edge_colors.keys()))
            old_c = edge_colors[e_recolor]
            # Try to find a different valid color
            new_c = find_best_color(e_recolor, edge_colors)
            if new_c is not None and new_c != old_c:
                # Check if it actually changes anything? 
                # If new_c is valid, it's a safe move.
                edge_colors[e_recolor] = new_c
                # Recoloring doesn't change edge count, but might enable adds
                # So we don't save as "better" unless it leads to an add later?
                # Actually, keeping the same count is fine, it might help.
                # But we only save if count increases.
                pass

        # 3. Swap (1 remove, 1 add)
        if edge_colors and missing_set and random.random() < 0.5:
            e_remove = random.choice(list(edge_colors.keys()))
            # Pick a missing edge that is "close" to e_remove? 
            # Or just random.
            e_add = random.choice(list(missing_set))
            
            old_c = edge_colors[e_remove]
            del edge_colors[e_remove]
            missing_set.add(e_remove)
            
            c_add = find_best_color(e_add, edge_colors)
            if c_add is not None:
                edge_colors[e_add] = c_add
                missing_set.discard(e_add)
                save_if_better()
                
                # Try to re-add e_remove
                c_rem = find_best_color(e_remove, edge_colors)
                if c_rem is not None:
                    edge_colors[e_remove] = c_rem
                    missing_set.discard(e_remove)
                    save_if_better()
            else:
                # Revert
                c_rem = find_best_color(e_remove, edge_colors)
                if c_rem is not None:
                    edge_colors[e_remove] = c_rem
                    missing_set.discard(e_remove)
                else:
                    edge_colors[e_remove] = old_c

        # 4. Double Swap (2 remove, 2 add)
        if random.random() < 0.2 and len(edge_colors) >= 2 and len(missing_set) >= 2:
            e1, e2 = random.sample(list(edge_colors.keys()), 2)
            e3, e4 = random.sample(list(missing_set), 2)
            
            c1 = edge_colors[e1]
            c2 = edge_colors[e2]
            del edge_colors[e1]
            del edge_colors[e2]
            missing_set.add(e1)
            missing_set.add(e2)
            
            c3 = find_best_color(e3, edge_colors)
            if c3 is not None:
                edge_colors[e3] = c3
                missing_set.discard(e3)
                c4 = find_best_color(e4, edge_colors)
                if c4 is not None:
                    edge_colors[e4] = c4
                    missing_set.discard(e4)
                    save_if_better()
                    
                    # Try to re-add e1, e2
                    c1_new = find_best_color(e1, edge_colors)
                    if c1_new is not None:
                        edge_colors[e1] = c1_new
                        missing_set.discard(e1)
                        save_if_better()
                    c2_new = find_best_color(e2, edge_colors)
                    if c2_new is not None:
                        edge_colors[e2] = c2_new
                        missing_set.discard(e2)
                        save_if_better()
                else:
                    # Revert e4
                    del edge_colors[e3]
                    missing_set.add(e3)
            # Revert e1, e2 if not added back
            if e1 not in edge_colors:
                 edge_colors[e1] = c1
                 missing_set.discard(e1)
            if e2 not in edge_colors:
                 edge_colors[e2] = c2
                 missing_set.discard(e2)

    # Restore best state
    edge_colors = best_state
    
    # Final Polish: Try to add any remaining edges
    missing_set = set(e for e in all_edges if e not in edge_colors)
    for e in list(missing_set):
        if time.time() > deadline:
            break
        c = find_best_color(e, edge_colors)
        if c is not None:
            edge_colors[e] = c
            missing_set.discard(e)
            
    # Final Cleanup: Remove any conflicts (should be none if logic is correct, but safe)
    # Since can_add_edge ensures validity, conflicts shouldn't exist.
    # But if we did any "force" moves, we might have them. 
    # The moves above only add if valid. So result is valid.
    
    result = []
    for (u, v), c in edge_colors.items():
        result.append([u, v, c])
    return result
