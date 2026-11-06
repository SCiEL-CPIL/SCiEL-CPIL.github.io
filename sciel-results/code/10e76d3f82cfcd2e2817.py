import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 20
    num_colors = 20
    
    # Precompute all 4-cycles
    # A 4-cycle is defined by 4 distinct vertices u, v, w, x.
    # The cycles are (u,v,w,x), (u,v,x,w), (u,w,v,x) in terms of edge sets.
    # Edges are stored as (min, max).
    
    all_cycles = []
    edge_to_cycles = {}
    
    vertices = list(range(n))
    for comb in itertools.combinations(vertices, 4):
        u, v, w, x = comb
        # Cycle 1: u-v-w-x-u
        c1 = [(u,v), (v,w), (w,x), (x,u)]
        # Cycle 2: u-v-x-w-u
        c2 = [(u,v), (v,x), (x,w), (w,u)]
        # Cycle 3: u-w-v-x-u
        c3 = [(u,w), (w,v), (v,x), (x,u)]
        
        for cycle in [c1, c2, c3]:
            # Normalize edges
            norm_cycle = []
            for e in cycle:
                norm_e = (min(e[0], e[1]), max(e[0], e[1]))
                norm_cycle.append(norm_e)
            # Check for duplicates in the cycle itself (shouldn't happen with distinct vertices)
            if len(set(norm_cycle)) == 4:
                all_cycles.append(norm_cycle)
                for e in norm_cycle:
                    if e not in edge_to_cycles:
                        edge_to_cycles[e] = []
                    edge_to_cycles[e].append(len(all_cycles) - 1)

    # State
    # edge_colors: dict (u,v) -> color
    # valid_edges: set of (u,v)
    
    # Initialize with a random graph
    # Density 0.4 is a good start
    p = 0.4
    edge_colors = {}
    for u in range(n):
        for v in range(u+1, n):
            if random.random() < p:
                edge_colors[(u,v)] = random.randint(0, num_colors - 1)
    
    def get_conflict_count():
        count = 0
        for cycle in all_cycles:
            # Check if all edges in cycle exist
            if all(e in edge_colors for e in cycle):
                colors = [edge_colors[e] for e in cycle]
                if len(set(colors)) < 4:
                    count += 1
        return count

    def is_color_valid_for_edge(e, c):
        """Check if assigning color c to edge e results in all cycles containing e being rainbow."""
        if e not in edge_to_cycles:
            return True
        for idx in edge_to_cycles[e]:
            cycle = all_cycles[idx]
            # Check if all other edges in this cycle exist
            if not all(other_e in edge_colors for other_e in cycle if other_e != e):
                continue
            # Check colors
            other_colors = [edge_colors[other_e] for other_e in cycle if other_e != e]
            # The new color c must be distinct from all other_colors
            if c in other_colors:
                return False
            # Also, the other_colors must be distinct from each other? 
            # No, the condition is that the 4 edges have distinct colors.
            # If other_colors have duplicates, the cycle is already invalid regardless of c.
            # But wait, if other_colors have duplicates, the cycle is ALREADY bad.
            # Changing e's color won't fix a duplicate among the other 3.
            # So if len(set(other_colors)) < 3, this cycle is unsalvageable by changing e alone.
            # However, for the purpose of "is this move valid", we usually check if the move *introduces* a conflict or *fixes* one.
            # If the cycle was already bad due to other edges, changing e might not fix it, but it also doesn't make it *worse* in terms of distinctness count?
            # Actually, if other_colors = [1, 1, 2], and we pick c=3, colors are [3,1,1,2] -> set size 3. Still bad.
            # If we pick c=1, colors are [1,1,1,2] -> set size 2. Worse.
            # So we just need to ensure that IF the other 3 are distinct, c is not in them.
            # If the other 3 are NOT distinct, the cycle is already bad.
            # Does adding c make it "more bad"? Not really, it's just bad.
            # But our goal is to reach a state with 0 bad cycles.
            # So if a cycle is already bad, we can't "fix" it by just changing e unless the badness was caused by e's previous color (which we are changing away from) AND the other 3 are distinct.
            # If the other 3 are not distinct, this cycle will remain bad.
            # For the heuristic "try to add edge", we want to know if we can add it such that it doesn't *create* new bad cycles that were previously good.
            # So: For all cycles containing e:
            #   If the other 3 edges are present:
            #     If their colors are distinct (len(set)==3):
            #        Then c must NOT be in their colors.
            #     If their colors are NOT distinct:
            #        The cycle is already bad. Adding e doesn't change the fact that it's bad.
            #        So we don't care about this cycle for the validity of the *move*?
            #        Actually, if we are in a state with 0 bad cycles, all other 3 MUST be distinct.
            #        So if we are growing from a valid state, this check is sufficient.
            #        If we are repairing, we might have bad cycles.
            
            if len(set(other_colors)) == 3:
                if c in other_colors:
                    return False
            # If len(set(other_colors)) < 3, the cycle is already invalid.
            # We assume we are trying to reach a valid state.
            # If we are in a valid state, this case doesn't happen.
            # If we are repairing, we might accept a move that doesn't fix this cycle but doesn't break others?
            # For simplicity, let's assume we are operating from a valid state or trying to maintain validity.
            
        return True

    # Phase 1: Repair initial random graph
    # We want to reach a state with 0 conflicts.
    
    # To speed up, we can just restart if it takes too long?
    # Or use a more aggressive repair.
    
    # Let's try a simpler approach: 
    # Start with empty graph. Add edges one by one if possible.
    # This guarantees validity at every step.
    
    edge_colors = {}
    
    # Shuffle all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    random.shuffle(all_edges)
    
    # Greedy Addition
    for e in all_edges:
        # Try to find a valid color
        # Shuffle colors for diversity
        colors = list(range(num_colors))
        random.shuffle(colors)
        
        added = False
        for c in colors:
            if is_color_valid_for_edge(e, c):
                edge_colors[e] = c
                added = True
                break
        
        if not added:
            # Could try to fix existing edges to make room, but that's complex.
            # For now, skip.
            pass
    
    # Current edge count
    current_edges = len(edge_colors)
    
    # Phase 2: Local Search to improve
    # If we have time, we can try to remove some edges and re-add them in a different order or with different colors to potentially fit more.
    # Or, try to add skipped edges by removing conflicting ones.
    
    # Let's identify skipped edges
    skipped_edges = [e for e in all_edges if e not in edge_colors]
    
    # We can iterate through skipped edges and try to "force" them in.
    # To force an edge e:
    # 1. Find a cycle containing e that is blocking it (i.e., other 3 edges have colors covering all 20? No, just covering c).
    # 2. Actually, if is_color_valid_for_edge returns False for all c, it means for every c, there is a cycle where other 3 colors include c.
    # 3. This is very unlikely with 20 colors unless the graph is very dense around e.
    
    # A better strategy for Phase 2: 
    # Randomly pick an existing edge, remove it.
    # Then try to re-add it and other skipped edges.
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Pick a random existing edge to remove
        if not edge_colors:
            break
            
        e_to_remove = random.choice(list(edge_colors.keys()))
        del edge_colors[e_to_remove]
        
        # Now try to re-add e_to_remove and maybe some skipped edges
        # We want to maximize edges, so we should try to add e_to_remove back, 
        # but perhaps with a different color that allows other edges to be added?
        # Or just try to add e_to_remove back.
        
        # Let's try to add e_to_remove back first.
        added = False
        colors = list(range(num_colors))
        random.shuffle(colors)
        for c in colors:
            if is_color_valid_for_edge(e_to_remove, c):
                edge_colors[e_to_remove] = c
                added = True
                break
        
        # If we didn't add it back, we lost an edge. 
        # But maybe we can add a DIFFERENT edge that was previously skipped?
        # Let's try to add a random skipped edge.
        if skipped_edges:
            # Pick a random skipped edge
            e_skip = random.choice(skipped_edges)
            # Try to add it
            colors = list(range(num_colors))
            random.shuffle(colors)
            for c in colors:
                if is_color_valid_for_edge(e_skip, c):
                    edge_colors[e_skip] = c
                    skipped_edges.remove(e_skip)
                    # We added an edge. Good.
                    # We might have removed one, so net change could be 0 or +1 (if we added back e_to_remove too).
                    break
        
        # If we are stuck, maybe do a small "shake"
        # But the above loop effectively does random restarts of local regions.

    # Final check and cleanup
    # Ensure no conflicts remain (should be none by construction)
    
    result = []
    for (u, v), c in edge_colors.items():
        result.append([u, v, c])
        
    return result
