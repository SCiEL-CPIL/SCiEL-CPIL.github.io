import time
import random
import itertools
from collections import Counter

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLOR = 120

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # 1. Precomputation
    # Map edge (i,j) to index
    edge_to_idx = {}
    idx_to_edge = []
    for i in range(N):
        for j in range(i + 1, N):
            edge_to_idx[(i, j)] = len(idx_to_edge)
            idx_to_edge.append((i, j))
    
    # Precompute all 4-cycles
    # Each cycle is a tuple of 4 edge indices.
    cycles = []
    for v1, v2, v3, v4 in itertools.combinations(range(N), 4):
        e12 = edge_to_idx[(v1, v2)]
        e13 = edge_to_idx[(v1, v3)]
        e14 = edge_to_idx[(v1, v4)]
        e23 = edge_to_idx[(v2, v3)]
        e24 = edge_to_idx[(v2, v4)]
        e34 = edge_to_idx[(v3, v4)]
        
        # Cycle 1: v1-v2-v3-v4-v1
        cycles.append((e12, e23, e34, e14))
        # Cycle 2: v1-v2-v4-v3-v1
        cycles.append((e12, e24, e34, e13))
        # Cycle 3: v1-v3-v2-v4-v1
        cycles.append((e13, e23, e24, e14))
    
    # For each edge, store the list of cycle indices it belongs to.
    # Also store the positions (0,1,2,3) in those cycles to help with constraint checking.
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cycle in enumerate(cycles):
        for pos, e_idx in enumerate(cycle):
            edge_to_cycles[e_idx].append((ci, pos))
    
    # Helper to check validity of a coloring
    def is_valid(coloring):
        for c in cycles:
            c1, c2, c3, c4 = coloring[c[0]], coloring[c[1]], coloring[c[2]], coloring[c[3]]
            if len({c1, c2, c3, c4}) == 4:
                return False
        return True

    # Helper to get allowed colors for an edge given current colors
    # This is crucial for fast local search.
    def get_allowed_colors(e_idx, colors):
        allowed = set(range(MAX_COLOR))
        for ci, pos in edge_to_cycles[e_idx]:
            c = cycles[ci]
            # Get the other 3 edges in this cycle
            other_pos = [p for p in range(4) if p != pos]
            c1 = colors[c[other_pos[0]]]
            c2 = colors[c[other_pos[1]]]
            c3 = colors[c[other_pos[2]]]
            
            # If c1, c2, c3 are all distinct, the new color MUST be one of them
            if c1 != c2 and c1 != c3 and c2 != c3:
                # Intersect allowed with {c1, c2, c3}
                if len(allowed) > 3:
                    allowed.intersection_update({c1, c2, c3})
                else:
                    # Optimization: if allowed is small, just check intersection
                    if c1 not in allowed and c2 not in allowed and c3 not in allowed:
                        return set()
                    new_allowed = set()
                    if c1 in allowed: new_allowed.add(c1)
                    if c2 in allowed: new_allowed.add(c2)
                    if c3 in allowed: new_allowed.add(c3)
                    allowed = new_allowed
                if not allowed:
                    return set()
        return allowed

    # 2. Hierarchical Clustering Construction
    # We build a hierarchical clustering (dendrogram) of the 16 vertices.
    # Each internal node corresponds to a "cluster" and is assigned a unique color.
    # The edges connecting vertices in different child clusters of a node are assigned that node's color.
    
    def hierarchical_construct():
        # Start with 16 singletons
        cluster_vertices = [set([i]) for i in range(N)]
        active_clusters = list(range(N))
        color_map = {}  # edge -> color
        next_color = 0
        
        # Random merge order
        for _ in range(N - 1):
            if time.time() > deadline:
                break
            # Pick two random active clusters
            i = random.randint(0, len(active_clusters) - 1)
            j = random.randint(0, len(active_clusters) - 1)
            while j == i:
                j = random.randint(0, len(active_clusters) - 1)
            
            c1 = active_clusters[i]
            c2 = active_clusters[j]
            
            # Assign new color to edges between c1 and c2
            new_color = next_color
            next_color += 1
            
            for v1 in cluster_vertices[c1]:
                for v2 in cluster_vertices[c2]:
                    e_idx = edge_to_idx[(min(v1, v2), max(v1, v2))]
                    color_map[e_idx] = new_color
            
            # Merge c1 and c2 into c1
            cluster_vertices[c1].update(cluster_vertices[c2])
            active_clusters.remove(c2)
            active_clusters.remove(c1)
            active_clusters.append(c1)
            
        # Initialize color array
        colors = [0] * NUM_EDGES
        for e_idx, c in color_map.items():
            colors[e_idx] = c
            
        # Check validity (should be valid)
        if not is_valid(colors):
            return None
            
        return colors

    # 3. Local Search Improvement
    def local_search(initial_colors, time_limit):
        colors = initial_colors[:]
        freq = [0] * MAX_COLOR
        for c in colors:
            freq[c] += 1
        
        used_colors = set(c for c in colors if freq[c] > 0)
        best_colors = colors[:]
        best_score = len(used_colors)
        
        end_time = time.time() + time_limit
        stall_count = 0
        max_stall = 2000
        
        while time.time() < end_time:
            # Randomly select an edge to try recoloring
            e = random.randint(0, NUM_EDGES - 1)
            allowed = get_allowed_colors(e, colors)
            
            if not allowed:
                continue
                
            current_color = colors[e]
            
            # Prioritize introducing new colors
            unused_in_allowed = [c for c in allowed if freq[c] == 0]
            
            if unused_in_allowed:
                new_color = random.choice(unused_in_allowed)
                # Apply move
                freq[current_color] -= 1
                if freq[current_color] == 0:
                    used_colors.discard(current_color)
                
                colors[e] = new_color
                freq[new_color] += 1
                used_colors.add(new_color)
                
                score = len(used_colors)
                if score > best_score:
                    best_score = score
                    best_colors = colors[:]
                stall_count = 0
            else:
                # Neutral move
                if stall_count < max_stall:
                    candidates = [c for c in allowed if c != current_color]
                    if candidates:
                        new_color = random.choice(candidates)
                    else:
                        new_color = current_color
                        
                    if new_color != current_color:
                        freq[current_color] -= 1
                        if freq[current_color] == 0:
                            used_colors.discard(current_color)
                        
                        colors[e] = new_color
                        freq[new_color] += 1
                        used_colors.add(new_color)
                        
                        score = len(used_colors)
                        if score > best_score:
                            best_score = score
                            best_colors = colors[:]
                    stall_count += 1
                else:
                    # Restart from best_colors
                    if random.random() < 0.05:
                        colors = best_colors[:]
                        freq = [0] * MAX_COLOR
                        for c in colors:
                            freq[c] += 1
                        used_colors = set(c for c in colors if freq[c] > 0)
                        stall_count = 0
        
        return best_colors, best_score

    # 4. Main Loop
    best_overall = [0] * NUM_EDGES
    best_overall_score = 0
    
    # Try hierarchical construction multiple times with different random seeds/orders
    # Then improve with local search
    
    # Pre-generate a few base colorings to avoid time spent on generation inside loop?
    # Or just generate on the fly.
    while time.time() < deadline:
        # Get a base coloring from hierarchical construction
        base_colors = hierarchical_construct()
        if base_colors is None:
            continue
            
        # Check score
        used = set(base_colors)
        base_score = len(used)
        
        if base_score > best_overall_score:
            best_overall = base_colors[:]
            best_overall_score = base_score
            
        # Allocate remaining time to local search
        remaining = deadline - time.time()
        if remaining < 5:
            break
            
        # Use a significant fraction of remaining time for local search
        # We want to spend more time searching than generating
        search_time = remaining * 0.7
        improved_colors, improved_score = local_search(base_colors, search_time)
        
        if improved_score > best_overall_score:
            best_overall = improved_colors
            best_overall_score = improved_score
            
        # If we have plenty of time, we can restart
        if time.time() > deadline - 2:
            break

    return best_overall
