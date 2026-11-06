import random
import time
import itertools
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    n = 20
    num_colors = 20
    
    # Initialize complete graph with random colours
    edge_colour = {}
    for u in range(n):
        for v in range(u+1, n):
            key = (u, v)
            edge_colour[key] = random.randint(0, num_colors - 1)
        
    # Generate all 4-cycles
    # A 4-cycle is defined by 4 distinct vertices.
    # For each set of 4 vertices, there are 3 cycles.
    cycles = []
    vertices = list(range(n))
    for comb in itertools.combinations(vertices, 4):
        u, v, w, x = comb
        # Cycle 1: u-v-w-x-u
        cycles.append([(u,v), (v,w), (w,x), (x,u)])
        # Cycle 2: u-v-x-w-u
        cycles.append([(u,v), (v,x), (x,w), (w,u)])
        # Cycle 3: u-w-v-x-u
        cycles.append([(u,w), (w,v), (v,x), (x,u)])
    
    def get_edge_key(u, v):
        return (min(u, v), max(u, v))
    
    def is_rainbow(cycle):
        colours = []
        for e in cycle:
            u, v = e
            key = get_edge_key(u, v)
            if key not in edge_colour:
                return None  # Edge doesn't exist
            colours.append(edge_colour[key])
        return len(set(colours)) == 4
    
    # Greedy deletion: find bad cycles, remove edge in most bad cycles
    while time.time() < deadline:
        # Find all bad cycles
        bad_cycles = []
        for cycle in cycles:
            rainbow = is_rainbow(cycle)
            if rainbow is None:
                continue  # Edge missing, skip
            if not rainbow:
                bad_cycles.append(cycle)
        
        if not bad_cycles:
            break
        
        # Count how many bad cycles each edge participates in
        edge_bad_count = Counter()
        for cycle in bad_cycles:
            for e in cycle:
                key = get_edge_key(e[0], e[1])
                edge_bad_count[key] += 1
        
        # Remove the edge that appears in the most bad cycles
        if edge_bad_count:
            # Pick the edge with highest count (break ties randomly)
            max_count = max(edge_bad_count.values())
            candidates = [key for key, count in edge_bad_count.items() if count == max_count]
            e_to_remove = random.choice(candidates)
            del edge_colour[e_to_remove]
    
    # Reconstruct list of edges from edge_colour
    result = []
    for (u, v), c in edge_colour.items():
        result.append([u, v, c])
        
    return result
