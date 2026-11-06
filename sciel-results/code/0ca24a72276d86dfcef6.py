import time
import random
import itertools

def construct():
    """
    Constructs a dense graph on 20 vertices with edge colors 0..19
    such that every 4-cycle is rainbow.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 20
    num_colors = 20
    
    # Initialize with complete graph K_20
    edges = set()
    colors = {}
    
    for u in range(n):
        for v in range(u + 1, n):
            edges.add((u, v))
            colors[(u, v)] = random.randint(0, num_colors - 1)
            
    # Precompute all 4-cycles
    all_4_cycles = []
    vertices = list(range(n))
    
    for quad in itertools.combinations(vertices, 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        e1 = (min(a,b), max(a,b))
        e2 = (min(b,c), max(b,c))
        e3 = (min(c,d), max(c,d))
        e4 = (min(d,a), max(d,a))
        if all(e in edges for e in [e1, e2, e3, e4]):
            all_4_cycles.append([e1, e2, e3, e4])
            
        # Cycle 2: a-b-d-c-a
        e1 = (min(a,b), max(a,b))
        e2 = (min(b,d), max(b,d))
        e3 = (min(c,d), max(c,d))
        e4 = (min(a,c), max(a,c))
        if all(e in edges for e in [e1, e2, e3, e4]):
            all_4_cycles.append([e1, e2, e3, e4])
            
        # Cycle 3: a-c-b-d-a
        e1 = (min(a,c), max(a,c))
        e2 = (min(b,c), max(b,c))
        e3 = (min(b,d), max(b,d))
        e4 = (min(a,d), max(a,d))
        if all(e in edges for e in [e1, e2, e3, e4]):
            all_4_cycles.append([e1, e2, e3, e4])

    def is_valid_cycle(cycle):
        c1 = colors[cycle[0]]
        c2 = colors[cycle[1]]
        c3 = colors[cycle[2]]
        c4 = colors[cycle[3]]
        return len({c1, c2, c3, c4}) == 4

    def find_violation():
        for cycle in all_4_cycles:
            if not is_valid_cycle(cycle):
                return cycle
        return None

    while True:
        if time.time() > deadline:
            break
            
        violation = find_violation()
        if violation is None:
            break
            
        fixed = False
        cycle_edges = violation[:]
        random.shuffle(cycle_edges)
        
        for edge in cycle_edges:
            for new_color in range(num_colors):
                if colors[edge] == new_color:
                    continue
                
                other_colors = set()
                for e in violation:
                    if e != edge:
                        other_colors.add(colors[e])
                
                if new_color in other_colors:
                    continue
                
                old_color = colors[edge]
                colors[edge] = new_color
                
                broken = False
                for cycle in all_4_cycles:
                    if edge in cycle:
                        c_others = [colors[e] for e in cycle if e != edge]
                        if new_color in c_others:
                            broken = True
                            break
                            
                if not broken:
                    fixed = True
                    break
                else:
                    colors[edge] = old_color
                    
            if fixed:
                break
        
        if not fixed:
            edge_to_remove = random.choice(cycle_edges)
            del colors[edge_to_remove]
            edges.discard(edge_to_remove)
            
            new_cycles = []
            for cycle in all_4_cycles:
                if all(e in edges for e in cycle):
                    new_cycles.append(cycle)
            all_4_cycles = new_cycles

    result = []
    for u, v in edges:
        c = colors[(u, v)]
        result.append([u, v, c])
        
    return result
