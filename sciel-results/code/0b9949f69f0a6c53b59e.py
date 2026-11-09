import time
import random
import itertools

N = 16
NUM_EDGES = N * (N - 1) // 2

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # Precompute edge indices
    edge_to_idx = {}
    idx_to_edge = []
    for i in range(N):
        for j in range(i + 1, N):
            edge_to_idx[(i, j)] = len(idx_to_edge)
            idx_to_edge.append((i, j))
    
    # Precompute 4-cycles
    cycles = []
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    
    for v1, v2, v3, v4 in itertools.combinations(range(N), 4):
        e12 = edge_to_idx[(v1, v2)]
        e13 = edge_to_idx[(v1, v3)]
        e14 = edge_to_idx[(v1, v4)]
        e23 = edge_to_idx[(v2, v3)]
        e24 = edge_to_idx[(v2, v4)]
        e34 = edge_to_idx[(v3, v4)]
        
        c1 = [e12, e23, e34, e14]
        cycles.append(c1)
        for e in c1:
            edge_to_cycles[e].append(len(cycles) - 1)
            
        c2 = [e12, e24, e34, e13]
        cycles.append(c2)
        for e in c2:
            edge_to_cycles[e].append(len(cycles) - 1)
            
        c3 = [e13, e23, e24, e14]
        cycles.append(c3)
        for e in c3:
            edge_to_cycles[e].append(len(cycles) - 1)
    
    # Precompute for each edge, the other edges in each cycle it belongs to
    edge_to_other_edges = [[] for _ in range(NUM_EDGES)]
    for ci, cyc in enumerate(cycles):
        for k in range(4):
            e_idx = cyc[k]
            others = [cyc[j] for j in range(4) if j != k]
            edge_to_other_edges[e_idx].append(tuple(others))
    
    def get_allowed_colors(colors, e_idx):
        """Get the set of colors that can be assigned to edge e_idx without creating a violation."""
        allowed = None
        for others in edge_to_other_edges[e_idx]:
            c1 = colors[others[0]]
            c2 = colors[others[1]]
            c3 = colors[others[2]]
            
            # If c1, c2, c3 are all distinct, new color must be in {c1, c2, c3}
            if c1 != c2 and c1 != c3 and c2 != c3:
                s = {c1, c2, c3}
                if allowed is None:
                    allowed = s
                else:
                    allowed = allowed.intersection(s)
                    if not allowed:
                        break
        
        if allowed is None:
            return set(range(120))
        return allowed
    
    def is_valid(colors):
        """Check if no 4-cycle has 4 distinct colors."""
        for cyc in cycles:
            c0 = colors[cyc[0]]
            c1 = colors[cyc[1]]
            c2 = colors[cyc[2]]
            c3 = colors[cyc[3]]
            if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
                return False
        return True
    
    def count_colors(colors):
        return len(set(colors))
    
    random.seed(42)
    
    # Structural initialization: partition vertices into groups
    def smart_initial():
        colors = [0] * NUM_EDGES
        # Try different partition sizes
        # Partition 16 vertices into 4 groups of 4
        groups = [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15]]
        vertex_group = [0]*N
        for gi, g in enumerate(groups):
            for v in g:
                vertex_group[v] = gi
        
        # Color edges: within-group edges get colors 0-3, between-group edges get unique colors
        color_counter = 4
        for i in range(N):
            for j in range(i+1, N):
                eidx = edge_to_idx[(i,j)]
                gi = vertex_group[i]
                gj = vertex_group[j]
                if gi == gj:
                    colors[eidx] = (i + j) % 4
                else:
                    colors[eidx] = color_counter
                    color_counter += 1
                    if color_counter >= 120:
                        color_counter = 4
        return colors
    
    # Try multiple structural initializations with random permutations
    best_score = -1
    best_colors = None
    
    for attempt in range(20):
        if time.time() > deadline - 30:
            break
        
        colors = smart_initial()
        # Apply random permutation to break symmetry
        perm = list(range(120))
        random.shuffle(perm)
        colors = [perm[c] for c in colors]
        
        if is_valid(colors):
            score = count_colors(colors)
            if score > best_score:
                best_score = score
                best_colors = colors[:]
    
    if best_colors is None:
        # Fallback: all 0s
        best_colors = [0] * NUM_EDGES
        best_score = 1
    
    current = best_colors[:]
    current_score = best_score
    current_used = set(current)
    
    # Local search with sideways moves
    stagnation_counter = 0
    max_stagnation_passes = 15
    
    while time.time() < deadline - 5:
        improved_in_pass = False
        best_score_in_pass = current_score
        
        edges_order = list(range(NUM_EDGES))
        random.shuffle(edges_order)
        
        for e_idx in edges_order:
            if time.time() > deadline - 5:
                break
                
            allowed = get_allowed_colors(current, e_idx)
            if not allowed:
                continue
                
            current_color = current[e_idx]
            
            # Prefer unused colors to increase score
            candidates = [c for c in allowed if c not in current_used]
            
            if candidates:
                new_color = random.choice(candidates)
            else:
                new_color = random.choice(list(allowed))
                
            if new_color == current_color:
                continue
                
            old_color = current[e_idx]
            current[e_idx] = new_color
            
            new_used = set(current)
            new_score = len(new_used)
            
            if new_score > current_score:
                current_score = new_score
                current_used = new_used
                improved_in_pass = True
                if current_score > best_score:
                    best_score = current_score
                    best_colors = current[:]
            elif new_score == current_score:
                current_used = new_used
                improved_in_pass = True
            else:
                current[e_idx] = old_color
                
        if not improved_in_pass:
            break
            
        if current_score > best_score:
            stagnation_counter = 0
        else:
            stagnation_counter += 1
            
        if stagnation_counter > max_stagnation_passes:
            # Perturb: change a few edges to random allowed colors
            for _ in range(10):
                e_idx = random.randint(0, NUM_EDGES - 1)
                allowed = get_allowed_colors(current, e_idx)
                if allowed:
                    current[e_idx] = random.choice(list(allowed))
            
            current_used = set(current)
            current_score = len(current_used)
            
            if current_score > best_score:
                best_score = current_score
                best_colors = current[:]
            
            stagnation_counter = 0

    return best_colors
