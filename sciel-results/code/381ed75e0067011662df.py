import time
import random
import itertools

N = 16
NUM_EDGES = N * (N - 1) // 2
DEADLINE = 85  # seconds, safety margin from 90s limit

def edge_index(i, j):
    """Return index of edge (i,j) in lexicographic ordering."""
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

def get_four_cycles():
    """Generate all 4-cycles (as tuples of 4 edge indices) for all 4-vertex subsets."""
    cycles = []
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        # Three 4-cycles on {a,b,c,d}:
        # (a,b,c,d): edges (a,b),(b,c),(c,d),(d,a)
        # (a,b,d,c): edges (a,b),(b,d),(d,c),(c,a)
        # (a,c,b,d): edges (a,c),(c,b),(b,d),(d,a)
        cycles.append((
            edge_index(a,b), edge_index(b,c), edge_index(c,d), edge_index(d,a)
        ))
        cycles.append((
            edge_index(a,b), edge_index(b,d), edge_index(d,c), edge_index(c,a)
        ))
        cycles.append((
            edge_index(a,c), edge_index(c,b), edge_index(b,d), edge_index(d,a)
        ))
    return cycles

def is_valid(colors, cycles):
    """Check if no 4-cycle has four distinct edge colors."""
    for e0, e1, e2, e3 in cycles:
        c0, c1, c2, c3 = colors[e0], colors[e1], colors[e2], colors[e3]
        if len({c0, c1, c2, c3}) == 4:
            return False
    return True

def count_colors(colors):
    return len(set(colors))

def construct():
    start_time = time.time()
    cycles = get_four_cycles()
    num_cycles = len(cycles)
    
    # Pre-compute for each edge, which cycles it participates in
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, (e0, e1, e2, e3) in enumerate(cycles):
        for e in (e0, e1, e2, e3):
            edge_to_cycles[e].append(ci)
    
    # For each cycle, store the positions of each edge
    cycle_edge_positions = []
    for ci, (e0, e1, e2, e3) in enumerate(cycles):
        positions = {}
        for pos, e in enumerate((e0, e1, e2, e3)):
            positions[e] = pos
        cycle_edge_positions.append(positions)
    
    def check_assignment_valid(colors, edge_idx, new_color):
        """Check if assigning new_color to edge_idx keeps all cycles valid."""
        for ci in edge_to_cycles[edge_idx]:
            e0, e1, e2, e3 = cycles[ci]
            pos = cycle_edge_positions[ci][edge_idx]
            # Get the 4 colors with the new assignment
            if pos == 0:
                c0, c1, c2, c3 = new_color, colors[e1], colors[e2], colors[e3]
            elif pos == 1:
                c0, c1, c2, c3 = colors[e0], new_color, colors[e2], colors[e3]
            elif pos == 2:
                c0, c1, c2, c3 = colors[e0], colors[e1], new_color, colors[e3]
            else:
                c0, c1, c2, c3 = colors[e0], colors[e1], colors[e2], new_color
            
            # Check for 4 distinct colors
            if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
                return False
        return True
    
    def try_assign(colors, edge_idx, color_val):
        """Try assigning color_val to edge_idx, return True if valid."""
        return check_assignment_valid(colors, edge_idx, color_val)
    
    # Phase 1: Greedy construction to get a good starting point
    random.seed(42)
    
    best_colors = [0] * NUM_EDGES
    best_count = 1
    
    for restart in range(50):
        if time.time() - start_time > DEADLINE:
            break
        
        colors = [0] * NUM_EDGES
        used_colors = {0}
        
        # Random edge ordering
        edges = list(range(NUM_EDGES))
        random.shuffle(edges)
        
        # Try to assign increasingly new colors
        for e in edges:
            if time.time() - start_time > DEADLINE:
                break
            # Try to assign a new color (max used + 1)
            new_color = max(used_colors) + 1
            if try_assign(colors, e, new_color):
                colors[e] = new_color
                used_colors.add(new_color)
        
        curr_count = count_colors(colors)
        if curr_count > best_count:
            best_count = curr_count
            best_colors = colors[:]
    
    # Phase 2: Simulated annealing with color swaps
    if time.time() - start_time < DEADLINE:
        # Start from best greedy solution
        sa_colors = best_colors[:]
        sa_used = set(sa_colors)
        
        # Temperature schedule
        T_start = 10.0
        T_end = 0.01
        total_iterations = 50000
        cooldown_rate = (T_end / T_start) ** (1.0 / max(1, total_iterations))
        
        T = T_start
        best_sa_count = best_count
        best_sa_colors = sa_colors[:]
        
        # Precompute edge pairs for swap moves
        # We'll swap colors between two edges that don't share a 4-cycle
        # or more generally, try all pairs but check validity
        
        for iteration in range(total_iterations):
            if time.time() - start_time > DEADLINE:
                break
            
            # Choose two random edges to swap colors
            e1 = random.randint(0, NUM_EDGES - 1)
            e2 = random.randint(0, NUM_EDGES - 1)
            
            if e1 == e2:
                continue
            
            c1 = sa_colors[e1]
            c2 = sa_colors[e2]
            
            if c1 == c2:
                continue
            
            # Try the swap: check if both assignments are valid
            # First, check if assigning c2 to e1 is valid (with e2 still having c2)
            # We need to be careful: when checking e1, e2 still has its old color c2
            # When checking e2, e1 still has its old color c1
            
            # Check if c2 can go to e1 (e2 currently has c2)
            valid1 = check_assignment_valid(sa_colors, e1, c2)
            
            # Check if c1 can go to e2 (e1 currently has c1)
            valid2 = check_assignment_valid(sa_colors, e2, c1)
            
            if valid1 and valid2:
                # Perform the swap
                sa_colors[e1] = c2
                sa_colors[e2] = c1
                
                # Update used colors
                if c1 not in sa_colors:
                    sa_used.discard(c1)
                if c2 not in sa_used:
                    sa_used.add(c2)
                
                curr_count = len(sa_used)
                if curr_count > best_sa_count:
                    best_sa_count = curr_count
                    best_sa_colors = sa_colors[:]
            else:
                # Accept with probability exp(-delta/T) where delta is how "bad" it is
                # For simplicity, just reject if invalid
                pass
            
            # Cool down
            T *= cooldown_rate
        
        if best_sa_count > best_count:
            best_count = best_sa_count
            best_colors = best_sa_colors[:]
    
    # Phase 3: Local search polish - try to improve by changing individual edges
    if time.time() - start_time < DEADLINE:
        polish_colors = best_colors[:]
        polish_used = set(polish_colors)
        
        for _ in range(100):
            if time.time() - start_time > DEADLINE:
                break
            
            improved = False
            edges_order = list(range(NUM_EDGES))
            random.shuffle(edges_order)
            
            for e in edges_order:
                if time.time() - start_time > DEADLINE:
                    break
                
                curr = polish_colors[e]
                
                # Try to change to a new color
                new_color = max(polish_used) + 1
                if try_assign(polish_colors, e, new_color):
                    polish_colors[e] = new_color
                    polish_used.add(new_color)
                    improved = True
                
                # Try to change to an existing different color to free up current
                for alt in sorted(polish_used - {curr}):
                    if try_assign(polish_colors, e, alt):
                        polish_colors[e] = alt
                        if curr not in polish_colors:
                            polish_used.discard(curr)
                        improved = True
                        break
            
            if not improved:
                break
        
        curr_count = len(polish_used)
        if curr_count > best_count:
            best_count = curr_count
            best_colors = polish_colors[:]
    
    return best_colors
