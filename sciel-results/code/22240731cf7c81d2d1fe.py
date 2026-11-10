import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 119

# Precompute edges in lexicographic order
EDGES = [(i, j) for i in range(N) for j in range(i + 1, N)]
EDGE_IDX = {}
for idx, (i, j) in enumerate(EDGES):
    EDGE_IDX[(i, j)] = idx
    EDGE_IDX[(j, i)] = idx

# Precompute all 4-cycles and their edge indices
CYCLES = []
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            for d in range(c + 1, N):
                # Cycle 1: a-b-c-d-a
                e1 = EDGE_IDX[(a, b)]
                e2 = EDGE_IDX[(b, c)]
                e3 = EDGE_IDX[(c, d)]
                e4 = EDGE_IDX[(d, a)]
                CYCLES.append((e1, e2, e3, e4))
                # Cycle 2: a-b-d-c-a
                e1 = EDGE_IDX[(a, b)]
                e2 = EDGE_IDX[(b, d)]
                e3 = EDGE_IDX[(d, c)]
                e4 = EDGE_IDX[(c, a)]
                CYCLES.append((e1, e2, e3, e4))
                # Cycle 3: a-c-b-d-a
                e1 = EDGE_IDX[(a, c)]
                e2 = EDGE_IDX[(c, b)]
                e3 = EDGE_IDX[(b, d)]
                e4 = EDGE_IDX[(d, a)]
                CYCLES.append((e1, e2, e3, e4))

NUM_CYCLES = len(CYCLES)  # 3 * C(16,4) = 3 * 1820 = 5460

# For each edge, which cycles does it belong to?
EDGE_TO_CYCLES = [[] for _ in range(NUM_EDGES)]
for ci, cyc in enumerate(CYCLES):
    for ei in cyc:
        EDGE_TO_CYCLES[ei].append(ci)

# Precompute for each edge the other edges in each cycle it belongs to
OTHER_EDGES = [None] * NUM_EDGES
for e in range(NUM_EDGES):
    OTHER_EDGES[e] = []
    for cyc_idx in EDGE_TO_CYCLES[e]:
        cyc = CYCLES[cyc_idx]
        others = tuple(x for x in cyc if x != e)
        OTHER_EDGES[e].append(others)


def count_violations(colors):
    """Count number of 4-cycles with 4 distinct edge colors."""
    violations = 0
    for e1, e2, e3, e4 in CYCLES:
        c1, c2, c3, c4 = colors[e1], colors[e2], colors[e3], colors[e4]
        if len(set((c1, c2, c3, c4))) == 4:
            violations += 1
    return violations


def count_distinct_colors(colors):
    return len(set(colors))


def make_initial_coloring():
    """Create a valid coloring using sum-based coloring.
    Color of edge (i,j) = i+j (range 1..31).
    This guarantees the property because in any 4-cycle, diagonals share a color.
    """
    colors = [0] * NUM_EDGES
    for idx, (i, j) in enumerate(EDGES):
        colors[idx] = i + j
    return colors


def check_edge_change(coloring, edge_idx, new_color):
    """
    Check if changing coloring[edge_idx] to new_color creates a rainbow cycle.
    Returns True if valid, False otherwise.
    """
    if coloring[edge_idx] == new_color:
        return True
    
    for others in OTHER_EDGES[edge_idx]:
        c1 = coloring[others[0]]
        c2 = coloring[others[1]]
        c3 = coloring[others[2]]
        
        # If any two of the other colors are the same, it can't be a rainbow cycle
        if c1 == c2 or c2 == c3 or c1 == c3:
            continue
            
        # If the new color matches any of the other colors, it's not a rainbow cycle
        if new_color == c1 or new_color == c2 or new_color == c3:
            continue
            
        # If we get here, all 4 colors are distinct
        return False
        
    return True


def compute_entropy(coloring):
    """Compute the entropy of the color distribution."""
    from collections import Counter
    counts = Counter(coloring)
    total = len(coloring)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * np.log2(p)
    return entropy


def simulated_annealing(coloring, rng, deadline, max_restarts=10):
    """
    Simulated annealing with entropy-aware acceptance.
    The energy function penalizes violations and rewards color diversity.
    """
    current = list(coloring)
    current_violations = count_violations(current)
    current_distinct = count_distinct_colors(current)
    current_entropy = compute_entropy(current)
    
    best = list(current)
    best_violations = current_violations
    best_distinct = current_distinct
    best_entropy = current_entropy
    
    # Temperature schedule
    T0 = 10.0
    T_min = 0.01
    cooling_rate = 0.995
    
    T = T0
    steps = 0
    max_steps = 50000
    
    # Track recent best to detect stagnation
    recent_best = best_distinct
    stagnation_count = 0
    
    while time.time() < deadline - 2.0 and steps < max_steps:
        # Pick a random edge
        e = rng.randint(0, NUM_EDGES - 1)
        
        # Choose a new color: bias towards unused colors to increase diversity
        if rng.random() < 0.7:
            # Try to introduce a new color
            used_colors = set(current)
            unused = [c for c in range(MAX_COLOR + 1) if c not in used_colors]
            if unused:
                new_color = rng.choice(unused)
            else:
                new_color = rng.randint(0, MAX_COLOR)
        else:
            # Random existing or new color
            new_color = rng.randint(0, MAX_COLOR)
        
        if new_color == current[e]:
            steps += 1
            T *= cooling_rate
            continue
        
        # Check if the change is valid
        if not check_edge_change(current, e, new_color):
            steps += 1
            T *= cooling_rate
            continue
        
        # Compute new metrics
        old_color = current[e]
        current[e] = new_color
        
        new_violations = count_violations(current)
        new_distinct = count_distinct_colors(current)
        new_entropy = compute_entropy(current)
        
        # Energy: minimize violations, maximize distinct colors and entropy
        old_energy = current_violations * 100 - current_distinct * 10 - current_entropy
        new_energy = new_violations * 100 - new_distinct * 10 - new_entropy
        
        delta_energy = new_energy - old_energy
        
        # Acceptance criterion
        if delta_energy < 0 or (T > T_min and rng.random() < np.exp(-delta_energy / T)):
            # Accept the move
            current_violations = new_violations
            current_distinct = new_distinct
            current_entropy = new_entropy
            
            # Update best
            if (current_violations < best_violations or
                (current_violations == best_violations and
                 (current_distinct > best_distinct or
                  (current_distinct == best_distinct and current_entropy > best_entropy)))):
                best = list(current)
                best_violations = current_violations
                best_distinct = current_distinct
                best_entropy = current_entropy
                
                if best_distinct > recent_best:
                    recent_best = best_distinct
                    stagnation_count = 0
                else:
                    stagnation_count += 1
        else:
            # Reject the move
            current[e] = old_color
        
        steps += 1
        T *= cooling_rate
        
        # Restart if stuck too long
        if stagnation_count > 5000 and T > T_min:
            T = max(T * 2, 1.0)
            stagnation_count = 0
        
        # Early exit if perfect
        if best_violations == 0 and best_distinct == MAX_COLOR + 1:
            break
    
    return best, best_distinct


def construct():
    """Return 120 integer colors in 0..119 for lexicographically ordered pairs (i,j), 0<=i<j<16.
    No four-cycle may have four distinct edge colors.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    rng = random.Random(42)
    
    # Start with the known valid sum-based coloring
    best_colors = make_initial_coloring()
    best_score = count_distinct_colors(best_colors)
    
    # Phase 1: Try multiple simulated annealing runs with different initial conditions
    num_runs = 5
    time_per_run = (deadline - start_time - 2.0) / num_runs
    
    for run in range(num_runs):
        run_deadline = start_time + time_per_run * (run + 1)
        
        # Create a diverse initial coloring for this run
        if run == 0:
            initial = make_initial_coloring()
        elif run == 1:
            # Random valid-ish coloring
            initial = [0] * NUM_EDGES
            used = {0}
            for idx, (i, j) in enumerate(EDGES):
                # Use a combination of i and j with some randomness
                base = (i * 3 + j * 5) % (MAX_COLOR + 1)
                if base not in used and rng.random() < 0.5:
                    initial[idx] = base
                    used.add(base)
                else:
                    initial[idx] = rng.choice(list(used))
        else:
            # Perturb the best found so far
            initial = list(best_colors)
            for _ in range(rng.randint(10, 30)):
                e = rng.randint(0, NUM_EDGES - 1)
                new_c = rng.randint(0, MAX_COLOR)
                if check_edge_change(initial, e, new_c):
                    initial[e] = new_c
        
        # Run simulated annealing
        result, score = simulated_annealing(initial, rng, run_deadline)
        
        # Validate
        if count_violations(result) == 0:
            if score > best_score:
                best_score = score
                best_colors = result
        
        # Check time
        if time.time() > deadline - 3.0:
            break
        
        # If we've used all colors, we're done
        if best_score == MAX_COLOR + 1:
            break
    
    # Final safety check
    if count_violations(best_colors) != 0:
        best_colors = make_initial_coloring()
    
    return list(best_colors)
