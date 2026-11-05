import time
import numpy as np
import random
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLORS = 120

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    # 1. Precompute all 4-cycles and edge indices
    def get_idx(i, j):
        if i > j:
            i, j = j, i
        return i * N - i * (i + 1) // 2 + (j - i - 1)

    # Map edge index to (u, v)
    edge_map = {}
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edge_map[idx] = (i, j)
            idx += 1

    # Precompute cycles
    all_cycles = []
    vertices = list(range(N))

    for quad in combinations(vertices, 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        e1, e2, e3, e4 = get_idx(a, b), get_idx(b, c), get_idx(c, d), get_idx(d, a)
        all_cycles.append((e1, e2, e3, e4))
        # Cycle 2: a-b-d-c-a
        e1, e2, e3, e4 = get_idx(a, b), get_idx(b, d), get_idx(d, c), get_idx(c, a)
        all_cycles.append((e1, e2, e3, e4))
        # Cycle 3: a-c-b-d-a
        e1, e2, e3, e4 = get_idx(a, c), get_idx(c, b), get_idx(b, d), get_idx(d, a)
        all_cycles.append((e1, e2, e3, e4))

    num_cycles = len(all_cycles)
    cycles_arr = np.array(all_cycles, dtype=np.int32)

    # Precompute for each edge, the cycles it belongs to and its position in them
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for c_idx in range(num_cycles):
        e1, e2, e3, e4 = all_cycles[c_idx]
        edge_to_cycles[e1].append((c_idx, 0))
        edge_to_cycles[e2].append((c_idx, 1))
        edge_to_cycles[e3].append((c_idx, 2))
        edge_to_cycles[e4].append((c_idx, 3))

    # Precompute cycle edge positions for fast lookup
    # For each cycle, store the 4 edge indices
    cycle_edges = cycles_arr  # shape (num_cycles, 4)

    # 2. Fast evaluation using precomputed structure
    def count_violations(colors):
        """Count rainbow 4-cycles efficiently."""
        c0 = colors[cycle_edges[:, 0]]
        c1 = colors[cycle_edges[:, 1]]
        c2 = colors[cycle_edges[:, 2]]
        c3 = colors[cycle_edges[:, 3]]
        distinct = (c0 != c1) & (c0 != c2) & (c0 != c3) & (c1 != c2) & (c1 != c3) & (c2 != c3)
        return int(np.sum(distinct))

    def evaluate(colors):
        """Returns (num_unique_colors, num_violations)"""
        num_unique = len(np.unique(colors))
        num_violations = count_violations(colors)
        return num_unique, num_violations

    # 3. Base Construction: Disjoint Triangles
    current_colors = np.zeros(NUM_EDGES, dtype=np.int32)
    best_colors = current_colors.copy()
    best_score = 1
    best_violations = 0

    triangles = []
    remaining = list(range(N))
    while len(remaining) >= 3:
        t = (remaining.pop(), remaining.pop(), remaining.pop())
        triangles.append(t)
        
    color_id = 1
    for u, v, w in triangles:
        e_uv = get_idx(u, v)
        e_vw = get_idx(v, w)
        e_wu = get_idx(w, u)
        current_colors[e_uv] = color_id
        current_colors[e_vw] = color_id + 1
        current_colors[e_wu] = color_id + 2
        color_id += 3
        
    curr_score, curr_violations = evaluate(current_colors)
    if curr_violations == 0:
        best_score = curr_score
        best_colors = current_colors.copy()
        best_violations = 0
    else:
        current_colors = np.zeros(NUM_EDGES, dtype=np.int32)
        best_colors = current_colors.copy()
        best_score = 1
        best_violations = 0
        curr_score = 1
        curr_violations = 0

    # 4. Local Search (Simulated Annealing) with incremental violation tracking
    rng = random.Random(42)
    T = 10.0
    cooling = 0.999
    T_min = 0.01
    
    edge_indices = list(range(NUM_EDGES))
    
    # Track violations incrementally
    # violation_count[cycle_idx] = number of distinct colors in that cycle (0-4)
    # We'll track which cycles are "rainbow" (4 distinct)
    
    def is_rainbow(cycle_idx, colors):
        """Check if a specific cycle has 4 distinct colors."""
        e1, e2, e3, e4 = cycle_edges[cycle_idx]
        c1, c2, c3, c4 = colors[e1], colors[e2], colors[e3], colors[e4]
        return len({c1, c2, c3, c4}) == 4

    def update_violation_delta(cycle_idx, colors):
        """Return whether this cycle is currently rainbow."""
        return is_rainbow(cycle_idx, colors)

    steps = 0
    stall_count = 0
    MAX_STALL = 5000
    
    while time.time() < deadline:
        # Pick a random edge
        e = rng.choice(edge_indices)
        old_c = current_colors[e]
        
        # Pick a new color with bias toward unused
        used_set = set(current_colors.tolist())
        if rng.random() < 0.75 and len(used_set) < MAX_COLORS:
            unused = [c for c in range(MAX_COLORS) if c not in used_set]
            new_c = rng.choice(unused)
        else:
            new_c = rng.randint(0, MAX_COLORS - 1)
            
        if new_c == old_c:
            continue
            
        # Compute violation delta incrementally
        # Only cycles containing edge e are affected
        affected_cycles = edge_to_cycles[e]
        
        old_violations = 0
        for c_idx, pos in affected_cycles:
            if is_rainbow(c_idx, current_colors):
                old_violations += 1
        
        current_colors[e] = new_c
        
        new_violations = 0
        for c_idx, pos in affected_cycles:
            if is_rainbow(c_idx, current_colors):
                new_violations += 1
        
        delta_violations = new_violations - old_violations
        total_violations = curr_violations + delta_violations
        
        new_score = curr_score
        if new_c not in set(current_colors[:e].tolist()) and new_c not in set(current_colors[e+1:].tolist()):
            # This is a rough check; better to track properly
            pass
        # Recompute score only if needed (expensive)
        if total_violations == 0:
            new_score = len(np.unique(current_colors))
        else:
            new_score = curr_score  # approximate
        
        # Check if this is a new best valid solution
        if total_violations == 0 and new_score > best_score:
            best_score = new_score
            best_colors = current_colors.copy()
            best_violations = 0
            stall_count = 0
        elif total_violations == 0:
            stall_count += 1
        
        # SA Acceptance
        old_fitness = curr_score - 100.0 * curr_violations
        new_fitness = new_score - 100.0 * total_violations
        
        delta = new_fitness - old_fitness
        
        if delta >= 0:
            pass
        else:
            if rng.random() < np.exp(delta / T):
                pass
            else:
                current_colors[e] = old_c
                T = max(T_min, T * cooling)
                steps += 1
                continue
        
        curr_score = new_score if total_violations == 0 else curr_score
        curr_violations = total_violations
        T = max(T_min, T * cooling)
        steps += 1

        # If stuck, try a restart perturbation
        if stall_count > MAX_STALL:
            # Perturb: randomly reassign a few edges
            for _ in range(rng.randint(3, 8)):
                pe = rng.choice(edge_indices)
                current_colors[pe] = rng.randint(0, MAX_COLORS - 1)
            curr_score, curr_violations = evaluate(current_colors)
            T = 5.0
            stall_count = 0

        if time.time() > deadline - 2.0:
            break

    # 5. Final polishing pass: try to improve by swapping colors on edges
    if time.time() < deadline - 3.0:
        polish_deadline = min(deadline - 0.5, time.time() + 8.0)
        for _ in range(200):
            if time.time() > polish_deadline:
                break
            # Try to find two edges that can be swapped to reduce violations or increase colors
            e1 = rng.choice(edge_indices)
            e2 = rng.choice(edge_indices)
            if e1 == e2:
                continue
            c1, c2 = current_colors[e1], current_colors[e2]
            if c1 == c2:
                continue
            
            # Try swapping
            current_colors[e1], current_colors[e2] = c2, c1
            new_v = count_violations(current_colors)
            new_s = len(np.unique(current_colors))
            
            if new_v < curr_violations or (new_v == curr_violations and new_s > curr_score):
                curr_score = new_s
                curr_violations = new_v
                if new_v == 0 and new_s > best_score:
                    best_score = new_s
                    best_colors = current_colors.copy()
            else:
                current_colors[e1], current_colors[e2] = c1, c2

    # Final Verification
    final_score, final_violations = evaluate(best_colors)
    if final_violations != 0:
        # Fallback to triangle base
        current_colors = np.zeros(NUM_EDGES, dtype=np.int32)
        color_id = 1
        remaining = list(range(N))
        while len(remaining) >= 3:
            t = (remaining.pop(), remaining.pop(), remaining.pop())
            u, v, w = t
            e_uv = get_idx(u, v)
            e_vw = get_idx(v, w)
            e_wu = get_idx(w, u)
            current_colors[e_uv] = color_id
            current_colors[e_vw] = color_id + 1
            current_colors[e_wu] = color_id + 2
            color_id += 3
        return current_colors.tolist()
        
    return best_colors.tolist()

if __name__ == "__main__":
    colors = construct()
    print(f"Max colors: {len(set(colors))}")
