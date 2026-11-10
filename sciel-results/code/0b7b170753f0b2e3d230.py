import time
import itertools
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

# Precompute edge map
def build_edge_map():
    edge_map = {}
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edge_map[(i, j)] = idx
            idx += 1
    return edge_map

EDGE_MAP = build_edge_map()

# Precompute for each edge, the list of 4-cycles containing it
# A 4-cycle containing edge (i,j) has form i-j-k-l-i where k,l are distinct from i,j and from each other
def build_cycle_map():
    """For each edge index, precompute the list of (other_edge_indices, positions) for all 4-cycles containing it."""
    cycle_map = {}
    for (i, j), idx in EDGE_MAP.items():
        cycles = []
        others = [v for v in range(N) if v != i and v != j]
        for k in others:
            for l in others:
                if l == k:
                    continue
                # 4-cycle: i-j-k-l-i
                # Edges: (i,j)=idx, (j,k), (k,l), (l,i)
                e2 = EDGE_MAP.get((j, k), EDGE_MAP.get((k, j)))
                e3 = EDGE_MAP.get((k, l), EDGE_MAP.get((l, k)))
                e4 = EDGE_MAP.get((l, i), EDGE_MAP.get((i, l)))
                cycles.append((e2, e3, e4))
        cycle_map[idx] = cycles
    return cycle_map

CYCLE_MAP = build_cycle_map()

def _count_distinct(colors, edge_idx, new_color):
    """Count how many 4-cycles containing edge_idx would have 4 distinct colors if we set colors[edge_idx]=new_color."""
    violations = 0
    for (e2, e3, e4) in CYCLE_MAP[edge_idx]:
        c2 = colors[e2]
        c3 = colors[e3]
        c4 = colors[e4]
        if len({new_color, c2, c3, c4}) == 4:
            violations += 1
    return violations

def _is_valid(colors):
    """Check if the entire coloring is valid (no 4-cycle has 4 distinct colors)."""
    for idx in range(NUM_EDGES):
        if _count_distinct(colors, idx, colors[idx]) > 0:
            return False
    return True

def _num_colors(colors):
    return len(set(colors))

def _initial_valid_coloring():
    """Create an initial valid coloring using a structured approach."""
    colors = [0] * NUM_EDGES
    # Use a simple algebraic structure: color edge (i,j) based on (i+j) mod some number
    # This tends to create few distinct colors but is valid
    for i in range(N):
        for j in range(i + 1, N):
            idx = EDGE_MAP[(i, j)]
            colors[idx] = (i + j) % 5
    return colors

def _local_search(colors, time_limit):
    """Perform local search to maximize number of distinct colors while maintaining validity."""
    colors = list(colors)
    best_colors = list(colors)
    best_score = _num_colors(colors)
    
    deadline = time.time() + time_limit
    rng = random.Random(42)
    
    # Build list of edges
    edge_list = list(range(NUM_EDGES))
    
    # Try to improve by changing colors
    stuck_counter = 0
    while time.time() < deadline:
        # Pick a random edge
        edge_idx = rng.choice(edge_list)
        current_color = colors[edge_idx]
        
        # Try to assign a new color that increases diversity
        # First, try unused colors
        used = set(colors)
        available = [c for c in range(120) if c not in used]
        
        if available:
            # Try a few random unused colors
            candidates = rng.sample(available, min(3, len(available)))
        else:
            candidates = []
        
        # Also try swapping with another edge's color
        other_edge = rng.choice(edge_list)
        if other_edge != edge_idx:
            candidates.append(colors[other_edge])
        
        best_candidate = None
        best_cand_score = -1
        
        for new_color in candidates:
            if new_color == current_color:
                continue
            viol = _count_distinct(colors, edge_idx, new_color)
            if viol == 0:
                # This is valid
                # Score: prefer colors that add new distinct colors
                if new_color not in set(colors):
                    score = 1000 + rng.random()
                else:
                    score = rng.random()
                if score > best_cand_score:
                    best_cand_score = score
                    best_candidate = new_color
        
        if best_candidate is not None:
            colors[edge_idx] = best_candidate
            new_score = _num_colors(colors)
            if new_score > best_score:
                best_score = new_score
                best_colors = list(colors)
                stuck_counter = 0
            else:
                stuck_counter += 1
        else:
            stuck_counter += 1
        
        # If stuck for too long, try a different strategy
        if stuck_counter > 200:
            stuck_counter = 0
            # Try to find an edge that can take a new color
            found = False
            for edge_idx in rng.sample(edge_list, min(50, len(edge_list))):
                current_color = colors[edge_idx]
                used = set(colors)
                available = [c for c in range(120) if c not in used]
                if available:
                    for new_color in rng.sample(available, min(5, len(available))):
                        if _count_distinct(colors, edge_idx, new_color) == 0:
                            colors[edge_idx] = new_color
                            new_score = _num_colors(colors)
                            if new_score > best_score:
                                best_score = new_score
                                best_colors = list(colors)
                            found = True
                            break
                if found:
                    break
    
    return best_colors, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Start with initial valid coloring
    colors = _initial_valid_coloring()
    best_colors = list(colors)
    best_score = _num_colors(colors)
    
    # Multi-start local search
    num_starts = 0
    while time.time() < deadline:
        # Create a varied initial coloring
        if num_starts == 0:
            init_colors = _initial_valid_coloring()
        else:
            # Perturb the best coloring
            init_colors = list(best_colors)
            rng = random.Random(num_starts)
            for _ in range(5):
                edge_idx = rng.choice(range(NUM_EDGES))
                new_color = rng.randint(0, 119)
                if _count_distinct(init_colors, edge_idx, new_color) == 0:
                    init_colors[edge_idx] = new_color
        
        # Run local search with remaining time
        remaining = deadline - time.time()
        if remaining < 2.0:
            break
        
        search_time = min(remaining - 1.0, 8.0)  # Cap each search at 8s
        result_colors, result_score = _local_search(init_colors, search_time)
        
        if result_score > best_score:
            best_score = result_score
            best_colors = result_colors
        
        num_starts += 1
        if time.time() > deadline - 2.0:
            break
    
    return best_colors
