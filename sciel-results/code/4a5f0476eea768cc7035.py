import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def build_edge_index():
    idx = [[-1]*N for _ in range(N)]
    counter = 0
    for i in range(N):
        for j in range(i+1, N):
            idx[i][j] = counter
            idx[j][i] = counter
            counter += 1
    return idx

EDGE_IDX = build_edge_index()

def build_cycles():
    cycles = []
    for a in range(N-3):
        for b in range(a+1, N-2):
            for c in range(b+1, N-1):
                for d in range(c+1, N):
                    cycles.append((EDGE_IDX[a][b], EDGE_IDX[b][c], EDGE_IDX[c][d], EDGE_IDX[d][a]))
                    cycles.append((EDGE_IDX[a][b], EDGE_IDX[b][d], EDGE_IDX[d][c], EDGE_IDX[c][a]))
                    cycles.append((EDGE_IDX[a][c], EDGE_IDX[c][b], EDGE_IDX[b][d], EDGE_IDX[d][a]))
    return cycles

CYCLES = build_cycles()
NUM_CYCLES = len(CYCLES)

EDGE_CYCLES = [[] for _ in range(NUM_EDGES)]
for ci, cycle in enumerate(CYCLES):
    for e in cycle:
        EDGE_CYCLES[e].append(ci)

def is_valid(colors):
    for cycle in CYCLES:
        if len(set(colors[e] for e in cycle)) > 3:
            return False
    return True

def can_take_color(colors, edge, new_color):
    """Check if assigning new_color to edge is valid (no 4-cycle gets 4 distinct colors)."""
    for ci in EDGE_CYCLES[edge]:
        cycle = CYCLES[ci]
        c0 = new_color if cycle[0] == edge else colors[cycle[0]]
        c1 = new_color if cycle[1] == edge else colors[cycle[1]]
        c2 = new_color if cycle[2] == edge else colors[cycle[2]]
        c3 = new_color if cycle[3] == edge else colors[cycle[3]]
        # Check if all four are distinct
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            return False
    return True

def local_search(colors, deadline):
    colors = list(colors)
    used_colors = set(colors)
    best_score = len(used_colors)
    best_colors = list(colors)
    
    no_improve_count = 0
    max_no_improve = 100 # Lower to trigger kicks more often
    consecutive_no_improve = 0
    
    # Precompute list of all colors for shuffling
    all_colors = list(range(120))
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Kick: randomly recolor several edges to escape local optimum
            num_kicks = random.randint(5, 15)
            for _ in range(num_kicks):
                e = random.randint(0, NUM_EDGES - 1)
                shuffled_colors = all_colors[:]
                random.shuffle(shuffled_colors)
                for nc in shuffled_colors:
                    if can_take_color(colors, e, nc):
                        colors[e] = nc
                        break
            
            used_colors = set(colors)
            no_improve_count = 0
            max_no_improve = random.randint(50, 200)
            consecutive_no_improve = 0
            continue

        # Try to improve: find an edge that can take an unused color
        improved = False
        edges = list(range(NUM_EDGES))
        random.shuffle(edges)
        
        for e in edges:
            if time.time() >= deadline:
                break
            
            # Try unused colors first (shuffled for diversity)
            unused = [c for c in range(120) if c not in used_colors]
            if not unused:
                break # No unused colors left to try
            
            random.shuffle(unused)
            
            found = False
            for nc in unused:
                if can_take_color(colors, e, nc):
                    colors[e] = nc
                    used_colors.add(nc)
                    improved = True
                    found = True
                    break
            
            if found:
                no_improve_count = 0
                consecutive_no_improve = 0
                if len(used_colors) > best_score:
                    best_score = len(used_colors)
                    best_colors = list(colors)
            else:
                no_improve_count += 1
        
        if not improved:
            consecutive_no_improve += 1
            # If we've had many consecutive failed passes, do a more aggressive kick
            if consecutive_no_improve >= 3:
                num_aggressive = random.randint(10, 20)
                for _ in range(num_aggressive):
                    e = random.randint(0, NUM_EDGES - 1)
                    shuffled_colors = all_colors[:]
                    random.shuffle(shuffled_colors)
                    for nc in shuffled_colors:
                        if can_take_color(colors, e, nc):
                            colors[e] = nc
                            break
                used_colors = set(colors)
                no_improve_count = 0
                consecutive_no_improve = 0

    return best_colors, best_score

def construct():
    deadline = time.time() + 55  # 55s safety
    
    best_colors = [0] * NUM_EDGES
    best_score = 0
    
    # Strategy 1: Deterministic Star Coloring (Guaranteed 16 colors)
    star_colors = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i+1, N):
            e = EDGE_IDX[i][j]
            if i == 0:
                star_colors[e] = j
            else:
                star_colors[e] = 0
    if is_valid(star_colors):
        best_colors = star_colors
        best_score = len(set(star_colors))
    
    # Generate diverse starting colorings for search
    strategies = [
        "star", "random", "structured", "min_perm"
    ]
    
    num_restarts = 20
    for strat_iter in range(num_restarts):
        if time.time() >= deadline:
            break
        
        strategy = strategies[strat_iter % len(strategies)]
        
        if strategy == "star":
            # Perturb the star coloring slightly
            colors = list(star_colors)
            for _ in range(10):
                e = random.randint(0, NUM_EDGES - 1)
                nc = random.randint(0, 119)
                if can_take_color(colors, e, nc):
                    colors[e] = nc
        elif strategy == "random":
            colors = [random.randint(0, 119) for _ in range(NUM_EDGES)]
        elif strategy == "structured":
            colors = [0] * NUM_EDGES
            for i in range(N):
                for j in range(i+1, N):
                    e = EDGE_IDX[i][j]
                    colors[e] = (i * 7 + j * 3) % 120
        else:  # min_perm
            perm = list(range(N))
            random.shuffle(perm)
            colors = [0] * NUM_EDGES
            for i in range(N):
                for j in range(i+1, N):
                    e = EDGE_IDX[i][j]
                    colors[e] = min(perm[i], perm[j])
        
        if not is_valid(colors):
            # Try to fix by recoloring invalid edges
            fixed = False
            for e in range(NUM_EDGES):
                if time.time() >= deadline:
                    break
                for nc in range(120):
                    if can_take_color(colors, e, nc):
                        colors[e] = nc
                        fixed = True
                        break
            if not is_valid(colors):
                continue
        
        # Run local search
        # Allocate time based on remaining budget
        remaining = deadline - time.time()
        ls_deadline = time.time() + remaining / (num_restarts - strat_iter)
        
        improved_colors, score = local_search(colors, ls_deadline)
        
        if score > best_score:
            best_score = score
            best_colors = improved_colors
            
        # Early stop if we reach a high score
        if best_score >= 35:
            break

    # Final verification
    if not is_valid(best_colors):
        return [0] * NUM_EDGES
        
    return best_colors

# EVOLVE-BLOCK-START
# EVOLVE-BLOCK-END
