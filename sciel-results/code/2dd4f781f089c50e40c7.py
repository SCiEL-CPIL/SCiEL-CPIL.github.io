import time
import random
import math
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLOR = 119

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

# Precompute all 4-cycles
CYCLES = []
for a in range(N):
    for b in range(a+1, N):
        for c in range(b+1, N):
            for d in range(c+1, N):
                # Cycle 1: a-b, b-c, c-d, d-a
                CYCLES.append((edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a)))
                # Cycle 2: a-b, b-d, d-c, c-a
                CYCLES.append((edge_idx(a,b), edge_idx(b,d), edge_idx(d,c), edge_idx(c,a)))
                # Cycle 3: a-c, c-b, b-d, d-a
                CYCLES.append((edge_idx(a,c), edge_idx(c,b), edge_idx(b,d), edge_idx(d,a)))

# Precompute for each edge, which cycles it belongs to
EDGE_CYCLES = [[] for _ in range(NUM_EDGES)]
for ci, cyc in enumerate(CYCLES):
    for e in cyc:
        EDGE_CYCLES[e].append(ci)

# Precompute cycle edge tuples as numpy arrays for faster access? 
# Actually, lists are fine for small sizes. Let's stick to lists for simplicity in the tight loop.

def count_distinct_colors(colors):
    return len(set(colors))

def is_valid_fast(colors, changed_edge):
    """Check validity only for cycles containing the changed edge."""
    for ci in EDGE_CYCLES[changed_edge]:
        cyc = CYCLES[ci]
        c0 = colors[cyc[0]]
        c1 = colors[cyc[1]]
        c2 = colors[cyc[2]]
        c3 = colors[cyc[3]]
        # If all 4 are distinct, it's invalid
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s limit
    
    best_score = 0
    best_colors = None
    
    # We will run multiple independent trials
    # Each trial: 
    # 1. Greedy construction with randomization
    # 2. Local search refinement (hill climbing / SA)
    
    # To maximize distinct colors, we want to avoid reusing colors if possible.
    # The constraint is "no C4 with 4 distinct colors".
    # This is equivalent to: For any C4, the 4 edges must share at least one color.
    
    # Strategy: 
    # Instead of starting with a valid coloring and trying to add colors,
    # we can try to assign colors to edges such that we maximize the number of unique colors.
    
    # Let's use a "Greedy + Local Search" hybrid.
    
    # Precompute a list of unused colors to pick from
    all_colors = list(range(120))
    
    trial_count = 0
    
    while time.time() < deadline:
        trial_count += 1
        # Randomize the color assignment order or initial state
        random.shuffle(all_colors)
        
        # Phase 1: Greedy Construction
        # We want to assign colors to all 120 edges.
        # To maximize distinct colors, we should try to use new colors first.
        # However, we must maintain validity.
        
        # Let's try a different greedy approach:
        # Sort edges by some heuristic (e.g., degree in cycle hypergraph) 
        # or just random order.
        
        edge_order = list(range(NUM_EDGES))
        random.shuffle(edge_order)
        
        colors = [0] * NUM_EDGES
        used_colors = set()
        
        # We will try to assign the next available unused color to each edge.
        # If it leads to a violation, we backtrack or try an existing color.
        # Since backtracking in Python for 120 edges can be slow, we use a simpler heuristic:
        # Assign a color, if invalid, try another. If all fail, keep the best valid one.
        
        # Actually, a simpler robust approach:
        # Start with a known valid coloring (e.g., all 0s)
        # Then, iteratively pick a random edge and try to change it to a NEW unused color.
        # If valid, accept. If not, try to change it to a RANDOM existing color (to move out of local opt).
        # This is essentially the current program but with a stronger bias towards new colors.
        
        # Let's improve the current program's strategy:
        # 1. Start with all 0s.
        # 2. Loop:
        #    a. Pick a random edge.
        #    b. If there are unused colors, pick one at random.
        #    c. Check if assigning this color to the edge is valid.
        #    d. If valid, accept and mark color as used.
        #    e. If invalid, try a few random existing colors to see if we can improve diversity elsewhere? 
        #       No, just skip.
        #    f. If no unused colors left, stop or do local search to shuffle colors.
        
        # The problem with the current program is it gets stuck at 16. 
        # Why 16? 
        # Maybe the greedy "try new color" fails often, so it stops adding colors early.
        # We need a smarter way to introduce new colors.
        
        # Alternative Strategy: "Color Packing"
        # We want to partition the 120 edges into sets, each set having a unique color, 
        # such that no C4 contains edges from 4 different sets.
        
        # Let's try a "Simulated Annealing" but with a different objective:
        # Objective: Maximize sum of (1 if color is new else 0) 
        # Actually, just maximizing distinct count is fine.
        
        # The key improvement: **Restarting** and **Diversity**.
        
        current_colors = [0] * NUM_EDGES
        used_colors = {0}
        
        # Randomize initial state slightly? No, 0s is a good base.
        # But maybe we should start with a random valid coloring?
        # Generating a random valid coloring is hard.
        
        # Let's stick to the "Incremental Addition" but make it smarter.
        # We maintain a list of "candidate edges" that haven't been assigned a unique color yet?
        # No, every edge has a color.
        
        # Let's try this:
        # 1. Start with all 0.
        # 2. Repeat until no progress or time out:
        #    - Pick a random edge e.
        #    - Pick a random unused color c.
        #    - If assigning c to e is valid, do it.
        #    - If not, try to find an edge e2 such that swapping colors of e and e2 (or changing e2) allows c to be used?
        #       Too complex.
        
        # Let's try a **Local Search** that specifically targets increasing the number of distinct colors.
        # Move: Change color of edge e from c1 to c2.
        # Gain: +1 if c2 is new and c1 is not used elsewhere.
        #       -1 if c1 is used only here and c2 is not new (reduces diversity? No, c2 exists, c1 disappears).
        #       0 otherwise.
        # Constraint: Validity.
        
        # We will use a "Hill Climbing" with random restarts.
        
        T = 10.0
        T_min = 0.01
        alpha = 0.999
        
        # To avoid getting stuck in the same local optimum as the seed,
        # we will randomize the order of edge selection and color selection.
        
        # Also, we will periodically "shuffle" the colors of a random subset of edges to escape local minima.
        
        for step in range(5000): # Max steps per trial
            if time.time() > deadline:
                break
                
            # Pick random edge
            e = random.randint(0, NUM_EDGES - 1)
            old_color = current_colors[e]
            
            # Decide new color
            # 80% chance: try an unused color
            # 20% chance: try a random used color (to mix)
            
            unused = [c for c in range(120) if c not in used_colors]
            
            if random.random() < 0.8 and unused:
                new_color = random.choice(unused)
            else:
                new_color = random.randint(0, 119)
                
            if new_color == old_color:
                continue
                
            # Try the change
            current_colors[e] = new_color
            
            if is_valid_fast(current_colors, e):
                # Calculate score change
                old_score = len(used_colors)
                
                # Update used_colors
                if new_color not in used_colors:
                    used_colors.add(new_color)
                if current_colors.count(old_color) == 0:
                    used_colors.discard(old_color)
                    
                new_score = len(used_colors)
                
                # Acceptance
                if new_score >= old_score:
                    # Always accept improvement or neutral
                    pass
                else:
                    # Deterioration: accept with prob exp(delta/T)
                    if random.random() < math.exp((new_score - old_score) / T):
                        pass
                    else:
                        # Reject
                        current_colors[e] = old_color
                        if new_color in used_colors and current_colors.count(new_color) == 0:
                            used_colors.discard(new_color)
                        if old_color not in used_colors:
                            used_colors.add(old_color)
                        T *= alpha
                        continue
            else:
                # Invalid
                current_colors[e] = old_color
                T *= alpha
                continue
            
            # Update best
            if len(used_colors) > best_score:
                best_score = len(used_colors)
                best_colors = current_colors[:]
            
            # Cooling
            T *= alpha
            if T < T_min:
                T = T_min
                # Reheat occasionally
                if random.random() < 0.1:
                    T = 5.0
        
        # End of trial. 
        # We can also try a "Polish" phase here if we have time.
        
    if best_colors is None:
        # Fallback
        return [0] * NUM_EDGES
        
    return best_colors

# Verification helper (not used in construct, but good for debugging)
def verify(colors):
    for cyc in CYCLES:
        c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
        if len({c0, c1, c2, c3}) == 4:
            return False
    return True

if __name__ == "__main__":
    res = construct()
    print("Score:", len(set(res)))
    print("Valid:", verify(res))
